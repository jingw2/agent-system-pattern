#-*-coding:utf-8-*-
#author: Jing Wang (零点未来)

import json

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

llm = ChatOpenAI(
    model="deepseek-v4-flash",
    temperature=0
)

# ---------------------
# Tools (simple Python functions)
# ---------------------

def lookup_order(order_id: str) -> str:
    '''
    Look up an order by its ID, return order status and amount.
    '''
    fake_db = {
        "ORD-1001": "Shipped, total $129.99",
        "ORD-1002": "Delivered, total $45.50",
        "ORD-1003": "Processing, total $89.00"
    }
    return fake_db.get(order_id, "Order not found")


def get_return_policy() -> str:
    '''
    Return the store return policy as text.
    '''
    return ("Items can be returned within 30 days of delivery "
            "in their original condition for a full refund.")


def initiate_refund(order_id: str) -> str:
    '''
    Initiate a refund for the given order, return confirmation.
    '''
    return f"Refund initiated for {order_id}. Money will be returned in 5-7 business days."


# tool registry: name -> function
TOOL_REGISTRY = {
    "lookup_order": lookup_order,
    "get_return_policy": get_return_policy,
    "initiate_refund": initiate_refund
}


# ---------------------
# Agent loop
# ---------------------

def run_agent(user_query: str, max_iterations: int = 5) -> str:
    '''
    Autonomous single-agent loop: LLM decides tool calls or a final answer.
    Exit when: model gives a final answer, or max_iterations is reached.
    '''
    # keep the whole conversation so the model sees tool feedback
    messages = [
        SystemMessage(
            content="""
            You are a customer service agent.

            You have these tools:
            - lookup_order(order_id): check order status
            - get_return_policy(): get the return policy
            - initiate_refund(order_id): start a refund for an order

            Rules:
            - If you need to call a tool, output ONLY JSON like:
              {"tool": "lookup_order", "args": {"order_id": "ORD-1001"}}
            - Otherwise, output the final answer directly to the customer.

            Do not invent order information; always call lookup_order first.
            """
        ),
        HumanMessage(content=user_query)
    ]

    for i in range(max_iterations):
        # 1. let the LLM decide the next step
        response = llm.invoke(messages)
        content = response.content.strip()

        # 2. try to parse it as a tool-call JSON
        tool_call = _parse_tool_call(content)

        # 3. no tool call -> treat as the final answer and stop the loop
        if tool_call is None:
            return content

        # 4. execute the tool and feed the result back into the loop
        tool_name = tool_call["tool"]
        args = tool_call.get("args", {})

        if tool_name not in TOOL_REGISTRY:
            result = f"Unknown tool: {tool_name}"
        else:
            # guard against malformed args (e.g. missing/extra parameters):
            # feed the error back to the loop instead of crashing the agent
            try:
                result = TOOL_REGISTRY[tool_name](**args)
            except Exception as e:
                result = f"Tool call failed: {e}"

        print(f"[iteration {i+1}] calling {tool_name}({args}) -> {result}")

        messages.append(HumanMessage(content=content))      # the tool-call request
        messages.append(SystemMessage(content=result))      # the tool result (environment feedback)

    # safety net: max iterations reached, return the last output
    # NOTE: when the loop exhausts, messages[-1] is the last tool feedback
    # (SystemMessage), not the model's final summary -- callers should treat
    # it as a truncated result rather than a final answer
    return messages[-1].content


def _parse_tool_call(text: str):
    '''
    Try to parse model output as a tool-call JSON.
    Return a dict {"tool": ..., "args": {...}} or None if not a tool call.
    '''
    # extract the outermost {...} block via the first '{' and the last '}',
    # keeping it robust to extra text around the JSON
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        return None

    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None

    # a tool call must have a "tool" field, and args must be a dict
    if "tool" not in data:
        return None

    args = data.get("args", {})
    if not isinstance(args, dict):
        args = {}

    return {"tool": str(data["tool"]), "args": args}


if __name__ == "__main__":

    query = "I want to return my order ORD-1001. What is your return policy and can I get a refund?"

    print("Customer query:", query)
    print("-" * 60)
    print("Final answer:", run_agent(query, max_iterations=5))
