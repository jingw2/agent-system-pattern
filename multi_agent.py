#-*-coding:utf-8-*-
#author: Jing Wang (零点未来)

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage


load_dotenv()


llm = ChatOpenAI(
    model="deepseek-v4-flash",
    temperature=0
)


# ---------------------
# Expert tools
# ---------------------
# each expert owns its own tools; the manager only delegates,
# it never calls these tools directly (agents as tools)

def search_catalog(keyword: str) -> str:
    '''search product catalog by keyword, return matching products'''
    catalog = {
        "tv": "Xiaomi TV A Pro 55 (TV-X1, $499), Hisense 55E7N (TV-H1, $549)",
        "camera": "FotoSnap Instant Camera (FS-IC10, $69.99), Sony Alpha A6700 (CAM-S1, $1299)",
        "phone": "iPhone 16 (PH-I1, $799), Xiaomi 14 Ultra (PH-M1, $899)",
    }
    for key, value in catalog.items():
        if keyword.lower() in key or key in keyword.lower():
            return value
    return f"No products found for: {keyword}"


def place_order(product_id: str) -> str:
    '''place an order for the given product id, return the order number'''
    order_id = "ORD-" + product_id.upper()
    return f"Order {order_id} placed successfully."


def lookup_ticket(ticket_id: str) -> str:
    '''look up a support ticket status'''
    return f"Ticket {ticket_id}: in review, expected reply within 24h."


def initiate_refund(order_id: str) -> str:
    '''start a refund for the given order'''
    return f"Refund for {order_id} initiated, money back in 5-7 business days."


def track_order(order_id: str) -> str:
    '''track the shipping status of an order'''
    return f"Order {order_id}: out for delivery, arrives tomorrow."


# ---------------------
# ExpertAgent: LLM + tools + memory
# ---------------------
# Each expert is a full agent entity: it has its own system
# instructions (memory[0]), its own tools, and its own conversation
# history that persists across calls. This is what distinguishes
# multi-agent from routing: the manager delegates to a living agent
# with state, not to a stateless chain.

class ExpertAgent:
    '''an expert agent with own instructions, tools and memory'''

    def __init__(self, name: str, instructions: str, tools: dict):
        self.name = name
        self.tools = tools
        # memory: first message is the system instructions,
        # following messages keep the conversation history
        self.memory = [SystemMessage(content=instructions)]

    def _run_tool_call(self, text: str) -> str:
        '''parse a {"tool": ..., "args": {...}} JSON and execute the tool'''
        import json

        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            return None

        try:
            call = json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return None

        tool_name = call.get("tool")
        args = call.get("args", {})
        if not isinstance(args, dict):
            args = {}

        if tool_name not in self.tools:
            return f"Unknown tool: {tool_name}"
        try:
            return self.tools[tool_name](**args)
        except Exception as e:
            # guard against malformed args: feed the error back
            # to the loop instead of crashing the agent
            return f"Tool call failed: {e}"

    def run(self, user_msg: str, max_iterations: int = 4) -> str:
        '''handle a user message with tools and memory'''
        # remember what the user said
        self.memory.append(HumanMessage(content=user_msg))

        for i in range(max_iterations):
            response = llm.invoke(self.memory)
            text = response.content

            tool_result = self._run_tool_call(text)
            if tool_result is None:
                # no tool call: this is the final answer,
                # remember it for future turns, then return
                self.memory.append(SystemMessage(content=f"Assistant: {text}"))
                return text

            # tool executed: feed the environment feedback back to the loop
            self.memory.append(
                SystemMessage(content=f"Tool result: {tool_result}")
            )

        # safety net: max iterations reached, force a final answer
        # instead of returning raw tool feedback
        self.memory.append(SystemMessage(
            content="Stop using tools now. "
                    "Give your final answer based on the information you already have."
        ))
        return llm.invoke(self.memory).content


# ---------------------
# Expert agents (each with own domain, tools and memory)
# ---------------------

sales_agent = ExpertAgent(
    name="sales",
    instructions="""
    You are the sales expert.
    Your job: help customers pick products and place orders.
    Tools you can use:
    - search_catalog: find products by keyword
    - place_order: place an order for a product id
    To call a tool, output ONLY JSON: {"tool": "<tool name>", "args": {...}}
    Otherwise output your final answer directly.
    You remember the whole conversation: use earlier context
    (e.g. which product you recommended) in later turns.
    """,
    tools={"search_catalog": search_catalog, "place_order": place_order}
)

support_agent = ExpertAgent(
    name="support",
    instructions="""
    You are the customer support expert.
    Your job: handle returns, refunds and support tickets.
    Tools you can use:
    - lookup_ticket: check a support ticket status
    - initiate_refund: start a refund for an order
    To call a tool, output ONLY JSON: {"tool": "<tool name>", "args": {...}}
    Otherwise output your final answer directly.
    """,
    tools={"lookup_ticket": lookup_ticket, "initiate_refund": initiate_refund}
)

shipping_agent = ExpertAgent(
    name="shipping",
    instructions="""
    You are the shipping expert.
    Your job: answer delivery and logistics questions.
    Tool you can use:
    - track_order: check the shipping status of an order
    To call a tool, output ONLY JSON: {"tool": "<tool name>", "args": {...}}
    Otherwise output your final answer directly.
    """,
    tools={"track_order": track_order}
)

EXPERTS = {
    "sales": sales_agent,
    "support": support_agent,
    "shipping": shipping_agent,
}


# ---------------------
# Manager: coordinates experts via tool calls
# ---------------------

manager_llm = ChatOpenAI(
    model="deepseek-v4-flash",
    temperature=0
)


def manager_workflow(user_msg: str) -> str:
    '''manager decides which expert to delegate to, then calls that agent'''
    # step 1: manager classifies the request (which expert is responsible)
    decision = manager_llm.invoke([
        SystemMessage(content="""
        You are the manager of a customer service team.

        Decide which expert should handle the request.

        Output ONLY one word:

        sales
        support
        shipping
        unclear

        Rules:
        - buying / product questions -> sales
        - returns / refunds / tickets -> support
        - delivery / logistics -> shipping
        - otherwise -> unclear
        """),
        HumanMessage(content=user_msg)
    ]).content.strip().lower()

    # step 2: delegate to the chosen expert agent (agents as tools).
    # the expert runs its own LLM + tools + memory loop;
    # the manager never executes the tools itself.
    if decision not in EXPERTS:
        return ("Sorry, I could not route your request. "
                "Please ask about products, support, or shipping.")

    return EXPERTS[decision].run(user_msg)


if __name__ == "__main__":

    # 1. sales: first turn picks a product with search_catalog
    print("Sales turn 1: ...")
    print(manager_workflow("I want a 55-inch TV under $600, what do you recommend?"))
    print("-" * 50)

    # 2. sales: second turn uses the memory of turn 1 to place the order
    # (proves the expert keeps its own conversation state across calls)
    print("Sales turn 2 (same agent, uses memory): ...")
    print(manager_workflow("OK, buy the one you just recommended."))
    print("-" * 50)

    # 3. support: different expert, its own tools (lookup_ticket)
    print("Support request: ...")
    print(manager_workflow("My camera arrived broken, I want a refund for order ORD-TV-X1."))
    print("-" * 50)

    # 4. shipping: a third expert (track_order)
    print("Shipping request: ...")
    print(manager_workflow("Where is my order ORD-TV-X1?"))
