#-*-coding:utf-8-*-
#author: Jing Wang (零点未来)

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableBranch,
    RunnablePassthrough
)


load_dotenv()


llm = ChatOpenAI(
    model="deepseek-v4-flash",
    temperature=0
)


# ---------------------
# Specialized agents
# ---------------------

query_chain = (
    ChatPromptTemplate.from_messages([
        (
            "system",
            """
            You are a product assistant.
            Answer product related questions.
            """
        ),
        (
            "human",
            "{input}"
        )
    ])
    | llm
    | StrOutputParser()
)


order_chain = (
    ChatPromptTemplate.from_messages([
        (
            "system",
            """
            You help customers place orders.
            Answer order related questions.
            """
        ),
        (
            "human",
            "{input}"
        )
    ])
    | llm
    | StrOutputParser()
)


unclear_chain = (
    ChatPromptTemplate.from_messages([
        (
            "system",
            """
            You handle unclear customer questions.
            Ask user to provide product or order related questions.
            """
        ),
        (
            "human",
            "{input}"
        )
    ])
    | llm
    | StrOutputParser()
)


# ---------------------
# Router
# ---------------------

router_chain = (
    ChatPromptTemplate.from_messages([
        (
            "system",
            """
            Classify customer request.

            Output ONLY one word:

            product
            order
            unclear

            Rules:
            - product questions -> product
            - buying/order questions -> order
            - otherwise -> unclear
            """
        ),
        (
            "human",
            "{input}"
        )
    ])
    | llm
    | StrOutputParser()
)


# ---------------------
# Router workflow
# ---------------------

agent_router = {
    "decision": router_chain,
    "input": RunnablePassthrough()
}


workflow = (
    agent_router
    |
    RunnableBranch(
        (
            lambda x:x["decision"].strip()=="product",
            query_chain
        ),

        (
            lambda x:x["decision"].strip()=="order",
            order_chain
        ),

        unclear_chain
    )
)



def route(user_msg):
    '''run router'''
    return workflow.invoke(user_msg) # if here using {"input": user_msg}, you need to use x['input']['input'] as the real input for specialized agent



if __name__=="__main__":

    print("Query Information: ...")
    print(
        route(
            "What is the average price of a 65-inch Smart LCD TV?"
        )
    )
    print("-" * 50)

    print("Order Information: ...")
    print(
        route(
            "Help me book a flight from Shenzhen to Beijing tomorrow"
        )
    )
    print("-" * 50)


    print("Unclear Information: ...")
    print(
        route(
            "I do not know what to say"
        )
    )
    print("-" * 50)