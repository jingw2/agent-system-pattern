#-*-coding:utf-8-*-
#author: Jing Wang (零点未来)

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import SystemMessage, HumanMessage

from dotenv import load_dotenv

# load .env file
load_dotenv()

# create LLM 
llm = ChatOpenAI(
    model="deepseek-v4-flash",
    temperature=0
)


def prompt_chain_iter(user_input: str) -> str:
    '''
    Iterate prompts
    '''

    prompt_chain = [
        """
        Extract product information from user input:
        product code, product name, category,
        price, brand, warranty, features.
        """,

        """
        Convert extracted product information into valid JSON.

        Output ONLY JSON.
        """
    ]

    current_input = user_input

    for sys_prompt in prompt_chain:

        output = llm.invoke([
            SystemMessage(content=sys_prompt),
            HumanMessage(content=current_input)
        ])

        current_input = output.content

    return current_input

def prompt_langchain(user_msg: str) -> str: 
    '''Use LSEL to achieve prompt chain'''
    extract_prompt = ChatPromptTemplate.from_template(
        """
            Extract product information from user input:
            product code, product name, category,
            price, brand, warranty, features.
        """
    )
    transform_prompt = ChatPromptTemplate.from_template(
        """
            Convert extracted product information into valid JSON.
            Output ONLY JSON.
        """
    )
    extract_chain = extract_prompt | llm | StrOutputParser()
    full_chain = {"output": extract_chain} | transform_prompt | llm | StrOutputParser()

    response = full_chain.invoke({"input": user_msg})
    return response


if __name__ == "__main__":

    user_msg = """
        I want to buy a FotoSnap Instant Camera (FS-IC10).
        It is a FotoSnap camera in Cameras and Camcorders category,
        priced at $69.99.
    """

    print("Prompt Chain Iteration: ")
    print(prompt_chain_iter(user_msg))
    
    print("Prompt Langchain...")
    print(prompt_langchain(user_msg))