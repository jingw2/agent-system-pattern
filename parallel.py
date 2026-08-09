#-*-coding:utf-8-*-
#author: Jing Wang (零点未来)

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    Runnable,
    RunnableParallel,
    RunnablePassthrough
)


load_dotenv()


llm = ChatOpenAI(
    model="deepseek-v4-flash",
    temperature=0
)



def create_chain(instruction:str) -> Runnable:

    return (
        ChatPromptTemplate.from_messages([
            (
                "system",
                instruction
            ),
            (
                "human",
                "{topic}"
            )
        ])
        | llm
        | StrOutputParser()
    )


# summarize LLM
summarize_chain = create_chain(
    """
    You are an AI research assistant.

    Summarize the topic.
    Include:
    - definition
    - key concepts
    - practical examples
    """
)

# questions LLM
questions_chain = create_chain(
    """
    Generate three important questions
    that someone should understand about this topic.
    """
)

# key terms LLM
terms_chain = create_chain(
    """
    Generate five important technical terms
    related to this topic.
    """
)



parallel_chain = RunnableParallel(
    {
        "summary": summarize_chain,
        "questions": questions_chain,
        "key_terms": terms_chain,
        "topic": RunnablePassthrough()
    }
)



synthesis_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are an expert AI assistant.

        Based on the research notes:

        Summary:
        {summary}

        Questions:
        {questions}

        Key Terms:
        {key_terms}

        Provide a comprehensive answer.
        """
    ),
    (
        "human",
        "{topic}"
    )
])



full_chain = (
    parallel_chain
    | synthesis_prompt
    | llm
    | StrOutputParser()
)



if __name__ == "__main__":

    result = full_chain.invoke(
        {
            "topic":"AI Agent System Design Pattern"
        }
    )

    print(result)