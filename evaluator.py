#-*-coding:utf-8-*-
#author: Jing Wang (零点未来)

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

generator_llm = ChatOpenAI(
    model="deepseek-v4-flash",
    temperature=0.2
)

reviewer_llm = ChatOpenAI(
    model="deepseek-v4-flash",
    temperature=0
)

def evaluator_loop(task_prompt: str, max_iterations:int=3):
    '''evaluator-reviewer loop'''
    current_code = ""
    feedback = ""

    generator_context = [
        SystemMessage(
            content="""
            You are an expert Python engineer.
            Optimize algorithms.
            """
        ),
        HumanMessage(
            content=f'''
             Task: {task_prompt},

             Current Code: {current_code},

             Reviewer feedback:{feedback}
             '''
        )
    ]


    for i in range(max_iterations):
        # Note that if iterations are too big, context window limit is overloaded

        # 1. Generate / improve code
        response = generator_llm.invoke(generator_context)

        current_code = response.content

        # 2. Evaluate
        evaluator_response = reviewer_llm.invoke([
            SystemMessage(
                content="""
                You are a strict competitive programming reviewer.

                Analyze:

                1. Algorithm correctness
                2. Time complexity
                3. Space complexity
                4. Edge cases

                Rules:

                - If any issue exists:
                give concise improvement instructions.

                - Only output PERFECT when:
                - algorithm is optimal
                - complexity satisfies constraints
                - no edge case failure
                """
            ),
            HumanMessage(
                content=f"""
                Task:
                <task>{task_prompt}</task>

                Code:
                <code>
                {current_code}
                </code>
                """
            )
        ])

        feedback = evaluator_response.content

        if feedback.strip()=="PERFECT":
            return current_code

    return current_code

if __name__=="__main__":

    task="""
        Write Python solution for Leetcode Group Anagrams.
        Constraints:
        strs.length <= 10^4
        lowercase English letters
    """

    print(evaluator_loop(task, max_iterations=2))