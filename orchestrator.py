#-*-coding:utf-8-*-
#author: Jing Wang (零点未来)

import json

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()


llm = ChatOpenAI(
    model="deepseek-v4-flash",
    temperature=0
)


# ---------------------
# Planner: dynamically decompose the task into subtasks
# ---------------------

planner_chain = (
    ChatPromptTemplate.from_messages([
        (
            "system",
            """
            You are an AI research task planner.

            Break the given task into 3 to 4 independent subtasks.
            Each subtask should cover one aspect of the task.

            Output ONLY a valid JSON array of strings, for example:
            ["subtask 1", "subtask 2", "subtask 3"]
            """
        ),
        (
            "human",
            "{task}"
        )
    ])
    | llm
    | StrOutputParser()
)


# ---------------------
# Worker: execute a single subtask
# ---------------------

worker_chain = (
    ChatPromptTemplate.from_messages([
        (
            "system",
            """
            You are a research worker LLM.

            Answer the subtask concisely.
            Focus only on the subtask.
            """
        ),
        (
            "human",
            "{subtask}"
        )
    ])
    | llm
    | StrOutputParser()
)


# ---------------------
# Synthesizer: merge all subtask results into the final answer
# ---------------------

synthesizer_chain = (
    ChatPromptTemplate.from_messages([
        (
            "system",
            """
            You are an expert AI assistant.

            Based on the research notes of all subtasks:

            {notes}

            Write a comprehensive and well-organized final answer
            for the original task.
            """
        ),
        (
            "human",
            "Original task: {task}"
        )
    ])
    | llm
    | StrOutputParser()
)


def parse_subtasks(raw: str) -> list[str]:
    '''parse planner JSON output into a list of subtasks'''
    cleaned = raw.strip()
    # strip markdown code fences, e.g. ```json ["a", "b"] ```
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").removeprefix("json").strip()
    try:
        subtasks = json.loads(cleaned)
    except json.JSONDecodeError:
        subtasks = None
    # only accept a list of strings, otherwise fall back to comma splitting
    if not (isinstance(subtasks, list) and all(isinstance(s, str) for s in subtasks)):
        subtasks = [s.strip() for s in cleaned.split(",") if s.strip()]
    return subtasks


def orchestrator_workflow(task: str) -> str:
    '''orchestrator: plan subtasks, run workers, synthesize results'''

    # step 1: planner dynamically decomposes the task
    subtasks = parse_subtasks(planner_chain.invoke({"task": task}))

    # step 2: each worker executes one subtask
    worker_results = []
    for subtask in subtasks:
        worker_results.append(
            worker_chain.invoke({"subtask": subtask})
        )

    # step 3: synthesizer merges all worker outputs
    notes = "\n\n".join(
        f"Subtask: {subtask}\nResult: {result}"
        for subtask, result in zip(subtasks, worker_results)
    )

    final_answer = synthesizer_chain.invoke({
        "task": task,
        "notes": notes
    })

    return final_answer


if __name__ == "__main__":

    task = "AI Agent System Design Pattern"

    print(orchestrator_workflow(task))
