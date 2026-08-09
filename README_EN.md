# Agent System Design Patterns

> Understand the 7 major design patterns of agent systems through minimal, runnable code examples.

**English | [中文](README.md)**

---

## 📖 About

This repository demonstrates the mainstream design patterns of agent systems, one pattern per file, each a **minimal single-file Python example** that actually runs (real LLM calls). Just `python3 xxx.py`.

The patterns are ordered along the **autonomy spectrum** — from fully controlled code flows to highly autonomous multi-agent systems:

```
Controlled · Predictable ─────────────────────────── Autonomous · Flexible
Prompt Chaining → Routing → Parallelization → Evaluator-Optimizer → Orchestrator-Workers → Single Agent → Multi-Agent
```

**Core insight**: don't build an agent first. Start with the simplest pattern and add complexity only on evidence.

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install langchain-openai langchain-core python-dotenv

# 2. Configure environment (copy the template and fill in your own key)
cp .env.example .env
# then edit .env: OPENAI_API_KEY=your-api-key
# .env is git-ignored and will never be pushed to the repository

# 3. Run any pattern
python3 prompt_chain.py
```

## 🗺️ Pattern Overview

| # | Pattern | File | Core Idea | Autonomy |
|---|---------|------|-----------|----------|
| 1 | Prompt Chaining | [`prompt_chain.py`](prompt_chain.py) | Fixed steps in sequence; each step processes the previous output, with optional gates | Zero |
| 2 | Routing | [`routing.py`](routing.py) | LLM classifies input and dispatches to the best-suited chain | Zero |
| 3 | Parallelization | [`parallel.py`](parallel.py) | Independent subtasks run in parallel, then results are aggregated | Zero |
| 4 | Evaluator-Optimizer | [`evaluator.py`](evaluator.py) | Generator + evaluator loop until the output passes (a.k.a. Reflection) | In-loop |
| 5 | Orchestrator-Workers | [`orchestrator.py`](orchestrator.py) | A central LLM dynamically decomposes tasks, delegates to workers, synthesizes | In-task |
| 6 | Single Agent | [`single_agent.py`](single_agent.py) | LLM + tools act autonomously in a while loop | Full |
| 7 | Multi-Agent | [`multi_agent.py`](multi_agent.py) | A manager LLM coordinates expert agents, each with its own tools and memory | Ceiling |

## 📂 Pattern Details

### 1. Prompt Chaining — `prompt_chain.py`

**What**: decompose a task into a fixed sequence of steps; each LLM call processes the previous step's output, with programmatic gates between steps that stop the flow when something is off.

**When**: the task decomposes cleanly into fixed subtasks; you'd trade latency for accuracy. Errors don't snowball because every step is checked.

**Example**: a user describes a product to buy → step 1 extracts product info (code / name / category / price / brand) → step 2 converts it into valid JSON. Two implementations included: `prompt_chain_iter` (message loop) and `prompt_langchain` (LCEL pipeline).

```bash
python3 prompt_chain.py
```

### 2. Routing — `routing.py`

**What**: a smart switch — the LLM classifies the input and sends it down the most suitable channel (different prompts, models, or flows).

**When**: requests span diverse topics and difficulty, and classification is reliably accurate. Like a hospital triage nurse: doesn't treat, just decides which department you belong to.

**Example**: product support — `router_chain` classifies requests as product / order / unclear, and `RunnableBranch` dispatches to the matching expert chain.

```bash
python3 routing.py
```

### 3. Parallelization — `parallel.py`

**What**: several LLMs work simultaneously and results are aggregated. Two variants: **sectioning** (split into independent subtasks) and **voting** (run the same task multiple times).

**When**: subtasks are genuinely independent and parallelizable, or you need multiple perspectives for confidence. Two benefits only: lower latency, higher quality.

**Example**: researching a topic — `RunnableParallel` runs "summary / questions / key terms" in parallel, then a synthesis chain composes the final answer.

```bash
python3 parallel.py
```

### 4. Evaluator-Optimizer — `evaluator.py`

**What**: a generator plus an evaluator, looping until the evaluation passes — the LLM judges its own work (i.e., the Reflection pattern).

**When**: evaluation criteria must be clearly writable and iteration must measurably improve output. With vague criteria, the loop spins in place.

**Example**: `evaluator_loop` has a generator write a LeetCode solution while the evaluator checks correctness / complexity / edge cases; on failure it rewrites with feedback, until PERFECT or max iterations.

```bash
python3 evaluator.py
```

### 5. Orchestrator-Workers — `orchestrator.py`

**What**: a central LLM dynamically decomposes a task, delegates subtasks to worker LLMs, and synthesizes results. Unlike parallelization, subtasks aren't pre-defined — the orchestrator decides them per input (an implementation of the Planning pattern).

**When**: the shape of subtasks depends on the input and can't be hardcoded — multi-file code changes, cross-source research.

**Example**: `planner_chain` breaks a topic into 3-4 subtasks (JSON output with tolerant parsing) → each worker executes one subtask → `synthesizer_chain` merges everything into a final report.

```bash
python3 orchestrator.py
```

### 6. Single Agent — `single_agent.py`

**What**: one LLM plus a set of tools acting autonomously in a loop until the task is done. Essentially a while loop: the LLM emits a tool-call JSON → the tool runs → environment feedback is fed back → decide again; no tool call means final answer.

**When**: open-ended tasks with unpredictable step counts and latency tolerance. Always set a max-iteration cap.

**Example**: a customer-service agent with tools `lookup_order` / `get_return_policy` / `initiate_refund`, running the full loop "look up order → check return policy → answer the refund question."

```bash
python3 single_agent.py
```

### 7. Multi-Agent — `multi_agent.py`

**What**: multiple specialized agents collaborate (Manager pattern). Each expert is a **full agent entity: its own LLM instructions + its own tools + its own memory**; the manager LLM does only two things — decide which expert to delegate to, then hand the request to that agent (agents as tools). The essential difference from Routing: routing is stateless dispatch, multi-agent delegates to living agents with state.

**When**: domains diverge too much for one agent, agents need independent contexts and tool sets, or you need generate-and-verify collaboration. Max out a single agent first.

**Example**: a customer-service center — `sales_agent` (search_catalog / place_order), `support_agent` (lookup_ticket / initiate_refund), `shipping_agent` (track_order), dispatched by `manager_workflow`. Turn 2 of the demo shows **cross-turn memory**: the expert remembers the product it recommended and places the order directly.

```bash
python3 multi_agent.py
```

## ⚠️ Notes

- **Key safety**: the repo only contains `.env.example` (a template with a placeholder key). Put your real key in the local `.env` (git-ignored, never pushed) — never commit any real credential to the repository.
- Every file makes real LLM calls (configure `.env` first); the model defaults to `deepseek-v4-flash`, switchable to any OpenAI-compatible model at the top of each file.
- These are minimal teaching implementations — no production components (logging, retries, evals). See the reference article below for production-grade concerns.
- Consistent style across files: author header, `load_dotenv()`, LCEL pipelines, English comments/docstrings, `__main__` test blocks.

## 📚 References

- **The full explainer (Chinese)**: Agent System Design Patterns — autonomy spectrum, per-pattern rationale and selection — *Agentic Design Patterns* (adp.xindoo.xyz), Anthropic *Building Effective Agents*, OpenAI *A Practical Guide to Building Agents*, MongoDB *7 Practical Design Patterns for Agentic Systems*, Databricks *Agent system design patterns*
- **Agent anatomy**: Lilian Weng, *LLM Powered Autonomous Agents* (LLM + Planning + Memory + Tools)
- **Four agentic patterns**: Andrew Ng, *Agentic AI* course (Reflection / Tool Use / Planning / Multi-Agent Collaboration)
