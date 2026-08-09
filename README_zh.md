# Agent System Design Patterns

> 用可运行的代码示例,理解 Agent 系统的 7 种主要设计模式。

**[English](README.md) | 中文**

---

## 📖 项目简介

本仓库用**最小可运行的单文件 Python 示例**,逐一演示 Agent 系统的主流设计模式。每个文件对应一种模式,`python3 xxx.py` 即可看到完整效果(真实调用 LLM)。

这些模式按「自主性光谱」排列——从完全受控的代码流程,到高度自主的多智能体系统:

```
受控 · 可预测 ─────────────────────────── 自主 · 灵活
Prompt Chaining → Routing → Parallelization → Evaluator-Optimizer → Orchestrator-Workers → Single Agent → Multi-Agent
```

**核心观点**:别一上来就造 Agent。从最简单的模式开始,有证据再加复杂度。

## 🚀 快速开始

```bash
# 1. 安装依赖
pip install langchain-openai langchain-core python-dotenv

# 2. 配置环境变量(从模板复制,填入你自己的 Key)
cp .env.example .env
# 然后编辑 .env:OPENAI_API_KEY=你的API Key
# .env 已被 .gitignore 忽略,绝不会被推送到仓库

# 3. 运行任意模式示例
python3 prompt_chain.py
```

## 🗺️ 设计模式总览

| # | 模式 | 文件 | 一句话核心 | 自主性 |
|---|------|------|-----------|--------|
| 1 | 提示词链 Prompt Chaining | [`prompt_chain.py`](prompt_chain.py) | 固定步骤串联,每步处理上一步输出,中间可加检查闸门 | 零 |
| 2 | 路由 Routing | [`routing.py`](routing.py) | LLM 分类输入,分流到最合适的处理链 | 零 |
| 3 | 并行化 Parallelization | [`parallel.py`](parallel.py) | 独立子任务并行执行,再聚合结果 | 零 |
| 4 | 评估-优化 Evaluator-Optimizer | [`evaluator.py`](evaluator.py) | 生成器 + 评估器循环,直到评估通过 | 循环内 |
| 5 | 编排-工人 Orchestrator-Workers | [`orchestrator.py`](orchestrator.py) | 中央 LLM 动态拆解任务、派给工人、再合成 | 任务内 |
| 6 | 自主单 Agent Single Agent | [`single_agent.py`](single_agent.py) | LLM + 工具在一个 while 循环里自主行动 | 全程 |
| 7 | 多 Agent 协作 Multi-Agent | [`multi_agent.py`](multi_agent.py) | 经理 LLM 协调各有工具和记忆的专家 Agent | 上限 |

## 📂 各模式详解

### 1. 提示词链 Prompt Chaining — `prompt_chain.py`

**是什么**:把任务拆成一串固定步骤,每一步的 LLM 调用处理上一步的输出,中间可加程序化检查(闸门),流程不对立即停止。

**何时用**:任务能干净地拆成固定子任务;想用延迟换准确率;每一步都被检查,错误不会滚雪球。

**示例**:用户描述想买的产品 → 第一步提取产品信息(编号/名称/类别/价格/品牌)→ 第二步转成合法 JSON。文件内含两种实现:`prompt_chain_iter`(迭代消息列表)和 `prompt_langchain`(LCEL 管道)。

```bash
python3 prompt_chain.py
```

### 2. 路由 Routing — `routing.py`

**是什么**:一个「智能交换机」——LLM 给输入分类,然后送进最合适的下游通道(不同提示词、不同模型、不同流程)。

**何时用**:输入覆盖的主题和难度分散,且分类能可靠做对。像医院分诊台:护士不治病,只判断你该挂哪个科。

**示例**:产品客服——`router_chain` 把请求分为 product / order / unclear,`RunnableBranch` 分流到对应的专家链。

```bash
python3 routing.py
```

### 3. 并行化 Parallelization — `parallel.py`

**是什么**:多个 LLM 同时干活再汇总。两种形态:切分(独立子任务并行)和投票(同一任务跑多次取多数)。

**何时用**:子任务确实独立、可并行;或需要多视角提高置信度。收益只有两个:降延迟、提质量。

**示例**:研究一个主题——`RunnableParallel` 同时跑「总结 / 提问 / 提取术语」三条链,再由合成链汇总成完整答案。

```bash
python3 parallel.py
```

### 4. 评估-优化 Evaluator-Optimizer — `evaluator.py`

**是什么**:一个生成器 + 一个评估器,循环直到评估通过——LLM 自己当评委(即 Reflection 反思模式)。

**何时用**:评估标准必须清晰可写,迭代能带来可衡量的改进。标准含糊,循环会原地打转。

**示例**:`evaluator_loop` 让生成器写 LeetCode 题解,评估器按正确性/复杂度/边界检查,不通过就带着反馈重写,直到 PERFECT 或达最大轮数。

```bash
python3 evaluator.py
```

### 5. 编排-工人 Orchestrator-Workers — `orchestrator.py`

**是什么**:中央 LLM 动态拆解任务、把子任务派给工人 LLM、再汇总结果。与并行化的区别:子任务不是预先定义好的,而是编排者针对具体输入现场决定(即 Planning 规划模式的一种实现)。

**何时用**:子任务形态取决于输入、无法硬编码——比如一次改多个文件的编码任务、跨来源检索。

**示例**:`planner_chain` 把主题拆成 3-4 个子任务(JSON 输出,容错解析)→ 每个 worker 执行一个子任务 → `synthesizer_chain` 汇总成最终报告。

```bash
python3 orchestrator.py
```

### 6. 自主单 Agent Single Agent — `single_agent.py`

**是什么**:一个 LLM + 一堆工具,在一个循环里自主决定下一步,直到任务完成。本质是 while 循环:LLM 输出工具调用 JSON → 执行工具 → 环境反馈回喂 → 再决策;不调工具即为最终回答。

**何时用**:开放式任务、步数不可预测、对延迟有容忍度。必须设迭代上限防死循环。

**示例**:客服 Agent——工具 `lookup_order` / `get_return_policy` / `initiate_refund`,完整跑完「查订单 → 核对退货政策 → 给出退款答复」。

```bash
python3 single_agent.py
```

### 7. 多 Agent 协作 Multi-Agent — `multi_agent.py`

**是什么**:多个各有专长的 Agent 分工协作(经理模式 Manager Pattern)。每个专家是**完整的 Agent 实体:自己的 LLM 指令 + 自己的工具 + 自己的记忆**;经理 LLM 只做两件事——判断该委托给谁,然后把请求交给那个专家执行(agents as tools)。这与路由的本质区别:路由是无状态分流,多 Agent 是委托给有工具有状态的 agent。

**何时用**:领域差异大到单 Agent 装不下、需要独立上下文与工具集、需要生成-验证协作。先最大化单 Agent 的能力,再考虑多 Agent。

**示例**:客服中心——`sales_agent`(search_catalog / place_order)、`support_agent`(lookup_ticket / initiate_refund)、`shipping_agent`(track_order),`manager_workflow` 分派。示例里第 2 轮对话展示专家**跨轮记忆**:记住上一轮推荐的产品并直接下单。

```bash
python3 multi_agent.py
```

## ⚠️ 注意事项

- **密钥安全**:仓库只包含 `.env.example`(模板,Key 为空占位)。真实 Key 放在本地 `.env`(由 `.gitignore` 忽略,绝不推送)——请勿把任何真实 Key 提交进仓库。
- 每个文件都真实调用 LLM(需配置 `.env`);模型默认 `deepseek-v4-flash`,可在文件头部换成任意 OpenAI 兼容模型。
- 全部为教学最小实现,不含生产级组件(日志、重试、评估)——生产级要点见下方参考的科普长文。
- 代码风格统一:文件头署名、`load_dotenv()`、LCEL 管道、英文注释与 docstring、`__main__` 测试块。

## ⭐ Star History 星标历史

![Star History 图表](/jingw2/agent-system-pattern/raw/main/assets/star-history-light.png)

由 [`scripts/gen_star_history.py`](/jingw2/agent-system-pattern/blob/main/scripts/gen_star_history.py) 通过 [GitHub Actions](/jingw2/agent-system-pattern/blob/main/.github/workflows/star-history.yml) 每日自动更新

## 📚 参考

- **科普长文**:Agent 系统设计模式(自主性光谱、各模式原理与选型)——《Agentic Design Patterns》(adp.xindoo.xyz)、Anthropic《Building Effective Agents》、OpenAI《A Practical Guide to Building Agents》、MongoDB《7 Practical Design Patterns for Agentic Systems》、Databricks《Agent system design patterns》
- **Agent 构成框架**:Lilian Weng《LLM Powered Autonomous Agents》(LLM + Planning + Memory + Tools)
- **四大 Agentic 模式**:Andrew Ng《Agentic AI》课程(Reflection / Tool Use / Planning / Multi-Agent Collaboration)
