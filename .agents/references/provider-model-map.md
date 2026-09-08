# Provider Model Map

This dated map resolves the provider-neutral tiers in `.agents/references/model-routing.md` to
currently documented models. Re-verify availability before use and update this file when a model
is added, deprecated, repriced, or materially re-benchmarked.

Last verified: 2026-09-07.

## Codex / OpenAI

| Tier | Default mapping | Typical use |
|---|---|---|
| frontier | GPT-6 Astra, high reasoning | orchestration, architecture, difficult escalation |
| balanced | GPT-5.6 Terra, medium reasoning | implementation, research, review, debugging |
| fast | GPT-5.6 Luna, low or medium reasoning | scans, mechanical edits, test execution, extraction |

Use GPT-5.6 Sol when the runtime offers it and the task needs stronger coding performance than
Terra without requiring Astra. Runtime availability is authoritative.

OpenAI documents Astra as the flagship for difficult end-to-end work, Terra as the balanced
intelligence/cost option, and Luna as the cost-sensitive high-volume option. OpenAI also advises
tuning reasoning effort with evaluations rather than defaulting every task to maximum effort.

Sources:

- https://developers.openai.com/api/docs/models
- https://developers.openai.com/api/docs/guides/latest-model
- https://openai.com/index/gpt-6-astra/
- https://openai.com/index/gpt-5-6/

## Claude Code / Anthropic

| Tier | Default mapping | Typical use |
|---|---|---|
| frontier | Claude Opus 5 | orchestration, difficult implementation, high-risk review |
| balanced | Claude Sonnet 5 | most implementation, research, review, debugging |
| fast | Claude Haiku 4.5 | narrow scans, deterministic execution, extraction |

Use Claude Fable 5.1 only for the hardest long-horizon reasoning when its additional cost is
justified by evaluation. Claude Code supports model selection in subagent frontmatter and at
invocation, but subagents inherit whether extended thinking is enabled in the main conversation;
the framework therefore cannot assume independent per-agent thinking controls.

Sources:

- https://platform.claude.com/docs/en/models/overview
- https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-opus-5
- https://platform.claude.com/docs/en/docs/about-claude/models/choosing-a-model
- https://code.claude.com/docs/en/sub-agents
- https://www.anthropic.com/news/claude-sonnet-5

## Antigravity / Google

| Tier | Default mapping | Typical use |
|---|---|---|
| frontier | Gemini 3.8 Flash, high effort | ordinary orchestration and complex agentic coding |
| balanced | Gemini 3.8 Flash, medium effort | implementation, research, review, debugging |
| fast | Gemini 3.8 Flash, low effort | scans, deterministic tasks, test execution |

Use an available Pro model only when repository-specific evaluation demonstrates an advantage
for the task. Run `agy models` before resolving the mapping because available models and effort
variants change. Official CLI documentation confirms session-level model selection; do not claim
role-level enforcement inside the IDE unless the current runtime exposes it. Separate CLI
sessions may be used for task-specific selection when appropriate.

Sources:

- https://codelabs.developers.google.com/antigravity-cli-hands-on
- https://deepmind.google/models/model-cards/gemini-3-8-flash/
- https://deepmind.google/models/gemini/flash/

## Benchmark Interpretation

Vendor benchmarks are directional evidence, not a durable routing contract. Benchmark versions,
agent harnesses, tool access, and scoring differ, and aggregate coding scores do not directly
measure performance as a repo researcher, reviewer, or tester. Subscription limits also do not
necessarily map directly to public API prices.

Use this map as a starting prior, then prefer results from
`.agents/references/model-evaluation.md` for the repositories and task profiles that matter to
the user.
