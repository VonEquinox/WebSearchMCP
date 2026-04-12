---
name: iterative-web-research
description: Conduct broad-to-deep web research with iterative search refinement, primary-source verification, and sourced Markdown reporting. Use when the user asks for deep research, extensive search, “search broadly first then go deeper”, latest information with citations, or a final report/MD document. Also use for Chinese requests such as “深度研究”, “广泛搜索”, “先广后深”, “越搜越深越搜越广”, “给出来源链接”, or “最后写成 md 文档”.
---

# Iterative Web Research

## Overview

Use this skill to turn an open-ended research question into a disciplined, plan-led web-research workflow: initialize living research artifacts, search multiple facets in parallel, verify key claims with primary sources, deepen based on what is learned, and deliver a sourced Markdown report.

## Use This Workflow

Apply this skill when the request has **all or most** of these properties:

- Need current or time-sensitive information
- Need many searches rather than one lookup
- Need breadth first, then depth
- Need source links or direct attribution
- Need a synthesized report, especially in Markdown
- Need comparison across papers, blogs, docs, benchmarks, or official reports

Do **not** use this skill for simple single-fact lookups, casual chat, or tasks that can be answered from local repo context alone.

## Required Intermediate Artifacts

Before writing the final report, create and maintain these living artifacts from the templates in `templates/`:

1. `research-plan.md` — the current list of research questions, priorities, evidence needs, statuses, and next actions
2. `evidence-ledger.md` — the source ledger that tracks claims, evidence strength, verification state, quantitative findings, and caveats
3. `iteration-log.md` — the wave-by-wave record of searches run, what was learned, plan changes, remaining gaps, and next targeted searches

Update all three after each meaningful search or fetch wave. Do **not** keep the evolving plan only in working memory.

## Workflow

### 1. Ground the search toolchain

Start with the WebSearch MCP flow:

1. Run `mcp__websearch__doctor`
2. Run `mcp__websearch__get_usage_guide`
3. Run `mcp__websearch__recommend_command` with the actual research task
4. Execute the recommended `uv run --project ... web-search-cli ...` command from shell

Prefer the MCP-guided repository workflow over ad hoc search.

### 2. Initialize the research workspace

Before searching, create a working copy of the three templates and fill in:

- research date
- research topic or title
- scope and requested output format
- known constraints or budget limits, if the user specified any
- the initial highest-risk unknowns

### 3. Create a living research plan

Before searching, split the request into 4 categories:

1. **Core claim questions** — the main thing to answer
2. **Mechanism questions** — why it happens
3. **Mitigation / method questions** — what people do about it
4. **Boundary / counterexample questions** — when it does not hold

For each question or claim, record:

- why it matters
- priority: high / medium / low
- what type of evidence is needed
- current status
- next action

Recommended statuses include:

- `Open`
- `Searching`
- `Partially answered`
- `Supported`
- `Refuted`
- `Inconclusive but bounded`
- `Deprioritized`

### 4. Search wide first

Run multiple discovery searches in parallel for the highest-priority open questions. Cover:

- Primary papers
- Official documentation or technical reports
- Benchmarks / evaluations
- Strong secondary explainers only if they point to primary sources

Good first-pass search prompt pattern:

```text
Find primary sources on <topic>. Prioritize original papers, arXiv pages, conference pages, official lab blogs, technical reports, benchmarks, and official repositories. Return grouped source links with 1-line relevance notes.
```

Do **not** trust the initial search summary as final evidence. Treat it as source discovery only.

### 5. Build and update the evidence ledger

After each discovery or fetch wave, add sources to the ledger and group them into buckets such as:

- Direct evidence
- Mechanistic evidence
- Mitigation evidence
- Counterevidence / caveats
- Surveys for orientation

For each source, record at least:

- Source title and link
- Date
- Source type
- Main claim supported or challenged
- Whether the evidence is direct, adjacent, or framing
- Evidence strength: strong / moderate / lower weight
- Whether it has been fetched
- Whether it has been verified
- Key quantitative findings when available
- Caveats or limitations

### 6. Fetch and verify the highest-value sources

Fetch the highest-value sources before writing conclusions.

Prioritize fetching in this order:

1. Original paper abstract / PDF / HTML
2. Official technical report
3. Official benchmark or repo
4. Survey for framing

Use fetches to extract:

- Exact task setup
- What was actually measured
- Whether the evidence is direct or only analogous
- Quantitative results when available
- Important caveats or limitations

Be careful with papers whose titles overstate conclusions. Verify from the abstract, setup, and results sections.

### 7. Run the iteration loop

After each search or fetch wave, update the research plan, evidence ledger, and iteration log **before** starting the next wave.

Repeat this loop until the stop criteria are met:

1. Identify the highest-priority unanswered claims or evidence gaps
2. Run targeted searches for those gaps
3. Fetch and verify the highest-value new sources
4. Update question statuses and next actions in the research plan
5. Update evidence-strength and verification fields in the evidence ledger
6. Append a new wave entry to the iteration log
7. Re-rank the remaining open questions

Common gap-driven follow-ups:

- If a paper shows degradation, search for its proposed mechanism
- If a mechanism is suggested, search for mitigation papers targeting that mechanism
- If the user asked about one specific domain pair but only adjacent evidence exists, search analogous domain-specialization papers and mark them as analogy
- If a survey reveals benchmark names, search those benchmark papers directly

### 8. Prefer primary-source reasoning patterns

Use these source priorities by topic:

- **Research claims:** papers, arXiv, conference pages
- **Model or product behavior:** official model cards, technical reports, official docs
- **Implementation details:** official repos or repo files
- **Benchmarks:** benchmark paper or official leaderboard docs

Use blogs only when they are official lab blogs or when they help locate primary sources.

### 9. Check stop criteria before writing

Do not write the final report until the **Stop Criteria** section below is satisfied, or until a user-imposed budget limit is reached and the remaining gaps are explicitly documented.

### 10. Write only after source coverage is sufficient

Do not write the final report until you can answer:

- What is well-supported?
- What is only partially supported?
- What is inference rather than direct evidence?
- What important evidence is still missing?

Trace every core conclusion back to verified ledger entries rather than ad hoc memory.

## Stop Criteria

Stop the research loop only when **all** applicable conditions below are true:

1. Every **high-priority** research question is marked as one of:
   - `Supported`
   - `Refuted`
   - `Inconclusive but bounded`
2. Every high-priority conclusion has either:
   - at least **2 direct high-value sources**, or
   - **1 direct primary source** plus **1 official or benchmark corroborating source**
3. At least **1 caveat, counterexample, or limitation source** has been logged for the main conclusion set
4. Major unresolved gaps are explicitly listed in the research plan or final report draft
5. If you stop because of time, tool, or token budget limits, the final output must clearly label:
   - which conclusions remain incomplete
   - what evidence is still missing
   - whether current claims are direct evidence, strong analogy, or inference

Do **not** treat “one broad search plus one follow-up search” as automatically sufficient.

## Writing the Final Markdown Report

Use this structure by default:

```md
# <clear title>

> Research date: YYYY-MM-DD
> Scope: <1-2 lines>

## Executive Summary
- 4-8 bullets with direct answers

## Questions
- Restate the user’s research questions cleanly

## Core Conclusions
### Conclusion 1
...
Evidence: ...

## Analysis
### 1. <theme>
...

## What is directly supported vs inferred
- Direct evidence:
- Strong analogy:
- Open questions:

## Key Sources
- [Title](URL) — why it matters
```

Write the report from the artifacts rather than from memory:

- The **research plan** determines which questions were in scope and how they were resolved
- The **evidence ledger** determines the evidence strength, verification status, and caveats for each claim
- The **iteration log** explains how the investigation deepened and why some open questions remain

## Reporting Rules

### Make evidence status explicit

Use labels like:

- **[Strong evidence]** multiple direct primary sources
- **[Moderate evidence]** one direct source plus supporting mechanism or survey
- **[Inference]** reasonable synthesis, not directly tested
- **[Newer result]** useful but not yet broadly replicated
- **[Lower weight]** withdrawn, informal, or otherwise weaker source

Ensure the label in the final report matches the status recorded in the evidence ledger.

### Separate direct evidence from analogy

If the exact asked question lacks direct experiments, say so explicitly. Then use nearby evidence with labels such as:

- “directly shown”
- “supported by adjacent evidence”
- “inferred from related domains”

### Cite important claims inline

For any nontrivial claim, include a link in the same paragraph or bullet.

### State dates concretely

Use absolute dates for recent papers or reports when the time context matters.

## Search Prompt Patterns

### Keep search prompts narrow and targeted

Do **not** dump the entire research task into the search or retrieval model and ask it to solve the whole problem in one prompt.

Instead:

- Use search prompts for **source discovery**, **claim checking**, or **gap filling**
- Keep each prompt focused on **one question, claim, mechanism, benchmark, or evidence gap**
- Break a large research task into multiple small prompts aligned with the research plan
- Save synthesis, comparison, and final judgment for the main research workflow after sources have been fetched and verified

Bad pattern:

```text
Research whether model X is better than model Y in every practical way, explain why, find all benchmarks, tell me what experts think, and give me a final recommendation.
```

Better pattern:

```text
Find direct benchmark sources comparing model X and model Y on coding tasks. Prioritize official reports, benchmark papers, and leaderboard documentation. Return links with 1-line relevance notes.
```

```text
Find primary sources explaining why model X underperforms on long-context retrieval. Prioritize mechanistic analysis, ablations, and technical reports. Return source links with brief notes.
```

### Breadth prompt

```text
Find primary sources on <topic>. Prioritize original papers, arXiv pages, conference pages, official lab blogs, technical reports, benchmarks, and official repositories. Return grouped source links with 1-line relevance notes.
```

### Mechanism prompt

```text
Find primary mechanistic or theoretical sources explaining <phenomenon>. Prioritize papers with direct measurements, ablations, or mechanistic analysis. Return source links with notes on the proposed mechanism.
```

### Mitigation prompt

```text
Find surveys and primary papers on methods to mitigate <problem>. Prioritize replay, regularization, parameter isolation, routing/MoE, curriculum, synthetic data, and evaluation benchmarks. Return grouped source links with notes.
```

### Evidence-gap prompt

```text
Find direct evidence for <very specific claim>. If direct evidence is scarce, find the closest adjacent domains and note that they are analogical rather than direct.
```

## Quality Bar

Before finishing, verify that the report:

- Starts from initialized intermediate artifacts
- Covers breadth before depth
- Uses fetched primary sources for core conclusions
- Distinguishes direct evidence from inference
- Includes links for all major claims
- Names limitations and missing evidence
- Matches the user’s requested output format, especially Markdown

## Common Failure Modes

Avoid these mistakes:

- Skip the research plan and search only from memory
- Stop after one search wave without checking stop criteria
- Dump the entire research brief into one giant search prompt and treat the retrieval model as the final analyst
- Quote the search engine summary as if it were a source
- Rely on secondary summaries without fetching primary sources
- Overclaim from adjacent evidence
- Mix product claims, benchmark claims, and mechanistic claims without labeling them
- Write the report before verifying the highest-value sources

## Minimal Execution Checklist

- Run doctor / usage guide / recommend command
- Initialize `research-plan.md`, `evidence-ledger.md`, and `iteration-log.md`
- Search 3–6 facets in parallel for the highest-priority questions
- Fetch top primary sources
- Update the evidence ledger and iteration log
- Run targeted search loops until the stop criteria are met or the budget is exhausted
- Draft the sourced Markdown report from the maintained artifacts
- Mark evidence strength, caveats, and unresolved gaps
