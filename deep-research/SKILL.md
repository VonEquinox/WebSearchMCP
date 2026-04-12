---
name: deep-research
description: Open-ended deep research with repeated broadening/deepening loops, explicit reframing, falsification, and hypothesis/contradiction tracking to produce a sourced research narrative. Use when the user asks for open-ended research, exploratory investigation, "deep research", "开放式研究", "做假设/证伪/找反例", or wants iterative broadening and deepening until insight stabilizes.
---

# Deep Research

## Overview

Use this skill to run open-ended research as an iterative program rather than a one-off lookup: repeatedly broaden the search space, deepen via primary-source verification, reframe the question as understanding improves, and actively stress-test conclusions with counterevidence.

This skill is intentionally heavier than a normal search workflow. It requires maintaining living artifacts (plan, ledger, hypotheses, contradictions, and decisions) so the investigation stays auditable and does not collapse into ad hoc intuition.

## How this differs from `iterative-web-research`

- `iterative-web-research` is optimized for **sourced web research**: broad-to-deep search, verify key claims, then write a report.
- `deep-research` is optimized for **open-ended research** where:
  - the best framing is unknown at the start,
  - competing explanations exist,
  - contradictions and boundary conditions matter,
  - you must iterate until insight stabilizes (not just until you have “some sources”).

## Use This Workflow

Use this skill when the request has most of these properties:

- The question is ambiguous, ill-posed, or likely to need reframing.
- The user wants an exploratory investigation rather than a quick answer.
- Multiple competing explanations/hypotheses are plausible.
- Contradictions, caveats, and boundary conditions are central to correctness.
- The investigation needs multiple loops of:
  - broadening (new terms/adjacent fields/benchmarks)
  - deepening (fetch + verify)
  - reframing (split/merge questions)
  - falsification (actively seek counterevidence)

Do **not** use this skill for simple single-fact lookups, casual chat, or tasks that can be answered from local repo context alone.

## Required Intermediate Artifacts

Create working copies from `templates/` and maintain them throughout the research. Update them after each loop.

1. `research-plan.md` — evolving question backlog (priority, status, next actions, mapping to hypotheses)
2. `evidence-ledger.md` — audited source ledger (verification state, settings/conditions, findings, caveats)
3. `iteration-log.md` — wave-by-wave narrative (what changed, what was learned, next loop)
4. `term-map.md` — user terms to research terms, synonyms, definitions, query expansions
5. `hypothesis-map.md` — competing hypotheses, predictions, evidence for/against, confidence, next tests
6. `contradiction-matrix.md` — conflicting findings, differing conditions, reconciliation status
7. `decision-log.md` — explicit reframing decisions (split/merge/drop questions, retire hypotheses, stop decisions)

Do **not** keep the evolving framing only in working memory.

## Workflow

### 1. Ground the search toolchain

Start with the WebSearch MCP flow:

1. Run `mcp__websearch__doctor`
2. Run `mcp__websearch__get_usage_guide`
3. Run `mcp__websearch__recommend_command` with the actual research task
4. Execute the recommended `uv run --project ... web-search-cli ...` command from shell

Prefer the MCP-guided repository workflow over ad hoc search.

### 2. Initialize the research workspace

Create a working copy of the 7 templates and fill in:

- research date
- topic/title
- requested output format and audience
- constraints/budget (time, number of loops, or must-cover sources)
- current framing (initial)
- key unknowns and risks

### 3. Build an initial framing and term map

Before searching, build an initial “framing” that makes the problem researchable:

- define the key terms the user used (in `term-map.md`)
- list 5–15 research synonyms/keywords per key term
- identify adjacent communities/keywords that might use different language
- write 2–4 alternative framings (ways to ask the question)
- select one as `Current framing` and record the choice in `decision-log.md`

### 4. Create a living research plan

Translate the current framing into a research plan. For each research question/claim, record:

- why it matters
- priority: high / medium / low
- what evidence is needed
- which hypothesis IDs it tests (if applicable)
- current status and next action

Recommended statuses:

- `Open`
- `Searching`
- `Partially answered`
- `Supported`
- `Refuted`
- `Inconclusive but bounded`
- `Deprioritized`

### 5. Seed competing hypotheses

Create 2–6 competing hypotheses in `hypothesis-map.md`.

For each hypothesis, write:

- a clear statement
- 2–5 predictions or observable implications
- what evidence would falsify it

Avoid single-hypothesis tunnel vision.

### 6. Run the Deep Research loop (repeat)

Repeat the loop below until the stop criteria are met.

#### Loop A — Broadening (discovery)

Broaden the search space based on the current plan and term map:

- expand synonyms, alternative terms, benchmark names, and adjacent fields
- search multiple facets in parallel (core claims, mechanisms, mitigations, boundaries)
- treat search output as **source discovery**, not final evidence

#### Loop B — Deepening (verification)

Fetch and verify the highest-value sources before updating conclusions:

- prefer original papers, official technical reports, official benchmarks/leaderboards, and official repos
- extract: task setup, metrics, conditions, quantitative results, and limitations
- record in `evidence-ledger.md` with `Fetched` and `Verified` kept explicit

#### Loop C — Reframing (plan and hypothesis updates)

Update the artifacts **before** starting the next loop:

- update question statuses and next actions in `research-plan.md`
- add/rename terms in `term-map.md` as you learn the field’s real vocabulary
- update hypothesis confidence and status in `hypothesis-map.md`
- record major reframes as explicit decisions in `decision-log.md`

Common reframes:

- split one vague question into 2–4 narrower questions
- replace user language with the field’s canonical terms
- separate “direct evidence” from “strong analogy” from “inference”

#### Loop D — Falsification / stress-testing

Actively search for counterevidence and contradictions:

- search for negative results, limitations, failure modes, and “does not hold when…”
- look for alternative mechanisms that explain the same observations
- populate `contradiction-matrix.md` and attempt reconciliation via conditions

#### Loop bookkeeping

After each loop, append to `iteration-log.md`:

- what you searched (and why)
- what you fetched/verified
- what changed in the framing/hypotheses
- remaining gaps and the next loop plan

### 7. Stop Criteria (insight plateau, not wave count)

Stop the loop only when the investigation reaches an insight plateau under the current constraints.

Minimum stop gate (all must hold):

1. **Evidence sufficiency**: each high-priority conclusion is supported/refuted/inconclusive-bounded with verified sources and logged caveats.
2. **Falsification sufficiency**: major competing hypotheses have been stress-tested; counterevidence has been actively sought.
3. **Framing stability**: the framing is stable (no major reframe in the most recent loop), or reframes have converged.
4. **Contradictions handled**: key contradictions are reconciled by conditions, or explicitly marked unresolved with next checks.
5. **Diminishing returns**: the most recent broadening pass yields mostly duplicates/low-value sources, and new sources no longer change the hypothesis map materially.

If you must stop due to budget limits, explicitly include:

- what remains unknown
- what evidence is still missing
- what would change your mind

Do **not** treat “two waves” as automatically sufficient.

## Writing the Final Markdown Research Report

Write the report from the artifacts (not from memory). Use this structure by default:

```md
# <clear title>

> Research date: YYYY-MM-DD
> Scope: <1-2 lines>
> Constraints: <if any>

## Executive Summary
- 5-10 bullets with confidence labels

## Current Framing and Definitions
- Key definitions (from term-map)
- Why this framing (from decision-log)

## Hypotheses (and where they stand)
- H1: ...
- H2: ...

## Findings
### 1) <theme or question>
...
Evidence: ...

## Contradictions and Boundary Conditions
- What appears to conflict
- How (or whether) it reconciles

## What remains unknown (and what would change my mind)
- Unknowns:
- Missing evidence:
- If we found X, we would update conclusion Y

## Key Sources
- [Title](URL) — why it matters
```

## Reporting Rules

### Make evidence status explicit

Use labels like:

- **[Strong evidence]** multiple direct primary sources
- **[Moderate evidence]** one direct source plus supporting mechanism or survey
- **[Inference]** reasonable synthesis, not directly tested
- **[Newer result]** useful but not yet broadly replicated
- **[Lower weight]** withdrawn, informal, or otherwise weaker source

Ensure labels match `evidence-ledger.md` and the contradiction matrix.

### Separate direct evidence from analogy

If direct experiments for the exact question are scarce, say so explicitly, then use nearby evidence with labels:

- “directly shown”
- “supported by adjacent evidence”
- “inferred from related domains”

### Cite important claims inline

For any nontrivial claim, include a link in the same paragraph or bullet.

### State dates concretely

Use absolute dates for recent papers or reports when time context matters.

## Search Prompt Patterns

### Keep search prompts narrow and targeted

Do **not** dump the entire research task into the search or retrieval model and ask it to solve the whole problem in one prompt.

Instead:

- use search prompts for **source discovery**, **claim checking**, or **gap filling**
- keep each prompt focused on **one question, claim, mechanism, benchmark, or evidence gap**
- break large research tasks into multiple small prompts aligned with the research plan
- save synthesis and judgment for after sources have been fetched and verified

### Breadth prompt

```text
Find primary sources on <topic>. Prioritize original papers, arXiv pages, conference pages, official lab blogs, technical reports, benchmarks, and official repositories. Return grouped source links with 1-line relevance notes.
```

### Falsification prompt

```text
Find counterevidence, limitations, negative results, or failure cases for <specific claim>. Prioritize primary sources and benchmark papers that show when the claim does not hold.
```

### Term-expansion prompt

```text
List research synonyms and canonical terms used by the field for <user term>. Provide 10-20 keywords and 3-5 related benchmark or dataset names to use in follow-up searches.
```

## Common Failure Modes

Avoid these mistakes:

- Lock onto the first framing and stop reframing
- Stop after one or two waves without checking stop criteria
- Dump the entire research brief into one giant search prompt and treat the retrieval model as the final analyst
- Avoid counterevidence and never run falsification searches
- Mix direct evidence and analogy without labeling
- Update conclusions without fetching/verifying the most valuable sources
- Keep hypotheses implicit rather than tracked

## Minimal Execution Checklist

- Run doctor / usage guide / recommend command
- Initialize the 7 intermediate artifacts from templates
- Build an initial framing and term map
- Seed 2–6 competing hypotheses
- Run repeated loops: broadening → deepening → reframing → falsification
- Stop only when the stop criteria are satisfied or the budget is exhausted
- Draft the final report from the maintained artifacts
