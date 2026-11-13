# AGENTS.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" -> "Write tests for invalid inputs, then make them pass"
- "Fix the bug" -> "Write a test that reproduces it, then make it pass"
- "Refactor X" -> "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] -> verify: [check]
2. [Step] -> verify: [check]
3. [Step] -> verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

## 5. Autonomous Long-Running Tasks

When executing a task:

- Work autonomously until the requested objective is fully completed.
- Do not ask the user clarification questions during execution.
- If there is ambiguity, inspect the repository, documentation, tests, and existing conventions first.
- If ambiguity remains, choose the most reasonable reversible assumption and continue.
- Do not stop merely to report intermediate progress.
- Do not stop after identifying a problem if it can reasonably be fixed.
- When an implementation attempt fails, diagnose the failure and try another reasonable approach.
- Continue fixing issues discovered by tests, linting, type checking, builds, or runtime verification.
- Run all relevant verification before declaring completion.
- Prefer executable verification over assumptions based only on code inspection.
- Do not declare the goal complete while known relevant failures remain.
- Only stop when the objective and success criteria are satisfied, or when a genuine external blocker makes further progress impossible.

## 6. Project Evaluation Conventions

- POC53 (English-dominant pathology): primary metric is corpus WER.
- radiology95 and holdout19 (code-switch-heavy): primary metric is corpus MER; WER is secondary/report-only.
- MER contract: apply the English medical normalizer first (`normalize_english`, which handles `c m`→`cm`, `乘`→`x`, alphanumeric boundaries), then `masr_eval_pkg.normalizers.get_normalizer("zh", zh_convert="none", number_normalize="to_arabic").normalize_for_mer`, then `compute_corpus_mer(..., text_preprocessed=True)`.
- Do not score raw medical text with the default ChineseNormalizer: formatting artifacts (`26SS12099`, `c m`, `乘`) inflate MER. Historical baselines: `evaluation/baselines/poc_mer_history_hybrid.json`.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

## Communication Style

- Respond in Chinese by default, retaining English technical terminology.
- Lead with the conclusion and answer directly without repeating the user's question.
- Keep standard responses within 5–10 lines and final reports for complex tasks within 20 lines whenever possible.
- After modifying code, report only: what changed, verification results, and unresolved issues.
- Do not narrate individual tool calls, file reads, or routine execution steps.
- Avoid lengthy logs, full diffs, or unnecessary code unless explicitly requested.
- Avoid pleasantries, repetitive summaries, irrelevant suggestions, and decorative headings.
- Maintain thorough reasoning, implementation, testing, and code review; only minimize user-facing communication.
- Clearly report errors, risks, test failures, and uncertainties. Never omit critical information for the sake of brevity.
- When the user explicitly requests a detailed explanation, the above length limits do not apply.
