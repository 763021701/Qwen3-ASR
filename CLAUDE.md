# CLAUDE.md

Project instructions for coding agents. These rules override default model habits when working in this repository.

## 0. Instruction Priority

Follow these instructions in this order:

1. Safety and system/developer instructions.
2. Tool calling rules in this file.
3. Project-specific instructions in this file.
4. General coding behavior guidelines.
5. Default model behavior.

When rules conflict, choose the more specific and safer rule.

---

## 1. Tool Calling Rules

When calling tools, follow these rules strictly. They override any conflicting habits from chat training.

### Argument formatting

1. Omit optional fields you don't need. Do not send null, "", {}, or [] as a placeholder. If a field is optional and you have no value, leave it out of the JSON entirely.

2. Match the container type exactly.

   - Array fields take JSON arrays: ["a", "b"], never "[\"a\",\"b\"]" as a string, never {} as an object, never "foo" as a bare string.
   - Single-element arrays still need brackets: ["foo"], not "foo".
   - Object fields take JSON objects, not arrays or strings.

3. Strings are raw strings. Do not wrap values in extra quotes, code fences, or markdown.

4. Numbers and booleans are unquoted.

   - Correct: 30
   - Wrong: "30"
   - Correct: true
   - Wrong: "true"

### Paths and identifiers

5. File paths, URLs, IDs, and similar fields go to system functions, not chat output. Never format them as markdown links, never wrap them in backticks, never add explanatory parentheses.

   Correct:

   /Users/me/notes.md

   Wrong:

   [notes.md](http://notes.md)

   Wrong:

   `/Users/me/notes.md`

   Wrong:

   /Users/me/notes.md (the notes file)

6. If a tool description says "path", treat it as input to a filesystem call. No formatting, no decoration.

### Related parameters

7. When a tool has paired parameters, such as offset + limit, start + end, from + to, provide both or neither.

### Recovery

8. If a tool returns a validation error, read the error message carefully and fix only what it complains about. Do not rewrite the whole call. Do not retry the same arguments.

9. If a tool returns a "Note:" with a defaulted value, that is informational, not an error. Continue the task. If the default is wrong, retry with the correct explicit value.

### Tool selection

10. Use the tool whose description matches your intent most specifically. Do not reach for a shell command if a dedicated tool exists. Do not reach for code execution for things a single tool call can handle.

---

## 2. Behavioral Guidelines

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

Tradeoff: These guidelines bias toward caution over speed. For trivial tasks, use judgment.

### Think Before Coding

Do not assume. Do not hide confusion. Surface tradeoffs.

Before implementing:

- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them; do not pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what is confusing. Ask.

### Simplicity First

Minimum code that solves the problem. Nothing speculative.

- No features beyond what was asked.
- No abstractions for single-use code.
- No flexibility or configurability that was not requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

### Surgical Changes

Touch only what you must. Clean up only your own mess.

When editing existing code:

- Do not improve adjacent code, comments, or formatting.
- Do not refactor things that are not broken.
- Match existing style, even if you would do it differently.
- If you notice unrelated dead code, mention it; do not delete it.

When your changes create orphans:

- Remove imports, variables, and functions that your changes made unused.
- Do not remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

### Goal-Driven Execution

Define success criteria. Loop until verified.

Transform tasks into verifiable goals:

- "Add validation" -> "Write tests for invalid inputs, then make them pass."
- "Fix the bug" -> "Write a test that reproduces it, then make it pass."
- "Refactor X" -> "Ensure tests pass before and after."

For multi-step tasks, state a brief plan:

1. [Step] -> verify: [check]
2. [Step] -> verify: [check]
3. [Step] -> verify: [check]

Strong success criteria let you loop independently. Weak criteria, such as "make it work", require clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

---

## 3. Project-Specific Instructions

- Multiple conda environments exist on this server. Different projects use different conda environments.
- Before executing any Python code, check which conda environment the current project requires and activate it.
