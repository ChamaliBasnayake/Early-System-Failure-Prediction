# Declaration of generative-AI tool use

(Required by Section 7 of the coursework brief. Keep this updated as you go.)

| Tool | Used for | How the output was verified |
|---|---|---|
| Claude (Anthropic) | Topic selection discussion, literature triage, drafting the approval request and work plan, starter code for repository scaffolding (Phase 0) | Every file read and understood before use; code run and tested locally; all citations opened and checked |
| Gemini (Google) | Assessed foundation model in Question 6 (documented in the report). Planned model `gemini-2.5-flash` was retired by Google for new API keys before the run (HTTP 404, 28 Oct 2026). Probe on 28 Oct 2026: `gemini-3.8-flash` free tier = 20 requests/day; `gemini-3.6-flash` 35 s/call; `gemini-3.5-flash-lite` 4 s/call. The study was run on `gemini-3.8-flash` (Interactions API, thinking level `low`, seed 11) after upgrading the key to Tier 1 (paid; prompts not used by Google for training on the paid tier); list price $0.75/M input, $3.75/M output. All responses cached under `results/genai/cache/` | n/a — it is the subject of evaluation |
| (add others: Copilot, ChatGPT, …) | | |
