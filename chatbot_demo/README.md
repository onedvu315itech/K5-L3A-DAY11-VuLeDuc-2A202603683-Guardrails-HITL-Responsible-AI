# VinBank Guardrailed Chatbot Demo

Single-screen chat UI demo backed by `gpt-4o-mini` (`OPENAI_API_KEY` from `.env`) and the repository's defense-in-depth guardrails:

1. **Rate Limiter** (`assignment.rate_limiter.RateLimitPlugin`) — sliding-window limit per session.
2. **Input Guardrails** (`guardrails.input_guardrails` + `agents.guards_agent`) — Unicode-normalized prompt-injection/jailbreak detection and VinBank banking topic filtering.
3. **LLM (`gpt-4o-mini`)** — OpenAI Chat Completions with `BLUE_INSTRUCTION`.
4. **Output Guardrails** (`guardrails.output_guardrails` + `agents.guards_agent`) — internal secret leak prevention and PII redaction (`[REDACTED]`).

## Run

From the repository root:

```powershell
.\.venv\Scripts\python.exe chatbot_demo/app.py
```

Then open `http://127.0.0.1:8000` in your browser.
