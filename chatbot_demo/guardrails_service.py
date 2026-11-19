"""Backend service wiring gpt-4o-mini with the repository's Guardrails pipeline."""
from __future__ import annotations

import os
import re
import sys
import time
import unicodedata
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google.genai import types
from openai import AsyncOpenAI

# Ensure <repo>/src and <repo>/.env are available regardless of launch directory.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

load_dotenv(REPO_ROOT / ".env")

from agents.agent import BLUE_INSTRUCTION  # noqa: E402
from agents.guards_agent import (  # noqa: E402
    check_secret_leak,
    content_filter_strong,
    detect_injection_strong,
)
from assignment.audit_log import AuditLogPlugin  # noqa: E402
from assignment.monitoring import MonitoringAlert  # noqa: E402
from assignment.rate_limiter import RateLimitPlugin  # noqa: E402
from guardrails.input_guardrails import detect_injection, topic_filter  # noqa: E402
from guardrails.output_guardrails import content_filter  # noqa: E402

_POLITE_GREETINGS = {
    "hi",
    "hello",
    "hey",
    "good morning",
    "good afternoon",
    "good evening",
    "xin chao",
    "chao",
    "chao ban",
    "thanks",
    "thank you",
    "cam on",
}


def _strip_for_greeting(text: str) -> str:
    nfd = unicodedata.normalize("NFD", text.strip().lower())
    no_marks = "".join(ch for ch in nfd if unicodedata.category(ch) != "Mn")
    no_marks = no_marks.replace("đ", "d")
    return re.sub(r"[^\w\s]", "", no_marks).strip()


def _is_polite_greeting(text: str) -> bool:
    return _strip_for_greeting(text) in _POLITE_GREETINGS


@dataclass
class _InvocationCtx:
    user_id: str


class GuardrailedChatbotService:
    """Single-screen chat service with Rate Limiting, Input Guardrails, gpt-4o-mini, and Output Guardrails."""

    def __init__(self, *, max_requests: int = 15, window_seconds: int = 60) -> None:
        self.api_key = (
            os.environ.get("OPENAI_API_KEY")
            or os.environ.get("OPEN_API_KEY")
            or ""
        ).strip()
        self.model = (os.environ.get("OPENAI_MODEL") or "gpt-4o-mini").strip() or "gpt-4o-mini"

        self.rate_limiter = RateLimitPlugin(
            max_requests=max_requests, window_seconds=window_seconds
        )
        self.audit = AuditLogPlugin()
        self.monitor = MonitoringAlert()
        self.sessions: dict[str, list[dict[str, str]]] = {}

    def _get_client(self) -> AsyncOpenAI:
        if not self.api_key:
            raise RuntimeError(
                "Missing OPENAI_API_KEY (or OPEN_API_KEY) in .env."
            )
        return AsyncOpenAI(api_key=self.api_key)

    def reset_session(self, session_id: str) -> None:
        self.sessions.pop(session_id, None)
        self.rate_limiter.user_windows.pop(session_id, None)

    async def chat(self, message: str, session_id: str = "default") -> dict[str, Any]:
        started = time.perf_counter()
        request_id = f"demo-{uuid.uuid4().hex[:8]}"
        clean_message = message or ""
        self.audit.record_input(
            user_id=session_id, text=clean_message, request_id=request_id
        )

        checks: list[dict[str, str]] = []

        # 1. Empty message validation
        if not clean_message.strip():
            checks = [
                {"name": "Rate Limiter", "status": "SKIP", "detail": "Empty input"},
                {"name": "Input Guardrail", "status": "BLOCK", "detail": "Empty message"},
                {"name": "Topic Filter", "status": "SKIP", "detail": "Skipped"},
                {"name": "Output Guardrail", "status": "SKIP", "detail": "Skipped"},
            ]
            return self._finalize(
                session_id=session_id,
                request_id=request_id,
                reply="Please enter a banking question to get started.",
                status="BLOCKED",
                layer="input_guardrail",
                reason="Empty message",
                checks=checks,
                started=started,
            )

        # 2. Rate Limiter
        user_content = types.Content(
            role="user", parts=[types.Part.from_text(text=clean_message)]
        )
        rl_result = await self.rate_limiter.on_user_message_callback(
            invocation_context=_InvocationCtx(user_id=session_id),
            user_message=user_content,
        )
        if rl_result is not None:
            self.monitor.rate_limit_hits += 1
            rl_msg = rl_result.parts[0].text if rl_result.parts else "Rate limit exceeded."
            human_rl_msg = (
                "You're sending messages a bit too quickly right now. "
                "Please wait a moment and try again shortly."
            )
            checks = [
                {"name": "Rate Limiter", "status": "BLOCK", "detail": rl_msg},
                {"name": "Injection Guard", "status": "SKIP", "detail": "Skipped"},
                {"name": "Topic Filter", "status": "SKIP", "detail": "Skipped"},
                {"name": "Output Guardrail", "status": "SKIP", "detail": "Skipped"},
            ]
            return self._finalize(
                session_id=session_id,
                request_id=request_id,
                reply=human_rl_msg,
                status="BLOCKED",
                layer="rate_limiter",
                reason="Rate limit exceeded",
                checks=checks,
                started=started,
            )

        used_count = len(self.rate_limiter.user_windows.get(session_id, ()))
        checks.append(
            {
                "name": "Rate Limiter",
                "status": "PASS",
                "detail": f"{used_count}/{self.rate_limiter.max_requests} req / {self.rate_limiter.window_seconds}s",
            }
        )

        # 3. Input Guardrail — Prompt Injection & Jailbreak detection
        if detect_injection(clean_message) == "BLOCK" or detect_injection_strong(clean_message):
            checks.extend(
                [
                    {
                        "name": "Injection Guard",
                        "status": "BLOCK",
                        "detail": "Prompt injection or jailbreak pattern detected",
                    },
                    {"name": "Topic Filter", "status": "SKIP", "detail": "Skipped"},
                    {"name": "Output Guardrail", "status": "SKIP", "detail": "Skipped"},
                ]
            )
            return self._finalize(
                session_id=session_id,
                request_id=request_id,
                reply=(
                    "I cannot answer or fulfill that request, as I'm unable to override my "
                    "safety guidelines or share internal instructions and credentials. "
                    "If you have questions about VinBank accounts, savings, transfers, or cards, "
                    "I'd be happy to help!"
                ),
                status="BLOCKED",
                layer="input_guardrail",
                reason="Prompt injection detected",
                checks=checks,
                started=started,
            )

        checks.append(
            {
                "name": "Injection Guard",
                "status": "PASS",
                "detail": "No prompt injection detected",
            }
        )

        # 4. Input Guardrail — Topic Filter
        is_greeting = _is_polite_greeting(clean_message)
        if not is_greeting and topic_filter(clean_message) == "BLOCK":
            checks.extend(
                [
                    {
                        "name": "Topic Filter",
                        "status": "BLOCK",
                        "detail": "Off-topic or restricted subject",
                    },
                    {"name": "Output Guardrail", "status": "SKIP", "detail": "Skipped"},
                ]
            )
            return self._finalize(
                session_id=session_id,
                request_id=request_id,
                reply=(
                    "I'm sorry, I cannot answer that question. I'm a VinBank assistant and can "
                    "only help with banking-related topics such as accounts, transfers, savings "
                    "rates, loans, and credit cards."
                ),
                status="BLOCKED",
                layer="input_guardrail",
                reason="Off-topic or prohibited topic",
                checks=checks,
                started=started,
            )

        checks.append(
            {
                "name": "Topic Filter",
                "status": "PASS",
                "detail": "Greeting allowed" if is_greeting else "VinBank banking topic",
            }
        )

        # 5. Call gpt-4o-mini via OpenAI API
        history = self.sessions.setdefault(session_id, [])
        messages: list[dict[str, str]] = [{"role": "system", "content": BLUE_INSTRUCTION}]
        messages.extend(history[-10:])
        messages.append({"role": "user", "content": clean_message})

        try:
            client = self._get_client()
            completion = await client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.4,
            )
            raw_reply = (completion.choices[0].message.content or "").strip()
        except Exception as exc:
            checks.append(
                {
                    "name": "Output Guardrail",
                    "status": "SKIP",
                    "detail": f"Model call failed ({type(exc).__name__})",
                }
            )
            return self._finalize(
                session_id=session_id,
                request_id=request_id,
                reply="Unable to reach gpt-4o-mini right now. Please check OPENAI_API_KEY in .env and try again.",
                status="BLOCKED",
                layer="model_error",
                reason=type(exc).__name__,
                checks=checks,
                started=started,
            )

        if not raw_reply:
            raw_reply = "I'm sorry, I couldn't generate a response. Please try asking your banking question again."

        # 6. Output Guardrails — Secret Leak Block & PII Redaction
        strong_check = content_filter_strong(raw_reply)
        if check_secret_leak(raw_reply) or not strong_check["safe"]:
            checks.append(
                {
                    "name": "Output Guardrail",
                    "status": "BLOCK",
                    "detail": "Internal credential or secret leak prevented",
                }
            )
            return self._finalize(
                session_id=session_id,
                request_id=request_id,
                reply=(
                    "I cannot share internal system details or sensitive credentials. "
                    "How else can I assist you with your VinBank banking needs?"
                ),
                status="BLOCKED",
                layer="output_guardrail",
                reason="Secret leak prevented",
                checks=checks,
                started=started,
            )

        pii_check = content_filter(raw_reply)
        if not pii_check["safe"]:
            final_reply = pii_check["redacted"]
            issues_str = ", ".join(pii_check["issues"])
            checks.append(
                {
                    "name": "Output Guardrail",
                    "status": "REDACT",
                    "detail": f"Redacted: {issues_str}",
                }
            )
            self._append_history(history, clean_message, final_reply)
            return self._finalize(
                session_id=session_id,
                request_id=request_id,
                reply=final_reply,
                status="REDACTED",
                layer="output_guardrail",
                reason=f"PII/sensitive pattern redacted ({issues_str})",
                checks=checks,
                started=started,
            )

        checks.append(
            {
                "name": "Output Guardrail",
                "status": "PASS",
                "detail": "No PII or secret leak detected",
            }
        )
        self._append_history(history, clean_message, raw_reply)
        return self._finalize(
            session_id=session_id,
            request_id=request_id,
            reply=raw_reply,
            status="ALLOWED",
            layer=None,
            reason="All guardrails passed",
            checks=checks,
            started=started,
        )

    @staticmethod
    def _append_history(
        history: list[dict[str, str]], user_msg: str, assistant_msg: str
    ) -> None:
        history.append({"role": "user", "content": user_msg})
        history.append({"role": "assistant", "content": assistant_msg})
        if len(history) > 20:
            del history[:-20]

    def _finalize(
        self,
        *,
        session_id: str,
        request_id: str,
        reply: str,
        status: str,
        layer: str | None,
        reason: str,
        checks: list[dict[str, str]],
        started: float,
    ) -> dict[str, Any]:
        is_blocked_or_redacted = status in {"BLOCKED", "REDACTED"}
        self.monitor.total_requests += 1
        if is_blocked_or_redacted:
            self.monitor.blocked_requests += 1

        self.audit.record_output(
            user_id=session_id,
            text=reply,
            blocked=is_blocked_or_redacted,
            layer=layer,
            request_id=request_id,
        )
        latency_ms = round((time.perf_counter() - started) * 1000, 1)
        return {
            "reply": reply,
            "status": status,
            "layer": layer,
            "reason": reason,
            "model": self.model,
            "latency_ms": latency_ms,
            "checks": checks,
        }
