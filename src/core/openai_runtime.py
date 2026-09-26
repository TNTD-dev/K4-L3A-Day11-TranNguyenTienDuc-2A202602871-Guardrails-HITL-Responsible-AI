"""
OpenAI SDK runtime — dùng cho:

  Blue Team → OpenRouter liquid/lfm-2.5-2.6b (create_blue_pair)
  Red Team  → OpenAI gpt-4o-mini (create_openai_pair) khi RED_TEAM_PROVIDER=openai

Gemini Red Team dùng Google ADK trong agents/*.py — không đi qua file này.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
from typing import Any, Callable
import uuid
import time

from core.config import (
    get_red_model,
    get_red_provider,
    get_blue_model,
    get_blue_provider,
    blue_client_kwargs,
    get_openai_api_key,
    red_openai_client_kwargs,
)


@dataclass
class ChatResult:
    response: str
    blocked: bool = False
    layer: str | None = None
    redacted: bool = False
    usage: dict[str, int | float | str | None] = field(default_factory=dict)
    error: dict[str, str] | None = None


class UsageBudget:
    """Persistent conservative token-cost ledger for lab API requests."""

    LIMITS = {"required": 2.0, "bonus": 3.0}
    RATES = {
        "gpt-4o-mini": (0.15, 0.60),
        "gpt-5.6-luna": (0.20, 1.20),
    }

    def __init__(self):
        self.path = Path(__file__).resolve().parents[2] / "outputs" / "api_usage.json"
        self.entries: list[dict] = []
        if self.path.is_file():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                self.entries = data.get("entries", []) if isinstance(data, dict) else []
            except (OSError, json.JSONDecodeError):
                raise RuntimeError("outputs/api_usage.json is unreadable; inspect or move it before API calls")

    def _totals(self) -> tuple[float, dict[str, float]]:
        spent = 0.0
        buckets = {key: 0.0 for key in self.LIMITS}
        for entry in self.entries:
            cost = float(entry.get("cost_usd", 0.0))
            bucket = entry.get("bucket", "required")
            if entry.get("status") in {"completed", "reserved", "unknown"}:
                spent += cost
                if bucket in buckets:
                    buckets[bucket] += cost
        return spent, buckets

    def _persist(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"budget_usd": 5.0, "entries": self.entries}, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    def reserve(self, *, model: str, prompt: str, max_output_tokens: int, bucket: str) -> str:
        if bucket not in self.LIMITS:
            raise ValueError("budget bucket must be required or bonus")
        normalized = model.lower()
        if normalized.startswith("liquid/lfm-2.5-2.6b"):
            if not normalized.endswith(":free"):
                raise RuntimeError(
                    "Cannot enforce the $5 cap: OpenRouter has no priced catalog entry for the configured model ID. "
                    "Use the listed :free variant only if it is the same lab-approved model."
                )
            input_rate = output_rate = 0.0
        else:
            matched = next((rate for name, rate in self.RATES.items() if normalized == name), None)
            if matched is None:
                raise RuntimeError(f"No verified token pricing is configured for {model}; refusing an unbounded API call")
            input_rate, output_rate = matched
        # UTF-8 byte estimate is intentionally conservative for mixed-language prompts.
        input_tokens = math.ceil(len(prompt.encode("utf-8")) / 2) + 16
        reserve = input_tokens * input_rate / 1_000_000 + max_output_tokens * output_rate / 1_000_000
        spent, buckets = self._totals()
        if spent + reserve > 5.0 or buckets[bucket] + reserve > self.LIMITS[bucket]:
            raise RuntimeError(
                f"API budget would exceed limit: need up to ${reserve:.4f}; "
                f"total reserved/spent ${spent:.4f}/$5.00, {bucket} bucket ${buckets[bucket]:.4f}/${self.LIMITS[bucket]:.2f}"
            )
        call_id = uuid.uuid4().hex
        self.entries.append({"id": call_id, "status": "reserved", "bucket": bucket,
                             "model": model, "cost_usd": reserve,
                             "max_reserved_usd": reserve, "max_output_tokens": max_output_tokens})
        self._persist()
        return call_id

    def finish(self, call_id: str, *, prompt_tokens: int, completion_tokens: int, usage_cost: float | None = None) -> float:
        entry = next(item for item in self.entries if item["id"] == call_id)
        rates = next((rate for name, rate in self.RATES.items() if entry["model"].lower() == name), (0.0, 0.0))
        if usage_cost is None and prompt_tokens == 0 and completion_tokens == 0 and rates != (0.0, 0.0):
            entry["status"] = "unknown"
            self._persist()
            return float(entry["max_reserved_usd"])
        actual = float(usage_cost) if usage_cost is not None else (
            prompt_tokens * rates[0] + completion_tokens * rates[1]
        ) / 1_000_000
        entry.update(status="completed", prompt_tokens=prompt_tokens,
                     completion_tokens=completion_tokens, cost_usd=actual)
        self._persist()
        return actual

    def fail(self, call_id: str) -> None:
        entry = next(item for item in self.entries if item["id"] == call_id)
        # Preserve the full reservation because a provider may have processed a request
        # before the client lost the response.
        entry["status"] = "unknown"
        self._persist()


_usage_budget = UsageBudget()
_openrouter_free_lock = asyncio.Lock()
_openrouter_free_last_completion: float | None = None


async def _wait_for_openrouter_free_slot() -> None:
    """Respect the shared Liquid free-route cooldown (one request per minute)."""
    global _openrouter_free_last_completion
    async with _openrouter_free_lock:
        if _openrouter_free_last_completion is not None:
            wait = 61.0 - (time.monotonic() - _openrouter_free_last_completion)
            if wait > 0:
                print(f"OpenRouter free-route cooldown: waiting {wait:.0f}s")
                await asyncio.sleep(wait)


@dataclass
class OpenAIAgent:
    name: str
    instruction: str
    provider: str = "openai"


@dataclass
class _MockInvocationContext:
    user_id: str = "student"


@dataclass
class OpenAIRunner:
    """Optional ADK-style plugins + Chat Completions."""

    app_name: str
    model: str
    plugins: list = field(default_factory=list)
    provider: str = "openai"
    temperature: float = 0.4
    max_output_tokens: int = 256
    budget_bucket: str = "required"
    client_kwargs: dict = field(default_factory=dict)
    input_hooks: list[Callable[[str], str | None]] = field(default_factory=list)
    output_hooks: list[Callable[[str], str]] = field(default_factory=list)

    def _client(self):
        from openai import OpenAI

        kwargs = dict(self.client_kwargs or {})
        kwargs.setdefault("timeout", 30.0)
        kwargs.setdefault("max_retries", 0)
        return OpenAI(**kwargs)

    async def chat(self, agent: OpenAIAgent, user_message: str) -> str:
        result = await self.chat_with_result(agent, user_message)
        if result.error:
            raise RuntimeError(f"{result.error['type']}: {result.error['message']}")
        return result.response

    async def chat_with_result(
        self, agent: OpenAIAgent, user_message: str, *, user_id: str = "student"
    ) -> ChatResult:
        try:
            return await self._chat_with_result(agent, user_message, user_id=user_id)
        except Exception as exc:
            message = str(exc)
            keys = [os.getenv(name, "") for name in (
                "OPENAI_API_KEY", "OPENROUTER_API_KEY", "GOOGLE_API_KEY"
            )]
            try:
                keys.append(get_openai_api_key())
            except Exception:
                pass
            for api_key in filter(None, set(keys)):
                message = message.replace(api_key, "[API_KEY_REDACTED]")
            return ChatResult(
                response="", blocked=False, layer="error",
                error={"type": type(exc).__name__, "message": message},
            )

    async def _chat_with_result(
        self, agent: OpenAIAgent, user_message: str, *, user_id: str = "student"
    ) -> ChatResult:
        decision = {"blocked": False, "layer": None, "redacted": False}
        for hook in self.input_hooks:
            blocked = hook(user_message)
            if blocked:
                return ChatResult(blocked, blocked=True, layer="input_guardrail")

        block_msg, plugin_decision = await self._run_input_plugins(user_message, user_id)
        if block_msg is not None:
            return ChatResult(block_msg, blocked=True, layer=plugin_decision.get("layer"))

        client = self._client()
        messages = [
            {"role": "system", "content": agent.instruction},
            {"role": "user", "content": user_message},
        ]
        prompt_for_budget = "\n".join(message["content"] for message in messages)
        reservation = _usage_budget.reserve(
            model=self.model, prompt=prompt_for_budget,
            max_output_tokens=self.max_output_tokens, bucket=self.budget_bucket,
        )
        request = {"model": self.model, "messages": messages,
                   "max_completion_tokens": self.max_output_tokens}
        if self.model.startswith("gpt-5.6"):
            request["reasoning_effort"] = "low"
        else:
            request["temperature"] = self.temperature
        is_openrouter_free = self.provider == "openrouter" and self.model.endswith(":free")
        try:
            if is_openrouter_free:
                await _wait_for_openrouter_free_slot()
            for attempt in range(2):
                try:
                    completion = client.chat.completions.create(**request)
                    break
                except Exception as exc:
                    status = getattr(exc, "status_code", None)
                    headers = getattr(getattr(exc, "response", None), "headers", {}) or {}
                    retry_after = headers.get("retry-after") or headers.get("Retry-After")
                    if not is_openrouter_free or attempt or status != 429 or not retry_after:
                        raise
                    delay = min(65.0, max(0.0, float(retry_after)))
                    if delay <= 0:
                        raise
                    print(f"OpenRouter returned 429; one retry after {delay:.0f}s")
                    await asyncio.sleep(delay)
            if is_openrouter_free:
                global _openrouter_free_last_completion
                async with _openrouter_free_lock:
                    _openrouter_free_last_completion = time.monotonic()
        except Exception:
            _usage_budget.fail(reservation)
            raise
        text = (completion.choices[0].message.content or "").strip()
        usage_obj = getattr(completion, "usage", None)
        prompt_tokens = int(getattr(usage_obj, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage_obj, "completion_tokens", 0) or 0)
        provider_cost = getattr(usage_obj, "cost", None)
        estimated_cost = _usage_budget.finish(
            reservation, prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            usage_cost=float(provider_cost) if provider_cost is not None else None,
        )
        usage = {"model": self.model, "prompt_tokens": prompt_tokens,
                 "completion_tokens": completion_tokens, "estimated_cost_usd": estimated_cost,
                 "returned_model": getattr(completion, "model", None)}

        for hook in self.output_hooks:
            filtered = hook(text)
            if filtered != text:
                decision.update(redacted=True, layer="output_guardrail")
            text = filtered

        text, output_decision = await self._run_output_plugins(text)
        decision.update(output_decision)
        return ChatResult(text, blocked=bool(decision["blocked"]),
                          layer=decision["layer"], redacted=bool(decision["redacted"]),
                          usage=usage)

    async def _run_input_plugins(self, user_message: str, user_id: str) -> tuple[str | None, dict]:
        if not self.plugins:
            return None, {"blocked": False, "layer": None, "redacted": False}
        try:
            from google.genai import types
        except ImportError as exc:
            raise RuntimeError("google-genai is required to run configured guardrail plugins") from exc

        user_content = types.Content(
            role="user",
            parts=[types.Part.from_text(text=user_message)],
        )
        ctx = _MockInvocationContext(user_id=user_id)
        for plugin in self.plugins:
            cb = getattr(plugin, "on_user_message_callback", None)
            if cb is None:
                continue
            result = await cb(invocation_context=ctx, user_message=user_content)
            decision = getattr(plugin, "last_decision", {}) or {}
            if result is None:
                continue
            return _content_to_text(result), decision
        return None, {"blocked": False, "layer": None, "redacted": False}

    async def _run_output_plugins(self, text: str) -> tuple[str, dict]:
        if not self.plugins or not text:
            return text, {"blocked": False, "layer": None, "redacted": False}
        try:
            from google.genai import types
        except ImportError as exc:
            raise RuntimeError("google-genai is required to run configured guardrail plugins") from exc

        content = types.Content(
            role="model", parts=[types.Part.from_text(text=text)]
        )

        class _Resp:
            pass

        llm_response = _Resp()
        llm_response.content = content

        class _Ctx:
            pass

        decision = {"blocked": False, "layer": None, "redacted": False}
        for plugin in self.plugins:
            cb = getattr(plugin, "after_model_callback", None)
            if cb is None:
                continue
            out = await cb(callback_context=_Ctx(), llm_response=llm_response)
            decision.update(getattr(plugin, "last_decision", {}) or {})
            if out is not None and getattr(out, "content", None) is not None:
                llm_response = out
        return _content_to_text(llm_response.content) or text, decision


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = getattr(content, "parts", None) or []
    chunks = []
    for part in parts:
        t = getattr(part, "text", None)
        if t:
            chunks.append(t)
    return "".join(chunks)


def _make_pair(
    *,
    name: str,
    instruction: str,
    app_name: str,
    model: str,
    provider: str,
    client_kwargs: dict,
    plugins: list | None = None,
    input_hooks: list | None = None,
    output_hooks: list | None = None,
    temperature: float = 0.4,
    max_output_tokens: int = 256,
    budget_bucket: str = "required",
) -> tuple[OpenAIAgent, OpenAIRunner]:
    agent = OpenAIAgent(name=name, instruction=instruction, provider=provider)
    runner = OpenAIRunner(
        app_name=app_name,
        model=model,
        provider=provider,
        client_kwargs=client_kwargs,
        plugins=list(plugins or []),
        input_hooks=list(input_hooks or []),
        output_hooks=list(output_hooks or []),
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        budget_bucket=budget_bucket,
    )
    return agent, runner


def create_blue_pair(
    *,
    name: str,
    instruction: str,
    app_name: str,
    plugins: list | None = None,
    input_hooks: list | None = None,
    output_hooks: list | None = None,
    temperature: float = 0.4,
    max_output_tokens: int = 256,
    budget_bucket: str = "required",
) -> tuple[OpenAIAgent, OpenAIRunner]:
    """Blue Team — always OpenRouter liquid/lfm-2.5-2.6b."""
    return _make_pair(
        name=name,
        instruction=instruction,
        app_name=app_name,
        # OpenRouter's currently listed route is the free variant of this same model.
        model=f"{get_blue_model()}:free",
        provider=get_blue_provider(),
        client_kwargs=blue_client_kwargs(),
        plugins=plugins,
        input_hooks=input_hooks,
        output_hooks=output_hooks,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        budget_bucket=budget_bucket,
    )


def create_openai_pair(
    *,
    name: str,
    instruction: str,
    app_name: str,
    plugins: list | None = None,
    input_hooks: list | None = None,
    output_hooks: list | None = None,
    temperature: float = 0.4,
    model: str | None = None,
    max_output_tokens: int = 256,
    budget_bucket: str | None = None,
) -> tuple[OpenAIAgent, OpenAIRunner]:
    """Red Team OpenAI path (default = soft model; advance may pass harder)."""
    return _make_pair(
        name=name,
        instruction=instruction,
        app_name=app_name,
        model=model or get_red_model(),
        provider=get_red_provider(),
        client_kwargs=red_openai_client_kwargs(),
        plugins=plugins,
        input_hooks=input_hooks,
        output_hooks=output_hooks,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        budget_bucket=budget_bucket or os.environ.get("API_BUDGET_BUCKET", "required"),
    )
