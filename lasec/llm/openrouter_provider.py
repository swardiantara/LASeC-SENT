"""LLM access through OpenRouter (one OpenAI-compatible API for OpenAI, Anthropic and Google models).

The grouping answer is requested as strict JSON schema; `provider.require_parameters` makes OpenRouter
route only to endpoints that honour it. The answer is validated afterwards anyway (grouping.validate).
"""
import json
import os

from openai import OpenAI

from .prompts import SCHEMA

BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterGrouper:
    def __init__(self, model, reasoning_effort="high", temperature=None, max_tokens=32000,
                 api_key_env="OPENROUTER_API_KEY", timeout=600):
        key = os.environ.get(api_key_env)
        if not key:
            raise RuntimeError(f"set {api_key_env} to use OpenRouter")
        self.client = OpenAI(api_key=key, base_url=BASE_URL, timeout=timeout, max_retries=3,
                             default_headers={"X-Title": "LASeC drone log triage"})
        self.model, self.reasoning_effort, self.temperature, self.max_tokens = model, reasoning_effort, temperature, max_tokens

    def __call__(self, system: str, user: str) -> dict:
        extra = {"provider": {"require_parameters": True}, "usage": {"include": True}}
        if self.reasoning_effort:
            extra["reasoning"] = {"effort": self.reasoning_effort}
        kwargs = dict(model=self.model, max_tokens=self.max_tokens, extra_body=extra,
                      response_format={"type": "json_schema",
                                       "json_schema": {"name": "event_groups", "strict": True, "schema": SCHEMA}},
                      messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        resp = self.client.chat.completions.create(**kwargs)
        if not resp.choices:
            raise RuntimeError(f"empty response: {resp}")
        choice = resp.choices[0]
        if choice.finish_reason == "length":
            raise RuntimeError("output truncated; reduce the batch size or raise max_tokens")
        text = (choice.message.content or "").strip()
        if text.startswith("```"):
            text = text.strip("`").removeprefix("json").strip()
        usage = resp.usage.model_dump() if resp.usage else None
        return dict(parsed=json.loads(text), raw=text, served_by=resp.model, usage=usage, request_id=resp.id,
                    finish_reason=choice.finish_reason)
