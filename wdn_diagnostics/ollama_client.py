from __future__ import annotations

import json
import socket
from urllib import request


class OllamaClient:
    """Tiny Ollama chat client using only the Python standard library."""

    def __init__(self, model: str = "qwen3:8b", host: str = "http://localhost:11434"):
        self.model = model
        self.host = host.rstrip("/")
        self.last_response_metadata = {}

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        timeout_seconds: int = 600,
        seed: int = 10,
        num_predict: int = 512,
        num_ctx: int = 4096,
        think: bool = False,
    ) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "format": "json",
            # Qwen 3 otherwise spends the structured-output budget in its hidden
            # reasoning channel and can return an empty JSON content field.  The
            # diagnostic agent needs a concise action object, not chain-of-thought.
            "think": think,
            "keep_alive": "10m",
            "options": {
                "temperature": temperature,
                "seed": seed,
                "num_predict": num_predict,
                "num_ctx": num_ctx,
            },
        }
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(
            f"{self.host}/api/chat",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=timeout_seconds) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except socket.timeout as exc:
            raise TimeoutError(
                "Ollama did not respond before the timeout. The first local model call can "
                "be slow while the model loads. Try again, or use a smaller model such as "
                "qwen3:4b."
            ) from exc
        self.last_response_metadata = {
            "done": body.get("done"),
            "done_reason": body.get("done_reason"),
            "prompt_eval_count": body.get("prompt_eval_count"),
            "eval_count": body.get("eval_count"),
            "total_duration_ns": body.get("total_duration"),
            "load_duration_ns": body.get("load_duration"),
            "prompt_eval_duration_ns": body.get("prompt_eval_duration"),
            "eval_duration_ns": body.get("eval_duration"),
        }
        return body["message"]["content"]


def extract_json_object(text: str) -> dict:
    """Extract the first JSON object from an LLM response."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"No JSON object found in LLM response: {text}")
    return json.loads(text[start : end + 1])
