"""The LLM-as-judge backend. Wraps the local vLLM server so deepeval can call
it as an evaluation model."""

import asyncio
import json
import logging
import re
from typing import Any, Optional

from pydantic import BaseModel
from deepeval.models import DeepEvalBaseLLM
from openai import AsyncOpenAI

from .config import CONFIG


class VLLMJudge(DeepEvalBaseLLM):
    """deepeval judge backed by an OpenAI-compatible vLLM server."""

    def __init__(self, model_name: str = CONFIG["judge_model"]):
        self.model_name = model_name
        self.client = AsyncOpenAI(
            base_url=CONFIG["vllm_url"],
            api_key=CONFIG["vllm_api_key"],
        )
        logging.info(f"Judge client pointed at {CONFIG['vllm_url']}")

    def load_model(self):
        return self.client

    def get_model_name(self) -> str:
        return self.model_name

    def _extract_json(self, text: str) -> str:
        """Return the first valid JSON object found in a model response."""
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if m:
            try:
                json.loads(m.group(1))
                return m.group(1)
            except json.JSONDecodeError:
                pass
        for start in (i for i, c in enumerate(text) if c == "{"):
            depth, in_str, esc = 0, False, False
            for i, c in enumerate(text[start:], start):
                if esc:
                    esc = False
                    continue
                if c == "\\":
                    esc = True
                    continue
                if c == '"':
                    in_str = not in_str
                    continue
                if in_str:
                    continue
                if c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        cand = text[start:i + 1]
                        try:
                            json.loads(cand)
                            return cand
                        except json.JSONDecodeError:
                            break
        return text

    def generate(self, prompt: str, schema: Optional[BaseModel] = None) -> Any:
        return asyncio.run(self.a_generate(prompt, schema))

    async def a_generate(self, prompt: str, schema: Optional[BaseModel] = None) -> Any:
        if schema:
            prompt = (
                f"{prompt}\n\nYou MUST respond with valid JSON only. "
                "No explanation, no markdown, just the JSON object."
            )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a precise evaluator. Always respond in valid JSON "
                    "when requested. Output only the JSON object, nothing else."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        try:
            resp = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.0,
                max_tokens=CONFIG["judge_max_tokens"],
                timeout=CONFIG["judge_timeout"],
            )
            content = resp.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"Judge generation error: {e}")
            if schema and hasattr(schema, "model_fields"):
                return schema(**{f: None for f in schema.model_fields})
            return "Error"

        if schema:
            try:
                return schema(**json.loads(self._extract_json(content)))
            except Exception as e:
                logging.warning(f"Judge JSON parse failed: {e}. Content: {content[:100]}...")
                if hasattr(schema, "model_fields"):
                    return schema(**{f: None for f in schema.model_fields})
                return content
        return content
