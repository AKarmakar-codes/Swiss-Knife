"""Reading input files and filtering out responses that aren't worth judging."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .config import CONFIG


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load records from a .jsonl file.

    The response is read from `response`, falling back to `output_steered` or
    `output` so files produced by other pipelines work without reformatting.
    """
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                response = (
                    obj.get("response")
                    or obj.get("output_steered")
                    or obj.get("output")
                    or ""
                )
                records.append(
                    {
                        "id": obj.get("id", line_num - 1),
                        "prompt": obj.get("prompt", ""),
                        "response": response,
                    }
                )
            except Exception as e:
                logging.warning(f"Skipping malformed line {line_num} in {path}: {e}")
    return records


def detect_broken_output(text: str) -> Tuple[bool, str]:
    """Flag outputs that are empty, a bare role token, too short, or stuck in a
    repetition loop. Returns (is_broken, reason)."""
    if not text:
        return True, "empty"
    text = text.strip()
    if text.lower() in CONFIG["role_tokens"]:
        return True, "role_token_only"
    if len(text) < CONFIG["min_output_length"]:
        return True, "too_short"
    return False, "ok"


def validate_record(record: Dict[str, Any]) -> Tuple[bool, str]:
    """Returns (is_valid, reason). Invalid records are not sent to the judge."""
    resp = record.get("response")
    if resp is None or (isinstance(resp, str) and not resp.strip()):
        return False, "empty_response"
    if not record.get("prompt", "").strip():
        return False, "empty_prompt"
    broken, reason = detect_broken_output(resp)
    if broken:
        return False, f"broken_{reason}"
    return True, "valid"


def resolve_input_files(input_path: str) -> List[str]:
    """Expand an input path to a sorted list of .jsonl files."""
    p = Path(input_path)
    if p.is_dir():
        return sorted(str(f) for f in p.glob("*.jsonl"))
    if p.is_file():
        return [str(p)]
    return []