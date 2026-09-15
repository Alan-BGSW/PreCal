from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pymupdf as fitz
import pandas as pd
import requests


_API_KEY_ALIASES = {
    "GENAIPLATFORM-FARM-SUBSCRIPTION-KEY": "GENAIPLATFORM_FARM_SUBSCRIPTION_KEY",
    "genaiplatform-farm-subscription-key": "GENAIPLATFORM_FARM_SUBSCRIPTION_KEY",
    "subscription-key": "GENAIPLATFORM_FARM_SUBSCRIPTION_KEY",
    "BMF_API_KEY": "GENAIPLATFORM_FARM_SUBSCRIPTION_KEY",
    "FARM_API_KEY": "GENAIPLATFORM_FARM_SUBSCRIPTION_KEY",
    "MODEL_FARM_SUBSCRIPTION_KEY": "GENAIPLATFORM_FARM_SUBSCRIPTION_KEY",
}


def _load_dotenv() -> None:
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        return

    with env_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = line.strip()
            if not row or row.startswith("#") or "=" not in row:
                continue
            key, _, value = row.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value

            alias_target = _API_KEY_ALIASES.get(key)
            if alias_target and alias_target not in os.environ and value:
                os.environ[alias_target] = value


def _get_subscription_key() -> str | None:
    for env_name in [
        "FARM_API_KEY",
        "MODEL_FARM_SUBSCRIPTION_KEY",
        "BMF_API_KEY",
        "GENAIPLATFORM_FARM_SUBSCRIPTION_KEY",
        "genaiplatform-farm-subscription-key",
        "GENAIPLATFORM-FARM-SUBSCRIPTION-KEY",
        "subscription-key",
        "CLAUDE_API_KEY",
        "API_KEY",
    ]:
        value = os.getenv(env_name)
        if value and value.strip():
            return value.strip()
    return None


_load_dotenv()


_HEADER_PATTERNS = [
    re.compile(r"FC\s*:\s*([A-Za-z][A-Za-z0-9_]*)\s*/\s*([\d.]+\s*;\s*\d+)"),
    re.compile(r"\[\s*([A-Za-z][A-Za-z0-9_]*)\s+([\d.]+\s*;\s*\d+)\s*\]"),
    re.compile(r"^\s*([A-Za-z][A-Za-z0-9_]*)\s+([\d.]+\s*;\s*\d+)\b", re.MULTILINE),
]


class _Tokenizer:
    """Token counting helper using cl100k, with char estimate fallback."""

    def __init__(self) -> None:
        self._enc = None
        try:
            import tiktoken

            self._enc = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self._enc = None

    def count(self, text: str) -> int:
        if self._enc is not None:
            return len(self._enc.encode(text))
        return max(1, len(text) // 4)


_TOK = _Tokenizer()


@dataclass
class FunctionBlock:
    function_name: str
    version: str
    text: str = ""
    pages: list[int] = field(default_factory=list)


@dataclass
class LabelRow:
    function_name: str
    label_name: str
    description: str
    label_value: str
    value_structure: str
    suffix_identified: str
    context: str
    pages: str


_KNOWN_SUFFIX_TYPES = {
    "_c": "Scalar",
    "_cw": "Scalar",
    "_ca": "Array",
    "_map": "2D Map",
    "_m": "2D Map",
    "_t": "Curve/Table",
    "_cur": "Curve/Table",
}

_HINT_SECTION_RE = re.compile(r"standard\s+calibration\s+hints?", re.IGNORECASE)
_HEADING_RE = re.compile(r"^\s*\d+(?:\.\d+){1,6}\s+\S")
_LABEL_TOKEN_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+\b")
_VALUE_MARKER_RE = re.compile(r"\b(?:start|standard|default|initial)\s*value\s*:", re.IGNORECASE)
_PART_ID_RE = re.compile(r"^P\d{3,}_[A-Za-z0-9_]+$")

_VERSION_LIKE_RE = re.compile(r"^\d+(?:\.\d+)*(?:;\d+)?$")


def _normalize_label_candidate(value: str) -> str:
    # Trim punctuation that often leaks from table cells or sentences.
    cleaned = value.strip().strip(".,;:()[]{}<>\"'")
    return cleaned


def _is_version_like(value: str) -> bool:
    compact = re.sub(r"\s+", "", value)
    return bool(_VERSION_LIKE_RE.match(compact))


def _looks_like_ecu_label(value: str) -> bool:
    if not value:
        return False
    if _is_version_like(value):
        return False
    if " " in value:
        return False
    if "/" in value or "\\" in value or ";" in value:
        return False
    if not re.match(r"^[A-Za-z][A-Za-z0-9_\-−]*$", value):
        return False
    suffix, suffix_category = _analyze_suffix(value)
    if not suffix or suffix_category == "Unknown":
        return False
    return True


def _analyze_suffix(label: str) -> tuple[str, str]:
    if "_" not in label:
        return "", "Unknown"

    for suffix in sorted(_KNOWN_SUFFIX_TYPES.keys(), key=len, reverse=True):
        if label.lower().endswith(suffix):
            return label[-len(suffix):], _KNOWN_SUFFIX_TYPES[suffix]

    return "", "Unknown"


def _normalize_whitespace(value: str) -> str:
    return " ".join(value.split())


def _infer_value_structure(value: str, suffix_category: str, suffix_identified: str) -> str:
    cleaned = _normalize_whitespace(value)
    if not cleaned:
        if suffix_category != "Unknown":
            return suffix_category
        return "Unknown"

    lower = cleaned.lower()
    if any(tok in lower for tok in [" x ", " y ", "breakpoint", "axis", "->", "=>"]):
        return "Curve/Table"
    if ":" in cleaned and any(ch.isdigit() for ch in cleaned):
        return "Curve/Table"
    if any(sep in cleaned for sep in [";", "|"]):
        return "Array"
    if cleaned.startswith("[") and cleaned.endswith("]"):
        return "Array"
    if re.search(r"\{.*\}", cleaned):
        return "2D Map"
    if suffix_category != "Unknown":
        return suffix_category
    if suffix_identified.lower() in {"_map", "_m"}:
        return "2D Map"
    if suffix_identified.lower() in {"_t", "_cur"}:
        return "Curve/Table"
    if suffix_identified.lower() in {"_ca"}:
        return "Array"
    return "Scalar"


def _merge_context_parts(parts: list[str]) -> str:
    merged: list[str] = []
    for part in parts:
        text = _normalize_whitespace(part)
        if text and text not in merged:
            merged.append(text)
    return " | ".join(merged)


def _trim_text(value: str, max_len: int = 220) -> str:
    text = _normalize_whitespace(value)
    if len(text) <= max_len:
        return text
    return text[: max_len - 3].rstrip() + "..."


def _is_noise_line(text: str) -> bool:
    lower = text.lower()
    noise_markers = [
        "calibration of function labels",
        "must be edited by calibration parameter editor",
        "do not edit these tables manually",
        "label name",
        "description",
        "caution",
        "apph_labtab",
    ]
    return any(marker in lower for marker in noise_markers)


def _extract_primary_value(value_parts: list[str]) -> str:
    if not value_parts:
        return ""

    value_text = _normalize_whitespace(" ".join(value_parts))
    match = re.search(
        r"(?:start|standard|default|initial)\s*value\s*:\s*([^\[]+?)(?:\s*\[[^\]]*\])?(?:\s|$)",
        value_text,
        flags=re.IGNORECASE,
    )
    if match:
        return _trim_text(match.group(1).strip(), max_len=120)

    numbers = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", value_text)
    if numbers:
        compact = ", ".join(numbers[:24])
        return _trim_text(compact, max_len=120)

    return _trim_text(value_text, max_len=120)


def _is_relevant_label(
    *,
    label: str,
    function_name: str,
    suffix_category: str,
    description: str,
    value: str,
) -> bool:
    if not label:
        return False
    if label.lower() == function_name.lower():
        return False
    if suffix_category == "Unknown":
        return False
    if _PART_ID_RE.match(label):
        return False

    has_signal = bool(value) or len(description) >= 24
    starts_like_calib = label.startswith(("Eem_", "FId_", "FID_", "EEM_"))
    return starts_like_calib and has_signal


def _extract_label_rows_from_section_text(
    function_name: str,
    section_text: str,
    page_str: str,
) -> list[LabelRow]:
    lines = [_normalize_whitespace(line) for line in section_text.splitlines()]
    lines = [line for line in lines if line]
    if not lines:
        return []

    extracted: list[LabelRow] = []
    current_label = ""
    desc_parts: list[str] = []
    value_parts: list[str] = []

    def flush_current() -> None:
        nonlocal current_label, desc_parts, value_parts
        if not current_label:
            return

        description = _trim_text(" ".join(desc_parts), max_len=240)
        value = _extract_primary_value(value_parts)
        suffix_identified, suffix_category = _analyze_suffix(current_label)
        value_type = _infer_value_structure(value, suffix_category, suffix_identified)
        context = _trim_text(description or value, max_len=220)

        if not _is_relevant_label(
            label=current_label,
            function_name=function_name,
            suffix_category=suffix_category,
            description=description,
            value=value,
        ):
            current_label = ""
            desc_parts = []
            value_parts = []
            return

        if context:
            extracted.append(
                LabelRow(
                    function_name=function_name,
                    label_name=current_label,
                    description=description,
                    label_value=value,
                    value_structure=value_type,
                    suffix_identified=suffix_identified,
                    context=context,
                    pages=page_str,
                )
            )

        current_label = ""
        desc_parts = []
        value_parts = []

    for line in lines:
        labels = [tok for tok in _LABEL_TOKEN_RE.findall(line) if _looks_like_ecu_label(tok)]
        if labels:
            # Keep the first valid label in the row; PDF extraction can duplicate tokens.
            label = labels[0]
            flush_current()
            current_label = label

            tail = line.split(label, 1)[-1].strip(" |:-")
            if tail and not _is_noise_line(tail):
                if _VALUE_MARKER_RE.search(tail):
                    value_parts.append(tail)
                else:
                    desc_parts.append(tail)
            continue

        if not current_label:
            continue

        if _is_noise_line(line):
            continue
        if _VALUE_MARKER_RE.search(line):
            value_parts.append(line)
            continue
        if value_parts and re.search(r"\d", line) and len(value_parts) < 4:
            value_parts.append(line)
            continue
        desc_parts.append(line)

    flush_current()

    # Deduplicate by label and keep the richest signal.
    by_label: dict[str, LabelRow] = {}
    for row in extracted:
        existing = by_label.get(row.label_name)
        if existing is None:
            by_label[row.label_name] = row
            continue
        old_score = len(existing.description) + len(existing.label_value) + len(existing.context)
        new_score = len(row.description) + len(row.label_value) + len(row.context)
        if new_score > old_score:
            by_label[row.label_name] = row

    return list(by_label.values())


def _is_heading_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if _HEADING_RE.match(stripped):
        return True
    if len(stripped) < 90 and stripped.endswith(":"):
        return True
    return False


def _extract_standard_hint_sections(block_text: str) -> list[str]:
    lines = block_text.splitlines()
    if not lines:
        return []

    section_starts = [i for i, line in enumerate(lines) if _HINT_SECTION_RE.search(line)]
    sections: list[str] = []

    for start in section_starts:
        end = len(lines)
        for i in range(start + 1, len(lines)):
            if _is_heading_line(lines[i]):
                end = i
                break
        snippet = "\n".join(lines[max(0, start - 2):end]).strip()
        if snippet:
            sections.append(snippet)

    if sections:
        return sections
    return []


def _build_decoder_prompts(function_name: str, chunk_text: str) -> tuple[str, str]:
    system_prompt = (
        "You are an Automotive ECU calibration parser. Parse ONLY calibration labels "
        "from the provided function-component section. The section is expected to be "
        "from Calibration -> Standard calibration hints.\n\n"
        "Output ONLY valid JSON array with objects using this schema:\n"
        "{\"label\":\"...\",\"description\":\"...\",\"value\":\"...\","
        "\"value_structure\":\"...\",\"section_context\":\"...\"}.\n\n"
        "Rules:\n"
        "1) Exclude function headers, version numbers, paths, and section names.\n"
        "2) Keep label text exactly as seen.\n"
        "3) For _t/_T/_CUR labels with table values, serialize value compactly as a list "
        "or x:y pairs in one string.\n"
        "4) If value is missing, set value to \"\".\n"
        "5) Do not output markdown or explanations."
    )
    user_prompt = (
        f"Function: {function_name}\n"
        "Extract labels from this Standard calibration hints text:\n\n"
        f"{chunk_text}"
    )
    return system_prompt, user_prompt


def normalize_function_name(raw_name: str) -> str:
    return raw_name.strip()


def normalize_version(raw_version: str) -> str:
    return re.sub(r"\s+", "", raw_version.strip())


def detect_function_header(page_text: str) -> tuple[str, str] | None:
    for line in page_text.splitlines():
        for pattern in _HEADER_PATTERNS:
            match = pattern.search(line)
            if match:
                return normalize_function_name(match.group(1)), normalize_version(match.group(2))
    return None


def extract_pages(pdf_path: str) -> list[str]:
    doc = fitz.open(pdf_path)
    pages: list[str] = []
    try:
        for page in doc:
            pages.append(page.get_text("text"))
    finally:
        doc.close()
    return pages


def segment_into_function_blocks(pages: list[str]) -> list[FunctionBlock]:
    blocks: list[FunctionBlock] = []
    current: FunctionBlock | None = None

    for page_no, page_text in enumerate(pages, start=1):
        header = detect_function_header(page_text)
        if header is None:
            if current is not None:
                current.text += "\n" + page_text
                current.pages.append(page_no)
            continue

        name, version = header
        identity = (name, version)
        current_identity = (current.function_name, current.version) if current else None

        if identity != current_identity:
            current = FunctionBlock(function_name=name, version=version)
            blocks.append(current)

        current.text += "\n" + page_text
        current.pages.append(page_no)

    return blocks


def chunk_block_text(text: str, max_tokens: int, overlap_lines: int) -> list[str]:
    lines = text.splitlines()
    chunks: list[str] = []
    i = 0

    while i < len(lines):
        token_count = 0
        j = i
        window_lines: list[str] = []

        while j < len(lines):
            candidate = lines[j]
            next_tokens = _TOK.count(candidate + "\n")
            if window_lines and token_count + next_tokens > max_tokens:
                break
            window_lines.append(candidate)
            token_count += next_tokens
            j += 1

        if not window_lines:
            window_lines = [lines[i]]
            j = i + 1

        chunks.append("\n".join(window_lines))

        if j >= len(lines):
            break

        i = max(i + 1, j - overlap_lines)

    return chunks


def _claude_endpoint() -> str:
    direct = os.getenv("CLAUDE_ENDPOINT", "").strip() or os.getenv("VERTEX_ENDPOINT", "").strip()
    if direct:
        return direct

    base = os.getenv("CLAUDE_BASE_URL", "https://aoai-farm.bosch-temp.com").rstrip("/")
    model = os.getenv("VERTEX_ANTHROPIC_MODEL", "claude-haiku-4-5@20251001")
    return f"{base}/api/google/v1/publishers/anthropic/models/{model}:rawPredict"


def _azure_openai_endpoint() -> str:
    direct = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
    if direct:
        return direct

    base = os.getenv("BMF_BASE_URL", "https://aoai-farm.bosch-temp.com/api").rstrip("/")
    deployment = os.getenv(
        "AZURE_OPENAI_DEPLOYMENT",
        "askbosch-prod-farm-openai-gpt-41-2025-04-14",
    ).strip()
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview").strip()
    return f"{base}/openai/deployments/{deployment}/chat/completions?api-version={api_version}"


def _build_auth_headers(api_key: str, mode: str) -> dict[str, str]:
    mode_norm = (mode or "bearer").strip().lower()
    if mode_norm in {"all", "all-headers", "all-subscription-headers"}:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "api-key": api_key,
            "x-api-key": api_key,
            "Ocp-Apim-Subscription-Key": api_key,
            "subscription-key": api_key,
            "x-subscription-key": api_key,
            "genaiplatform-farm-subscription-key": api_key,
        }
    if mode_norm in {"genaiplatform-farm-subscription-key", "farm-subscription-key"}:
        return {
            "Content-Type": "application/json",
            "genaiplatform-farm-subscription-key": api_key,
        }
    if mode_norm == "api-key":
        return {"Content-Type": "application/json", "api-key": api_key}
    if mode_norm == "x-api-key":
        return {"Content-Type": "application/json", "x-api-key": api_key}
    if mode_norm in {"ocp", "ocp-apim", "ocp-apim-subscription-key"}:
        return {"Content-Type": "application/json", "Ocp-Apim-Subscription-Key": api_key}
    if mode_norm in {"subscription-key", "subscription"}:
        return {"Content-Type": "application/json", "subscription-key": api_key}
    if mode_norm in {"x-subscription-key", "x-subscription"}:
        return {"Content-Type": "application/json", "x-subscription-key": api_key}
    return {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}


def _extract_text_from_llm_response(data: dict[str, Any]) -> str:
    if isinstance(data.get("content"), list):
        texts = [
            block.get("text", "")
            for block in data["content"]
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        merged = "\n".join(t for t in texts if t).strip()
        if merged:
            return merged

    msg = data.get("choices", [{}])[0].get("message", {}).get("content")
    if isinstance(msg, str):
        return msg.strip()
    if isinstance(msg, list):
        return "\n".join(part.get("text", "") for part in msg if isinstance(part, dict)).strip()

    return json.dumps(data)


def _post_with_auth_modes(
    endpoint: str,
    payload: dict[str, Any],
    api_key: str,
    auth_mode: str,
) -> tuple[requests.Response, list[str]]:
    candidate_modes = [
        auth_mode.strip().lower() or "bearer",
        "genaiplatform-farm-subscription-key",
        "all-subscription-headers",
        "bearer",
        "api-key",
        "x-api-key",
        "ocp-apim-subscription-key",
        "subscription-key",
        "x-subscription-key",
    ]

    tried_modes: list[str] = []
    response: requests.Response | None = None
    for mode in candidate_modes:
        if mode in tried_modes:
            continue
        tried_modes.append(mode)
        headers = _build_auth_headers(api_key, mode)
        attempt = requests.post(endpoint, headers=headers, json=payload, timeout=120)
        response = attempt
        if attempt.status_code == 200:
            break
        if attempt.status_code not in (401, 403):
            break

    if response is None:
        raise RuntimeError("No LLM response received.")

    if response.status_code == 401:
        body = response.text.lower()
        if "invalid subscription key" in body:
            raise RuntimeError(
                "BMF authentication failed: invalid subscription key for this endpoint/subscription. "
                "Verify key validity and assigned BMF product access with IT, or use the correct BMF environment URL."
            )

    return response, tried_modes


def _ask_via_azure_openai(
    *,
    endpoint: str,
    function_name: str,
    chunk_text: str,
    temperature: float,
    max_tokens: int,
    api_key: str,
    auth_mode: str,
) -> str:
    system_prompt, user_prompt = _build_decoder_prompts(function_name, chunk_text)

    deployment = os.getenv(
        "AZURE_OPENAI_DEPLOYMENT",
        "askbosch-prod-farm-openai-gpt-41-2025-04-14",
    ).strip()
    payload = {
        "model": deployment,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    response, tried_modes = _post_with_auth_modes(endpoint, payload, api_key, auth_mode)
    if response.status_code != 200:
        raise RuntimeError(
            "Azure OpenAI request failed "
            f"[{response.status_code}] endpoint={endpoint} modes={tried_modes}: "
            f"{response.text[:1000]}"
        )
    return _extract_text_from_llm_response(response.json())


def _ask_via_claude_vertex(
    *,
    endpoint: str,
    function_name: str,
    chunk_text: str,
    temperature: float,
    max_tokens: int,
    api_key: str,
    auth_mode: str,
) -> str:
    system_prompt, user_prompt = _build_decoder_prompts(function_name, chunk_text)

    payload = {
        "anthropic_version": "vertex-2023-10-16",
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    response, tried_modes = _post_with_auth_modes(endpoint, payload, api_key, auth_mode)
    if response.status_code != 200:
        raise RuntimeError(
            "Claude request failed "
            f"[{response.status_code}] endpoint={endpoint} modes={tried_modes}: "
            f"{response.text[:1000]}"
        )
    return _extract_text_from_llm_response(response.json())


def ask_llm_extract_labels(function_name: str, chunk_text: str, *, temperature: float, max_tokens: int) -> str:
    api_key = _get_subscription_key()
    if not api_key:
        raise RuntimeError(
            "Set one of: BMF_API_KEY, GENAIPLATFORM_FARM_SUBSCRIPTION_KEY, "
            "genaiplatform-farm-subscription-key, CLAUDE_API_KEY, API_KEY."
        )

    auth_mode = os.getenv("LLM_FARM_AUTH_MODE", "bearer")
    backend = os.getenv("LLM_BACKEND", "azure-openai").strip().lower()

    if backend in {"azure", "azure-openai", "openai"}:
        return _ask_via_azure_openai(
            endpoint=_azure_openai_endpoint(),
            function_name=function_name,
            chunk_text=chunk_text,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            auth_mode=auth_mode,
        )

    if backend in {"claude", "vertex", "anthropic"}:
        return _ask_via_claude_vertex(
            endpoint=_claude_endpoint(),
            function_name=function_name,
            chunk_text=chunk_text,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            auth_mode=auth_mode,
        )

    if backend == "auto":
        errors: list[str] = []
        try:
            return _ask_via_azure_openai(
                endpoint=_azure_openai_endpoint(),
                function_name=function_name,
                chunk_text=chunk_text,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=api_key,
                auth_mode=auth_mode,
            )
        except Exception as exc:
            errors.append(f"azure-openai: {exc}")

        try:
            return _ask_via_claude_vertex(
                endpoint=_claude_endpoint(),
                function_name=function_name,
                chunk_text=chunk_text,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=api_key,
                auth_mode=auth_mode,
            )
        except Exception as exc:
            errors.append(f"claude: {exc}")

        raise RuntimeError("All LLM backends failed. " + " | ".join(errors))

    raise RuntimeError(
        "Invalid LLM_BACKEND. Use one of: azure-openai, claude, auto."
    )


def _parse_llm_json_array(text: str) -> list[dict[str, str]]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9_\-]*\n", "", cleaned)
        cleaned = re.sub(r"\n```$", "", cleaned)

    try:
        data = json.loads(cleaned)
        if isinstance(data, list):
            return [d for d in data if isinstance(d, dict)]
    except json.JSONDecodeError:
        pass

    # Fallback: capture the first JSON array if extra text leaked into the response.
    match = re.search(r"\[.*\]", cleaned, flags=re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            if isinstance(data, list):
                return [d for d in data if isinstance(d, dict)]
        except json.JSONDecodeError:
            return []

    return []


def _compress_pages(pages: list[int]) -> str:
    if not pages:
        return ""
    sorted_pages = sorted(set(pages))
    ranges: list[str] = []
    start = sorted_pages[0]
    prev = sorted_pages[0]

    for p in sorted_pages[1:]:
        if p == prev + 1:
            prev = p
            continue
        ranges.append(f"{start}-{prev}" if start != prev else str(start))
        start = p
        prev = p

    ranges.append(f"{start}-{prev}" if start != prev else str(start))
    return ", ".join(ranges)


def extract_label_rows_from_pdf(
    pdf_path: str,
    *,
    max_input_tokens: int,
    overlap_lines: int,
    temperature: float,
    max_output_tokens: int,
) -> list[LabelRow]:
    pages = extract_pages(pdf_path)
    blocks = segment_into_function_blocks(pages)
    rows: dict[tuple[str, str], LabelRow] = {}

    print(f"[info] Processing {Path(pdf_path).name}: {len(blocks)} function blocks")

    for block_idx, block in enumerate(blocks, start=1):
        section_texts = _extract_standard_hint_sections(block.text)
        page_str = _compress_pages(block.pages)

        if not section_texts:
            print(
                "[info] "
                f"{Path(pdf_path).name} block {block_idx}/{len(blocks)} "
                f"function={block.function_name} skipped (no Standard calibration hints section)"
            )
            continue

        print(
            "[info] "
            f"{Path(pdf_path).name} block {block_idx}/{len(blocks)} "
            f"function={block.function_name} sections={len(section_texts)}"
        )

        for section_idx, section_text in enumerate(section_texts, start=1):
            deterministic_rows = _extract_label_rows_from_section_text(
                function_name=block.function_name,
                section_text=section_text,
                page_str=page_str,
            )

            if deterministic_rows:
                for drow in deterministic_rows:
                    key = (drow.function_name, drow.label_name)
                    existing = rows.get(key)
                    if existing is None:
                        rows[key] = drow
                        continue
                    existing_score = len(existing.context) + len(existing.description) + len(existing.label_value)
                    new_score = len(drow.context) + len(drow.description) + len(drow.label_value)
                    if new_score > existing_score:
                        rows[key] = drow
                continue

            chunk_texts = chunk_block_text(
                section_text,
                max_tokens=max_input_tokens,
                overlap_lines=overlap_lines,
            )

            for chunk_idx, chunk_text in enumerate(chunk_texts, start=1):
                print(
                    "[info] "
                    f"{Path(pdf_path).name} block {block_idx}/{len(blocks)} "
                    f"section {section_idx}/{len(section_texts)} "
                    f"chunk {chunk_idx}/{len(chunk_texts)} "
                    f"function={block.function_name}"
                )

                llm_text = ask_llm_extract_labels(
                    function_name=block.function_name,
                    chunk_text=chunk_text,
                    temperature=temperature,
                    max_tokens=max_output_tokens,
                )
                candidates = _parse_llm_json_array(llm_text)

                for item in candidates:
                    label = _normalize_label_candidate(str(item.get("label", "")))
                    if not _looks_like_ecu_label(label):
                        continue

                    suffix_identified, suffix_category = _analyze_suffix(label)
                    description = _normalize_whitespace(str(item.get("description", "")).strip())
                    label_value = _normalize_whitespace(str(item.get("value", "")).strip())
                    section_context = _normalize_whitespace(str(item.get("section_context", "")).strip())
                    model_value_structure = _normalize_whitespace(
                        str(item.get("value_structure", "")).strip()
                    )
                    value_structure = model_value_structure or _infer_value_structure(
                        label_value,
                        suffix_category,
                        suffix_identified,
                    )

                    if not _is_relevant_label(
                        label=label,
                        function_name=block.function_name,
                        suffix_category=suffix_category,
                        description=description,
                        value=label_value,
                    ):
                        continue

                    base_context = section_context or description
                    context = _trim_text(base_context)

                    if not context:
                        continue

                    key = (block.function_name, label)
                    candidate_row = LabelRow(
                        function_name=block.function_name,
                        label_name=label,
                        description=description,
                        label_value=label_value,
                        value_structure=value_structure,
                        suffix_identified=suffix_identified,
                        context=context,
                        pages=page_str,
                    )
                    existing = rows.get(key)
                    if existing is None:
                        rows[key] = candidate_row
                        continue

                    existing_score = len(existing.context) + len(existing.description) + len(existing.label_value)
                    new_score = len(candidate_row.context) + len(candidate_row.description) + len(candidate_row.label_value)
                    if new_score > existing_score:
                        rows[key] = candidate_row

    return list(rows.values())


def rows_to_dataframe(rows: list[LabelRow]) -> pd.DataFrame:
    records = [
        {
            "Function Component": r.function_name,
            "Label Name": r.label_name,
            "Description": r.description,
            "Value": r.label_value,
            "Value Type": r.value_structure,
            "Suffix": r.suffix_identified,
            "Context": r.context,
            "Section Page(s)": r.pages,
        }
        for r in rows
    ]

    df = pd.DataFrame(records)
    if df.empty:
        return pd.DataFrame(
            columns=[
                "Function Component",
                "Label Name",
                "Description",
                "Value",
                "Value Type",
                "Suffix",
                "Context",
                "Section Page(s)",
            ]
        )

    return df.sort_values(by=["Function Component", "Label Name"]).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract labels/context from source and destination software docs using LLM and write to one Excel file with two sheets."
    )
    parser.add_argument("--source-pdf", required=True, help="Path to source PDF document.")
    parser.add_argument("--destination-pdf", required=True, help="Path to destination PDF document.")
    parser.add_argument("--output-xlsx", required=True, help="Path to output Excel file (.xlsx).")
    parser.add_argument("--max-input-tokens", type=int, default=2800, help="Approx input token budget per chunk.")
    parser.add_argument("--overlap-lines", type=int, default=6, help="Line overlap between chunks.")
    parser.add_argument("--temperature", type=float, default=0.0, help="LLM temperature.")
    parser.add_argument("--max-output-tokens", type=int, default=1200, help="LLM max output tokens per chunk.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    source_rows = extract_label_rows_from_pdf(
        args.source_pdf,
        max_input_tokens=args.max_input_tokens,
        overlap_lines=args.overlap_lines,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
    )
    destination_rows = extract_label_rows_from_pdf(
        args.destination_pdf,
        max_input_tokens=args.max_input_tokens,
        overlap_lines=args.overlap_lines,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
    )

    source_df = rows_to_dataframe(source_rows)
    destination_df = rows_to_dataframe(destination_rows)

    output_path = Path(args.output_xlsx)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        source_df.to_excel(writer, sheet_name="Source", index=False)
        destination_df.to_excel(writer, sheet_name="Destination", index=False)

    print(f"[done] Wrote workbook: {output_path}")
    print(f"[done] Source rows: {len(source_df)}")
    print(f"[done] Destination rows: {len(destination_df)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
