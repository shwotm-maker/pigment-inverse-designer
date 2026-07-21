"""Optional, provider-agnostic AI language layer.

Numeric λmax prediction always stays with the local ML model (more accurate for
quantitative prediction). The LLM only does the "language" parts:

1. parse_natural_language_target() : free-text → target nm + solvent + constraints
2. interpret_results()             : Korean markdown report over the candidates
3. comment_on_candidate()          : short Korean note per candidate

Backends (choose in the UI / config.AI_PROVIDERS):
* Claude (Anthropic)  — best quality, paid.
* Google Gemini, Groq, OpenRouter, Cerebras, Mistral, NVIDIA NIM — free tiers,
  all reached through the OpenAI-compatible Chat Completions API (one `openai`
  client, different base_url + model). Quality is lower than Claude but fine for
  this language layer.

Design rules (README section 15):
* The app must run fully WITHOUT any AI — every function degrades to a clear
  "AI 미사용" fallback when no key is configured or a call fails.
* The model is instructed NOT to invent synthesis procedures / reaction
  conditions and to preserve the research-disclaimer framing.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

from . import config
from .utils import get_logger

logger = get_logger("ai_assistant")

MAX_TOKENS_PARSE = 1024
MAX_TOKENS_REPORT = 3500
MAX_TOKENS_COMMENT = 600

_SAFETY_SYSTEM = (
    "당신은 유기 발색단·안료 연구를 돕는 화학정보학 어시스턴트입니다. "
    "이 도구는 공개 데이터와 로컬 머신러닝 모델에 기반한 '가상 스크리닝' 개념검증 도구입니다.\n"
    "반드시 지켜야 할 제약:\n"
    "1) 구체적인 합성 절차, 시약, 반응조건, 촉매, 온도/시간 등 실험 레시피를 절대 생성하지 마십시오.\n"
    "2) 실제 합성 가능성·색상 정확도·독성/안전성·특허 신규성·산업 성능을 보증하지 마십시오.\n"
    "3) 정량적 최대흡수파장(λmax) 예측치는 이미 로컬 ML 모델이 제공합니다. 그 값을 재발명하지 말고 "
    "해석·요약·정성 코멘트에만 사용하십시오.\n"
    "4) 항상 한국어로, 간결하고 실무적으로 답하십시오.\n"
    "5) 색상과 흡수파장은 일대일 대응이 아니며 실제 색은 여러 요인에 좌우됨을 필요한 곳에서 언급하십시오."
)

# Text schema description used for JSON parsing across all providers.
_TARGET_JSON_SPEC = (
    "다음 키를 가진 JSON 객체 '하나만' 출력하세요(코드펜스·설명 없이 순수 JSON):\n"
    '{\n'
    '  "target_absorption_nm": number|null,   // 목표 최대흡수파장(nm). 색상만 있으면 보색 기준 근사치(400~700).\n'
    '  "solvent_name": string|null,           // 아래 옵션 중 하나 또는 null\n'
    '  "min_mol_weight": number|null,\n'
    '  "max_mol_weight": number|null,\n'
    '  "max_logp": number|null,\n'
    '  "excluded_elements": string[],         // 제외 원소 기호, 없으면 []\n'
    '  "reasoning": string                    // 1-2문장 한국어 근거(색상→파장은 근사임 명시)\n'
    '}'
)


@dataclass
class AISettings:
    """Which backend to use, plus the key and (optional) model override."""

    provider: str = config.AI_DEFAULT_PROVIDER
    api_key: str | None = None
    model: str | None = None

    def spec(self) -> dict:
        return config.AI_PROVIDERS[self.provider]

    def resolved_key(self) -> str | None:
        if self.api_key and self.api_key.strip():
            return self.api_key.strip()
        env = self.spec().get("key_env")
        return os.environ.get(env) if env else None

    def resolved_model(self) -> str:
        return (self.model or "").strip() or self.spec()["default_model"]


@dataclass
class AIResult:
    """Either content, or a human-readable reason for its absence."""

    ok: bool
    content: object = None
    error: str = ""


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------
def is_available(settings: AISettings) -> bool:
    """True if the required SDK is importable and a key is resolvable."""
    if settings.resolved_key() is None:
        return False
    sdk = settings.spec()["sdk"]
    try:
        if sdk == "anthropic":
            import anthropic  # noqa: F401
        else:
            import openai  # noqa: F401
    except ImportError:
        return False
    return True


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------
def _anthropic_chat(settings: AISettings, system: str, user: str,
                    max_tokens: int, thinking: bool) -> str:
    import anthropic

    kwargs = {"api_key": settings.resolved_key()}
    base_url = os.environ.get("ANTHROPIC_BASE_URL")
    if base_url:
        kwargs["base_url"] = base_url
    client = anthropic.Anthropic(**kwargs)
    create_kwargs = {
        "model": settings.resolved_model(),
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }
    if thinking:
        create_kwargs["thinking"] = {"type": "adaptive"}
    resp = client.messages.create(**create_kwargs)
    for block in resp.content:
        if block.type == "text":
            return block.text
    return ""


def _openai_chat(settings: AISettings, system: str, user: str,
                 max_tokens: int, json_mode: bool) -> str:
    """Chat via any OpenAI-compatible endpoint (Gemini/Groq/OpenRouter/...)."""
    from openai import OpenAI

    spec = settings.spec()
    client = OpenAI(api_key=settings.resolved_key(), base_url=spec["base_url"])
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    common = {
        "model": settings.resolved_model(),
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.3,
    }
    # Try JSON mode where supported; fall back gracefully if the provider rejects it.
    if json_mode:
        try:
            resp = client.chat.completions.create(
                response_format={"type": "json_object"}, **common
            )
            return resp.choices[0].message.content or ""
        except Exception as exc:  # provider doesn't support response_format
            logger.info("json_object mode unsupported (%s); retrying plain.", exc)
    resp = client.chat.completions.create(**common)
    return resp.choices[0].message.content or ""


def _chat(settings: AISettings, system: str, user: str, max_tokens: int,
          json_mode: bool = False, thinking: bool = False) -> str:
    if settings.spec()["sdk"] == "anthropic":
        return _anthropic_chat(settings, system, user, max_tokens, thinking)
    return _openai_chat(settings, system, user, max_tokens, json_mode)


def _handle_error(exc: Exception) -> str:
    """Map SDK exceptions (either SDK) to a short Korean message."""
    name = type(exc).__name__.lower()
    if "authentication" in name or "permission" in name:
        return "API 키 인증에 실패했습니다. 공급자와 키를 확인하세요."
    if "ratelimit" in name or "rate_limit" in name:
        return "무료 한도/속도 제한에 걸렸습니다. 잠시 후 다시 시도하세요."
    if "connection" in name or "timeout" in name:
        return "네트워크 연결 오류입니다. 인터넷 연결을 확인하세요."
    if "notfound" in name:
        return "모델 이름이 잘못되었을 수 있습니다. 사이드바에서 모델명을 확인하세요."
    return f"AI 호출 실패: {exc}"


def _extract_json(text: str) -> dict:
    """Parse a JSON object out of an LLM response (tolerates code fences/prose)."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    else:
        brace = re.search(r"\{.*\}", text, re.DOTALL)
        if brace:
            text = brace.group(0)
    return json.loads(text)


# ---------------------------------------------------------------------------
# 1) Natural-language target parsing
# ---------------------------------------------------------------------------
def parse_natural_language_target(text: str, settings: AISettings) -> AIResult:
    """Parse free text into a structured target dict (see _TARGET_JSON_SPEC)."""
    if settings.resolved_key() is None:
        return AIResult(ok=False, error="API 키가 없습니다.")
    if not text or not text.strip():
        return AIResult(ok=False, error="입력 텍스트가 비어 있습니다.")

    solvents = ", ".join(config.COMMON_SOLVENTS.keys())
    prompt = (
        f"{_TARGET_JSON_SPEC}\n\n"
        f"사용 가능한 용매/상태 옵션: {solvents}\n"
        f"사용자 요청: \"{text.strip()}\""
    )
    try:
        raw = _chat(settings, _SAFETY_SYSTEM, prompt, MAX_TOKENS_PARSE, json_mode=True)
        data = _extract_json(raw)
        data.setdefault("excluded_elements", [])
        return AIResult(ok=True, content=data)
    except json.JSONDecodeError:
        return AIResult(ok=False, error="AI 응답을 JSON으로 해석하지 못했습니다. 다시 시도하세요.")
    except Exception as exc:  # noqa: BLE001
        logger.error("parse failed: %s", exc)
        return AIResult(ok=False, error=_handle_error(exc))


# ---------------------------------------------------------------------------
# 2) Result interpretation report
# ---------------------------------------------------------------------------
def interpret_results(target_nm: float, candidates_records: list[dict],
                      candidate_type: str, settings: AISettings) -> AIResult:
    """Korean markdown report interpreting the top candidates."""
    if settings.resolved_key() is None:
        return AIResult(ok=False, error="API 키가 없습니다.")
    if not candidates_records:
        return AIResult(ok=False, error="해석할 후보가 없습니다.")

    table = json.dumps(candidates_records[:15], ensure_ascii=False, indent=2)
    prompt = (
        f"목표 최대흡수파장: {target_nm:.0f} nm\n"
        f"후보 유형: {candidate_type}\n\n"
        f"아래는 로컬 ML 모델과 RDKit이 계산한 후보 목록(JSON)입니다. "
        f"이 수치를 근거로 한국어 마크다운 보고서를 작성하세요.\n"
        f"1) 상위 후보 3~5개가 왜 유망한지(목표차·불확실성·유사도 관점)\n"
        f"2) 신규성(유사도)과 예측 신뢰도의 트레이드오프\n"
        f"3) 실무자가 다음에 확인할 점(주의사항)\n"
        f"수치를 지어내지 말고 제공된 값만 사용하세요. 합성 절차는 절대 쓰지 마세요.\n\n"
        f"후보 데이터:\n{table}"
    )
    try:
        out = _chat(settings, _SAFETY_SYSTEM, prompt, MAX_TOKENS_REPORT, thinking=True)
        return AIResult(ok=True, content=out)
    except Exception as exc:  # noqa: BLE001
        logger.error("interpret failed: %s", exc)
        return AIResult(ok=False, error=_handle_error(exc))


# ---------------------------------------------------------------------------
# 3) Per-candidate qualitative comment
# ---------------------------------------------------------------------------
def comment_on_candidate(record: dict, target_nm: float, settings: AISettings) -> AIResult:
    """Short (2-3 sentence) Korean qualitative note for one candidate."""
    if settings.resolved_key() is None:
        return AIResult(ok=False, error="API 키가 없습니다.")
    prompt = (
        f"목표 λmax {target_nm:.0f} nm. 아래 단일 후보에 대해 구조적 특징과 예측 성질을 "
        f"2~3문장 한국어로 요약하세요. 수치는 제공값만 사용하고 합성법은 쓰지 마세요.\n\n"
        f"{json.dumps(record, ensure_ascii=False)}"
    )
    try:
        out = _chat(settings, _SAFETY_SYSTEM, prompt, MAX_TOKENS_COMMENT)
        return AIResult(ok=True, content=out.strip())
    except Exception as exc:  # noqa: BLE001
        logger.error("comment failed: %s", exc)
        return AIResult(ok=False, error=_handle_error(exc))
