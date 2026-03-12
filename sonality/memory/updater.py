from __future__ import annotations

import logging
from enum import StrEnum
from typing import Final

from pydantic import BaseModel

from .. import config
from ..ess import ESSResult
from ..llm.caller import llm_call
from ..prompts import INSIGHT_PROMPT

log = logging.getLogger(__name__)

MIN_SNAPSHOT_RETENTION: Final = 0.6
SNAPSHOT_CHAR_LIMIT: Final[int] = config.SPONGE_MAX_TOKENS * 5


class InsightDecision(StrEnum):
    EXTRACT = "EXTRACT"
    SKIP = "SKIP"


class InsightExtractionResponse(BaseModel):
    insight_decision: InsightDecision = InsightDecision.SKIP
    insight_text: str = ""


def validate_snapshot(old: str, new: str) -> bool:
    """Reject snapshots that lost too much content.

    Repeated LLM rewrites are lossy — minority opinions and distinctive traits
    can silently vanish. This check catches catastrophic content loss.
    (Open Character Training 2025: persona traits = neural activation patterns;
    losing a sentence = losing a trait)
    """
    if not new or len(new) < 30:
        log.warning("Snapshot validation failed: new snapshot too short (%d chars)", len(new))
        return False
    ratio = len(new) / max(len(old), 1)
    if ratio < MIN_SNAPSHOT_RETENTION:
        log.warning(
            "Snapshot validation failed: content ratio %.2f < %.2f (%d -> %d chars)",
            ratio,
            MIN_SNAPSHOT_RETENTION,
            len(old),
            len(new),
        )
        return False
    return True


def extract_insight(
    ess: ESSResult,
    user_message: str,
    agent_response: str,
    model: str = config.ESS_MODEL,
) -> str:
    """Extract a personality-relevant insight from an interaction.

    Accumulate-then-consolidate approach: insights are appended per-interaction
    and integrated into the snapshot during reflection. Avoids lossy per-interaction
    full rewrites. (ABBEL 2025: belief bottleneck; Park et al. 2023: reflection
    is the critical mechanism for believable agents)
    """
    prompt = INSIGHT_PROMPT.format(
        user_message=user_message[:300],
        agent_response=agent_response[:300],
        ess_score=f"{ess.score:.2f}",
    )
    result = llm_call(
        prompt=prompt,
        response_model=InsightExtractionResponse,
        fallback=InsightExtractionResponse(),
        model=model,
        max_tokens=config.FAST_LLM_MAX_TOKENS,
    )
    if not result.success:
        raise ValueError("Insight extraction returned invalid decision payload")
    response = result.value
    if response.insight_decision is not InsightDecision.EXTRACT:
        log.info("No personality insight extracted")
        return ""
    text = response.insight_text.strip()
    if not text:
        log.info("No personality insight extracted")
        return ""
    log.info("Insight extracted: %s", text[:80])
    return text
