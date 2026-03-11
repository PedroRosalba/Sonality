"""LLM-based belief evidence assessment and provenance tracking.

Replaces formula-based belief updating (log2 confidence, fixed contraction ratios)
with LLM semantic assessment of how new evidence affects existing beliefs.
Links beliefs to supporting/contradicting episodes via graph edges.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel

from ..llm.caller import llm_call
from ..llm.prompts import BELIEF_UPDATE_PROMPT
from .graph import EdgeType, MemoryGraph
from .sponge import BeliefMeta, SpongeState

log = logging.getLogger(__name__)


def _append_unique_uid(values: list[str], uid: str) -> None:
    """Append episode UID once while preserving insertion order."""
    if uid not in values:
        values.append(uid)


class UpdateMagnitude(StrEnum):
    MAJOR = "MAJOR"
    MINOR = "MINOR"


class ContractionAction(StrEnum):
    CONTRACT = "CONTRACT"
    NONE = "NONE"


class BeliefUpdateResponse(BaseModel):
    direction: float = 0.0
    evidence_strength: float = 0.5
    new_uncertainty: float = 0.5
    reasoning: str = ""
    update_magnitude: UpdateMagnitude = UpdateMagnitude.MINOR
    contraction_action: ContractionAction = ContractionAction.NONE


@dataclass(frozen=True, slots=True)
class ProvenanceUpdate:
    """Result of a belief evidence assessment."""

    topic: str
    direction: float
    evidence_strength: float
    new_uncertainty: float
    update_magnitude: UpdateMagnitude
    contraction_action: ContractionAction
    reasoning: str


async def assess_belief_evidence(
    *,
    topic: str,
    episode_uid: str,
    episode_content: str,
    ess_score: float,
    reasoning_type: str,
    source_reliability: str,
    sponge: SpongeState,
    graph: MemoryGraph,
) -> ProvenanceUpdate:
    """Use LLM to assess how new evidence affects a belief, then update provenance.

    Updates BeliefMeta with episode UIDs and uncertainty. Creates graph edges
    (SUPPORTS_BELIEF/CONTRADICTS_BELIEF) for provenance tracking.
    """
    meta = sponge.belief_meta.get(topic)
    current_value = sponge.opinion_vectors.get(topic, 0.0)

    prompt = BELIEF_UPDATE_PROMPT.format(
        topic=topic,
        current_value=f"{current_value:+.2f}",
        confidence=f"{meta.confidence:.2f}" if meta else "0.00",
        supporting_count=len(meta.supporting_episode_uids) if meta else 0,
        contradicting_count=len(meta.contradicting_episode_uids) if meta else 0,
        uncertainty=f"{meta.uncertainty:.2f}" if meta else "1.00",
        episode_content=episode_content[:1000],
        ess_score=f"{ess_score:.2f}",
        reasoning_type=reasoning_type,
        source_reliability=source_reliability,
    )
    result = llm_call(
        prompt=prompt,
        response_model=BeliefUpdateResponse,
        fallback=BeliefUpdateResponse(direction=0.0, evidence_strength=0.0),
    )
    if not result.success:
        raise ValueError("Belief evidence assessment returned invalid decision payload")
    response = result.value

    # Update provenance in BeliefMeta
    if meta is None:
        meta = BeliefMeta(formed_at=sponge.interaction_count)
        sponge.belief_meta[topic] = meta

    edge_type = EdgeType.SUPPORTS_BELIEF
    if response.direction > 0:
        _append_unique_uid(meta.supporting_episode_uids, episode_uid)
    else:
        edge_type = EdgeType.CONTRADICTS_BELIEF
        _append_unique_uid(meta.contradicting_episode_uids, episode_uid)
        meta.last_challenged_at = sponge.interaction_count

    meta.uncertainty = response.new_uncertainty

    # Create graph edge for provenance
    try:
        await graph.link_belief(
            episode_uid,
            topic,
            edge_type=edge_type,
            strength=response.evidence_strength,
            reasoning=response.reasoning[:200],
        )
    except Exception:
        log.exception("Failed to create belief provenance edge for %s", topic)

    return ProvenanceUpdate(
        topic=topic,
        direction=response.direction,
        evidence_strength=response.evidence_strength,
        new_uncertainty=response.new_uncertainty,
        update_magnitude=response.update_magnitude,
        contraction_action=response.contraction_action,
        reasoning=response.reasoning,
    )
