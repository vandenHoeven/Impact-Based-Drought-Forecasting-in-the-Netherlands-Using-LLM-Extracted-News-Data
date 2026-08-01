"""Final Pydantic schema and system prompt for LLMn drought-impact extraction."""

from __future__ import annotations

from typing import Annotated, List, Literal

from pydantic import BaseModel, Field, StringConstraints

DroughtImpactLabel = Literal[
    "Crop Failure & Yield Reduction",
    "Livestock Stress & Mortality",
    "Irrigation Shortage",
    "Groundwater Depletion",
    "Reservoir & Surface Water Shortage",
    "Water Use Restrictions",
    "Hydropower Reduction",
    "Thermal/Nuclear Cooling Constraints",
    "Industrial Water Shortages",
    "Inland Waterway Disruption",
    "Freshwater Ecosystem Degradation",
    "Forest Dieback & Vegetation Stress",
    "Wetland Loss",
    "Wildfire Occurrence",
    "Wildfire Risk Increase",
    "Heat & Air Quality Health Impacts",
    "Water Supply & Sanitation Issues",
    "Agricultural Economic Loss",
    "Broader Economic Disruption",
    "Social Impacts",
]

RecencyBucket = Literal[0, 1, 2, 3, 4, 5, 6, 12, 24]

LocationString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
EvidenceString = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=700)
]


class Impact(BaseModel):
    classification: DroughtImpactLabel
    severity: Literal[1, 2, 3] = Field(
        description="Severity scale: 1=Low/Minor, 2=Medium/Moderate, 3=High/Extreme."
    )
    evidence: EvidenceString = Field(
        description="Verbatim quote (≤50 words) supporting THIS impact only."
    )
    confidence: Literal[0.0, 0.25, 0.5, 0.75, 1.0] = Field(
        description=(
            "Directness of evidence: 1.0=Explicitly stated, 0.5=Strongly implied, "
            "0.0=Guesswork."
        )
    )


class DroughtImpactEvent(BaseModel):
    reasoning: str = Field(
        description=(
            "Explain why this location contains drought impacts and how the impacts "
            "are derived from the text."
        )
    )
    location: LocationString = Field(
        description="The exact place name as written in the text."
    )
    recency_in_months: RecencyBucket = Field(
        description=(
            "Estimated time since impact. Use 0 for 'current/ongoing', 1 for "
            "'within last month', 2-6 for months ago, 12 to indicate around 12 months, "
            "and 24 to indicate 24+ months."
        )
    )
    impacts: List[Impact] = Field(
        description=(
            "All drought impacts explicitly supported in the text for this "
            "location/context."
        )
    )


class DroughtImpactExtraction(BaseModel):
    events: List[DroughtImpactEvent] = Field(
        description=(
            "A list of distinct drought impact events. If the article mentions multiple "
            "locations or multiple contexts, create a separate entry for each."
        ),
        default_factory=list,
    )


SYSTEM_PROMPT = """Task: Extract drought-related impact events from a news article.

Return a structured JSON with one or more distinct drought impact events.
Each event captures: reasoning, location, recency, and a list of impacts.

IMPACT CATEGORIES:
- Crop Failure & Yield Reduction
- Livestock Stress & Mortality
- Irrigation Shortage
- Groundwater Depletion
- Reservoir & Surface Water Shortage
- Water Use Restrictions
- Hydropower Reduction
- Thermal/Nuclear Cooling Constraints
- Industrial Water Shortages
- Inland Waterway Disruption
- Freshwater Ecosystem Degradation
- Forest Dieback & Vegetation Stress
- Wetland Loss
- Wildfire Occurrence
- Wildfire Risk Increase
- Heat & Air Quality Health Impacts
- Water Supply & Sanitation Issues
- Agricultural Economic Loss
- Broader Economic Disruption
- Social Impacts

FIELD GUIDANCE:
- reasoning:
    Explain why this location contains drought impacts and how impacts are derived from the text.

- location:
    Exact place name as written in the text.

- recency_in_months:
    Estimate time since impact.

- impacts:
    Extract ALL distinct drought impacts explicitly supported by the text.

    For each impact:
        - classification: choose ONE category
        - severity: rate this specific impact only, using the scale:
            1 = Low/Minor: small, localized, manageable impacts handled by standard
                local management (minor financial loss or biological stress).
            2 = Medium/Moderate: noticeable regional-scale impacts with enforced
                restrictions or adaptive changes (water-use bans, reduced cargo
                loads, measurable regional yield reductions).
            3 = High/Extreme: widespread or catastrophic systemic failure triggering
                institutional emergency interventions (bailouts, complete crop
                failure, ecological collapse, power plant shutdowns).
        - evidence: quote ≤50 words (max 700 characters) supporting ONLY this impact
        - confidence: rate directness of evidence using EXACTLY one of
          0.0, 0.25, 0.5, 0.75, 1.0:
            1.0 = explicitly stated, 0.75 = clearly supported,
            0.5 = strongly implied, 0.25 = weakly implied, 0.0 = guesswork.
    When multiple impact categories are directly supported by the same evidence,
    include all applicable impacts.
    Prefer recall over selecting a single dominant category.
    If two categories are both explicitly stated, output both.

CRITICAL RULES:
1. Follow the provided JSON schema strictly.
2. Use location names EXACTLY as written in the text.
3. Each event = one location + multiple impact-specific claims.
4. Each impact must have its own evidence quote.
5. Only include real consequences of drought; avoid vague inferences.
6. Return an empty list if no meaningful drought impacts are present.
7. Exclude future adaptations, policy planning, and proposed changes (e.g., plans to build a reservoir next time).
"""
