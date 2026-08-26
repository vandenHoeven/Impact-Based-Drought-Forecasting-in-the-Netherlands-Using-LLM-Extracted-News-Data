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

# Using Annotated for cleaner, reusable constraints
# strip_whitespace=True is a lifesaver for data cleaning
LocationString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
EvidenceString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=250)]


# Pydantic classes to force structured output
class DroughtImpactEvent(BaseModel):
    reasoning: str = Field(
        description=(
            "First, identify if the text describes a *realized* consequence of drought. "
            "Then, explain why this quote supports the specific classification and severity you chose. "
            "If the impact is only a future prediction, explain why it should be excluded or marked as risk."
        )
    )
    location: LocationString = Field(
        description="The specific geographic place name as written in the text. "
                    "Be as specific as possible (e.g., 'Lower Murray-Darling Basin' instead of just 'Australia')."
    )
    drought_impact_classification: DroughtImpactLabel = Field(
        description="Select the most relevant category from the allowed list that defines the impact."
    )
    recency_in_months: RecencyBucket = Field(
        description=(
            "Estimated time since impact. Use 0 for 'current/ongoing', 1 for 'within last month', "
            "2-6 for months ago, 12 to indicate around 12 months, and 24 to indicate 24+ months."
        )
    )
    severity: Literal[1, 2, 3] = Field(
        description="1: Localized/Moderate, 2: Widespread/Severe, 3: Large-scale/Extreme/Cascading."
    )
    evidence: EvidenceString = Field(
        description="A verbatim quote of 25 words or less. It MUST contain the location and the impact."
    )


class DroughtImpactExtraction(BaseModel):
    events: List[DroughtImpactEvent] = Field(
        description="A list of distinct drought impact events. If the article mentions multiple "
                    "locations or multiple types of impacts, create a separate entry for each.",
        default_factory=list
    )

# Drought impact extraction task prompt
SYSTEM_PROMPT = """Task: Extract drought-related impact events from a news article.

Return a structured JSON with one or more distinct drought impact events.
Each event captures: reasoning, location, classification, recency, severity, and evidence.

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
- reasoning: Explain step-by-step why this constitutes a drought impact before filling other fields.
- location: The most specific geographic place name as written in the text (e.g., 'Lower Murray-Darling Basin' not just 'Australia').
- drought_impact_classification: Choose from the list above.
- recency_in_months: Use 0 for 'current/ongoing', 1 for 'within last month', 2-6 for months ago, 12 for around 12 months, and 24 for 24+ months.
- severity: 1 (Localized/Moderate), 2 (Widespread/Severe), 3 (Large-scale/Extreme/Cascading).
- evidence: A direct quote (≤25 words) that MUST mention both the location AND the impact. No paraphrasing.

CRITICAL RULES:
1. Follow the provided JSON schema strictly.
2. Use location names EXACTLY as written in the text.
3. Each event = one distinct location + impact pair.
4. Evidence must be a verbatim quote containing the location.
5. Only include real consequences of drought; avoid vague inferences.
6. Return an empty list if no meaningful drought impacts are present."""