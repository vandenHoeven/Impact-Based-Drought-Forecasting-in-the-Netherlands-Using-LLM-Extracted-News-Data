from __future__ import annotations

import re
import unicodedata

# Common shortenings → NUTS1 ID
NUTS1_ALIASES: dict[str, str] = {
    "noord": "NL1",
    "oost": "NL2",
    "west": "NL3",
    "zuid": "NL4",
}

# Common shortenings → NUTS2 ID
NUTS2_ALIASES: dict[str, str] = {
    "brabant": "NL41",
    "limburg": "NL42",
    "friesland": "NL12",
    "fryslan": "NL12",
}

# Spelling / prefix variants → NUTS3 ID
NUTS3_ALIASES: dict[str, str] = {
    "zeeuws vlaanderen": "NL341",
    "zeeuwsch vlaanderen": "NL341",
    "de achterhoek": "NL225",
    "oost achterhoek": "NL225",
}

# CBS 4-region macros and frequent sub-regions.
# Values are either {"nuts2": [...]} or {"nuts3": [...]}.
MACRO_REGIONS: dict[str, dict[str, list[str]]] = {
    "noord nederland": {"nuts2": ["NL11", "NL12", "NL13"]},
    "oost nederland": {"nuts2": ["NL21", "NL22", "NL23"]},
    "west nederland": {"nuts2": ["NL31", "NL32", "NL33", "NL34"]},
    "zuid nederland": {"nuts2": ["NL41", "NL42"]},
    "hollands midden": {
        "nuts3": ["NL332", "NL333", "NL337", "NL33A", "NL33B", "NL33C"],
    },
    "midden brabant": {
        "nuts3": ["NL411", "NL412", "NL413", "NL414"],
    },
    "midden en west brabant": {
        "nuts3": ["NL411", "NL412", "NL413", "NL414"],
    },
    "westhoek": {
        "nuts3": ["NL411", "NL412", "NL413", "NL414"],
    },
    "zuid limburg": {"nuts3": ["NL423"]},
    "noord limburg": {"nuts3": ["NL421"]},
    "gelderland midden": {"nuts2": ["NL22"]},
}


def normalize_location_name(name: str) -> str:
    text = (name or "").strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"^(de|het|den|in de|in het)\s+", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_nuts_name(name: str) -> str:
    text = normalize_location_name(name)
    text = re.sub(r"\s*\(nl\)\s*$", "", text)
    return text.strip()
