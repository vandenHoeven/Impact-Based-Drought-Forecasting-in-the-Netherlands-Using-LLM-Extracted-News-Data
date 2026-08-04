# Thesis-final geocoding outputs

These are the **correct thesis-final** products from the Gemini 3.5 Flash **flex** geocoding run used with the Chapter 7 baseline pipeline.

Prefer this folder over `data/processed/` when you need the real thesis spatial data.

## Layout

```text
final/
  points/     # point-coder (Nominatim + ranking)
  nuts3/      # nuts3-coder (NUTS-3 assignment from points)
```

| Path | In Git? | Role |
| --- | --- | --- |
| `points/..._geocoded.csv` | yes | Flat point-geocoded table (viewer default) |
| `points/..._geocoded_filtered.csv` | yes | Filtered point table |
| `nuts3/..._nuts3.csv` | yes | Flat NUTS-3 assigned table (viewer default) |
| `points/..._geocoded.json` | **local only** | Nested articles + geocode fields (~240 MB) |
| `nuts3/..._nuts3.json` | **local only** | Nested articles + NUTS fields (~250 MB) |

Matching full LLM input JSON (`../input/chapter7_merged_..._flex.json`, ~230 MB) is also **local only**. See the Chapter 06 README section *Local-only large files*.

## Not here

- Older package samples: `../processed/impacts_*.csv`, `../input/impacts_for_geocoding.json` (legacy; not this flex thesis run)
- NUTS-1 / NUTS-2 coder outputs (not part of Chapter 06)
