# Data directory

Repo-root shared data tree (not the Chapter 04 pipeline folder).

- `raw/`: large raw inputs (usually gitignored). Full Lexis / newspaper corpora are **not shipped** here for copyright reasons.
- `processed/`: shared frozen panels, impact databases, and geospatial outputs.

Chapter 04 acquisition → preprocessing → LLMn outputs live under:

`chapters/04_database_construction/data/`

See [`chapters/04_database_construction/README.md`](../chapters/04_database_construction/README.md).
