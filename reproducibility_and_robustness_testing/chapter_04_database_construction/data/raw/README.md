# Chapter 04 fixture raw newspaper data

Small **test sample** for preprocessing robustness checks—not redistributable full Lexis holdings (copyright). Place Lexis-like ZIP archives here for procedure tests only.

## Expected format
- File name: `Page_<start>_to_<end>.zip`
- Contents: one or more `.docx` articles

## Current fixture
- `Page_71_to_76.zip`

Tests in this folder discover all `*.zip` here and do **not** read from
`chapters/04_database_construction/data/raw/`.
