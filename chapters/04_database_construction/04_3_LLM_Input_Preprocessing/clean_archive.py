"""Global near-duplicate removal over raw archive data.

Pipeline:
1. Read all DOCX articles from every ZIP in chapter data/raw.
2. Write one combined JSON with all articles.
3. Run MinHash + LSH deduplication across the full corpus.
4. Extract article features such as title, top date, bottom date, publication date,
   and cleaned body text.
5. Write deduplicated output and a duplicate report JSON under chapter data/preprocessed.
"""

from __future__ import annotations

import json
import re
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path

from docx import Document
from tqdm import tqdm

try:
    from datasketch import MinHash, MinHashLSH
except ImportError as exc:
    raise SystemExit(
        "Missing dependency 'datasketch'. Install it with: pip install datasketch"
    ) from exc


MONTH_ALIASES = {
    "januari": "January",
    "februari": "February",
    "maart": "March",
    "april": "April",
    "mei": "May",
    "juni": "June",
    "juli": "July",
    "augustus": "August",
    "september": "September",
    "oktober": "October",
    "november": "November",
    "december": "December",
}

MONTHS = (
    "January|February|March|April|May|June|July|August|"
    "September|October|November|December"
)

DATE_LINE_PATTERN = re.compile(
    r"(?P<date>(?:\d{1,2}\s+[A-Za-zéèêëïîôöüç]+\s+\d{4})|(?:[A-Za-z]+\s+\d{1,2},?\s+\d{4}))",
    re.IGNORECASE,
)

PATTERN_LOAD_DATE = re.compile(
    rf"Load-Date:\s*(?P<date>(?:{MONTHS})\s+\d{{1,2}},?\s+\d{{4}})",
    re.IGNORECASE | re.MULTILINE,
)

PATTERN_TOP_DATE = re.compile(
    rf"\b(?P<date>(?:{MONTHS})\s+\d{{1,2}},?\s+\d{{4}})\b",
    re.IGNORECASE,
)

CLEAN_RATIO_LOW_WARNING = 0.25
CLEAN_RATIO_HIGH_WARNING = 0.95
DROP_CLEAN_RATIO_THRESHOLD = 0.30

CONTENT_START_MARKERS = (
    "body",
    "volledige tekst:",
    "volledige tekst",
    "full text:",
)

CONTENT_END_MARKERS = (
    "classification",
    "graphic",
    "end of document",
    "link naar pdf",
)

METADATA_PREFIXES = (
    "copyright",
    "section:",
    "length:",
    "byline:",
    "dateline:",
    "subject:",
    "industry:",
    "publication-type:",
    "journal code:",
    "language:",
    "geographic:",
    "load-date:",
)


def extract_text_from_docx(docx_path: Path) -> str:
    """Extract all paragraph text from a DOCX file."""
    doc = Document(docx_path)
    return "\n".join(p.text for p in doc.paragraphs)


def normalize_for_dedup(text: str) -> str:
    """Normalize text for robust near-duplicate matching."""
    lowered = text.lower()
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered.strip()


def parse_date_string(date_text: str) -> str | None:
    """Parse an English or Dutch article date into ISO format."""
    cleaned = date_text.strip().rstrip(".")
    cleaned = re.sub(r"\b(mon|tue|wed|thu|fri|sat|sun)(day)?\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b(maandag|dinsdag|woensdag|donderdag|vrijdag|zaterdag|zondag)\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    for dutch, english in MONTH_ALIASES.items():
        cleaned = re.sub(rf"\b{dutch}\b", english, cleaned, flags=re.IGNORECASE)

    try:
        return datetime.strptime(cleaned, "%B %d, %Y").strftime("%Y-%m-%d")
    except ValueError:
        try:
            return datetime.strptime(cleaned, "%d %B %Y").strftime("%Y-%m-%d")
        except ValueError:
            return None


def extract_title(text: str) -> str:
    """Extract the first non-empty line as the article title."""
    for line in text.splitlines():
        candidate = line.strip()
        if candidate:
            return candidate
    return ""


def extract_top_date(text: str) -> str | None:
    """Extract the first date found near the top of the article."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    # First pass: English month regex used in historical pipeline.
    head_text = "\n".join(lines[:16])
    top_match = PATTERN_TOP_DATE.search(head_text)
    if top_match:
        parsed = parse_date_string(top_match.group("date"))
        if parsed:
            return parsed

    # Fallback pass: broader parser (also supports Dutch month names).
    for line in lines[:12]:
        if line.lower().startswith("load-date:"):
            continue
        match = DATE_LINE_PATTERN.search(line)
        if match:
            return parse_date_string(match.group("date"))
    return None


def extract_bottom_date(text: str) -> str | None:
    """Extract the bottom publication/load date if present."""
    matches = list(PATTERN_LOAD_DATE.finditer(text))
    if not matches:
        return None
    return parse_date_string(matches[-1].group("date"))


def extract_publication_date(text: str) -> str | None:
    """Prefer the top date, otherwise fall back to the bottom load date."""
    return extract_top_date(text) or extract_bottom_date(text)


def prepend_publication_date(clean_text: str, publication_date: str | None) -> str:
    """Prefix clean text with publication date when available."""
    if not publication_date:
        return clean_text
    if not clean_text:
        return f"Publication date: {publication_date}"
    return f"Publication date: {publication_date}\n\n{clean_text}"


def extract_clean_article_text(text: str) -> str:
    """Keep only the main article text, excluding headers, metadata, and closing blocks."""
    lines = [line.rstrip() for line in text.splitlines()]
    marker_positions: dict[str, int] = {}
    highlight_line_index: int | None = None
    highlight_inline_text = ""

    for index, line in enumerate(lines):
        lowered = line.strip().lower()
        if lowered.startswith("volledige tekst"):
            marker_positions.setdefault("volledige tekst", index + 1)
        elif lowered.startswith("full text"):
            marker_positions.setdefault("full text", index + 1)
        elif lowered == "body" or lowered.startswith("body"):
            marker_positions.setdefault("body", index + 1)
        elif lowered.startswith("highlight:"):
            marker_positions.setdefault("highlight", index + 1)
            if highlight_line_index is None:
                highlight_line_index = index
                highlight_inline_text = line.split(":", 1)[1].strip() if ":" in line else ""

    # When present, Highlight is a trusted start signal for clean text.
    if "highlight" in marker_positions:
        start_index = marker_positions["highlight"]
        preserve_after_highlight = True
    elif "volledige tekst" in marker_positions:
        start_index = marker_positions["volledige tekst"]
        preserve_after_highlight = False
    elif "full text" in marker_positions:
        start_index = marker_positions["full text"]
        preserve_after_highlight = False
    elif "body" in marker_positions:
        start_index = marker_positions["body"]
        preserve_after_highlight = False
    else:
        start_index = 0
        preserve_after_highlight = False

    body_lines: list[str] = []
    if preserve_after_highlight and highlight_inline_text:
        body_lines.append(highlight_inline_text)

    # Main end marker: Classification. If absent, fall back to other end markers.
    end_index = len(lines)
    classification_index = None
    for index in range(start_index, len(lines)):
        lowered = lines[index].strip().lower()
        if lowered == "classification" or lowered.startswith("classification"):
            classification_index = index
            break

    if classification_index is not None:
        end_index = classification_index
    else:
        for index in range(start_index, len(lines)):
            lowered = lines[index].strip().lower()
            if lowered in CONTENT_END_MARKERS or any(
                lowered.startswith(marker) for marker in CONTENT_END_MARKERS
            ):
                end_index = index
                break

    for line in lines[start_index:end_index]:
        lowered = line.strip().lower()
        if lowered == "highlight:" or lowered.startswith("highlight:"):
            continue
        if lowered == "body" or lowered.startswith("body"):
            continue
        if (not preserve_after_highlight) and any(lowered.startswith(prefix) for prefix in METADATA_PREFIXES):
            continue
        if not line.strip():
            if body_lines:
                body_lines.append("")
            continue
        body_lines.append(line.strip())

    cleaned_lines: list[str] = []
    previous_blank = False
    for line in body_lines:
        if not line:
            if not previous_blank:
                cleaned_lines.append("")
            previous_blank = True
            continue
        cleaned_lines.append(line)
        previous_blank = False

    return "\n".join(cleaned_lines).strip()


def make_shingles(text: str, shingle_size: int = 5) -> set[str]:
    """Create word shingles used by MinHash."""
    tokens = re.findall(r"\w+", text)
    if not tokens:
        return set()
    if len(tokens) < shingle_size:
        return {" ".join(tokens)}
    return {
        " ".join(tokens[i : i + shingle_size])
        for i in range(len(tokens) - shingle_size + 1)
    }


def extract_blocking_title(text: str) -> str:
    """Extract normalized title for dedup blocking fallback."""
    for line in text.splitlines():
        candidate = line.strip().lower()
        if candidate:
            return candidate
    return ""


def jaccard_similarity(set_a: set[str], set_b: set[str]) -> float:
    """Compute exact Jaccard similarity between two shingle sets."""
    if not set_a or not set_b:
        return 0.0
    union_size = len(set_a | set_b)
    if union_size == 0:
        return 0.0
    return len(set_a & set_b) / union_size


def build_minhash(shingles: set[str], num_perm: int = 128) -> MinHash:
    """Build a MinHash signature from shingles."""
    minhash = MinHash(num_perm=num_perm)
    for shingle in shingles:
        minhash.update(shingle.encode("utf-8"))
    return minhash


def collect_articles_from_raw(raw_dir: Path) -> list[dict]:
    """Load all DOCX articles from all ZIP files under raw_dir."""
    zip_files = sorted(raw_dir.glob("*.zip"))
    if not zip_files:
        raise FileNotFoundError(f"No ZIP files found in: {raw_dir}")

    all_articles: list[dict] = []
    article_id = 0

    for zip_path in tqdm(zip_files, desc="ZIP files", unit="zip"):
        with zipfile.ZipFile(zip_path, "r") as archive:
            with tempfile.TemporaryDirectory() as tmpdir:
                archive.extractall(tmpdir)
                docx_files = sorted(Path(tmpdir).rglob("*.docx"))

                for docx_file in tqdm(
                    docx_files,
                    desc=f"DOCX in {zip_path.name}",
                    unit="docx",
                    leave=False,
                ):
                    text = extract_text_from_docx(docx_file)
                    dedup_text = normalize_for_dedup(text)

                    all_articles.append(
                        {
                            "article_id": article_id,
                            "source_zip": zip_path.name,
                            "source_file": docx_file.name,
                            "text": text,
                            "dedup_text": dedup_text,
                        }
                    )
                    article_id += 1

    return all_articles


def deduplicate_articles_lsh(
    articles: list[dict],
    threshold: float = 0.88,
    num_perm: int = 128,
    shingle_size: int = 5,
) -> tuple[list[dict], list[dict]]:
    """Deduplicate articles globally using MinHash + LSH."""
    # Use a lower threshold for candidate retrieval, then apply exact Jaccard
    # thresholding to avoid LSH false negatives on near-identical articles.
    candidate_threshold = max(0.5, threshold - 0.2)
    lsh = MinHashLSH(threshold=candidate_threshold, num_perm=num_perm)
    unique_articles: list[dict] = []
    duplicate_report: list[dict] = []
    signatures: dict[str, MinHash] = {}
    shingle_index: dict[str, set[str]] = {}
    title_index: dict[str, list[str]] = {}

    for article in tqdm(articles, desc="Deduplicating", unit="article"):
        key = str(article["article_id"])
        shingles = make_shingles(article["dedup_text"], shingle_size=shingle_size)

        if not shingles:
            duplicate_report.append(
                {
                    "duplicate_article_id": article["article_id"],
                    "kept_article_id": None,
                    "reason": "empty_text",
                    "source_zip": article["source_zip"],
                    "source_file": article["source_file"],
                }
            )
            continue

        signature = build_minhash(shingles, num_perm=num_perm)
        candidate_ids = set(lsh.query(signature))

        title_key = extract_blocking_title(article["text"])
        if title_key in title_index:
            candidate_ids.update(title_index[title_key])

        if candidate_ids:
            kept_key = None
            best_similarity = 0.0

            for candidate_key in sorted(candidate_ids, key=int):
                similarity = jaccard_similarity(shingles, shingle_index[candidate_key])
                if similarity >= threshold and similarity > best_similarity:
                    kept_key = candidate_key
                    best_similarity = similarity

            if kept_key is not None:
                duplicate_report.append(
                    {
                        "duplicate_article_id": article["article_id"],
                        "kept_article_id": int(kept_key),
                        "exact_jaccard": round(float(best_similarity), 4),
                        "source_zip": article["source_zip"],
                        "source_file": article["source_file"],
                    }
                )
                continue

        lsh.insert(key, signature)
        signatures[key] = signature
        shingle_index[key] = shingles
        title_index.setdefault(title_key, []).append(key)
        unique_articles.append(article)

    return unique_articles, duplicate_report


def enrich_article(article: dict) -> dict:
    """Add article-level features derived from the original text."""
    text = article["text"]
    clean_text = extract_clean_article_text(text)
    top_date = extract_top_date(text)
    bottom_date = extract_bottom_date(text)
    publication_date = top_date or bottom_date

    clean_text_with_date = prepend_publication_date(clean_text, publication_date)
    text_len = max(len(text.strip()), 1)
    clean_ratio = len(clean_text.strip()) / text_len

    warning = None
    if clean_ratio < CLEAN_RATIO_LOW_WARNING:
        warning = "clean_text_too_short_check_regex"
    elif clean_ratio > CLEAN_RATIO_HIGH_WARNING:
        warning = "clean_text_too_long_check_regex"

    enriched = dict(article)
    enriched["title"] = extract_title(text)
    enriched["top_date"] = top_date
    enriched["bottom_date"] = bottom_date
    enriched["publication_date"] = publication_date
    enriched["clean_text"] = clean_text_with_date
    enriched["clean_text_ratio"] = round(clean_ratio, 4)
    enriched["cleaning_warning"] = warning
    return enriched


def make_labeling_id(article: dict) -> str:
    """Create a stable labeling identifier from date and article id."""
    publication_date = article.get("publication_date")
    year = "UNK"
    if isinstance(publication_date, str) and len(publication_date) >= 4:
        year = publication_date[:4]
    return f"NEWS_{year}_{int(article['article_id']) + 1:06d}"


def build_feature_block(article: dict) -> dict:
    """Build the extracted feature block for downstream tasks."""
    return {
        "title": article.get("title") or "",
        "top_date": article.get("top_date") or "",
        "bottom_date": article.get("bottom_date") or "",
        "publication_date": article.get("publication_date") or "",
        "clean_text": article.get("clean_text") or "",
        "clean_text_ratio": article.get("clean_text_ratio", 0.0),
        "cleaning_warning": article.get("cleaning_warning") or "",
        "source_zip": article.get("source_zip") or "",
        "source_file": article.get("source_file") or "",
        "article_id": article.get("article_id"),
    }


def build_empty_spatial_target() -> dict:
    """Build an empty spatial target annotation template."""
    return {
        "mention": "",
        "induction_type": "",
        "nuts3_codes": [],
        "confidence_manual": None,
    }


def to_labeling_record(article: dict) -> dict:
    """Convert internal article schema into labeling-ready JSON schema."""
    return {
        "id": make_labeling_id(article),
        "meta": {
            "date": article.get("publication_date") or "",
            "source": article.get("source_zip") or "",
            "original_filename": article.get("source_file") or "",
        },
        "text_content": article.get("clean_text") or article.get("text") or "",
        "labels": {
            "relevance_score": None,
            "sectors": [],
            "spatial_targets": [],
        },
        "manual_notes": "",
        "features": build_feature_block(article),
    }


def to_newsjson_record(article: dict) -> dict:
    """Convert article into a compact annotation file schema."""
    return {
        "id": make_labeling_id(article),
        "meta": {
            "date": article.get("publication_date") or "",
            "source": article.get("source_zip") or "",
            "original_filename": article.get("source_file") or "",
        },
        "text_content": article.get("clean_text") or article.get("text") or "",
        "labels": {
            "relevance_score": None,
            "sectors": [],
            "spatial_targets": [build_empty_spatial_target()],
        },
        "manual_notes": "",
        "features": build_feature_block(article),
    }


def write_json(data: list[dict], output_path: Path) -> None:
    """Write JSON with stable UTF-8 encoding."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file_handle:
        json.dump(data, file_handle, indent=2, ensure_ascii=False)


def run_clean_archive(raw_dir: Path, output_dir: Path) -> None:
    """Run the full preprocessing pipeline for given raw/output directories."""
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_articles_path = output_dir / "all_articles_raw.json"
    dedup_articles_path = output_dir / "all_articles_deduplicated.json"
    dedup_report_path = output_dir / "dedup_report.json"
    newsjson_path = output_dir / "newsjson.json"

    print(f"Loading raw data from: {raw_dir}")
    articles = collect_articles_from_raw(raw_dir)
    write_json(articles, all_articles_path)
    print(f"Wrote combined corpus: {all_articles_path}")

    unique_articles, duplicate_report = deduplicate_articles_lsh(
        articles,
        threshold=0.88,
        num_perm=128,
        shingle_size=5,
    )

    # Remove internal field and add final feature-extraction columns.
    unique_articles = [enrich_article(article) for article in unique_articles]

    # Drop very low-quality cleaned texts from final output.
    pre_drop_unique_count = len(unique_articles)
    dropped_low_ratio_count = sum(
        1
        for article in unique_articles
        if float(article.get("clean_text_ratio", 0.0)) < DROP_CLEAN_RATIO_THRESHOLD
    )
    unique_articles = [
        article
        for article in unique_articles
        if float(article.get("clean_text_ratio", 0.0)) >= DROP_CLEAN_RATIO_THRESHOLD
    ]

    for article in articles:
        article.pop("dedup_text", None)

    labeling_records = [to_labeling_record(article) for article in unique_articles]
    newsjson_records = [to_newsjson_record(article) for article in unique_articles]

    write_json(labeling_records, dedup_articles_path)
    write_json(newsjson_records, newsjson_path)
    write_json(duplicate_report, dedup_report_path)

    short_warning_count = sum(
        1
        for article in unique_articles
        if article.get("cleaning_warning") == "clean_text_too_short_check_regex"
    )
    long_warning_count = sum(
        1
        for article in unique_articles
        if article.get("cleaning_warning") == "clean_text_too_long_check_regex"
    )
    warning_count = short_warning_count + long_warning_count
    dropped_low_ratio_pct = (
        (dropped_low_ratio_count / pre_drop_unique_count) * 100
        if pre_drop_unique_count
        else 0.0
    )

    print(f"Input articles: {len(articles)}")
    print(f"Unique articles: {len(unique_articles)}")
    print(f"Removed duplicates: {len(duplicate_report)}")
    print(
        "Cleaning warnings total: "
        f"{warning_count} (short: {short_warning_count}, long: {long_warning_count})"
    )
    print(
        "Dropped low clean_text_ratio articles "
        f"(< {DROP_CLEAN_RATIO_THRESHOLD:.2f}): "
        f"{dropped_low_ratio_count} ({dropped_low_ratio_pct:.2f}%)"
    )
    print(f"CleanWarning short_text priority count: {short_warning_count}")
    print(f"Wrote deduplicated corpus: {dedup_articles_path}")
    print(f"Wrote labeling newsjson: {newsjson_path}")
    print(f"Wrote duplicate report: {dedup_report_path}")


def main() -> None:
    chapter_root = Path(__file__).resolve().parents[1]
    data_dir = chapter_root / "data"
    run_clean_archive(raw_dir=data_dir / "raw", output_dir=data_dir / "preprocessed")


if __name__ == "__main__":
    main()
