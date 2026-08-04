"""Run the Chapter 5 evaluation set through the LLM extractor (any provider).

Requires private corpus + Chapter 04 LLMn framework and API keys.
Full article dumps are written under data/runs/ (gitignored); not part of the hand-in freeze.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

try:
	from dotenv import load_dotenv
except ImportError:  # pragma: no cover
	load_dotenv = None


CHAPTER_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
CH04_DATA = REPO_ROOT / "chapters" / "04_database_construction" / "data"
CH04_FRAMEWORK = (
	REPO_ROOT
	/ "chapters"
	/ "04_database_construction"
	/ "04_4_LLMn_Extraction_Framework"
)
DEFAULT_EVALUATION_DATASET = CHAPTER_ROOT / "data" / "evaluation_set.json"
DEFAULT_SOURCE_CORPUS = CH04_DATA / "preprocessed" / "all_articles_deduplicated.json"
DEFAULT_OUTPUT_DIR = CHAPTER_ROOT / "data" / "runs"
DEFAULT_MODEL_NAME = "gemini/gemini-3.5-flash"
ALLOWED_LABELS = {"Relevant", "Irrelevant"}

if load_dotenv is not None:
	load_dotenv(CH04_FRAMEWORK / ".env")
	load_dotenv(REPO_ROOT / ".env")


def slugify(value: str) -> str:
	text = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
	text = re.sub(r"_+", "_", text).strip("_")
	return text or "model"


def load_json(path: Path) -> Any:
	with path.open("r", encoding="utf-8") as handle:
		return json.load(handle)


def save_json(path: Path, payload: Any) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", encoding="utf-8") as handle:
		json.dump(payload, handle, indent=2, ensure_ascii=False)


def load_llmn_extraction():
	framework_dir = str(CH04_FRAMEWORK)
	if framework_dir not in sys.path:
		sys.path.insert(0, framework_dir)
	import llmn_extraction  # noqa: PLC0415 — path set just above

	return llmn_extraction


def load_evaluation_reviews(evaluation_dataset_path: Path) -> list[dict[str, Any]]:
	payload = load_json(evaluation_dataset_path)
	reviews = payload.get("article_reviews") if isinstance(payload, dict) else None
	if not isinstance(reviews, list):
		raise ValueError("Evaluation dataset must contain an 'article_reviews' list.")
	return [r for r in reviews if isinstance(r, dict) and r.get("label") in ALLOWED_LABELS]


def normalize_corpus_articles(payload: Any) -> list[dict[str, Any]]:
	if isinstance(payload, dict):
		if isinstance(payload.get("sample"), list):
			payload = payload["sample"]
		elif isinstance(payload.get("articles"), list):
			payload = payload["articles"]
	if not isinstance(payload, list):
		raise ValueError("Source corpus must be a JSON array or a sampled_articles payload.")

	articles: list[dict[str, Any]] = []
	for index, item in enumerate(payload):
		if not isinstance(item, dict):
			continue
		features = item.get("features", {}) if isinstance(item.get("features"), dict) else {}
		meta = item.get("meta", {}) if isinstance(item.get("meta"), dict) else {}
		article_id = str(item.get("id") or features.get("article_id") or item.get("article_id") or index).strip()
		articles.append(
			{
				"id": item.get("id", article_id),
				"article_id": article_id,
				"features": {
					"article_id": str(features.get("article_id") or article_id),
					"title": str(features.get("title") or item.get("title") or meta.get("title") or "").strip(),
					"publication_date": str(
						features.get("publication_date") or meta.get("date") or item.get("publication_date") or ""
					).strip(),
					"source_zip": str(features.get("source_zip") or meta.get("source") or item.get("source_zip") or "").strip(),
					"source_file": str(
						features.get("source_file") or meta.get("original_filename") or item.get("source_file") or ""
					).strip(),
					"clean_text": str(
						item.get("text_content") or features.get("clean_text") or item.get("clean_text") or item.get("text") or ""
					).strip(),
				},
				"meta": meta,
				"raw": item,
			}
		)
	return articles


def load_corpus_index(source_corpus_path: Path) -> dict[str, dict[str, Any]]:
	return {a["article_id"]: a for a in normalize_corpus_articles(load_json(source_corpus_path))}


def build_selected_articles(
	reviews: list[dict[str, Any]], article_index: dict[str, dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[str]]:
	selected: list[dict[str, Any]] = []
	missing: list[str] = []
	seen: set[str] = set()
	for review in reviews:
		article_id = str(review.get("article_id", "")).strip()
		if not article_id or article_id in seen:
			continue
		seen.add(article_id)
		article = article_index.get(article_id)
		if article is None:
			missing.append(article_id)
			continue
		selected.append(article)
	return selected, missing


def build_output_paths(output_dir: Path, model_name: str) -> tuple[Path, Path, Path]:
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	safe = slugify(model_name)
	run_dir = output_dir / safe / timestamp
	run_dir.mkdir(parents=True, exist_ok=True)
	return (
		run_dir / f"evaluation_selected_articles_{safe}_{timestamp}.json",
		run_dir / f"evaluation_llm_output_{safe}_{timestamp}.json",
		run_dir / f"evaluation_manifest_{safe}_{timestamp}.json",
	)


def require_api_key_for_model(model_name: str) -> None:
	lower = model_name.lower()
	if "anthropic" in lower or "claude" in lower:
		if not os.getenv("ANTHROPIC_API_KEY", "").strip():
			raise RuntimeError("ANTHROPIC_API_KEY is not set (needed for Claude models).")
	elif "gpt" in lower or "openai" in lower:
		if not os.getenv("OPENAI_API_KEY", "").strip():
			raise RuntimeError("OPENAI_API_KEY is not set (needed for OpenAI models).")
	elif "gemini" in lower or "google" in lower:
		if not (os.getenv("GOOGLE_API_KEY", "").strip() or os.getenv("GEMINI_API_KEY", "").strip()):
			raise RuntimeError("GOOGLE_API_KEY or GEMINI_API_KEY is not set (needed for Gemini models).")


def main() -> None:
	parser = argparse.ArgumentParser(description="Run Chapter 5 evaluation articles through an LLM extractor.")
	parser.add_argument("--evaluation-dataset", default=str(DEFAULT_EVALUATION_DATASET))
	parser.add_argument("--source-corpus", default=str(DEFAULT_SOURCE_CORPUS))
	parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
	parser.add_argument("--model", default=DEFAULT_MODEL_NAME, help="LiteLLM model id, e.g. gemini/gemini-3.5-flash")
	parser.add_argument("--limit", type=int, default=None)
	args = parser.parse_args()

	evaluation_dataset_path = Path(args.evaluation_dataset)
	source_corpus_path = Path(args.source_corpus)
	output_dir = Path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	if not evaluation_dataset_path.exists():
		raise FileNotFoundError(f"Evaluation dataset not found: {evaluation_dataset_path}")
	if not source_corpus_path.exists():
		raise FileNotFoundError(
			f"Source corpus not found: {source_corpus_path}. "
			"Provide the private Chapter 04 preprocessed corpus via --source-corpus."
		)
	if not (CH04_FRAMEWORK / "llmn_extraction.py").exists():
		raise FileNotFoundError(f"Could not find llmn_extraction.py at: {CH04_FRAMEWORK}")
	require_api_key_for_model(args.model)

	reviews = load_evaluation_reviews(evaluation_dataset_path)
	selected_articles, missing_ids = build_selected_articles(reviews, load_corpus_index(source_corpus_path))
	if args.limit is not None:
		selected_articles = selected_articles[: args.limit]

	selected_input_path, llm_output_path, manifest_path = build_output_paths(output_dir, args.model)
	save_json(selected_input_path, selected_articles)

	llmn = load_llmn_extraction()
	llmn.run_extraction(
		input_path=selected_input_path,
		output_path=llm_output_path,
		limit=args.limit,
		model_name=args.model,
	)

	save_json(
		manifest_path,
		{
			"evaluation_dataset": str(evaluation_dataset_path),
			"source_corpus": str(source_corpus_path),
			"model_name": args.model,
			"selected_review_count": len(reviews),
			"selected_article_count": len(selected_articles),
			"missing_article_ids": missing_ids,
			"selected_input_path": str(selected_input_path),
			"llm_output_path": str(llm_output_path),
		},
	)

	print("=" * 70)
	print("Evaluation run completed")
	print(f"Selected reviews: {len(reviews)}")
	print(f"Selected articles: {selected_input_path}")
	print(f"LLM output: {llm_output_path}")
	print(f"Manifest: {manifest_path}")
	if missing_ids:
		print(f"Missing article IDs: {len(missing_ids)}")
	print("=" * 70)


if __name__ == "__main__":
	main()
