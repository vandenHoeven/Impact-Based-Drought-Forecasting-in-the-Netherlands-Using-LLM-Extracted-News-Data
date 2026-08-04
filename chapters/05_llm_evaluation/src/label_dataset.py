"""Streamlit app for building a two-stratum drought evaluation dataset."""

from __future__ import annotations

import ast
import json
import os
import random
import re
import shutil
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

if "streamlit" in sys.modules:
	import streamlit as st
	from streamlit.runtime.scriptrunner_utils.script_run_context import get_script_run_ctx
else:
	class _StreamlitStub:
		def cache_data(self, *args: Any, **kwargs: Any):
			def decorator(func: Any) -> Any:
				return func

			return decorator

	st = _StreamlitStub()

	def get_script_run_ctx() -> None:
		return None


# Package root: chapters/05_llm_evaluation/
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
MONOREPO_ROOT = Path(os.getenv("DROUGHT_MONOREPO_ROOT", str(REPO_ROOT)))
_DEFAULT_SCHEMA = (
	REPO_ROOT
	/ "chapters"
	/ "04_database_construction"
	/ "04_4_LLMn_Extraction_Framework"
	/ "schemas.py"
)
_DEFAULT_SOURCE = (
	REPO_ROOT
	/ "chapters"
	/ "04_database_construction"
	/ "data"
	/ "preprocessed"
	/ "all_articles_deduplicated.json"
)
LLM_SCHEMA_PATH = Path(os.getenv("EVAL_BUILDER_LLM_SCHEMA", str(_DEFAULT_SCHEMA)))
DEFAULT_SOURCE_JSON = Path(os.getenv("EVAL_BUILDER_SOURCE_JSON", str(_DEFAULT_SOURCE)))
OUTPUT_DIR = Path(os.getenv("EVAL_BUILDER_OUTPUT_DIR", str(PACKAGE_ROOT / "data")))
LABELLER_STATE_DIR = OUTPUT_DIR / ".labeller"
QUEUE_FILE = LABELLER_STATE_DIR / "queue.json"
STATE_FILE = LABELLER_STATE_DIR / "state.json"
REVIEWS_FILE = LABELLER_STATE_DIR / "reviews.json"
EVENTS_FILE = LABELLER_STATE_DIR / "events.json"
SUMMARY_FILE = OUTPUT_DIR / "evaluation_set.json"
DEFAULT_RANDOM_SEED = 42
DEFAULT_MIN_IMPACT_CLASSES = 10
DEFAULT_MIN_SEVERITY_OBSERVATIONS = 2
DEFAULT_MIN_RECENCY_OBSERVATIONS = 2


@lru_cache(maxsize=1)
def load_llm_schema() -> dict[str, list[Any]]:
	if not LLM_SCHEMA_PATH.exists():
		raise FileNotFoundError(f"Schema source file not found: {LLM_SCHEMA_PATH}")

	module = ast.parse(LLM_SCHEMA_PATH.read_text(encoding="utf-8"))
	impact_classes: list[str] = []
	recency_months: list[int] = []
	severity_classes: list[int] = []

	for node in module.body:
		if not isinstance(node, ast.Assign):
			continue
		if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
			continue
		name = node.targets[0].id
		value = node.value

		if name == "DroughtImpactLabel" and isinstance(value, ast.Subscript):
			slice_value = value.slice
			items = slice_value.elts if isinstance(slice_value, ast.Tuple) else [slice_value]
			for item in items:
				if isinstance(item, ast.Constant) and isinstance(item.value, str):
					impact_classes.append(item.value)
		elif name == "RecencyBucket" and isinstance(value, ast.Subscript):
			slice_value = value.slice
			items = slice_value.elts if isinstance(slice_value, ast.Tuple) else [slice_value]
			for item in items:
				if isinstance(item, ast.Constant) and isinstance(item.value, int):
					recency_months.append(item.value)

	for node in module.body:
		if not isinstance(node, ast.ClassDef) or node.name != "Impact":
			continue
		for item in node.body:
			if not isinstance(item, ast.AnnAssign):
				continue
			if not isinstance(item.target, ast.Name):
				continue
			if item.target.id != "severity":
				continue
			annotation = item.annotation
			if isinstance(annotation, ast.Subscript):
				slice_value = annotation.slice
				items = slice_value.elts if isinstance(slice_value, ast.Tuple) else [slice_value]
				for entry in items:
					if isinstance(entry, ast.Constant) and isinstance(entry.value, int):
						severity_classes.append(entry.value)

	if not impact_classes or not recency_months or not severity_classes:
		raise ValueError(
			f"Could not extract the canonical schema classes from {LLM_SCHEMA_PATH}"
		)

	return {
		"impact_classes": impact_classes,
		"recency_months": recency_months,
		"severity_classes": severity_classes,
	}


def get_impact_classes() -> list[str]:
	return list(load_llm_schema()["impact_classes"])


def get_recency_months() -> list[int]:
	return list(load_llm_schema()["recency_months"])


def get_severity_classes() -> list[int]:
	return list(load_llm_schema()["severity_classes"])


def format_recency_option(value: int) -> str:
	if value == 0:
		return "0 - current/ongoing"
	if value == 1:
		return "1 - within last month"
	if value == 12:
		return "12 - ~12 months"
	if value == 24:
		return "24 - 24+ months"
	return f"{value} months"


def format_severity_option(value: int) -> str:
	if value == 1:
		return "1 - Localized/Moderate"
	if value == 2:
		return "2 - Widespread/Severe"
	if value == 3:
		return "3 - Large-scale/Extreme/Cascading"
	return str(value)


def slugify(value: str, max_length: int = 80) -> str:
	text = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
	text = re.sub(r"_+", "_", text).strip("_")
	return (text[:max_length].rstrip("_") if text else "article") or "article"


def load_json(path: Path) -> Any:
	if not path.exists():
		return None
	with path.open("r", encoding="utf-8") as handle:
		return json.load(handle)


def save_json(path: Path, data: Any) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", encoding="utf-8") as handle:
		json.dump(data, handle, indent=2, ensure_ascii=False)


def normalize_source_articles(payload: Any) -> list[dict[str, Any]]:
	if isinstance(payload, dict):
		if isinstance(payload.get("sample"), list):
			payload = payload["sample"]
		elif isinstance(payload.get("articles"), list):
			payload = payload["articles"]

	if not isinstance(payload, list):
		raise ValueError("Expected the source corpus to be a JSON array or a sampled_articles payload.")

	articles: list[dict[str, Any]] = []
	for index, item in enumerate(payload):
		if not isinstance(item, dict):
			continue

		features = item.get("features", {}) if isinstance(item.get("features"), dict) else {}
		meta = item.get("meta", {}) if isinstance(item.get("meta"), dict) else {}
		article_id = str(item.get("id") or features.get("article_id") or item.get("article_id") or index).strip()
		text_content = item.get("text_content") or features.get("clean_text") or item.get("clean_text") or item.get("text") or ""
		articles.append(
			{
				"article_id": article_id,
				"title": str(features.get("title") or item.get("title") or meta.get("title") or "").strip(),
				"publication_date": str(features.get("publication_date") or meta.get("date") or item.get("publication_date") or "").strip(),
				"source_zip": str(features.get("source_zip") or meta.get("source") or item.get("source_zip") or "").strip(),
				"source_file": str(features.get("source_file") or meta.get("original_filename") or item.get("source_file") or "").strip(),
				"text_content": str(text_content).strip(),
				"raw": item,
			}
		)

	return articles


@st.cache_data(show_spinner=False)
def load_source_corpus(source_json: str) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
	source_path = Path(source_json)
	if not source_path.exists():
		raise FileNotFoundError(f"Source JSON not found: {source_path}")
	payload = load_json(source_path)
	articles = normalize_source_articles(payload)
	return articles, {article["article_id"]: article for article in articles}


def load_queue(article_index: dict[str, dict[str, Any]], seed: int) -> list[str]:
	queue_payload = load_json(QUEUE_FILE)
	if isinstance(queue_payload, list) and queue_payload:
		queue_ids = [str(item).strip() for item in queue_payload if str(item).strip()]
		return [article_id for article_id in queue_ids if article_id in article_index]

	queue_ids = list(article_index.keys())
	random.Random(seed).shuffle(queue_ids)
	save_json(QUEUE_FILE, queue_ids)
	return queue_ids


def load_state() -> dict[str, Any]:
	payload = load_json(STATE_FILE)
	if isinstance(payload, dict):
		return payload
	return {"queue_position": 0}


def save_state(queue_position: int) -> None:
	save_json(STATE_FILE, {"queue_position": queue_position})


def load_reviews() -> list[dict[str, Any]]:
	payload = load_json(REVIEWS_FILE)
	if isinstance(payload, list) and payload:
		return payload
	summary = load_json(SUMMARY_FILE)
	if isinstance(summary, dict) and isinstance(summary.get("article_reviews"), list):
		return list(summary["article_reviews"])
	return []


def load_events() -> list[dict[str, Any]]:
	payload = load_json(EVENTS_FILE)
	if isinstance(payload, list) and payload:
		return payload
	summary = load_json(SUMMARY_FILE)
	if isinstance(summary, dict) and isinstance(summary.get("event_records"), list):
		return list(summary["event_records"])
	return []


def save_reviews(reviews: list[dict[str, Any]]) -> None:
	save_json(REVIEWS_FILE, reviews)


def save_events(events: list[dict[str, Any]]) -> None:
	save_json(EVENTS_FILE, events)


def save_summary(summary: dict[str, Any]) -> None:
	save_json(SUMMARY_FILE, summary)


def current_article(queue: list[str], article_index: dict[str, dict[str, Any]], queue_position: int) -> dict[str, Any] | None:
	for position in range(queue_position, len(queue)):
		article_id = queue[position]
		article = article_index.get(article_id)
		if article is not None:
			return {"position": position, **article}
	return None


def build_empty_event() -> dict[str, Any]:
	return {
		"location": "",
		"impact_class": get_impact_classes()[0],
		"severity": get_severity_classes()[0],
		"recency": get_recency_months()[0],
		"notes": "",
	}


def ensure_event_form_state() -> None:
	if "event_rows" not in st.session_state:
		st.session_state.event_rows = [build_empty_event()]


def reset_event_form_state() -> None:
	st.session_state.event_rows = [build_empty_event()]
	st.session_state.event_form_reset_pending = True


def prepare_event_form_state() -> None:
	reset_pending = bool(st.session_state.pop("event_form_reset_pending", False))
	if reset_pending or "review_label" not in st.session_state:
		st.session_state.review_label = "Relevant"
	if reset_pending or "review_notes" not in st.session_state:
		st.session_state.review_notes = ""


def render_article(article: dict[str, Any]) -> None:
	st.subheader(article.get("title") or "Untitled article")
	meta_left, meta_right = st.columns(2)
	with meta_left:
		st.markdown(f"**Article ID:** {article.get('article_id', '')}")
		st.markdown(f"**Publication date:** {article.get('publication_date') or 'Unknown'}")
	with meta_right:
		st.markdown(f"**Source file:** {article.get('source_file') or 'Unknown'}")
		st.markdown(f"**Source archive:** {article.get('source_zip') or 'Unknown'}")

	body = article.get("text_content") or ""
	if not body and isinstance(article.get("raw"), dict):
		body = str(article["raw"].get("text_content") or article["raw"].get("text") or "")

	st.text_area("Article text", value=body, height=420, key=f"article_text_{article['article_id']}")


def select_default_index(options: list[Any], value: Any) -> int:
	try:
		return options.index(value)
	except ValueError:
		return 0


def append_or_replace_review(review: dict[str, Any]) -> list[dict[str, Any]]:
	reviews = load_reviews()
	updated = [item for item in reviews if str(item.get("article_id")) != review["article_id"]]
	updated.append(review)
	save_reviews(updated)
	return updated


def append_or_replace_events(article_id: str, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
	existing = load_events()
	remaining = [event for event in existing if str(event.get("article_id")) != article_id]
	remaining.extend(events)
	save_events(remaining)
	return remaining


def summarize_events(events: list[dict[str, Any]]) -> dict[str, Any]:
	impact_counts: dict[str, int] = {}
	severity_counts: dict[str, int] = {}
	recency_counts: dict[str, int] = {}
	for event in events:
		impact = str(event.get("impact_class") or "Other")
		severity = str(event.get("severity") or "Unknown")
		recency = str(event.get("recency") or "Unknown")
		impact_counts[impact] = impact_counts.get(impact, 0) + 1
		severity_counts[severity] = severity_counts.get(severity, 0) + 1
		recency_counts[recency] = recency_counts.get(recency, 0) + 1

	return {
		"event_count": len(events),
		"impact_class_counts": impact_counts,
		"severity_counts": severity_counts,
		"recency_counts": recency_counts,
	}


def rebuild_summary(reviews: list[dict[str, Any]], events: list[dict[str, Any]]) -> dict[str, Any]:
	relevant_reviews = [review for review in reviews if review.get("label") == "Relevant"]
	return {
		"article_reviews": reviews,
		"event_records": events,
		"review_count": len(reviews),
		"relevant_count": len(relevant_reviews),
		"irrelevant_count": sum(1 for review in reviews if review.get("label") == "Irrelevant"),
		"uncertain_count": sum(1 for review in reviews if review.get("label") == "Uncertain"),
		"event_summary": summarize_events(events),
	}


def reset_outputs() -> None:
	for path in [QUEUE_FILE, STATE_FILE, REVIEWS_FILE, EVENTS_FILE, SUMMARY_FILE]:
		if path.exists():
			path.unlink()

	if OUTPUT_DIR.exists():
		shutil.rmtree(OUTPUT_DIR)


def render_sidebar_settings(source_json: Path, seed: int, min_impact_classes: int, min_severity: int, min_recency: int) -> None:
	with st.sidebar:
		st.header("Dataset construction")
		st.write(f"Source corpus: {source_json}")
		st.write(f"Output directory: {OUTPUT_DIR}")
		st.write(f"Minimum impact classes: {min_impact_classes}")
		st.write(f"Minimum observations per severity class: {min_severity}")
		st.write(f"Minimum observations per recency class: {min_recency}")
		st.caption("Sampling continues until the event-level coverage requirements are satisfied.")
		if st.button("Reset dataset build", use_container_width=True):
			reset_outputs()
			st.session_state.clear()
			st.rerun()


def render_review_form(current: dict[str, Any], reviews: list[dict[str, Any]], events: list[dict[str, Any]]) -> None:
	prepare_event_form_state()
	ensure_event_form_state()
	if "review_label" not in st.session_state:
		st.session_state.review_label = "Relevant"
	if "review_notes" not in st.session_state:
		st.session_state.review_notes = ""
	impact_classes = get_impact_classes()
	severity_classes = get_severity_classes()
	recency_months = get_recency_months()

	st.markdown("### Stratum A: Relevance classification")
	st.radio(
		"Does this article contain a Dutch drought impact?",
		["Relevant", "Irrelevant", "Uncertain"],
		horizontal=True,
		key="review_label",
	)
	st.text_area("Review notes", key="review_notes", height=100)

	if st.session_state.review_label == "Relevant":
		st.markdown("### Stratum B: Event extraction")
		st.caption("Each row is one mentioned location-impact tuple. Use the location exactly as written in the article; keep all valid events without pruning during sampling.")
		for index, row in enumerate(st.session_state.event_rows):
			with st.expander(f"Event {index + 1}", expanded=index == 0):
				row["location"] = st.text_input(
					"Mentioned location",
					value=row["location"],
					help="Enter the location exactly as mentioned in the text. Use the most specific place name available.",
					key=f"location_{current['article_id']}_{index}",
				)
				row["impact_class"] = st.selectbox(
					"Impact class",
					impact_classes,
					index=select_default_index(impact_classes, row["impact_class"]),
					key=f"impact_class_{current['article_id']}_{index}",
				)
				row["severity"] = st.selectbox(
					"Severity",
					severity_classes,
					format_func=format_severity_option,
					index=select_default_index(severity_classes, row["severity"]),
					key=f"severity_{current['article_id']}_{index}",
				)
				row["recency"] = st.selectbox(
					"Recency (months)",
					recency_months,
					format_func=format_recency_option,
					index=select_default_index(recency_months, row["recency"]),
					key=f"recency_{current['article_id']}_{index}",
				)
				row["notes"] = st.text_area("Event notes", value=row["notes"], key=f"notes_{current['article_id']}_{index}", height=90)

		add_col, remove_col = st.columns(2)
		with add_col:
			if st.button("Add another event", use_container_width=True):
				st.session_state.event_rows.append(build_empty_event())
				st.rerun()
		with remove_col:
			if len(st.session_state.event_rows) > 1 and st.button("Remove last event", use_container_width=True):
				st.session_state.event_rows.pop()
				st.rerun()


def store_current_review(current: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
	label = st.session_state.review_label
	notes = st.session_state.review_notes.strip()
	review = {
		"article_id": current["article_id"],
		"title": current.get("title", ""),
		"publication_date": current.get("publication_date", ""),
		"source_zip": current.get("source_zip", ""),
		"source_file": current.get("source_file", ""),
		"label": label,
		"notes": notes,
	}
	reviews = append_or_replace_review(review)

	events: list[dict[str, Any]] = []
	if label == "Relevant":
		for row in st.session_state.event_rows:
			location = str(row.get("location", "")).strip()
			impact_class = str(row.get("impact_class") or get_impact_classes()[0]).strip()
			severity = int(row.get("severity") or get_severity_classes()[0])
			recency = int(row.get("recency") or get_recency_months()[0])
			event_notes = str(row.get("notes", "")).strip()
			if not location and not event_notes:
				continue
			events.append(
				{
					"article_id": current["article_id"],
					"title": current.get("title", ""),
					"location": location,
					"impact_class": impact_class,
					"severity": severity,
					"recency": recency,
					"notes": event_notes,
				}
			)
	append_or_replace_events(current["article_id"], events)
	save_summary(rebuild_summary(reviews, load_events()))
	return review, events


def render_progress(reviews: list[dict[str, Any]], events: list[dict[str, Any]], min_impact_classes: int, min_severity: int, min_recency: int) -> None:
	impact_classes = get_impact_classes()
	severity_classes = get_severity_classes()
	recency_months = get_recency_months()
	relevant_reviews = [review for review in reviews if review.get("label") == "Relevant"]
	irrelevant_reviews = [review for review in reviews if review.get("label") == "Irrelevant"]
	uncertain_reviews = [review for review in reviews if review.get("label") == "Uncertain"]
	event_summary = summarize_events(events)
	impact_class_total = len(event_summary["impact_class_counts"])
	severity_counts = event_summary["severity_counts"]
	recency_counts = event_summary["recency_counts"]
	severity_complete = all(severity_counts.get(str(level), 0) >= min_severity for level in severity_classes)
	recency_complete = all(recency_counts.get(str(bucket), 0) >= min_recency for bucket in recency_months)

	metric_cols = st.columns(4)
	metric_cols[0].metric("Reviewed articles", len(reviews))
	metric_cols[1].metric("Relevant", len(relevant_reviews))
	metric_cols[2].metric("Irrelevant / uncertain", len(irrelevant_reviews) + len(uncertain_reviews))
	metric_cols[3].metric("Extracted events", len(events))

	coverage_cols = st.columns(3)
	coverage_cols[0].metric("Distinct impact classes", f"{impact_class_total}/{min_impact_classes}")
	coverage_cols[1].metric("Severity classes", "Complete" if severity_complete else "Incomplete")
	coverage_cols[2].metric("Recency classes", "Complete" if recency_complete else "Incomplete")

	st.caption(
		"Coverage is evaluated per class for severity and recency; impact coverage uses a minimum count of distinct categories."
	)


def main() -> None:
	if get_script_run_ctx() is None:
		raise SystemExit(
			"This app must be launched with Streamlit: "
			"streamlit run chapter_05/src/label_dataset.py"
		)

	st.set_page_config(page_title="Evaluation Set Builder", layout="wide")
	st.title("Evaluation Set Builder")
	st.caption("Construct a two-stratum evaluation dataset: article relevance first, then event-level drought impact extraction.")

	source_json = Path(os.getenv("EVAL_BUILDER_SOURCE_JSON", str(DEFAULT_SOURCE_JSON)))
	seed = int(os.getenv("EVAL_BUILDER_RANDOM_SEED", str(DEFAULT_RANDOM_SEED)))
	min_impact_classes = int(os.getenv("EVAL_BUILDER_MIN_IMPACT_CLASSES", str(DEFAULT_MIN_IMPACT_CLASSES)))
	min_severity = int(os.getenv("EVAL_BUILDER_MIN_SEVERITY_OBSERVATIONS", str(DEFAULT_MIN_SEVERITY_OBSERVATIONS)))
	min_recency = int(os.getenv("EVAL_BUILDER_MIN_RECENCY_OBSERVATIONS", str(DEFAULT_MIN_RECENCY_OBSERVATIONS)))

	render_sidebar_settings(source_json, seed, min_impact_classes, min_severity, min_recency)

	try:
		articles, article_index = load_source_corpus(str(source_json))
	except Exception as exc:
		st.error(str(exc))
		return

	if not articles:
		st.warning("No articles were found in the source corpus.")
		return

	queue = load_queue(article_index, seed)
	state = load_state()
	reviews = load_reviews()
	events = load_events()
	reviewed_ids = {str(review.get("article_id", "")).strip() for review in reviews if str(review.get("article_id", "")).strip()}

	if "queue_position" not in st.session_state:
		st.session_state.queue_position = int(state.get("queue_position", 0) or 0)

	render_progress(reviews, events, min_impact_classes, min_severity, min_recency)
	event_summary = summarize_events(events)
	impact_classes = get_impact_classes()

	if all(event_summary.get("impact_class_counts", {}).get(impact_class, 0) > 0 for impact_class in impact_classes):
		coverage_status = "Impact-class coverage is complete."
	else:
		coverage_status = "Continue sampling until the event-level coverage requirements are satisfied."
	st.info(coverage_status)

	current = current_article(queue, article_index, int(st.session_state.queue_position))
	while current is not None and current["article_id"] in reviewed_ids:
		st.session_state.queue_position = current["position"] + 1
		save_state(st.session_state.queue_position)
		current = current_article(queue, article_index, int(st.session_state.queue_position))

	if current is None:
		st.warning("No more candidate articles are available in the queue.")
		return

	st.info(f"Reviewing article {current['position'] + 1} of {len(queue)}.")
	col_left, col_right = st.columns([1.3, 1])
	with col_left:
		render_article(current)
	with col_right:
		render_review_form(current, reviews, events)

	button_col_1, button_col_2, button_col_3 = st.columns([1, 1, 1])
	with button_col_1:
		if st.button("Save and continue", type="primary", use_container_width=True):
			review, extracted_events = store_current_review(current)
			st.session_state.queue_position = current["position"] + 1
			save_state(st.session_state.queue_position)
			reset_event_form_state()
			st.success(f"Saved {review['label'].lower()} review with {len(extracted_events)} event(s).")
			st.rerun()
	with button_col_2:
		if st.button("Skip article", use_container_width=True):
			st.session_state.queue_position = current["position"] + 1
			save_state(st.session_state.queue_position)
			reset_event_form_state()
			st.rerun()
	with button_col_3:
		if st.button("Export summary", use_container_width=True):
			save_summary(rebuild_summary(load_reviews(), load_events()))
			st.success(f"Summary written to {SUMMARY_FILE}")

	with st.expander("Post-hoc stratification overview", expanded=False):
		summary = rebuild_summary(reviews, events)
		st.json(summary["event_summary"])
		st.write("Impact classes present:", sorted(summary["event_summary"]["impact_class_counts"].keys()))
		st.write("Raw article review count:", summary["review_count"])


if __name__ == "__main__":
	main()


