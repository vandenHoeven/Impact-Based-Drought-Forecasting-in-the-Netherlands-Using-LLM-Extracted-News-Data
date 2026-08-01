"""
LLMn drought-impact extraction runner.

Input
-----
JSON list of articles (default: chapters/04_database_construction/data/preprocessed/newsjson.json),
as produced by 04_3_LLM_Input_Preprocessing/clean_archive.py.

Each article must provide:
  - id
  - features.title
  - features.clean_text  (fallback: text_content)

Output
------
JSON list of the same articles enriched under
chapters/04_database_construction/data/llm_extracted/ with:
  - model_name
  - features.llm_drought_impacts   (events matching DroughtImpactExtraction)
  - features.llm_drought_impact_error  (empty string on success)
  - per-impact evidence_found_verbatim

Schema
------
See schemas.py for the final Pydantic models and SYSTEM_PROMPT.

Requires API credentials in a .env file (see .env.example).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import litellm
from dotenv import load_dotenv
from litellm import acompletion

from schemas import SYSTEM_PROMPT, DroughtImpactExtraction

load_dotenv()

CHAPTER_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = CHAPTER_ROOT / "data"
RESULTS_DIR = DATA_DIR / "llm_extracted"
DEFAULT_INPUT_PATH = DATA_DIR / "preprocessed" / "newsjson.json"
DEFAULT_MODEL_NAME = "gemini/gemini-3.5-flash"

TARGET_RPM = 10
DEFAULT_MAX_CONCURRENT = 8
MAX_RETRIES = 85
BACKOFF_BASE_SECONDS = 4.0
BACKOFF_MAX_SECONDS = 900.0
BACKOFF_503_BASE_SECONDS = 30.0
BACKOFF_503_MAX_SECONDS = 900.0
PARSE_ATTEMPTS = 2
DEFAULT_SERVICE_TIER = "standard"
REQUEST_TIMEOUT_FLEX = 900
FLEX_COST_MULTIPLIER = 0.5


@dataclass
class UsageStats:
    api_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    failed_calls: int = 0

    def merge(self, other: "UsageStats | dict | None") -> None:
        if other is None:
            return
        if isinstance(other, dict):
            self.api_calls += int(other.get("api_calls", 0))
            self.prompt_tokens += int(other.get("prompt_tokens", 0))
            self.completion_tokens += int(other.get("completion_tokens", 0))
            self.total_tokens += int(other.get("total_tokens", 0))
            self.cost_usd += float(other.get("cost_usd", 0.0))
            self.failed_calls += int(other.get("failed_calls", 0))
            return
        self.api_calls += other.api_calls
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.total_tokens += other.total_tokens
        self.cost_usd += other.cost_usd
        self.failed_calls += other.failed_calls

    def to_dict(self) -> dict:
        return {
            "api_calls": self.api_calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": self.cost_usd,
            "failed_calls": self.failed_calls,
        }


def _extract_usage(
    response,
    model_name: str,
    service_tier: str = DEFAULT_SERVICE_TIER,
) -> UsageStats:
    usage = getattr(response, "usage", None)
    if usage is None:
        print("Warning: API response missing usage metadata; cost not recorded for this call.")
        return UsageStats(api_calls=1, failed_calls=1)

    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    total_tokens = int(getattr(usage, "total_tokens", 0) or 0)
    if total_tokens == 0:
        total_tokens = prompt_tokens + completion_tokens

    try:
        cost_usd = float(litellm.completion_cost(completion_response=response, model=model_name))
    except Exception:
        cost_usd = 0.0

    if service_tier == "flex":
        response_tier = str(getattr(response, "service_tier", "") or "").lower()
        if response_tier not in ("flex", "batch"):
            cost_usd *= FLEX_COST_MULTIPLIER

    return UsageStats(
        api_calls=1,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        cost_usd=cost_usd,
    )


def _print_usage_summary(
    stats: UsageStats,
    model_name: str,
    service_tier: str = DEFAULT_SERVICE_TIER,
) -> None:
    avg_tokens = stats.total_tokens / stats.api_calls if stats.api_calls else 0.0
    print(
        f"Token usage: prompt={stats.prompt_tokens:,} | "
        f"completion={stats.completion_tokens:,} | total={stats.total_tokens:,}"
    )
    print(f"Estimated cost (USD): ${stats.cost_usd:.4f}")
    print(f"API calls: {stats.api_calls:,} (avg {avg_tokens:.0f} tokens/call)")
    if stats.failed_calls:
        print(f"Calls without usage metadata: {stats.failed_calls}")
    tier_note = ""
    if service_tier == "flex":
        tier_note = (
            " Flex tier active (~50% of standard); estimate halved when LiteLLM "
            "did not apply flex pricing."
        )
    print(
        f"Pricing note: {model_name} via LiteLLM model cost table "
        f"(actual invoice may differ for cached input, batch tier, flex tier, etc.)."
        f"{tier_note}"
    )


def _build_completion_kwargs(model_name, messages, service_tier: str = DEFAULT_SERVICE_TIER):
    kwargs = {
        "model": model_name,
        "messages": messages,
        "response_format": DroughtImpactExtraction,
        "temperature": 0,
    }
    if service_tier != "standard":
        kwargs["service_tier"] = service_tier
        kwargs["timeout"] = REQUEST_TIMEOUT_FLEX
    return kwargs


class AsyncRateLimiter:
    """Limit how many API requests may start per minute (async-safe)."""

    def __init__(self, rpm: float):
        self._interval = 60.0 / float(rpm)
        self._lock = asyncio.Lock()
        self._next_slot = 0.0

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            wait_seconds = self._next_slot - now
            if wait_seconds > 0:
                await asyncio.sleep(wait_seconds)
                now = time.monotonic()
            self._next_slot = now + self._interval


def build_default_output_path(model_name):
    safe_model_name = model_name.replace("/", "_").replace("\\", "_")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return RESULTS_DIR / f"articles_with_llm_features_{safe_model_name}_{timestamp}.json"


def build_checkpoint_path(input_path, model_name, service_tier=DEFAULT_SERVICE_TIER):
    """Stable per-input/model checkpoint path so interrupted runs can resume."""
    safe_model_name = model_name.replace("/", "_").replace("\\", "_")
    tier_suffix = f"_{service_tier}" if service_tier != "standard" else ""
    return RESULTS_DIR / f"checkpoint_{safe_model_name}{tier_suffix}_{Path(input_path).stem}.jsonl"


def article_key(article, index):
    return article.get("id", f"index_{index}")


def load_checkpoint(checkpoint_path):
    """Load previously completed articles from a JSONL checkpoint, keyed by article id."""
    done = {}
    if not checkpoint_path.exists():
        return done
    with open(checkpoint_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict) and "key" in record and "article" in record:
                done[record["key"]] = record["article"]
    return done


def _format_duration(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _print_progress(pipeline_start: float, completed: int, total: int) -> None:
    elapsed = time.time() - pipeline_start
    avg_seconds = elapsed / completed
    remaining_articles = total - completed
    remaining_seconds = avg_seconds * remaining_articles
    expected_end = datetime.now() + timedelta(seconds=remaining_seconds)
    print(
        f"Timing: avg {avg_seconds:.1f}s/article | elapsed {_format_duration(elapsed)} | "
        f"left ~{_format_duration(remaining_seconds)} | ETA end {expected_end:%Y-%m-%d %H:%M:%S}"
    )


def _is_retryable_error(error_text):
    text = error_text.lower()
    retry_signals = [
        "429",
        "rate limit",
        "too many requests",
        "rpm",
        "503",
        "service unavailable",
        "temporarily unavailable",
        "timeout",
        "timed out",
        "connection reset",
        "connection aborted",
    ]
    return any(signal in text for signal in retry_signals)


def _compute_retry_backoff_seconds(attempt: int, err_text: str) -> float:
    text = err_text.lower()
    is_503 = "503" in text or "service unavailable" in text or "high demand" in text
    if is_503:
        base, cap = BACKOFF_503_BASE_SECONDS, BACKOFF_503_MAX_SECONDS
    else:
        base, cap = BACKOFF_BASE_SECONDS, BACKOFF_MAX_SECONDS
    return min(cap, base * (2 ** (attempt - 1)))


def _article_title_and_text(article: dict) -> tuple[str, str]:
    features = article.get("features", {}) or {}
    title = features.get("title", "") or ""
    clean_text = features.get("clean_text") or article.get("text_content") or ""
    return title, clean_text


async def classify_article_async(
    title,
    clean_text,
    model_name,
    rate_limiter: AsyncRateLimiter,
    article_id: str | None = None,
    on_retry=None,
    service_tier: str = DEFAULT_SERVICE_TIER,
):
    label = article_id or "article"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Title: {title}\n\nClean Text:\n{clean_text}",
        },
    ]
    for attempt in range(1, MAX_RETRIES + 1):
        await rate_limiter.acquire()
        try:
            response = await acompletion(
                **_build_completion_kwargs(model_name, messages, service_tier)
            )
            return response.choices[0].message.content, _extract_usage(
                response, model_name, service_tier
            )
        except Exception as e:
            err_text = str(e)
            if attempt == MAX_RETRIES or not _is_retryable_error(err_text):
                return f"Error: {err_text}", UsageStats(failed_calls=1)

            backoff = _compute_retry_backoff_seconds(attempt, err_text)
            jitter = random.uniform(0.1, 1.0)
            wait_seconds = backoff + jitter
            print(
                f"[{label}] Retry {attempt}/{MAX_RETRIES} in {wait_seconds:.1f}s "
                f"due to transient error: {err_text}"
            )
            if on_retry is not None:
                on_retry(
                    {
                        "article_id": label,
                        "attempt": attempt,
                        "max_retries": MAX_RETRIES,
                        "wait_seconds": wait_seconds,
                        "error_text": err_text[:200],
                    }
                )
            await asyncio.sleep(wait_seconds)


async def classify_and_parse_async(
    title,
    clean_text,
    model_name,
    rate_limiter: AsyncRateLimiter,
    article_id: str | None = None,
    on_retry=None,
    service_tier: str = DEFAULT_SERVICE_TIER,
):
    """Call the model and validate the response, re-calling once if validation fails."""
    events, parse_error = None, ""
    usage = UsageStats()
    for attempt in range(1, PARSE_ATTEMPTS + 1):
        raw_result, call_usage = await classify_article_async(
            title,
            clean_text,
            model_name,
            rate_limiter,
            article_id=article_id,
            on_retry=on_retry,
            service_tier=service_tier,
        )
        usage.merge(call_usage)
        events, parse_error = parse_llm_result(raw_result)
        if events is not None:
            break
        if isinstance(raw_result, str) and raw_result.startswith("Error:"):
            break
        if attempt < PARSE_ATTEMPTS:
            print(
                f"[{article_id or 'article'}] Response failed schema validation; "
                f"re-calling model (attempt {attempt + 1}/{PARSE_ATTEMPTS})."
            )
    return events, parse_error, usage


def classify_article(
    title,
    clean_text,
    model_name=DEFAULT_MODEL_NAME,
    target_rpm=TARGET_RPM,
    service_tier=DEFAULT_SERVICE_TIER,
):
    rate_limiter = AsyncRateLimiter(target_rpm)
    return asyncio.run(
        classify_and_parse_async(
            title, clean_text, model_name, rate_limiter, service_tier=service_tier
        )
    )


def _normalize_for_match(text):
    collapsed = " ".join(str(text).split()).casefold()
    return collapsed.strip(" \"'\u201c\u201d\u2018\u2019")


def annotate_evidence_verbatim(events, title, clean_text):
    """Flag each impact with whether its evidence quote actually appears in the article."""
    haystack = _normalize_for_match(f"{title}\n{clean_text}")
    missing = 0
    for event in events:
        if not isinstance(event, dict):
            continue
        for impact in event.get("impacts") or []:
            quote = _normalize_for_match(impact.get("evidence", ""))
            found = bool(quote) and quote in haystack
            impact["evidence_found_verbatim"] = found
            if not found:
                missing += 1
    return missing


def count_non_verbatim_impacts(output_data):
    return sum(
        1
        for article in output_data
        for event in article.get("features", {}).get("llm_drought_impacts") or []
        if isinstance(event, dict)
        for impact in event.get("impacts") or []
        if impact.get("evidence_found_verbatim") is False
    )


def _enrich_article(article, model_name, events, parse_error):
    features = dict(article.get("features", {}))
    features["llm_drought_impacts"] = events if events is not None else []
    features["llm_drought_impact_error"] = parse_error if events is None else ""
    enriched = dict(article)
    enriched["model_name"] = model_name
    enriched["features"] = features
    return enriched


async def _process_articles_async(
    pending,
    model_name,
    rate_limiter,
    semaphore,
    pipeline_start,
    total,
    checkpoint_path,
    on_article_progress=None,
    on_retry=None,
    service_tier=DEFAULT_SERVICE_TIER,
):
    progress_lock = asyncio.Lock()
    completed_count = 0
    batch_usage = UsageStats()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_file = open(checkpoint_path, "a", encoding="utf-8")

    async def process_one(index, article):
        nonlocal completed_count
        title, clean_text = _article_title_and_text(article)
        article_id = article_key(article, index)
        article_usage = UsageStats()

        async with semaphore:
            try:
                events, parse_error, article_usage = await classify_and_parse_async(
                    title,
                    clean_text,
                    model_name,
                    rate_limiter,
                    article_id=article_id,
                    on_retry=on_retry,
                    service_tier=service_tier,
                )
                had_error = events is None
                impact_count = 0
                if not had_error:
                    annotate_evidence_verbatim(events, title, clean_text)
                    impact_count = sum(
                        len(event.get("impacts") or [])
                        for event in events
                        if isinstance(event, dict)
                    )
                enriched = _enrich_article(article, model_name, events, parse_error)
            except Exception as exc:
                enriched = _enrich_article(
                    article,
                    model_name,
                    None,
                    f"Unexpected processing error: {exc}",
                )
                had_error = True
                impact_count = 0
                parse_error = enriched["features"]["llm_drought_impact_error"]
                events = None

        async with progress_lock:
            batch_usage.merge(article_usage)
            checkpoint_file.write(
                json.dumps({"key": article_id, "article": enriched}, ensure_ascii=False)
                + "\n"
            )
            checkpoint_file.flush()
            completed_count += 1
            done = completed_count
            print("-" * 70)
            print(f"[{done}/{total}] Finished article: {article_id}")
            if had_error:
                print(f"Result: ERROR -> {parse_error}")
            else:
                event_count = len(events)
                print(
                    f"Result: extracted_events={event_count} | extracted_impacts={impact_count}"
                )
            print(
                f"Article usage: prompt={article_usage.prompt_tokens:,} | "
                f"completion={article_usage.completion_tokens:,} | "
                f"cost=${article_usage.cost_usd:.4f} | api_calls={article_usage.api_calls}"
            )
            _print_progress(pipeline_start, done, total)
            if on_article_progress is not None:
                on_article_progress(
                    {
                        "article_id": article_id,
                        "batch_done": done,
                        "batch_total": total,
                        "had_error": had_error,
                        "parse_error": parse_error,
                        "impact_count": impact_count,
                        "prompt_tokens": article_usage.prompt_tokens,
                        "completion_tokens": article_usage.completion_tokens,
                        "total_tokens": article_usage.total_tokens,
                        "cost_usd": article_usage.cost_usd,
                        "api_calls": article_usage.api_calls,
                        "batch_prompt_tokens": batch_usage.prompt_tokens,
                        "batch_completion_tokens": batch_usage.completion_tokens,
                        "batch_total_tokens": batch_usage.total_tokens,
                        "batch_cost_usd": batch_usage.cost_usd,
                        "batch_api_calls": batch_usage.api_calls,
                    }
                )

        return index, enriched

    try:
        tasks = [process_one(index, article) for index, article in pending]
        results = await asyncio.gather(*tasks)
    finally:
        checkpoint_file.close()

    return {index: enriched for index, enriched in results}, batch_usage


def parse_llm_result(raw_result):
    if isinstance(raw_result, str) and raw_result.startswith("Error:"):
        return None, raw_result

    try:
        payload = json.loads(raw_result)
        extraction = DroughtImpactExtraction.model_validate(payload)
        return [event.model_dump() for event in extraction.events], ""
    except Exception as exc:
        return None, f"Could not parse model response: {exc}; raw={raw_result}"


def load_articles(input_path):
    """Load articles from a JSON list (newsjson.json from 04_3)."""
    with open(input_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, list):
        return payload

    raise ValueError(
        "Input JSON must be a list of articles (e.g. newsjson.json from 04_3)."
    )


def run_extraction(
    input_path=DEFAULT_INPUT_PATH,
    output_path=None,
    limit=None,
    model_name=DEFAULT_MODEL_NAME,
    max_concurrent=DEFAULT_MAX_CONCURRENT,
    target_rpm=TARGET_RPM,
    service_tier=DEFAULT_SERVICE_TIER,
    on_article_progress=None,
    on_retry=None,
):
    input_path = Path(input_path)
    if output_path is None:
        output_path = build_default_output_path(model_name)
    output_path = Path(output_path)

    print("=" * 70)
    print("Running LLMn drought-impact extraction")
    print("=" * 70)
    print(f"Input file: {input_path}")
    print(f"Output file: {output_path}")
    print(f"Model: {model_name}")
    print(f"Service tier: {service_tier}")
    if service_tier == "flex":
        print(f"Request timeout: {REQUEST_TIMEOUT_FLEX}s (Flex queue may wait several minutes)")

    if not os.path.exists(input_path):
        print(f"Error: Could not find input file: {input_path}")
        return None

    try:
        data = load_articles(input_path)
    except Exception as exc:
        print(f"Error: {exc}")
        return None

    if len(data) == 0:
        print("Error: Input JSON must contain at least one article.")
        return None

    total = len(data) if limit is None else min(limit, len(data))
    articles = data[:total]
    print(f"Articles to process: {total}")
    print(f"Concurrency: {max_concurrent} | Target RPM: {target_rpm}")

    checkpoint_path = build_checkpoint_path(input_path, model_name, service_tier)
    done_map = load_checkpoint(checkpoint_path)
    pending = [
        (index, article)
        for index, article in enumerate(articles)
        if article_key(article, index) not in done_map
    ]
    if done_map:
        print(f"Resuming from checkpoint: {checkpoint_path}")
        print(f"Already completed: {total - len(pending)} | Remaining: {len(pending)}")

    pipeline_start = time.time()
    print(f"Started at: {datetime.now():%Y-%m-%d %H:%M:%S}")

    new_results = {}
    batch_usage = UsageStats()
    if pending:
        rate_limiter = AsyncRateLimiter(target_rpm)
        semaphore = asyncio.Semaphore(max_concurrent)
        new_results, batch_usage = asyncio.run(
            _process_articles_async(
                pending,
                model_name,
                rate_limiter,
                semaphore,
                pipeline_start,
                len(pending),
                checkpoint_path,
                on_article_progress=on_article_progress,
                on_retry=on_retry,
                service_tier=service_tier,
            )
        )

    output_data = []
    for index, article in enumerate(articles):
        key = article_key(article, index)
        if key in done_map:
            output_data.append(done_map[key])
        else:
            output_data.append(new_results[index])

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    checkpoint_path.unlink(missing_ok=True)

    errors = sum(
        1
        for article in output_data
        if article.get("features", {}).get("llm_drought_impact_error")
    )
    total_impacts = sum(
        len(event.get("impacts") or [])
        for article in output_data
        for event in article.get("features", {}).get("llm_drought_impacts") or []
        if isinstance(event, dict)
    )
    non_verbatim = count_non_verbatim_impacts(output_data)

    total_elapsed = time.time() - pipeline_start
    print("=" * 70)
    print("Extraction completed")
    print(
        f"Finished at: {datetime.now():%Y-%m-%d %H:%M:%S} | "
        f"Total elapsed: {_format_duration(total_elapsed)}"
    )
    print(f"Saved enriched data to: {output_path}")
    print(
        f"Processed: {len(output_data)} | Errors: {errors} | "
        f"Total extracted impacts: {total_impacts}"
    )
    print(f"Impacts with non-verbatim evidence quotes: {non_verbatim}")
    _print_usage_summary(batch_usage, model_name, service_tier)
    print("=" * 70)

    return {
        "articles": len(output_data),
        "errors": errors,
        "impacts": total_impacts,
        "non_verbatim": non_verbatim,
        "elapsed_seconds": total_elapsed,
        "usage": batch_usage.to_dict(),
    }


def run_single_article_test(
    data_path=DEFAULT_INPUT_PATH,
    article_index=0,
    model_name=DEFAULT_MODEL_NAME,
    service_tier=DEFAULT_SERVICE_TIER,
):
    print("=" * 70)
    print("Running one-article LLMn extraction test")
    print("=" * 70)

    if not os.path.exists(data_path):
        print(f"Error: Could not find data file: {data_path}")
        return

    try:
        data = load_articles(data_path)
    except Exception as exc:
        print(f"Error: {exc}")
        return

    if not data:
        print("Error: Data file is empty.")
        return

    article_index = max(0, min(article_index, len(data) - 1))
    article = data[article_index]

    article_id = article.get("id", "<missing id>")
    title, text = _article_title_and_text(article)

    print(f"Article index: {article_index}")
    print(f"Article id: {article_id}")
    print(f"Title: {title}")
    print(f"Model: {model_name}")
    print(f"Service tier: {service_tier}")
    if service_tier == "flex":
        print(f"Request timeout: {REQUEST_TIMEOUT_FLEX}s (Flex queue may wait several minutes)")
    print(f"Text length: {len(text)} chars")
    print("Text preview:")
    print((text[:300] + "...") if len(text) > 300 else text)
    print("-" * 70)
    print("Calling model...")

    events, parse_error, usage = classify_article(
        title, text, model_name=model_name, service_tier=service_tier
    )

    print("-" * 70)
    print("Model result:")
    if events is None:
        print(f"ERROR: {parse_error}")
    else:
        annotate_evidence_verbatim(events, title, text)
        print(json.dumps(events, indent=2, ensure_ascii=False))
    print("-" * 70)
    _print_usage_summary(usage, model_name, service_tier)
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LLMn drought-impact extraction (schema in schemas.py)."
    )
    parser.add_argument("--mode", choices=["test-one", "extract"], default="extract")
    parser.add_argument("--input", default=str(DEFAULT_INPUT_PATH))
    parser.add_argument("--output", default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_MAX_CONCURRENT,
        help=f"Maximum number of in-flight API requests (default: {DEFAULT_MAX_CONCURRENT}).",
    )
    parser.add_argument(
        "--rpm",
        type=float,
        default=TARGET_RPM,
        help=f"Maximum API request starts per minute (default: {TARGET_RPM}).",
    )
    parser.add_argument(
        "--service-tier",
        choices=["standard", "flex"],
        default=DEFAULT_SERVICE_TIER,
        help="Gemini inference tier (flex = 50%% cheaper, higher latency, best-effort).",
    )
    args = parser.parse_args()

    if args.mode == "test-one":
        run_single_article_test(
            data_path=args.input,
            article_index=args.index,
            model_name=args.model,
            service_tier=args.service_tier,
        )
    else:
        run_extraction(
            input_path=args.input,
            output_path=args.output,
            limit=args.limit,
            model_name=args.model,
            max_concurrent=args.concurrency,
            target_rpm=args.rpm,
            service_tier=args.service_tier,
        )
