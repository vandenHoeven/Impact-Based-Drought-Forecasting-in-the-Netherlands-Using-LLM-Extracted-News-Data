"""Fuzzy title clustering + iterative medoid near-duplicate removal.

Archival reference only — do not run. Imported historically by
post_proccesing_batches.py; not part of the live Chapter 04 pipeline.
See ../README.md.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable

TITLE_SIM_THRESHOLD = 0.85
TEXT_DEDUP_THRESHOLD = 0.44
MIN_TEXT_CHARS_FOR_MATCH = 200
SHINGLE_SIZE = 3


@dataclass
class ArticleRecord:
    id: str
    batch: str
    title: str
    title_norm: str
    text: str
    text_chars: int
    date_sort_key: str = ""


@dataclass
class TitleCluster:
    member_ids: list[str]
    title: str
    size: int


class UnionFind:
    def __init__(self, keys: list[str]) -> None:
        self.parent = {key: key for key in keys}

    def find(self, key: str) -> str:
        while self.parent[key] != key:
            self.parent[key] = self.parent[self.parent[key]]
            key = self.parent[key]
        return key

    def union(self, left: str, right: str) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left != root_right:
            self.parent[root_right] = root_left


def normalize_for_similarity(text: str) -> str:
    lowered = text.lower()
    return re.sub(r"\s+", " ", lowered).strip()


def word_shingles(text: str, size: int = SHINGLE_SIZE) -> set[str]:
    words = normalize_for_similarity(text).split()
    if len(words) < size:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + size]) for i in range(len(words) - size + 1)}


def jaccard_similarity(set_a: set[str], set_b: set[str]) -> float:
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def text_similarity(text_a: str, text_b: str) -> float:
    return jaccard_similarity(word_shingles(text_a), word_shingles(text_b))


def title_similarity(title_a: str, title_b: str) -> float:
    return text_similarity(title_a, title_b)


def make_safe_text_similarity(
    text_by_id: dict[str, str],
    text_chars_by_id: dict[str, int],
) -> Callable[[str, str], float]:
    def safe_text_similarity(id_a: str, id_b: str) -> float:
        if text_chars_by_id.get(id_a, 0) < MIN_TEXT_CHARS_FOR_MATCH:
            return 0.0
        if text_chars_by_id.get(id_b, 0) < MIN_TEXT_CHARS_FOR_MATCH:
            return 0.0
        return text_similarity(text_by_id.get(id_a, ""), text_by_id.get(id_b, ""))

    return safe_text_similarity


def find_group_medoid(
    member_ids: list[str],
    sim_fn: Callable[[str, str], float],
) -> tuple[str, dict[str, float]]:
    if len(member_ids) == 1:
        return member_ids[0], {member_ids[0]: 1.0}

    mean_sims: dict[str, float] = {}
    for mid in member_ids:
        others = [oid for oid in member_ids if oid != mid]
        if not others:
            mean_sims[mid] = 1.0
            continue
        scores = [sim_fn(mid, oid) for oid in others]
        mean_sims[mid] = sum(scores) / len(scores)

    medoid_id = max(member_ids, key=lambda mid: mean_sims[mid])
    sim_to_medoid = {mid: sim_fn(mid, medoid_id) for mid in member_ids}
    return medoid_id, sim_to_medoid


def simulate_iterative_dedup(
    member_ids: list[str],
    sim_fn: Callable[[str, str], float],
    threshold: float = TEXT_DEDUP_THRESHOLD,
) -> dict:
    remaining = set(member_ids)
    used_centroids: set[str] = set()
    runs: list[dict] = []
    run_num = 0
    stop_reason = "no_runs_needed"
    terminal_detail = "No removals triggered."

    while True:
        if len(remaining) <= 1:
            stop_reason = "single_article_remaining"
            terminal_detail = (
                f"After run {run_num}: only {len(remaining)} article(s) remain."
            )
            break

        eligible = [mid for mid in remaining if mid not in used_centroids]
        if not eligible:
            stop_reason = "no_eligible_centroids"
            terminal_detail = (
                f"After run {run_num}: all {len(remaining)} remaining articles "
                "already served as centroid."
            )
            break

        centroid_id, _ = find_group_medoid(eligible, sim_fn=sim_fn)
        sim_to_centroid = {mid: sim_fn(mid, centroid_id) for mid in remaining}
        to_remove = [
            mid
            for mid in remaining
            if mid != centroid_id and sim_to_centroid[mid] >= threshold
        ]
        if not to_remove:
            others = [mid for mid in remaining if mid != centroid_id]
            max_sim = max((sim_to_centroid[mid] for mid in others), default=0.0)
            stop_reason = "no_removable_matches"
            terminal_detail = (
                f"After run {run_num}: centroid {centroid_id} had 0 removable matches "
                f"(max sim among others = {max_sim:.2f}, threshold = {threshold:.2f})."
            )
            break

        run_num += 1
        runs.append(
            {
                "run": run_num,
                "centroid_id": centroid_id,
                "removed_ids": sorted(to_remove),
                "kept_ids": sorted(mid for mid in remaining if mid not in to_remove),
            }
        )
        remaining -= set(to_remove)
        used_centroids.add(centroid_id)

    if runs and stop_reason == "no_runs_needed":
        stop_reason = "completed"
        terminal_detail = f"Stopped after run {len(runs)} with {len(remaining)} article(s) kept."

    return {
        "runs": runs,
        "final_kept": sorted(remaining),
        "total_removed": len(member_ids) - len(remaining),
        "stop_reason": stop_reason,
        "terminal_detail": terminal_detail,
    }


def build_fuzzy_title_clusters(
    articles: list[ArticleRecord],
    threshold: float = TITLE_SIM_THRESHOLD,
) -> list[TitleCluster]:
    titled = [article for article in articles if article.title_norm]
    if not titled:
        return []

    norm_to_title: dict[str, str] = {}
    norm_sizes: dict[str, int] = defaultdict(int)
    for article in titled:
        norm_sizes[article.title_norm] += 1
        if article.title_norm not in norm_to_title:
            norm_to_title[article.title_norm] = article.title

    norms = sorted(norm_sizes, key=lambda norm: (-norm_sizes[norm], norm))

    buckets: dict[str, list[str]] = defaultdict(list)
    for norm in norms:
        first_word = norm.split()[0] if norm.split() else norm[:1] or "_"
        buckets[first_word].append(norm)

    uf = UnionFind(norms)
    for bucket_norms in buckets.values():
        for i, left_norm in enumerate(bucket_norms):
            left_title = norm_to_title[left_norm]
            for right_norm in bucket_norms[i + 1 :]:
                if title_similarity(left_title, norm_to_title[right_norm]) >= threshold:
                    uf.union(left_norm, right_norm)

    components: dict[str, list[str]] = defaultdict(list)
    for norm in norms:
        components[uf.find(norm)].append(norm)

    articles_by_norm: dict[str, list[ArticleRecord]] = defaultdict(list)
    for article in titled:
        articles_by_norm[article.title_norm].append(article)

    clusters: list[TitleCluster] = []
    for norms_in_cluster in components.values():
        members: list[ArticleRecord] = []
        for norm in norms_in_cluster:
            members.extend(articles_by_norm[norm])
        if len(members) < 2:
            continue
        members.sort(key=lambda article: (article.date_sort_key, article.id))
        clusters.append(
            TitleCluster(
                member_ids=[article.id for article in members],
                title=members[0].title,
                size=len(members),
            )
        )
    return clusters


def collect_removals(
    articles: list[ArticleRecord],
    title_threshold: float = TITLE_SIM_THRESHOLD,
    text_threshold: float = TEXT_DEDUP_THRESHOLD,
) -> tuple[set[str], dict]:
    text_by_id = {article.id: article.text for article in articles}
    text_chars_by_id = {article.id: article.text_chars for article in articles}
    sim_fn = make_safe_text_similarity(text_by_id, text_chars_by_id)

    clusters = build_fuzzy_title_clusters(articles, threshold=title_threshold)
    ids_to_remove: set[str] = set()
    clusters_with_removals = 0
    total_dedup_runs = 0

    for cluster in clusters:
        dedup = simulate_iterative_dedup(
            cluster.member_ids,
            sim_fn=sim_fn,
            threshold=text_threshold,
        )
        remove = set(cluster.member_ids) - set(dedup["final_kept"])
        if remove:
            clusters_with_removals += 1
            total_dedup_runs += len(dedup["runs"])
        ids_to_remove |= remove

    stats = {
        "clusters": len(clusters),
        "clusters_with_removals": clusters_with_removals,
        "removed": len(ids_to_remove),
        "total_dedup_runs": total_dedup_runs,
        "title_sim_threshold": title_threshold,
        "text_dedup_threshold": text_threshold,
        "min_text_chars_for_match": MIN_TEXT_CHARS_FOR_MATCH,
    }
    return ids_to_remove, stats
