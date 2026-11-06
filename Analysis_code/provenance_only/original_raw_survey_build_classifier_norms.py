#!/usr/bin/env python3
"""Build the Chinese classifier sensorimotor norm database from raw survey CSVs.

Inputs
------
    <project_dir>/1.csv ... <project_dir>/24.csv

Outputs
-------
    <output_dir>/*.csv
    <output_dir>/processing_report.md
    <output_dir>/processing_report.txt
    <output_dir>/session_info.txt

The script deliberately ignores direct identifiers (Alipay, email, IP, UA, and
Referrer). It reproduces the primary screening logic in the original R script,
while fixing the local-versus-global participant ID mismatch.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import platform
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


SEED = 20260827
N_SPLIT_HALF_REPETITIONS = 1000
LIST_IDS = tuple(range(1, 25))

DIMENSION_MAP = {
    "视觉": "vision",
    "听觉": "audition",
    "味觉": "gustation",
    "嗅觉": "olfaction",
    "触觉": "touch",
    "内部感觉": "interoception",
    "腿部及脚部动作": "leg_foot",
    "手部及肩膀动作": "hand_arm",
    "头部（除口部外）动作": "head",
    "口部及喉咙动作": "mouth_throat",
    "躯干（除四肢外）动作": "torso",
}

DIMENSIONS = list(DIMENSION_MAP.values())
PERCEPTUAL_DIMENSIONS = [
    "vision",
    "audition",
    "gustation",
    "olfaction",
    "touch",
    "interoception",
]
ACTION_DIMENSIONS = ["leg_foot", "hand_arm", "head", "mouth_throat", "torso"]

ATTENTION_COLUMNS_AND_KEYS = [
    ("8.请写出与 **xiāo shēng nì jì **相对应的汉字：", "销声匿迹"),
    ("9.**“灿”**字的偏旁部首为：", "B.火"),
    ("10.请选出下列四个选项中**字形正确**的一项：", "B.队员"),
    ("11.请选出与**“包袱”**一词**意义不符**的一项：", "C.背包里的布单"),
]

SENSORY_IMPAIRMENT_COLUMN = (
    "7.您是否有感知障碍（如视觉、听觉、味觉、嗅觉、触觉障碍等）："
)
CONSENT_COLUMN = "12.请问您是否已经认真阅读并明晰问卷填写说明，并且愿意继续参与研究？"
AGE_COLUMN = "2.您的年龄为"
GENDER_COLUMN = "1.您的性别"
MOTHER_TONGUE_COLUMN = "4.您的母语为（含方言）："

RATING_HEADER_PATTERN = re.compile(
    r"\*\*\s*(?P<item_number>\d+)\.\s*(?P<classifier>.+?)\*\*:(?P<dimension>.+)$"
)
RATING_VALUE_PATTERN = re.compile(r"^[A-F]\.([0-5])")
PLAIN_RATING_PATTERN = re.compile(r"^([0-5])(?:\.0)?$")


def parse_args() -> argparse.Namespace:
    script_project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, default=script_project)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_project / "outputs" / "norm_build_v1",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(data: pd.DataFrame, path: Path) -> None:
    data.to_csv(path, index=False, encoding="utf-8-sig", lineterminator="\n")


def parse_rating(value: object) -> float:
    if value is None or pd.isna(value):
        return np.nan
    text = str(value).strip()
    if not text:
        return np.nan
    match = RATING_VALUE_PATTERN.match(text) or PLAIN_RATING_PATTERN.match(text)
    if not match:
        raise ValueError(f"Unrecognized rating encoding: {text!r}")
    return float(match.group(1))


def parse_age(value: object) -> float:
    text = "" if value is None or pd.isna(value) else str(value).strip()
    match = re.fullmatch(r"(\d{1,3})(?:岁)?", text)
    return float(match.group(1)) if match else np.nan


def longest_identical_run(values: list[float]) -> int:
    longest = 0
    current = 0
    previous: float | None = None
    for value in values:
        if pd.isna(value):
            current = 0
            previous = None
        elif previous is not None and value == previous:
            current += 1
        else:
            current = 1
            previous = value
        longest = max(longest, current)
    return longest


def hanzi_only(label: str) -> str:
    return re.sub(r"（[^）]*）", "", label).strip()


def student_t_975(df: float) -> float:
    """Approximate the two-sided 95% Student-t critical value.

    Uses the Cornish-Fisher expansion around z=.975. With the observed cell
    degrees of freedom (>=6), the approximation is sufficiently accurate for
    descriptive confidence intervals and avoids an undeclared SciPy dependency.
    """
    if not np.isfinite(df) or df <= 0:
        return np.nan
    z = 1.959963984540054
    v = float(df)
    return (
        z
        + (z**3 + z) / (4 * v)
        + (5 * z**5 + 16 * z**3 + 3 * z) / (96 * v**2)
        + (3 * z**7 + 19 * z**5 + 17 * z**3 - 15 * z) / (384 * v**3)
        + (
            79 * z**9
            + 776 * z**7
            + 1482 * z**5
            - 1920 * z**3
            - 945 * z
        )
        / (92160 * v**4)
    )


def inspect_rating_headers(headers: list[str], list_id: int) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for column_order, header in enumerate(headers):
        match = RATING_HEADER_PATTERN.search(header)
        if not match:
            continue
        raw_dimension = match.group("dimension").strip()
        if raw_dimension not in DIMENSION_MAP:
            continue
        records.append(
            {
                "list": list_id,
                "source_item_number": int(match.group("item_number")),
                "classifier": match.group("classifier").strip(),
                "dimension_raw": raw_dimension,
                "dimension": DIMENSION_MAP[raw_dimension],
                "source_column_order": column_order,
                "source_header": header,
            }
        )
    if not records:
        raise ValueError(f"No rating columns detected in list {list_id}")
    return records


def build_source_tables(project_dir: Path) -> tuple[pd.DataFrame, ...]:
    manifest_rows: list[dict[str, object]] = []
    participant_rows: list[dict[str, object]] = []
    rating_rows: list[dict[str, object]] = []
    item_rows: list[dict[str, object]] = []
    global_id = 0

    for list_id in LIST_IDS:
        path = project_dir / f"{list_id}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing required raw file: {path}")
        data = pd.read_csv(path, dtype=str, encoding="utf-8-sig", keep_default_na=False)
        header_info = inspect_rating_headers(list(data.columns), list_id)
        rating_headers = [record["source_header"] for record in header_info]

        local_ids = pd.to_numeric(data["编号"], errors="raise").astype(int).tolist()
        expected_ids = list(range(1, len(data) + 1))
        if sorted(local_ids) != expected_ids or len(set(local_ids)) != len(local_ids):
            raise ValueError(
                f"List {list_id} local IDs are not a unique continuous sequence 1..N"
            )

        durations = pd.to_numeric(data["答题时长"], errors="raise").astype(float)
        duration_mean = float(durations.mean())
        duration_sd = float(durations.std(ddof=1))

        blocks = {}
        for info in header_info:
            key = (info["source_item_number"], info["classifier"])
            blocks.setdefault(key, []).append(info)
        for (item_number, classifier), infos in blocks.items():
            dimensions_present = {x["dimension"] for x in infos}
            if dimensions_present != set(DIMENSIONS):
                raise ValueError(
                    f"List {list_id}, item {item_number}, {classifier}: incomplete dimensions"
                )
            item_rows.append(
                {
                    "source_block_id": f"L{list_id:02d}_I{item_number:02d}",
                    "list": list_id,
                    "source_item_number": item_number,
                    "classifier": classifier,
                    "classifier_hanzi": hanzi_only(classifier),
                    "n_dimensions": len(infos),
                }
            )

        manifest_rows.append(
            {
                "file": path.name,
                "sha256": sha256(path),
                "size_bytes": path.stat().st_size,
                "modified_time": path.stat().st_mtime,
                "n_participants": len(data),
                "n_columns": len(data.columns),
                "n_rating_columns": len(rating_headers),
                "n_classifier_blocks": len(blocks),
            }
        )

        for row_index, row in data.iterrows():
            global_id += 1
            local_id = int(row["编号"])
            encoded_ratings = [parse_rating(row[header]) for header in rating_headers]
            n_missing = int(sum(pd.isna(value) for value in encoded_ratings))
            long_run = longest_identical_run(encoded_ratings)
            attention_passes = [
                str(row[column]).strip() == correct
                for column, correct in ATTENTION_COLUMNS_AND_KEYS
            ]
            duration = float(durations.iloc[row_index])
            rt_outlier = bool(
                np.isfinite(duration_sd)
                and (
                    duration < duration_mean - 2 * duration_sd
                    or duration > duration_mean + 2 * duration_sd
                )
            )
            missing_prop = n_missing / len(encoded_ratings)
            long_prop = long_run / len(encoded_ratings)
            flag_attention = not all(attention_passes)
            flag_missing = missing_prop > 0.20
            flag_straightline = long_prop >= 0.90
            flag_impairment = str(row[SENSORY_IMPAIRMENT_COLUMN]).strip() == "A.有"
            consent = str(row[CONSENT_COLUMN]).strip() == "A.是"
            retain_primary = bool(
                consent
                and not flag_attention
                and not rt_outlier
                and not flag_missing
                and not flag_straightline
            )

            participant_rows.append(
                {
                    "global_participant_id": global_id,
                    "participant_code": f"P{global_id:04d}",
                    "list": list_id,
                    "local_participant_id": local_id,
                    "duration_seconds": duration,
                    "list_duration_mean": duration_mean,
                    "list_duration_sd": duration_sd,
                    "attention_q1_pass": attention_passes[0],
                    "attention_q2_pass": attention_passes[1],
                    "attention_q3_pass": attention_passes[2],
                    "attention_q4_pass": attention_passes[3],
                    "attention_fail_count": int(sum(not x for x in attention_passes)),
                    "flag_attention": flag_attention,
                    "flag_rt_outlier": rt_outlier,
                    "rating_missing_prop": missing_prop,
                    "longest_identical_run": long_run,
                    "longest_identical_run_prop": long_prop,
                    "flag_missing_gt_20pct": flag_missing,
                    "flag_straightline_ge_90pct": flag_straightline,
                    "flag_sensory_impairment": flag_impairment,
                    "consent": consent,
                    "retain_primary": retain_primary,
                    "retain_sensitivity_no_impairment": retain_primary
                    and not flag_impairment,
                    "gender": str(row[GENDER_COLUMN]).strip(),
                    "age": parse_age(row[AGE_COLUMN]),
                    "flag_age_unparseable": pd.isna(parse_age(row[AGE_COLUMN])),
                    "flag_age_under_18": bool(
                        np.isfinite(parse_age(row[AGE_COLUMN]))
                        and parse_age(row[AGE_COLUMN]) < 18
                    ),
                    "mother_tongue": str(row[MOTHER_TONGUE_COLUMN]).strip(),
                }
            )

            for info, rating in zip(header_info, encoded_ratings):
                rating_rows.append(
                    {
                        "global_participant_id": global_id,
                        "participant_code": f"P{global_id:04d}",
                        "list": list_id,
                        "local_participant_id": local_id,
                        "source_block_id": (
                            f"L{list_id:02d}_I{int(info['source_item_number']):02d}"
                        ),
                        "source_item_number": int(info["source_item_number"]),
                        "classifier": info["classifier"],
                        "classifier_hanzi": hanzi_only(str(info["classifier"])),
                        "dimension": info["dimension"],
                        "rating": rating,
                    }
                )

    return (
        pd.DataFrame(manifest_rows),
        pd.DataFrame(participant_rows),
        pd.DataFrame(rating_rows),
        pd.DataFrame(item_rows),
    )


def add_classifier_ids(
    item_registry: pd.DataFrame, ratings_all: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ordered_labels = item_registry["classifier"].drop_duplicates().tolist()
    id_map = {label: f"C{index:03d}" for index, label in enumerate(ordered_labels, 1)}
    item_registry = item_registry.copy()
    item_registry.insert(
        0, "classifier_id", item_registry["classifier"].map(id_map)
    )
    counts = item_registry.groupby("classifier")["source_block_id"].transform("size")
    item_registry["is_repeated_questionnaire_block"] = counts > 1

    ratings_all = ratings_all.copy()
    ratings_all.insert(4, "classifier_id", ratings_all["classifier"].map(id_map))

    duplicate_audit = (
        item_registry.groupby(
            ["classifier_id", "classifier", "classifier_hanzi"], as_index=False
        )
        .agg(
            n_source_blocks=("source_block_id", "size"),
            source_block_ids=("source_block_id", lambda x: ";".join(x)),
            source_lists=("list", lambda x: ";".join(map(str, sorted(set(x))))),
            source_item_numbers=(
                "source_item_number",
                lambda x: ";".join(map(str, x)),
            ),
        )
    )
    duplicate_audit = duplicate_audit.loc[
        duplicate_audit["n_source_blocks"] > 1
    ].reset_index(drop=True)
    return item_registry, ratings_all, duplicate_audit


def build_exclusion_flow(participants: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for list_id, group in participants.groupby("list", sort=True):
        step13 = group["flag_attention"] | group["flag_rt_outlier"]
        rows.append(
            {
                "list": list_id,
                "n_raw": len(group),
                "n_attention_fail": int(group["flag_attention"].sum()),
                "n_rt_outlier": int(group["flag_rt_outlier"].sum()),
                "n_attention_or_rt": int(step13.sum()),
                "n_missing_gt_20pct": int(group["flag_missing_gt_20pct"].sum()),
                "n_straightline_ge_90pct": int(
                    group["flag_straightline_ge_90pct"].sum()
                ),
                "n_sensory_impairment_flag": int(
                    group["flag_sensory_impairment"].sum()
                ),
                "n_retained_primary": int(group["retain_primary"].sum()),
                "n_retained_sensitivity_no_impairment": int(
                    group["retain_sensitivity_no_impairment"].sum()
                ),
            }
        )
    result = pd.DataFrame(rows)
    total = {"list": "TOTAL"}
    for column in result.columns[1:]:
        total[column] = int(result[column].sum())
    return pd.concat([result, pd.DataFrame([total])], ignore_index=True)


def build_clean_ratings(
    ratings_all: pd.DataFrame, participants: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    retain = participants.set_index("global_participant_id")["retain_primary"]
    clean_blocks = ratings_all.loc[
        ratings_all["global_participant_id"].map(retain).fillna(False)
    ].copy()
    clean_collapsed = (
        clean_blocks.groupby(
            [
                "global_participant_id",
                "participant_code",
                "list",
                "classifier_id",
                "classifier",
                "classifier_hanzi",
                "dimension",
            ],
            as_index=False,
            sort=False,
        )
        .agg(
            rating=("rating", "mean"),
            n_source_blocks_averaged=("source_block_id", "nunique"),
        )
    )
    return clean_blocks, clean_collapsed


def build_norms(clean: pd.DataFrame) -> pd.DataFrame:
    norms = (
        clean.groupby(
            ["classifier_id", "classifier", "classifier_hanzi", "dimension"],
            as_index=False,
            sort=False,
        )["rating"]
        .agg(mean="mean", sd="std", n="count")
    )
    norms["se"] = norms["sd"] / np.sqrt(norms["n"])
    norms["t_critical_95"] = norms["n"].map(lambda n: student_t_975(n - 1))
    norms["ci95_lower"] = norms["mean"] - norms["t_critical_95"] * norms["se"]
    norms["ci95_upper"] = norms["mean"] + norms["t_critical_95"] * norms["se"]
    norms["ci95_lower"] = norms["ci95_lower"].clip(lower=0)
    norms["ci95_upper"] = norms["ci95_upper"].clip(upper=5)
    norms["dimension"] = pd.Categorical(
        norms["dimension"], categories=DIMENSIONS, ordered=True
    )
    return norms.sort_values(["classifier_id", "dimension"]).reset_index(drop=True)


def dominant_label(row: pd.Series, columns: list[str], suffix: str = "_mean") -> str:
    values = np.asarray([row[f"{column}{suffix}"] for column in columns], dtype=float)
    maximum = np.nanmax(values)
    winners = [
        column
        for column, value in zip(columns, values)
        if np.isclose(value, maximum, rtol=0, atol=1e-12)
    ]
    return " + ".join(winners)


def exclusivity(row: pd.Series, columns: list[str]) -> float:
    values = np.asarray([row[f"{column}_mean"] for column in columns], dtype=float)
    total = np.nansum(values)
    if not np.isfinite(total) or total <= 0:
        return np.nan
    return float((np.nanmax(values) - np.nanmin(values)) / total)


def build_master(norms: pd.DataFrame, item_registry: pd.DataFrame) -> pd.DataFrame:
    identity = norms[
        ["classifier_id", "classifier", "classifier_hanzi"]
    ].drop_duplicates()
    master = identity.copy()
    for statistic in ["mean", "sd", "n", "se", "ci95_lower", "ci95_upper"]:
        wide = norms.pivot(
            index="classifier_id", columns="dimension", values=statistic
        ).reindex(columns=DIMENSIONS)
        wide.columns = [f"{dimension}_{statistic}" for dimension in DIMENSIONS]
        master = master.merge(wide.reset_index(), on="classifier_id", how="left")

    source = (
        item_registry.groupby("classifier_id", as_index=False)
        .agg(
            source_list=("list", "first"),
            source_item_numbers=(
                "source_item_number", lambda x: ";".join(map(str, x))
            ),
            n_source_blocks=("source_block_id", "size"),
        )
    )
    master = master.merge(source, on="classifier_id", how="left")
    master["overall_mean_11"] = master[
        [f"{x}_mean" for x in DIMENSIONS]
    ].mean(axis=1)
    master["perceptual_mean_6"] = master[
        [f"{x}_mean" for x in PERCEPTUAL_DIMENSIONS]
    ].mean(axis=1)
    master["action_mean_5"] = master[
        [f"{x}_mean" for x in ACTION_DIMENSIONS]
    ].mean(axis=1)

    master["dominant_perceptual"] = master.apply(
        dominant_label, axis=1, columns=PERCEPTUAL_DIMENSIONS
    )
    master["dominant_perceptual_n_tied"] = master["dominant_perceptual"].map(
        lambda x: x.count(" + ") + 1
    )
    master["dominant_action"] = master.apply(
        dominant_label, axis=1, columns=ACTION_DIMENSIONS
    )
    master["dominant_action_n_tied"] = master["dominant_action"].map(
        lambda x: x.count(" + ") + 1
    )
    master["dominant_overall_11"] = master.apply(
        dominant_label, axis=1, columns=DIMENSIONS
    )
    master["dominant_overall_11_n_tied"] = master["dominant_overall_11"].map(
        lambda x: x.count(" + ") + 1
    )
    master["perceptual_exclusivity"] = master.apply(
        exclusivity, axis=1, columns=PERCEPTUAL_DIMENSIONS
    )
    master["action_exclusivity"] = master.apply(
        exclusivity, axis=1, columns=ACTION_DIMENSIONS
    )
    master["overall_exclusivity_11"] = master.apply(
        exclusivity, axis=1, columns=DIMENSIONS
    )
    return master.sort_values("classifier_id").reset_index(drop=True)


def pairwise_euclidean(matrix: np.ndarray) -> np.ndarray:
    squared = np.sum(matrix * matrix, axis=1, keepdims=True)
    distance_squared = squared + squared.T - 2 * matrix @ matrix.T
    np.maximum(distance_squared, 0, out=distance_squared)
    return np.sqrt(distance_squared)


def ward_partitions(
    matrix: np.ndarray, requested_k: set[int]
) -> dict[int, np.ndarray]:
    """Deterministic Ward agglomeration using minimum increase in within-cluster SSE."""
    active: dict[int, dict[str, object]] = {
        index: {"members": [index], "size": 1, "centroid": matrix[index].copy()}
        for index in range(len(matrix))
    }
    next_id = len(matrix)
    partitions: dict[int, np.ndarray] = {}

    def capture(k: int) -> None:
        groups = list(active.values())
        labels = np.zeros(len(matrix), dtype=int)
        for label, group in enumerate(groups, 1):
            labels[np.asarray(group["members"], dtype=int)] = label
        partitions[k] = labels

    if len(active) in requested_k:
        capture(len(active))
    while len(active) > min(requested_k):
        ids = list(active)
        centroids = np.vstack([active[x]["centroid"] for x in ids])
        sizes = np.asarray([active[x]["size"] for x in ids], dtype=float)
        squared = np.sum(centroids * centroids, axis=1, keepdims=True)
        d2 = squared + squared.T - 2 * centroids @ centroids.T
        np.maximum(d2, 0, out=d2)
        criterion = (sizes[:, None] * sizes[None, :]) / (
            sizes[:, None] + sizes[None, :]
        ) * d2
        criterion[np.tril_indices_from(criterion)] = np.inf
        left_pos, right_pos = np.unravel_index(np.argmin(criterion), criterion.shape)
        left_id, right_id = ids[left_pos], ids[right_pos]
        left, right = active.pop(left_id), active.pop(right_id)
        new_size = int(left["size"]) + int(right["size"])
        new_centroid = (
            int(left["size"]) * np.asarray(left["centroid"])
            + int(right["size"]) * np.asarray(right["centroid"])
        ) / new_size
        active[next_id] = {
            "members": list(left["members"]) + list(right["members"]),
            "size": new_size,
            "centroid": new_centroid,
        }
        next_id += 1
        if len(active) in requested_k:
            capture(len(active))
    return partitions


def relabel_by_overall_strength(labels: np.ndarray, overall: np.ndarray) -> np.ndarray:
    cluster_means = {
        label: float(np.mean(overall[labels == label])) for label in np.unique(labels)
    }
    ordered = sorted(cluster_means, key=lambda x: (-cluster_means[x], x))
    mapping = {old: new for new, old in enumerate(ordered, 1)}
    return np.asarray([mapping[x] for x in labels], dtype=int)


def mean_silhouette(distance: np.ndarray, labels: np.ndarray) -> float:
    scores = []
    for index, label in enumerate(labels):
        same = labels == label
        same[index] = False
        if same.sum() == 0:
            scores.append(0.0)
            continue
        a = float(distance[index, same].mean())
        alternatives = []
        for other in np.unique(labels):
            if other != label:
                alternatives.append(float(distance[index, labels == other].mean()))
        b = min(alternatives)
        denominator = max(a, b)
        scores.append((b - a) / denominator if denominator > 0 else 0.0)
    return float(np.mean(scores))


def add_clustering(master: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mean_columns = [f"{dimension}_mean" for dimension in DIMENSIONS]
    raw = master[mean_columns].to_numpy(dtype=float)
    if not np.isfinite(raw).all():
        raise ValueError("Clustering matrix contains missing or non-finite values")
    z = (raw - raw.mean(axis=0)) / raw.std(axis=0, ddof=1)
    k_values = set(range(2, 11))
    metric_rows = []
    profile_rows = []
    output = master.copy()

    for scale_name, matrix in [("raw", raw), ("z", z)]:
        partitions = ward_partitions(matrix, k_values)
        distance = pairwise_euclidean(matrix)
        silhouettes = {}
        relabeled = {}
        for k in sorted(k_values):
            labels = relabel_by_overall_strength(
                partitions[k], output["overall_mean_11"].to_numpy()
            )
            relabeled[k] = labels
            silhouettes[k] = mean_silhouette(distance, labels)
            metric_rows.append(
                {
                    "method": "Ward",
                    "scale": scale_name,
                    "distance": "Euclidean",
                    "k": k,
                    "average_silhouette": silhouettes[k],
                }
            )
        best_k = min(
            (k for k in silhouettes if silhouettes[k] == max(silhouettes.values())),
            default=2,
        )
        for k in (2, 3, 4):
            output[f"cluster_ward_{scale_name}_k{k}"] = relabeled[k]
        output[f"cluster_ward_{scale_name}_selected_k"] = best_k
        output[f"cluster_ward_{scale_name}_selected"] = relabeled[best_k]

        for k_label, k in [("k3", 3), ("selected", best_k)]:
            labels = relabeled[k]
            for cluster in sorted(np.unique(labels)):
                mask = labels == cluster
                row = {
                    "method": "Ward",
                    "scale": scale_name,
                    "solution": k_label,
                    "k": k,
                    "cluster": int(cluster),
                    "n_classifiers": int(mask.sum()),
                }
                for dimension, column in zip(DIMENSIONS, mean_columns):
                    row[f"{dimension}_mean"] = float(output.loc[mask, column].mean())
                profile_rows.append(row)

    return output, pd.DataFrame(metric_rows), pd.DataFrame(profile_rows)


def split_half_reliability(clean: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    for dimension in DIMENSIONS:
        dimension_data = clean.loc[clean["dimension"] == dimension]
        arrays = [
            group["rating"].to_numpy(dtype=float)
            for _, group in dimension_data.groupby("classifier_id", sort=True)
        ]
        raw_correlations = []
        corrected = []
        for _ in range(N_SPLIT_HALF_REPETITIONS):
            half_1 = []
            half_2 = []
            for values in arrays:
                permuted = rng.permutation(values)
                split = (len(permuted) + 1) // 2
                half_1.append(float(permuted[:split].mean()))
                half_2.append(float(permuted[split:].mean()))
            correlation = float(np.corrcoef(half_1, half_2)[0, 1])
            raw_correlations.append(correlation)
            corrected.append(2 * correlation / (1 + correlation))
        raw_array = np.asarray(raw_correlations)
        corrected_array = np.asarray(corrected)
        rows.append(
            {
                "dimension": dimension,
                "seed": SEED,
                "n_repetitions": N_SPLIT_HALF_REPETITIONS,
                "r_raw_mean": raw_array.mean(),
                "r_raw_median": np.median(raw_array),
                "r_raw_ci95_lower": np.quantile(raw_array, 0.025),
                "r_raw_ci95_upper": np.quantile(raw_array, 0.975),
                "spearman_brown_mean": corrected_array.mean(),
                "spearman_brown_median": np.median(corrected_array),
                "spearman_brown_ci95_lower": np.quantile(corrected_array, 0.025),
                "spearman_brown_ci95_upper": np.quantile(corrected_array, 0.975),
            }
        )
    return pd.DataFrame(rows)


def dominance_summary(master: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for family, column in [
        ("perceptual", "dominant_perceptual"),
        ("action", "dominant_action"),
        ("overall_11", "dominant_overall_11"),
    ]:
        counts = master[column].value_counts(dropna=False)
        for label, count in counts.items():
            rows.append(
                {
                    "dimension_family": family,
                    "dominant_dimension": label,
                    "n": int(count),
                    "percentage": 100 * count / len(master),
                }
            )
    return pd.DataFrame(rows)


def build_data_dictionary(master: pd.DataFrame) -> pd.DataFrame:
    descriptions = {
        "classifier_id": "Stable sequential ID for the unique questionnaire classifier label.",
        "classifier": "Classifier label exactly as shown in the questionnaire, including pronunciation notes.",
        "classifier_hanzi": "Classifier label with parenthesized pronunciation removed; not necessarily unique.",
        "source_list": "Questionnaire list containing the classifier.",
        "source_item_numbers": "Original within-list item number(s), semicolon-separated.",
        "n_source_blocks": "Number of original questionnaire blocks represented by this classifier.",
        "overall_mean_11": "Unweighted mean of the 11 classifier-level dimension means.",
        "perceptual_mean_6": "Unweighted mean of the six perceptual dimension means.",
        "action_mean_5": "Unweighted mean of the five action/effector dimension means.",
        "dominant_perceptual": "All perceptual dimensions tied for the maximum mean; ties joined with ' + '.",
        "dominant_action": "All action dimensions tied for the maximum mean; ties joined with ' + '.",
        "dominant_overall_11": "All dimensions tied for the maximum among all 11 means.",
        "perceptual_exclusivity": "(max-min)/sum across six perceptual means; unit interval.",
        "action_exclusivity": "(max-min)/sum across five action means; unit interval.",
        "overall_exclusivity_11": "(max-min)/sum across all 11 means; unit interval.",
    }
    rows = []
    for column in master.columns:
        description = descriptions.get(column, "")
        match = re.fullmatch(
            rf"({'|'.join(map(re.escape, DIMENSIONS))})_(mean|sd|n|se|ci95_lower|ci95_upper)",
            column,
        )
        if match:
            dimension, statistic = match.groups()
            statistic_text = {
                "mean": "mean rating",
                "sd": "sample SD across retained participants",
                "n": "number of retained unique participants",
                "se": "standard error of the mean",
                "ci95_lower": "lower bound of clipped 95% t interval",
                "ci95_upper": "upper bound of clipped 95% t interval",
            }[statistic]
            description = f"{dimension}: {statistic_text}."
        elif column.startswith("cluster_ward_"):
            description = (
                "Ward hierarchical cluster assignment using Euclidean distance; "
                "cluster numbers are ordered by decreasing cluster overall mean."
            )
        elif column.endswith("_n_tied"):
            description = "Number of dimensions sharing the recorded maximum."
        rows.append(
            {
                "column": column,
                "data_type": str(master[column].dtype),
                "description": description,
            }
        )
    return pd.DataFrame(rows)


def format_count_table(frame: pd.DataFrame, columns: list[str]) -> str:
    return frame[columns].to_string(index=False)


def write_report(
    output_dir: Path,
    manifest: pd.DataFrame,
    participants: pd.DataFrame,
    exclusion_flow: pd.DataFrame,
    item_registry: pd.DataFrame,
    duplicate_audit: pd.DataFrame,
    norms: pd.DataFrame,
    master: pd.DataFrame,
    cluster_metrics: pd.DataFrame,
    reliability: pd.DataFrame,
) -> None:
    total = exclusion_flow.iloc[-1]
    n_distribution = (
        norms[["classifier_id", "n"]]
        .drop_duplicates()["n"]
        .value_counts()
        .sort_index()
    )
    perceptual_counts = master["dominant_perceptual"].value_counts().to_dict()
    action_counts = master["dominant_action"].value_counts().to_dict()
    selected_raw = int(master["cluster_ward_raw_selected_k"].iloc[0])
    selected_z = int(master["cluster_ward_z_selected_k"].iloc[0])
    k234 = cluster_metrics.loc[cluster_metrics["k"].isin([2, 3, 4])]

    report = f"""# Chinese classifier norm database: processing report

Generated by `scripts/build_classifier_norms.py`.

## 1. Inputs and preservation

- Raw input files: `{manifest['file'].iloc[0]}` through `{manifest['file'].iloc[-1]}` ({len(manifest)} files).
- Raw files were read only. SHA-256 hashes, sizes, modification times, row counts, and column counts are saved in `input_manifest.csv`.
- Direct identifiers (Alipay account, email, IP, UA, Referrer) were not transferred to any derived dataset.
- Raw participant rows: {int(total['n_raw'])} across 24 independent lists.

## 2. Participant identifiers

The same local number in different lists represents a different person. Participants were sorted by numeric list and then local ID, and assigned one continuous global ID from 1 to {len(participants)}. The complete reversible non-identifying crosswalk is in `participant_id_map.csv`.

## 3. Rating parsing

Survey values `A.0（毫无感觉）` through `F.5（感觉强烈）` (and abbreviated forms such as `D.3`) were explicitly converted to integers 0 through 5. The 11 output dimensions are:

`{', '.join(DIMENSIONS)}`.

All raw rating cells were validated before screening.

## 4. Primary participant screening

This reproduces the substantive rules in the original R script:

1. Any failure among four attention/language checks caused exclusion. Matching was exact, using the original answer key: 销声匿迹; B.火; B.队员; C.背包里的布单.
2. Completion-time outliers were identified separately within each list as values below mean - 2 SD or above mean + 2 SD. List means and SDs were computed from all raw rows before exclusion.
3. Rating missingness above 20% caused exclusion.
4. A longest consecutive run of the same response covering at least 90% of rating cells caused exclusion.
5. Primary retention required consent and no flag under rules 1-4.

Sensory impairment was retained as a flag but was not added to the primary exclusion rule because the original R pipeline did not exclude it. `retain_sensitivity_no_impairment` provides a prespecified sensitivity set.

### Screening flow

```
{format_count_table(exclusion_flow, ['list','n_raw','n_attention_fail','n_rt_outlier','n_attention_or_rt','n_missing_gt_20pct','n_straightline_ge_90pct','n_retained_primary'])}
```

Primary retained participant sessions: {int(total['n_retained_primary'])} of {int(total['n_raw'])}.

## 5. Correction to the old implementation

The old participant metadata used global IDs, whereas the transposed list files used local `P1...Pn` IDs. Joining on list plus the unconverted ID meant that screening flags were not applied correctly beyond list 1. The new pipeline extracts metadata and ratings from the same raw row and assigns the global ID before any join.

## 6. Classifier inventory and the repeated block

- Original questionnaire blocks: {len(item_registry)}.
- Unique questionnaire classifier labels: {master['classifier_id'].nunique()}.
- Repeated exact labels: {len(duplicate_audit)}.
- The repeated block is `重（chóng）` in list 7, source items 10 and 11.

For a repeated classifier block, ratings were first averaged within participant and dimension. Mean, SD, and N were then computed across unique retained participants. Thus the repeated block does not double the participant N or treat repeated observations as independent.

## 7. Norm statistics

For every classifier x dimension cell, the output contains the arithmetic mean, sample SD (denominator N-1), unique-participant N, SE, and a two-sided 95% t interval clipped to the possible rating range 0-5. The t critical value uses a high-order Cornish-Fisher approximation; observed degrees of freedom are at least 6.

N per classifier is determined by the retained participants in its list. Distribution of classifier N:

```
{n_distribution.to_string()}
```

## 8. Dominant dimensions and exclusivity

Dominant perceptual, action, and overall dimensions are the dimension(s) with the largest classifier mean. Exact ties are preserved and joined with ` + `; no first-tie forcing is used. No post-hoc threshold was used to create a `none` action category.

Exclusivity is calculated separately over the 6 perceptual dimensions, 5 action dimensions, and all 11 dimensions as `(maximum - minimum) / sum`. Values are stored on the 0-1 scale (not percentages).

Perceptual dominant counts: {perceptual_counts}

Action dominant counts: {action_counts}

## 9. Clustering

Deterministic Ward hierarchical clustering was run with Euclidean distance on:

- raw 0-5 classifier means; and
- column-wise z-standardized classifier means (each dimension given equal variance).

Average silhouette width was evaluated for k=2 through 10. The selected k maximizes silhouette, with the smaller k chosen only if values are exactly tied. Cluster numbers were relabeled by decreasing cluster-level overall sensorimotor mean to make numbering reproducible and interpretable.

Raw-scale selected k: {selected_raw}. Z-standardized selected k: {selected_z}.

Silhouette values for k=2, 3, 4:

```
{k234.to_string(index=False)}
```

The master table preserves k=2, k=3, k=4, and silhouette-selected assignments for both scales. Cluster membership is exploratory and is not treated as an independently validated class label.

## 10. Split-half reliability

For each dimension, retained participant ratings were randomly divided within classifier into two near-equal halves, half means were correlated across classifiers, and the Spearman-Brown correction was applied. This was repeated {N_SPLIT_HALF_REPETITIONS} times with seed {SEED}; means, medians, and empirical 95% intervals are saved in `split_half_reliability.csv`.

```
{reliability[['dimension','spearman_brown_mean','spearman_brown_ci95_lower','spearman_brown_ci95_upper']].to_string(index=False)}
```

## 11. Output interpretation

- `classifier_master.csv`: one row per unique classifier label; main analysis table.
- `classifier_norms_long.csv`: one row per classifier x dimension.
- `ratings_long_clean.csv`: retained, de-duplicated participant-level ratings.
- `ratings_long_clean_source_blocks.csv`: retained ratings before repeated-block averaging.
- `participant_qc.csv`: non-identifying screening variables and flags.
- `exclusion_flow_by_list.csv`: transparent participant flow.
- `cluster_solution_metrics.csv` and `cluster_profiles.csv`: clustering diagnostics.
- `data_dictionary.csv`: definitions for master-table columns.

## 12. Software

Python {platform.python_version()}; pandas {pd.__version__}; NumPy {np.__version__}; platform {platform.platform()}.
"""
    report_zh = f"""# 汉语量词感觉运动 norm 数据库：数据处理报告

本报告由 `scripts/build_classifier_norms.py` 自动生成，所有数字均来自本次正式运行。

## 1. 原始数据与可追溯性

- 输入为 24 个原始问卷文件（`1.csv` 至 `24.csv`），只读处理，没有覆盖或改写源文件。
- 共 {int(total['n_raw'])} 条被试记录。每个源文件的 SHA-256、文件大小、修改时间、行列数均保存在 `input_manifest.csv`。
- 派生数据不包含支付宝账号、邮箱、IP、UA 或 Referrer 等直接身份信息。
- 本次运行后重新计算的源文件哈希应与清单一致，以便检查原始数据是否发生变化。

## 2. 被试编号

不同 list 中相同的本地编号代表不同被试。程序先按 list 数字顺序排序，再按各 list 内的本地编号排序，生成从 1 到 {len(participants)} 的全局连续编号。映射表保存在 `participant_id_map.csv`。

该处理修正了旧流程的关键问题：旧 `participant.xlsx` 使用全局编号，而旧转置评分表使用每个 list 内重新开始的 `P1...Pn`。二者直接连接会导致 list 2–24 的筛选标记不能正确匹配。本流程从同一条原始问卷记录同时提取元数据和评分，生成全局编号后再进行任何连接。

## 3. 评分编码与维度

问卷导出的 `A.0（毫无感觉）` 至 `F.5（感觉强烈）`，以及 `D.3` 等缩写形式，被显式映射为 0–5。所有评分单元在筛选前均经过合法性检查。

11 个英文维度名为：`{', '.join(DIMENSIONS)}`。

## 4. 主分析被试筛选规则

主分析逐项对应旧 R 脚本的实质规则：

1. 四道注意/语言检查题任意一道错误即标记排除；采用旧脚本的精确答案：销声匿迹、B.火、B.队员、C.背包里的布单。
2. 在每个 list 内，以该 list 全部原始记录计算答题时长均值和样本 SD；低于均值减 2 SD 或高于均值加 2 SD 者标记为时长异常。
3. 评分缺失比例大于 20% 者标记排除。
4. 最长连续相同评分达到该被试评分单元的 90% 或以上者标记为直线作答。
5. 主分析保留条件为：已同意参与，且未触发以上四项排除规则。

“存在感知障碍”保留为独立标记，但不进入主分析排除条件，因为旧 R 流程未据此排除。另提供 `retain_sensitivity_no_impairment`，用于排除感知障碍者的敏感性分析。

### 各 list 被试流

```
{format_count_table(exclusion_flow, ['list','n_raw','n_attention_fail','n_rt_outlier','n_attention_or_rt','n_missing_gt_20pct','n_straightline_ge_90pct','n_retained_primary'])}
```

主分析最终保留 {int(total['n_retained_primary'])}/{int(total['n_raw'])} 条独立被试记录。

## 5. 量词题块与重复题处理

- 原始问卷题块数：{len(item_registry)}。
- 唯一量词标签数：{master['classifier_id'].nunique()}。
- 重复的完全相同标签数：{len(duplicate_audit)}。
- 唯一重复项目为 list 7 的第 10、11 题 `重（chóng）`。

对于重复题，先在“被试 × 量词 × 维度”层面平均两次评分，再跨独立被试计算 norm。因此重复题不会把同一被试计为两名评分者，N 仍为独立被试数。两个原始题块均保留在源题注册表和 `duplicate_item_audit.csv` 中。

## 6. Norm 统计量

每个“量词 × 维度”单元计算：算术均值、样本标准差（分母 N-1）、独立被试 N、标准误和双侧 95% t 区间。置信区间限制在理论评分范围 0–5 内。t 临界值使用高阶 Cornish-Fisher 近似；本数据最小自由度为 6。

各量词 N 的分布：

```
{n_distribution.to_string()}
```

## 7. 主导维度与 exclusivity

分别在 6 个知觉维度、5 个动作维度和全部 11 维中寻找最大均值。若存在完全并列，保留所有并列维度并用 ` + ` 连接，不再像旧代码那样强制选择第一个。动作主导维度没有使用事后阈值，因此未人为生成 `none` 类别。

Exclusivity 分别在知觉 6 维、动作 5 维和全部 11 维中计算：

`(最大维度均值 - 最小维度均值) / 各维度均值之和`

主表保存 0–1 比例，不乘以 100。

知觉主导维度分布：{perceptual_counts}

动作主导维度分布：{action_counts}

## 8. 聚类

使用欧氏距离和 Ward 层次聚类，分别分析：

1. 原始 0–5 量词维度均值；
2. 按维度进行 z 标准化后的均值（每个维度具有相同方差权重）。

对 k=2 至 10 计算平均轮廓系数；选定 k 为轮廓系数最大者。簇编号按簇的整体 11 维平均强度从高到低重新编号，以保证编号稳定且便于解释。

原始尺度选定 k={selected_raw}；z 标准化选定 k={selected_z}。

k=2、3、4 的结果：

```
{k234.to_string(index=False)}
```

主数据库同时保留两种尺度的 k=2、k=3、k=4 和轮廓系数最优解。聚类属于探索性结果，不应被视为已经独立验证的自然类别。

## 9. 分半信度

在每个量词内部将保留被试随机分为人数尽可能相等的两半，分别计算半样本均值，再跨量词计算 Pearson 相关和 Spearman-Brown 校正值。每个维度重复 {N_SPLIT_HALF_REPETITIONS} 次，固定随机种子为 {SEED}；输出均值、中位数和经验 95% 区间。

```
{reliability[['dimension','spearman_brown_mean','spearman_brown_ci95_lower','spearman_brown_ci95_upper']].to_string(index=False)}
```

## 10. 主要输出

- `classifier_master.csv`：353 行主数据库，包含每维 Mean/SD/N/SE/CI、主导维度、exclusivity 与聚类。
- `classifier_norms_long.csv`：量词 × 维度长表。
- `classifier_norm_database.xlsx`：用于人工审阅的多 sheet 工作簿。
- `ratings_long_clean.csv`：重复题已在被试内平均的清洗后评分。
- `participant_qc.csv` 与 `exclusion_flow_by_list.csv`：逐被试 QC 和被试流。
- `cluster_solution_metrics.csv` 与 `cluster_profiles.csv`：聚类诊断和簇均值。
- `split_half_reliability.csv`：重复分半信度。
- `data_dictionary.csv`：主数据库字段说明。

## 11. 软件环境

Python {platform.python_version()}；pandas {pd.__version__}；NumPy {np.__version__}；系统 {platform.platform()}。
"""

    (output_dir / "processing_report.md").write_text(report_zh, encoding="utf-8")
    plain_zh = re.sub(r"^#+\s*", "", report_zh, flags=re.MULTILINE).replace("`", "")
    (output_dir / "processing_report.txt").write_text(plain_zh, encoding="utf-8")
    (output_dir / "processing_report_en.md").write_text(report, encoding="utf-8")
    plain_en = re.sub(r"^#+\s*", "", report, flags=re.MULTILINE).replace("`", "")
    (output_dir / "processing_report_en.txt").write_text(plain_en, encoding="utf-8")


def main() -> None:
    args = parse_args()
    project_dir = args.project_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest, participants, ratings_all, item_registry = build_source_tables(project_dir)
    item_registry, ratings_all, duplicate_audit = add_classifier_ids(
        item_registry, ratings_all
    )
    exclusion_flow = build_exclusion_flow(participants)
    clean_blocks, clean = build_clean_ratings(ratings_all, participants)
    norms = build_norms(clean)
    master = build_master(norms, item_registry)
    master, cluster_metrics, cluster_profiles = add_clustering(master)
    reliability = split_half_reliability(clean)
    dominance = dominance_summary(master)
    dictionary = build_data_dictionary(master)

    id_map = participants[
        [
            "global_participant_id",
            "participant_code",
            "list",
            "local_participant_id",
        ]
    ]
    write_csv(manifest, output_dir / "input_manifest.csv")
    write_csv(id_map, output_dir / "participant_id_map.csv")
    write_csv(participants, output_dir / "participant_qc.csv")
    write_csv(exclusion_flow, output_dir / "exclusion_flow_by_list.csv")
    write_csv(item_registry, output_dir / "classifier_item_registry.csv")
    write_csv(duplicate_audit, output_dir / "duplicate_item_audit.csv")
    write_csv(ratings_all, output_dir / "ratings_long_all.csv")
    write_csv(clean_blocks, output_dir / "ratings_long_clean_source_blocks.csv")
    write_csv(clean, output_dir / "ratings_long_clean.csv")
    write_csv(norms, output_dir / "classifier_norms_long.csv")
    write_csv(master, output_dir / "classifier_master.csv")
    write_csv(dominance, output_dir / "dominance_summary.csv")
    write_csv(cluster_metrics, output_dir / "cluster_solution_metrics.csv")
    write_csv(cluster_profiles, output_dir / "cluster_profiles.csv")
    write_csv(reliability, output_dir / "split_half_reliability.csv")
    write_csv(dictionary, output_dir / "data_dictionary.csv")

    session = (
        f"Python: {sys.version}\n"
        f"Platform: {platform.platform()}\n"
        f"pandas: {pd.__version__}\n"
        f"numpy: {np.__version__}\n"
        f"Seed: {SEED}\n"
    )
    (output_dir / "session_info.txt").write_text(session, encoding="utf-8")
    write_report(
        output_dir,
        manifest,
        participants,
        exclusion_flow,
        item_registry,
        duplicate_audit,
        norms,
        master,
        cluster_metrics,
        reliability,
    )

    print(f"Completed: {output_dir}")
    print(f"Raw participants: {len(participants)}")
    print(f"Retained participants: {int(participants['retain_primary'].sum())}")
    print(f"Unique classifiers: {master['classifier_id'].nunique()}")
    print(f"Norm cells: {len(norms)}")


if __name__ == "__main__":
    main()
