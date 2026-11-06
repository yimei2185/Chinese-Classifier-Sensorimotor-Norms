#!/usr/bin/env python3
"""Build the audited classifier–noun analysis tables.

Inputs (read-only)
------------------
1. 260720/H-cls_D-noun_V1.xlsx: retained classifier–noun construction counts.
2. 260723/noun_norm.xlsx: independent noun sensorimotor norms.
3. Frozen classifier norm release: classifier_master.csv.

Outputs
-------
Machine-readable CSV tables, a variable dictionary, input manifest, metadata,
and a bilingual processing report in ``260827/outputs/classifier_noun_build_v1``.
The companion JavaScript builder creates the formatted Excel audit workbook.

Methodological decisions
------------------------
* Every source pair is retained in Pair_Master; exclusions are flags, not row
  deletions.
* A classifier is analysis-eligible only when it maps to exactly one frozen
  classifier ID. The explicit traditional mapping 個 -> 个 is also accepted.
* Homographic classifier forms with multiple frozen IDs are not duplicated or
  averaged. They remain ``ambiguous_homograph`` pending contextual review.
* Reduplicated forms and unmatched/measurement expressions are not silently
  normalized into a rated classifier.
* Nouns match the independent norms first by simplified form, then by a unique
  traditional form.
* Association statistics use margins of the supplied retained construction
  table. They are not full-corpus lexical association statistics.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


BUILD_ID = "classifier_noun_build_v1"
ROOT = Path(r"F:\S1\project\classifiers")
COLLOCATION_FILE = ROOT / "260720" / "H-cls_D-noun_V1.xlsx"
NOUN_NORM_FILE = ROOT / "260723" / "noun_norm.xlsx"
CLASSIFIER_RELEASE = (
    ROOT / "260827" / "releases" / "classifier_norm_v1_20260827"
)
CLASSIFIER_NORM_FILE = CLASSIFIER_RELEASE / "data" / "classifier_master.csv"
OUT_DIR = ROOT / "260827" / "outputs" / BUILD_ID

PAIR_SHEET = "clf_counts_output_head_dep"
NOUN_SHEET = "Sheet1"

DIMENSION_MAP = {
    "vision": "visual",
    "audition": "auditory",
    "gustation": "gustatory",
    "olfaction": "olfactory",
    "touch": "tactile",
    "interoception": "interoceptive",
    "leg_foot": "leg_foot",
    "hand_arm": "hand_arm",
    "head": "head",
    "mouth_throat": "mouth_throat",
    "torso": "torso",
}

# Only orthographic conversions that are unambiguous and explicitly reviewed.
CLASSIFIER_ORTHOGRAPHIC_MAP = {"個": "个"}

# Used only to describe unmatched tokens; membership never changes eligibility.
MEASUREMENT_OR_NON_TARGET_EXPRESSIONS = {
    "两", "亩", "元", "兆", "光伏", "克", "公分", "公斤", "公里", "公顷",
    "分钟", "千伏", "升", "吨", "圆", "型", "天", "寸", "平方公里",
    "平方米", "平米", "年", "年期", "批次", "折", "文", "斗", "斤", "日",
    "旬", "欧元", "毛", "毫克", "毫升", "毫安", "毫米", "盎司", "磅", "秒",
    "米", "纳米", "美元", "美金", "英寸", "英镑", "里", "马力",
}


def clean_token(value: object) -> str:
    if pd.isna(value):
        return ""
    return (
        str(value)
        .replace("\u200b", "")
        .replace("\u200c", "")
        .replace("\u200d", "")
        .replace("\ufeff", "")
        .strip()
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def csv_write(frame: pd.DataFrame, filename: str) -> None:
    frame.to_csv(OUT_DIR / filename, index=False, encoding="utf-8-sig")


def is_reduplicated(token: str) -> bool:
    return len(token) >= 2 and len(token) % 2 == 0 and token[: len(token) // 2] == token[len(token) // 2 :]


def safe_join(values: pd.Series) -> str:
    return "|".join(sorted({str(x) for x in values if pd.notna(x) and str(x) != ""}))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for path in (COLLOCATION_FILE, NOUN_NORM_FILE, CLASSIFIER_NORM_FILE):
        if not path.exists():
            raise FileNotFoundError(path)

    pairs_raw = pd.read_excel(COLLOCATION_FILE, sheet_name=PAIR_SHEET)
    required_pair = {"head", "dependent", "count_sum"}
    missing_pair = required_pair.difference(pairs_raw.columns)
    if missing_pair:
        raise ValueError(f"Missing pair columns: {sorted(missing_pair)}")

    pairs = pairs_raw.loc[:, ["head", "dependent", "count_sum"]].copy()
    pairs.insert(0, "source_pair_row", np.arange(2, len(pairs) + 2, dtype=int))
    pairs["noun_corpus"] = pairs["head"].map(clean_token)
    pairs["classifier_corpus"] = pairs["dependent"].map(clean_token)
    pairs["collocation_frequency"] = pd.to_numeric(pairs["count_sum"], errors="coerce")
    invalid = pairs[
        (pairs["noun_corpus"] == "")
        | (pairs["classifier_corpus"] == "")
        | pairs["collocation_frequency"].isna()
        | (pairs["collocation_frequency"] <= 0)
    ]
    if not invalid.empty:
        raise ValueError(f"Source has {len(invalid)} invalid pair rows; no silent deletion permitted.")
    pairs["collocation_frequency"] = pairs["collocation_frequency"].astype(int)
    if pairs.duplicated(["classifier_corpus", "noun_corpus"]).any():
        raise ValueError("Source contains duplicate cleaned classifier–noun pairs.")

    classifier_norm = pd.read_csv(CLASSIFIER_NORM_FILE, encoding="utf-8-sig")
    classifier_norm["classifier_hanzi"] = classifier_norm["classifier_hanzi"].map(clean_token)
    classifier_norm["classifier"] = classifier_norm["classifier"].map(clean_token)
    classifier_groups = {
        token: group.copy()
        for token, group in classifier_norm.groupby("classifier_hanzi", sort=False)
    }

    classifier_counts = (
        pairs.groupby("classifier_corpus", as_index=False)
        .agg(
            corpus_noun_type_count=("noun_corpus", "nunique"),
            corpus_token_frequency=("collocation_frequency", "sum"),
        )
        .sort_values(["corpus_token_frequency", "classifier_corpus"], ascending=[False, True])
    )

    classifier_crosswalk_rows: list[dict[str, object]] = []
    for row in classifier_counts.itertuples(index=False):
        token = row.classifier_corpus
        normalized = CLASSIFIER_ORTHOGRAPHIC_MAP.get(token, token)
        candidates = classifier_groups.get(normalized)
        if candidates is not None and len(candidates) == 1:
            status = "orthographic_variant" if normalized != token else "exact_unique"
            eligible = True
            matched_id = str(candidates.iloc[0]["classifier_id"])
            matched_label = str(candidates.iloc[0]["classifier"])
        elif candidates is not None and len(candidates) > 1:
            status = "ambiguous_homograph"
            eligible = False
            matched_id = ""
            matched_label = ""
        else:
            if is_reduplicated(token):
                status = "reduplicated_form"
            elif token in MEASUREMENT_OR_NON_TARGET_EXPRESSIONS:
                status = "measurement_or_non_target"
            else:
                status = "unmatched"
            eligible = False
            matched_id = ""
            matched_label = ""

        candidate_ids = "" if candidates is None else safe_join(candidates["classifier_id"])
        candidate_labels = "" if candidates is None else safe_join(candidates["classifier"])
        classifier_crosswalk_rows.append(
            {
                "classifier_corpus": token,
                "classifier_normalized": normalized,
                "classifier_match_status": status,
                "classifier_analysis_eligible": eligible,
                "classifier_id": matched_id,
                "classifier_norm_label": matched_label,
                "candidate_classifier_ids": candidate_ids,
                "candidate_classifier_labels": candidate_labels,
                "candidate_count": 0 if candidates is None else len(candidates),
                "corpus_noun_type_count": int(row.corpus_noun_type_count),
                "corpus_token_frequency": int(row.corpus_token_frequency),
                "review_note": (
                    "Contextual sense/pronunciation review required; do not duplicate or average."
                    if status == "ambiguous_homograph"
                    else ""
                ),
            }
        )
    classifier_crosswalk = pd.DataFrame(classifier_crosswalk_rows)

    noun_norm = pd.read_excel(NOUN_NORM_FILE, sheet_name=NOUN_SHEET)
    for required in ["simplified", "traditional", *DIMENSION_MAP.values()]:
        if required not in noun_norm.columns:
            raise ValueError(f"Missing noun norm column: {required}")
    noun_norm["simplified"] = noun_norm["simplified"].map(clean_token)
    noun_norm["traditional"] = noun_norm["traditional"].map(clean_token)
    if noun_norm["simplified"].duplicated().any():
        raise ValueError("noun_norm simplified forms are not unique.")

    simplified_lookup = noun_norm.set_index("simplified", drop=False)
    traditional_groups = {
        token: group.copy()
        for token, group in noun_norm.groupby("traditional", sort=False)
        if token != ""
    }
    noun_counts = (
        pairs.groupby("noun_corpus", as_index=False)
        .agg(
            corpus_classifier_type_count=("classifier_corpus", "nunique"),
            noun_frequency_in_retained_constructions=("collocation_frequency", "sum"),
        )
        .sort_values(
            ["noun_frequency_in_retained_constructions", "noun_corpus"],
            ascending=[False, True],
        )
    )

    noun_crosswalk_rows: list[dict[str, object]] = []
    for row in noun_counts.itertuples(index=False):
        token = row.noun_corpus
        if token in simplified_lookup.index:
            match = simplified_lookup.loc[token]
            status = "exact_simplified"
            eligible = True
            matched_simplified = str(match["simplified"])
        else:
            candidates = traditional_groups.get(token)
            if candidates is not None and len(candidates) == 1:
                status = "unique_traditional_variant"
                eligible = True
                matched_simplified = str(candidates.iloc[0]["simplified"])
            elif candidates is not None and len(candidates) > 1:
                status = "ambiguous_traditional"
                eligible = False
                matched_simplified = ""
            else:
                status = "unmatched"
                eligible = False
                matched_simplified = ""
        noun_crosswalk_rows.append(
            {
                "noun_corpus": token,
                "noun_match_status": status,
                "noun_analysis_eligible": eligible,
                "noun_norm_simplified": matched_simplified,
                "corpus_classifier_type_count": int(row.corpus_classifier_type_count),
                "noun_frequency_in_retained_constructions": int(
                    row.noun_frequency_in_retained_constructions
                ),
            }
        )
    noun_crosswalk = pd.DataFrame(noun_crosswalk_rows)

    pair_master = pairs.drop(columns=["head", "dependent", "count_sum"]).copy()
    pair_master = pair_master.merge(classifier_crosswalk, on="classifier_corpus", how="left", validate="many_to_one")
    pair_master = pair_master.merge(noun_crosswalk, on="noun_corpus", how="left", validate="many_to_one")

    total_tokens = int(pair_master["collocation_frequency"].sum())
    classifier_marginal = pair_master.groupby("classifier_corpus")["collocation_frequency"].transform("sum")
    noun_marginal = pair_master.groupby("noun_corpus")["collocation_frequency"].transform("sum")
    pair_master["classifier_frequency_in_retained_constructions"] = classifier_marginal
    pair_master["noun_frequency_in_retained_constructions"] = noun_marginal
    pair_master["total_retained_construction_tokens"] = total_tokens
    pair_master["p_classifier_given_noun_retained"] = pair_master["collocation_frequency"] / noun_marginal
    pair_master["p_noun_given_classifier_retained"] = pair_master["collocation_frequency"] / classifier_marginal
    expected = noun_marginal * classifier_marginal / total_tokens
    pair_master["expected_frequency_retained"] = expected
    pair_master["pmi_retained"] = np.log2(pair_master["collocation_frequency"] / expected)
    pair_master["logdice_retained"] = 14 + np.log2(
        2 * pair_master["collocation_frequency"] / (noun_marginal + classifier_marginal)
    )
    pair_master["t_score_retained"] = (
        pair_master["collocation_frequency"] - expected
    ) / np.sqrt(pair_master["collocation_frequency"])
    pair_master["observed_expected_ratio_retained"] = pair_master["collocation_frequency"] / expected
    for threshold in (10, 20, 50, 100):
        pair_master[f"frequency_ge_{threshold}"] = pair_master["collocation_frequency"] >= threshold

    pair_master["analysis_eligible"] = (
        pair_master["classifier_analysis_eligible"]
        & pair_master["noun_analysis_eligible"]
    )
    eligible_frequency = (
        pair_master["collocation_frequency"]
        .where(pair_master["analysis_eligible"], 0)
        .groupby(pair_master["noun_corpus"])
        .transform("sum")
    )
    eligible_type_count = (
        pair_master.assign(_eligible_id=pair_master["classifier_id"].where(pair_master["analysis_eligible"]))
        .groupby("noun_corpus")["_eligible_id"]
        .transform("nunique")
    )
    pair_master["eligible_classifier_token_frequency_for_noun"] = eligible_frequency
    pair_master["eligible_classifier_type_count_for_noun"] = eligible_type_count
    pair_master["classifier_norm_coverage_for_noun"] = eligible_frequency / noun_marginal
    pair_master["weight_frequency_among_eligible_classifiers"] = np.where(
        pair_master["analysis_eligible"] & (eligible_frequency > 0),
        pair_master["collocation_frequency"] / eligible_frequency,
        np.nan,
    )
    pair_master["weight_unweighted_among_eligible_classifiers"] = np.where(
        pair_master["analysis_eligible"] & (eligible_type_count > 0),
        1 / eligible_type_count,
        np.nan,
    )

    norm_selected_cols = [
        "classifier_id", "classifier", "classifier_hanzi",
        *[f"{d}_mean" for d in DIMENSION_MAP],
        "dominant_perceptual", "dominant_action", "dominant_overall_11",
        "perceptual_exclusivity", "action_exclusivity", "overall_exclusivity_11",
        "cluster_ward_raw_selected_k", "cluster_ward_raw_selected",
        "cluster_ward_z_selected_k", "cluster_ward_z_selected",
    ]
    norm_selected = classifier_norm[norm_selected_cols].copy()
    norm_selected = norm_selected.rename(
        columns={
            "classifier": "classifier_norm_full_label",
            "classifier_hanzi": "classifier_norm_hanzi",
            **{f"{d}_mean": f"classifier_norm_{d}" for d in DIMENSION_MAP},
        }
    )
    # classifier_id is empty for all ineligible rows, so the join cannot inflate rows.
    pair_master = pair_master.merge(norm_selected, on="classifier_id", how="left", validate="many_to_one")

    noun_selected = noun_norm[["simplified", "traditional", "pinyin", "english", "pos_cat", "respondents", "source", *DIMENSION_MAP.values()]].copy()
    noun_selected = noun_selected.rename(
        columns={
            "simplified": "noun_norm_simplified",
            "traditional": "noun_norm_traditional",
            "pinyin": "noun_norm_pinyin",
            "english": "noun_norm_english",
            "pos_cat": "noun_norm_pos_cat",
            "respondents": "noun_norm_respondents",
            "source": "noun_norm_source",
            **{noun_col: f"observed_noun_{classifier_dim}" for classifier_dim, noun_col in DIMENSION_MAP.items()},
        }
    )
    pair_master = pair_master.merge(noun_selected, on="noun_norm_simplified", how="left", validate="many_to_one")

    pair_master.insert(1, "pair_id", [f"P{i:05d}" for i in range(1, len(pair_master) + 1)])
    pair_master["classifier_noun"] = pair_master["classifier_corpus"] + "—" + pair_master["noun_corpus"]

    leading = [
        "pair_id", "source_pair_row", "classifier_corpus", "noun_corpus", "classifier_noun",
        "collocation_frequency", "analysis_eligible", "classifier_match_status",
        "noun_match_status", "classifier_id", "noun_norm_simplified",
    ]
    pair_master = pair_master[leading + [c for c in pair_master.columns if c not in leading]]
    eligible_pairs = pair_master[pair_master["analysis_eligible"]].copy()
    pair_index_columns = [
        "pair_id", "source_pair_row", "classifier_corpus", "noun_corpus",
        "collocation_frequency", "analysis_eligible", "classifier_match_status",
        "noun_match_status", "classifier_id", "classifier_norm_label",
        "candidate_classifier_ids", "candidate_classifier_labels",
        "noun_norm_simplified", "classifier_norm_coverage_for_noun",
    ]
    pair_match_index = pair_master[pair_index_columns].copy()
    eligible_index_columns = [
        "pair_id", "classifier_corpus", "noun_corpus", "collocation_frequency",
        "classifier_id", "classifier_norm_full_label", "noun_norm_simplified",
        "classifier_norm_coverage_for_noun",
        "weight_unweighted_among_eligible_classifiers",
        "weight_frequency_among_eligible_classifiers",
        "p_classifier_given_noun_retained", "p_noun_given_classifier_retained",
        "pmi_retained", "logdice_retained", "t_score_retained",
    ]
    eligible_pair_index = eligible_pairs[eligible_index_columns].copy()

    summary_rows = []
    def add(section: str, metric: str, value: object, note: str = "") -> None:
        summary_rows.append({"section": section, "metric": metric, "value": value, "note": note})

    add("source", "source_pair_rows", len(pair_master))
    add("source", "unique_corpus_nouns", pair_master["noun_corpus"].nunique())
    add("source", "unique_corpus_classifier_tokens", pair_master["classifier_corpus"].nunique())
    add("source", "total_retained_construction_tokens", total_tokens)
    add("source", "minimum_pair_frequency", int(pair_master["collocation_frequency"].min()), "Suggests an upstream frequency >= 10 extraction threshold.")
    add("matching", "eligible_pair_rows", len(eligible_pairs))
    add("matching", "eligible_unique_nouns", eligible_pairs["noun_corpus"].nunique())
    add("matching", "eligible_unique_classifier_tokens", eligible_pairs["classifier_corpus"].nunique())
    add("matching", "eligible_unique_classifier_ids", eligible_pairs["classifier_id"].nunique())
    eligible_tokens = int(eligible_pairs["collocation_frequency"].sum())
    add("matching", "eligible_construction_tokens", eligible_tokens)
    add("matching", "eligible_token_coverage", eligible_tokens / total_tokens)
    for status, group in classifier_crosswalk.groupby("classifier_match_status"):
        add("classifier_match_status", f"classifier_types__{status}", len(group))
        add("classifier_match_status", f"classifier_tokens__{status}", int(group["corpus_token_frequency"].sum()))
    for status, group in noun_crosswalk.groupby("noun_match_status"):
        add("noun_match_status", f"noun_types__{status}", len(group))
        add("noun_match_status", f"noun_tokens__{status}", int(group["noun_frequency_in_retained_constructions"].sum()))
    for threshold in (10, 20, 50, 100):
        subset = eligible_pairs[eligible_pairs["collocation_frequency"] >= threshold]
        add("eligible_threshold", f"pair_rows_ge_{threshold}", len(subset))
        add("eligible_threshold", f"unique_nouns_ge_{threshold}", subset["noun_corpus"].nunique())
        add("eligible_threshold", f"unique_classifier_ids_ge_{threshold}", subset["classifier_id"].nunique())
        add("eligible_threshold", f"tokens_ge_{threshold}", int(subset["collocation_frequency"].sum()))
    audit_summary = pd.DataFrame(summary_rows)

    dictionary_rows = [
        ("Pair_Master", "analysis_eligible", "TRUE only when both classifier and noun have accepted, unique norm matches."),
        ("Pair_Master", "collocation_frequency", "Observed frequency in the supplied retained classifier-construction table."),
        ("Pair_Master", "classifier_match_status", "Classifier crosswalk decision; ambiguous homographs are never averaged or duplicated."),
        ("Pair_Master", "noun_match_status", "Noun match to simplified form or unique traditional form in the independent norm table."),
        ("Pair_Master", "p_classifier_given_noun_retained", "Pair frequency divided by noun frequency within the retained construction table."),
        ("Pair_Master", "p_noun_given_classifier_retained", "Pair frequency divided by classifier frequency within the retained construction table."),
        ("Pair_Master", "expected_frequency_retained", "Expected pair frequency under independence using retained-table margins."),
        ("Pair_Master", "pmi_retained", "log2(observed/expected), calculated only within the retained construction table."),
        ("Pair_Master", "logdice_retained", "14 + log2(2*pair frequency/(noun marginal + classifier marginal)), retained table only."),
        ("Pair_Master", "t_score_retained", "(observed-expected)/sqrt(observed), retained table only."),
        ("Pair_Master", "classifier_norm_coverage_for_noun", "Share of the noun's retained construction tokens contributed by analysis-eligible classifiers."),
        ("Pair_Master", "weight_frequency_among_eligible_classifiers", "Pair frequency normalized across eligible classifiers of the noun; candidate Model 2/3 weight."),
        ("Pair_Master", "weight_unweighted_among_eligible_classifiers", "Equal weight over eligible classifier types of the noun; candidate Model 1 weight."),
        ("Pair_Master", "classifier_norm_*", "Frozen classifier norm value from classifier_norm_v1_20260827."),
        ("Pair_Master", "observed_noun_*", "Independent observed noun norm value from noun_norm.xlsx."),
        ("Classifier_Xwalk", "candidate_classifier_ids", "All frozen IDs sharing the normalized written form; multiple IDs require review."),
        ("Classifier_Xwalk", "classifier_analysis_eligible", "TRUE only for a unique frozen-ID match or explicit unique orthographic mapping."),
        ("Noun_Xwalk", "noun_analysis_eligible", "TRUE for exact simplified or unique traditional-form noun match."),
    ]
    data_dictionary = pd.DataFrame(dictionary_rows, columns=["dataset", "variable", "definition"])

    classifier_dim_cols = [f"classifier_norm_{d}" for d in DIMENSION_MAP]
    noun_dim_cols = [f"observed_noun_{d}" for d in DIMENSION_MAP]
    p_classifier_sums = pair_master.groupby("noun_corpus")["p_classifier_given_noun_retained"].sum()
    p_noun_sums = pair_master.groupby("classifier_corpus")["p_noun_given_classifier_retained"].sum()
    eligible_weight_sums = (
        eligible_pairs.groupby("noun_corpus")["weight_frequency_among_eligible_classifiers"].sum()
    )
    qc_rows = [
        ("source_pair_row_count_preserved", len(pair_master) == len(pairs_raw), len(pair_master), len(pairs_raw)),
        ("clean_pair_keys_unique", not pair_master.duplicated(["classifier_corpus", "noun_corpus"]).any(), int(pair_master.duplicated(["classifier_corpus", "noun_corpus"]).sum()), 0),
        ("total_frequency_reconciles", int(pair_master["collocation_frequency"].sum()) == total_tokens, int(pair_master["collocation_frequency"].sum()), total_tokens),
        ("classifier_crosswalk_has_one_row_per_token", len(classifier_crosswalk) == pair_master["classifier_corpus"].nunique(), len(classifier_crosswalk), pair_master["classifier_corpus"].nunique()),
        ("noun_crosswalk_has_one_row_per_token", len(noun_crosswalk) == pair_master["noun_corpus"].nunique(), len(noun_crosswalk), pair_master["noun_corpus"].nunique()),
        ("p_classifier_given_noun_sums_to_one", bool(np.allclose(p_classifier_sums, 1.0, atol=1e-12)), float(np.max(np.abs(p_classifier_sums - 1))), 0),
        ("p_noun_given_classifier_sums_to_one", bool(np.allclose(p_noun_sums, 1.0, atol=1e-12)), float(np.max(np.abs(p_noun_sums - 1))), 0),
        ("eligible_frequency_weights_sum_to_one", bool(np.allclose(eligible_weight_sums, 1.0, atol=1e-12)), float(np.max(np.abs(eligible_weight_sums - 1))), 0),
        ("eligible_classifier_norm_dimensions_complete", not eligible_pairs[classifier_dim_cols].isna().any().any(), int(eligible_pairs[classifier_dim_cols].isna().sum().sum()), 0),
        ("eligible_noun_norm_dimensions_complete", not eligible_pairs[noun_dim_cols].isna().any().any(), int(eligible_pairs[noun_dim_cols].isna().sum().sum()), 0),
        ("eligible_classifier_dimensions_within_0_5", bool(((eligible_pairs[classifier_dim_cols] >= 0) & (eligible_pairs[classifier_dim_cols] <= 5)).all().all()), "0-5", "0-5"),
        ("eligible_noun_dimensions_within_0_5", bool(((eligible_pairs[noun_dim_cols] >= 0) & (eligible_pairs[noun_dim_cols] <= 5)).all().all()), "0-5", "0-5"),
        ("ambiguous_homographs_have_no_assigned_id", bool(classifier_crosswalk.loc[classifier_crosswalk["classifier_match_status"] == "ambiguous_homograph", "classifier_id"].fillna("").eq("").all()), int(classifier_crosswalk.loc[classifier_crosswalk["classifier_match_status"] == "ambiguous_homograph", "classifier_id"].fillna("").ne("").sum()), 0),
    ]
    qc_checks = pd.DataFrame(qc_rows, columns=["check", "passed", "observed", "expected"])
    if not qc_checks["passed"].all():
        failed = qc_checks.loc[~qc_checks["passed"], "check"].tolist()
        raise AssertionError(f"QC checks failed: {failed}")

    manifest = pd.DataFrame(
        [
            {"role": "collocation_source", "path": str(COLLOCATION_FILE), "bytes": COLLOCATION_FILE.stat().st_size, "sha256": sha256(COLLOCATION_FILE)},
            {"role": "noun_norm_source", "path": str(NOUN_NORM_FILE), "bytes": NOUN_NORM_FILE.stat().st_size, "sha256": sha256(NOUN_NORM_FILE)},
            {"role": "frozen_classifier_norm", "path": str(CLASSIFIER_NORM_FILE), "bytes": CLASSIFIER_NORM_FILE.stat().st_size, "sha256": sha256(CLASSIFIER_NORM_FILE)},
        ]
    )

    csv_write(pair_master, "classifier_noun_master.csv")
    csv_write(eligible_pairs, "analysis_eligible_pairs.csv")
    csv_write(pair_match_index, "classifier_noun_match_index.csv")
    csv_write(eligible_pair_index, "analysis_eligible_pair_index.csv")
    csv_write(classifier_crosswalk, "classifier_crosswalk.csv")
    csv_write(noun_crosswalk, "noun_crosswalk.csv")
    csv_write(audit_summary, "match_audit_summary.csv")
    csv_write(data_dictionary, "data_dictionary.csv")
    csv_write(manifest, "input_manifest.csv")
    csv_write(qc_checks, "qc_checks.csv")

    metadata = {
        "build_id": BUILD_ID,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "dimensions": DIMENSION_MAP,
        "output_rows": {
            "classifier_noun_master.csv": len(pair_master),
            "analysis_eligible_pairs.csv": len(eligible_pairs),
            "classifier_noun_match_index.csv": len(pair_match_index),
            "analysis_eligible_pair_index.csv": len(eligible_pair_index),
            "classifier_crosswalk.csv": len(classifier_crosswalk),
            "noun_crosswalk.csv": len(noun_crosswalk),
            "match_audit_summary.csv": len(audit_summary),
            "data_dictionary.csv": len(data_dictionary),
            "input_manifest.csv": len(manifest),
            "qc_checks.csv": len(qc_checks),
        },
        "output_columns": {
            "classifier_noun_master.csv": len(pair_master.columns),
            "analysis_eligible_pairs.csv": len(eligible_pairs.columns),
            "classifier_noun_match_index.csv": len(pair_match_index.columns),
            "analysis_eligible_pair_index.csv": len(eligible_pair_index.columns),
            "classifier_crosswalk.csv": len(classifier_crosswalk.columns),
            "noun_crosswalk.csv": len(noun_crosswalk.columns),
            "match_audit_summary.csv": len(audit_summary.columns),
            "data_dictionary.csv": len(data_dictionary.columns),
            "input_manifest.csv": len(manifest.columns),
            "qc_checks.csv": len(qc_checks.columns),
        },
    }
    (OUT_DIR / "build_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    ambiguous = classifier_crosswalk[classifier_crosswalk["classifier_match_status"] == "ambiguous_homograph"]
    report = f"""# 量词—名词分析母表处理报告（{BUILD_ID}）

## 输入

- 语料搭配：`{COLLOCATION_FILE}`，工作表 `{PAIR_SHEET}`。
- 冻结量词 norm：`{CLASSIFIER_NORM_FILE}`。
- 独立名词 norm：`{NOUN_NORM_FILE}`，工作表 `{NOUN_SHEET}`。
- 输入文件 SHA-256 见 `input_manifest.csv`。

## 数据保留原则

源搭配表的 {len(pair_master):,} 行全部保留在 `classifier_noun_master.csv`，没有因匹配失败而删除。
源表共有 {total_tokens:,} 个搭配实例，最低单条频次为 {int(pair_master['collocation_frequency'].min())}；这表明频次下限可能在更早的语料提取阶段实施，需要结合原始语料提取说明确认。

## 匹配规则

1. 量词首先按规范化后的汉字形式匹配冻结量词 ID。
2. 唯一精确匹配可进入分析；经明确审核的 `個 -> 个` 也可进入分析。
3. 同一字形对应多个冻结 ID 时，标为 `ambiguous_homograph`，不复制语料边、不平均 norm。
4. 重叠形式、度量/非目标表达以及其他未匹配项只标记，不自动归并。
5. 名词先匹配简体形式；未匹配时再尝试唯一的繁体形式。

## 结果

- 可分析搭配：{len(eligible_pairs):,} 行。
- 可分析名词：{eligible_pairs['noun_corpus'].nunique():,} 个。
- 可分析语料量词形式：{eligible_pairs['classifier_corpus'].nunique():,} 个。
- 可分析冻结量词 ID：{eligible_pairs['classifier_id'].nunique():,} 个。
- 可分析搭配实例：{eligible_tokens:,}，占源搭配实例的 {eligible_tokens / total_tokens:.2%}。
- 待语境消歧的同形量词形式：{', '.join(ambiguous['classifier_corpus'].tolist()) if len(ambiguous) else '无'}。

## 关联强度的解释

条件概率、期望频次、PMI、logDice、t-score 和 observed/expected ratio 均使用本文件中“已保留量词结构”的边际频次。它们不能解释为完整语料库词频基础上的传统关联指标，变量名统一带有 `_retained`。

## 下一阶段使用方式

`analysis_eligible_pairs.csv` 是后续名词预测的输入。第一轮预测应分别保存：等权量词均值、搭配频次权重、`P(classifier|noun)` 权重，以及关联强度权重。预测模型与 `observed_noun_*` 独立名词评分比较时，需同时报告每个名词的 `classifier_norm_coverage_for_noun`。
"""
    (OUT_DIR / "processing_report.md").write_text(report, encoding="utf-8")
    (OUT_DIR / "processing_report.txt").write_text(report.replace("`", ""), encoding="utf-8")

    print(json.dumps({
        "output_dir": str(OUT_DIR),
        "source_pairs": len(pair_master),
        "source_tokens": total_tokens,
        "eligible_pairs": len(eligible_pairs),
        "eligible_nouns": int(eligible_pairs["noun_corpus"].nunique()),
        "eligible_classifier_tokens": int(eligible_pairs["classifier_corpus"].nunique()),
        "eligible_classifier_ids": int(eligible_pairs["classifier_id"].nunique()),
        "eligible_tokens": eligible_tokens,
        "eligible_token_coverage": eligible_tokens / total_tokens,
        "ambiguous_homographs": ambiguous["classifier_corpus"].tolist(),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
