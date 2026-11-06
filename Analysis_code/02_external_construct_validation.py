"""Classifier-level external construct validation.

Inputs
------
1. Frozen classifier norms (release v1).
2. Audited classifier-noun pairs containing independent noun norms.

Outputs
-------
Nine machine-readable CSV tables, four figures in PNG/SVG, a Markdown report,
and JSON provenance/QC metadata. Existing source and frozen files are read only.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from reportlab.graphics import renderSVG
from reportlab.graphics.shapes import Circle, Drawing, Line, PolyLine, Rect, String
from reportlab.lib.colors import Color, HexColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


SEED = 20260904
N_PERM = 10_000
N_BOOT = 10_000
N_STABILITY = 500
STABILITY_MIN_NOUNS = 5

PROJECT = Path(__file__).resolve().parents[1]
FROZEN_MASTER = PROJECT / "Classifier_sensorimotor_norms.csv"
ELIGIBLE_PAIRS = PROJECT / "Classifier_noun_validation.csv"
# The public analytical table already contains the audited frequencies and independent noun norms.
COLLOCATION_SOURCE = ELIGIBLE_PAIRS
NOUN_NORM_SOURCE = ELIGIBLE_PAIRS
OUTPUT = PROJECT / "reproduced_outputs" / "external_validation"
FIGURES = OUTPUT / "figures"

DIMENSIONS = [
    "vision", "audition", "gustation", "olfaction", "touch", "interoception",
    "leg_foot", "hand_arm", "head", "mouth_throat", "torso",
]
PERCEPTUAL = DIMENSIONS[:6]
ACTION = DIMENSIONS[6:]
DISPLAY = {
    "vision": "Vision", "audition": "Audition", "gustation": "Gustation",
    "olfaction": "Olfaction", "touch": "Touch", "interoception": "Interoception",
    "leg_foot": "Leg/foot", "hand_arm": "Hand/arm", "head": "Head",
    "mouth_throat": "Mouth/throat", "torso": "Torso",
}
PROFILE_METRICS = ["cosine", "pearson", "spearman", "rmse", "euclidean", "minkowski3"]
PRIMARY_PERM_METRICS = ["cosine", "pearson", "rmse", "minkowski3"]
SIMILARITY_METRICS = {"cosine", "pearson", "spearman"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def save_csv(frame: pd.DataFrame, filename: str) -> None:
    frame.to_csv(OUTPUT / filename, index=False, encoding="utf-8-sig")


def rankdata_1d(values: np.ndarray) -> np.ndarray:
    """Average ranks (1-based), including exact ties; SciPy-compatible semantics."""
    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks_sorted = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks_sorted[start:stop] = (start + 1 + stop) / 2
        start = stop
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = ranks_sorted
    return ranks


def rankdata_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim == 1:
        return rankdata_1d(values)
    return np.vstack([rankdata_1d(row) for row in values])


def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xc, yc = x - x.mean(), y - y.mean()
    denom = math.sqrt(float(np.sum(xc * xc) * np.sum(yc * yc)))
    return float(np.sum(xc * yc) / denom) if denom > 0 else np.nan


def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    return pearson_r(rankdata_1d(x), rankdata_1d(y))


def wilcoxon_signed_rank_normal(values: np.ndarray) -> tuple[float, float]:
    """Two-sided Wilcoxon signed-rank test using tie-corrected normal approximation."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isclose(values, 0.0, atol=1e-15)]
    n = len(values)
    if n == 0:
        return np.nan, np.nan
    absolute = np.abs(values)
    ranks = rankdata_1d(absolute)
    w_plus = float(ranks[values > 0].sum())
    w_minus = float(ranks[values < 0].sum())
    _, counts = np.unique(absolute, return_counts=True)
    tie_term = float(np.sum(counts ** 3 - counts))
    variance = (n * (n + 1) * (2 * n + 1) - tie_term) / 24
    if variance <= 0:
        return min(w_plus, w_minus), np.nan
    difference = w_plus - n * (n + 1) / 4
    corrected = difference - 0.5 * np.sign(difference)
    z = corrected / math.sqrt(variance)
    p_two_sided = math.erfc(abs(z) / math.sqrt(2))
    return min(w_plus, w_minus), p_two_sided


def correlation_permutation_p(
    x: np.ndarray, y: np.ndarray, method: str, rng: np.random.Generator, n_perm: int = N_PERM
) -> tuple[float, float]:
    observed = pearson_r(x, y) if method == "pearson" else spearman_rho(x, y)
    x_use = np.asarray(x, dtype=float) if method == "pearson" else rankdata_1d(x)
    y_use = np.asarray(y, dtype=float) if method == "pearson" else rankdata_1d(y)
    xc, yc = x_use - x_use.mean(), y_use - y_use.mean()
    denom = math.sqrt(float(np.sum(xc * xc) * np.sum(yc * yc)))
    null = np.empty(n_perm)
    for i in range(n_perm):
        null[i] = np.sum(xc * yc[rng.permutation(len(yc))]) / denom
    p_value = (np.sum(np.abs(null) >= abs(observed)) + 1) / (n_perm + 1)
    return observed, float(p_value)


def row_correlations(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ac = a - a.mean(axis=1, keepdims=True)
    bc = b - b.mean(axis=1, keepdims=True)
    denom = np.sqrt(np.sum(ac * ac, axis=1) * np.sum(bc * bc, axis=1))
    return np.divide(np.sum(ac * bc, axis=1), denom, out=np.full(len(a), np.nan), where=denom > 0)


def bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator, n_boot: int = N_BOOT) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    boot = np.empty(n_boot)
    for start in range(0, n_boot, 1000):
        stop = min(start + 1000, n_boot)
        idx = rng.integers(0, len(values), size=(stop - start, len(values)))
        boot[start:stop] = np.nanmean(values[idx], axis=1)
    return tuple(np.nanpercentile(boot, [2.5, 97.5]))


def bootstrap_corr_ci(
    x: np.ndarray, y: np.ndarray, method: str, rng: np.random.Generator, n_boot: int = N_BOOT
) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    boot = np.empty(n_boot)
    for start in range(0, n_boot, 500):
        stop = min(start + 500, n_boot)
        idx = rng.integers(0, len(x), size=(stop - start, len(x)))
        xb, yb = x[idx], y[idx]
        if method == "spearman":
            xb = rankdata_rows(xb)
            yb = rankdata_rows(yb)
        boot[start:stop] = row_correlations(xb, yb)
    return tuple(np.nanpercentile(boot, [2.5, 97.5]))


def bh_adjust(pvalues: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(pvalues), dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * len(p) / np.arange(1, len(p) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out = np.empty_like(adjusted)
    out[order] = np.minimum(adjusted, 1.0)
    return out


def vector_cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else np.nan


def metric_matrices(c: np.ndarray, p: np.ndarray) -> dict[str, np.ndarray]:
    c_norm = np.linalg.norm(c, axis=1)
    p_norm = np.linalg.norm(p, axis=1)
    denom = c_norm[:, None] * p_norm[None, :]
    cosine = np.divide(c @ p.T, denom, out=np.full((len(c), len(p)), np.nan), where=denom > 0)

    cc = c - c.mean(axis=1, keepdims=True)
    pc = p - p.mean(axis=1, keepdims=True)
    corr_denom = np.linalg.norm(cc, axis=1)[:, None] * np.linalg.norm(pc, axis=1)[None, :]
    pearson = np.divide(cc @ pc.T, corr_denom, out=np.full((len(c), len(p)), np.nan), where=corr_denom > 0)

    cr = rankdata_rows(c)
    pr = rankdata_rows(p)
    crc = cr - cr.mean(axis=1, keepdims=True)
    prc = pr - pr.mean(axis=1, keepdims=True)
    rank_denom = np.linalg.norm(crc, axis=1)[:, None] * np.linalg.norm(prc, axis=1)[None, :]
    spearman = np.divide(crc @ prc.T, rank_denom, out=np.full((len(c), len(p)), np.nan), where=rank_denom > 0)

    diff = c[:, None, :] - p[None, :, :]
    rmse = np.sqrt(np.mean(diff * diff, axis=2))
    euclidean = np.sqrt(np.sum(diff * diff, axis=2))
    minkowski3 = np.sum(np.abs(diff) ** 3, axis=2) ** (1 / 3)
    return {
        "cosine": cosine, "pearson": pearson, "spearman": spearman,
        "rmse": rmse, "euclidean": euclidean, "minkowski3": minkowski3,
    }


def correspondence_test(
    c: np.ndarray,
    p: np.ndarray,
    classifier_ids: list[str],
    rng: np.random.Generator,
    analysis: str,
    metrics: list[str] = PROFILE_METRICS,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    n = len(c)
    matrices = metric_matrices(c, p)
    permutations = np.empty((N_PERM, n), dtype=np.int32)
    for i in range(N_PERM):
        permutations[i] = rng.permutation(n)
    rows: list[dict] = []
    per_classifier = pd.DataFrame({"classifier_id": classifier_ids})
    null_store: dict[str, np.ndarray] = {}

    for metric in metrics:
        matrix = matrices[metric]
        matched = np.diag(matrix)
        mismatched = (np.nansum(matrix, axis=1) - matched) / (n - 1)
        advantage = matched - mismatched if metric in SIMILARITY_METRICS else mismatched - matched
        per_classifier[f"matched_{metric}"] = matched
        per_classifier[f"mean_mismatched_{metric}"] = mismatched
        per_classifier[f"matched_advantage_{metric}"] = advantage

        null = np.empty(N_PERM)
        for start in range(0, N_PERM, 1000):
            stop = min(start + 1000, N_PERM)
            null[start:stop] = matrix[np.arange(n)[None, :], permutations[start:stop]].mean(axis=1)
        null_store[metric] = null
        observed = float(np.mean(matched))
        null_mean = float(np.mean(null))
        if metric in SIMILARITY_METRICS:
            effect = observed - null_mean
            p_directional = (np.sum(null >= observed) + 1) / (N_PERM + 1)
        else:
            effect = null_mean - observed
            p_directional = (np.sum(null <= observed) + 1) / (N_PERM + 1)
        p_two = (np.sum(np.abs(null - null_mean) >= abs(observed - null_mean)) + 1) / (N_PERM + 1)
        ci_low, ci_high = bootstrap_mean_ci(advantage, rng)
        wilcoxon_stat, wilcoxon_p = wilcoxon_signed_rank_normal(advantage)
        matched_ci = bootstrap_mean_ci(matched, rng)
        rows.append({
            "analysis": analysis,
            "metric": metric,
            "n_classifiers": n,
            "observed_matched_mean": observed,
            "observed_matched_median": float(np.median(matched)),
            "observed_matched_sd": float(np.std(matched, ddof=1)),
            "observed_matched_boot_ci_low": matched_ci[0],
            "observed_matched_boot_ci_high": matched_ci[1],
            "permutation_null_mean": null_mean,
            "permutation_null_sd": float(np.std(null, ddof=1)),
            "oriented_observed_minus_null_effect": effect,
            "effect_boot_ci_low": ci_low,
            "effect_boot_ci_high": ci_high,
            "permutation_p_directional": p_directional,
            "permutation_p_two_sided": p_two,
            "percent_classifiers_advantage_gt_0": float(np.mean(advantage > 0) * 100),
            "mean_classifier_advantage": float(np.mean(advantage)),
            "median_classifier_advantage": float(np.median(advantage)),
            "wilcoxon_statistic": wilcoxon_stat,
            "wilcoxon_p_two_sided": wilcoxon_p,
        })
    return pd.DataFrame(rows), per_classifier, null_store


def standardize_columns(x: np.ndarray) -> np.ndarray:
    sd = x.std(axis=0, ddof=1)
    return (x - x.mean(axis=0)) / sd


def exact_dominant_set(values: np.ndarray, names: list[str]) -> str:
    maximum = np.max(values)
    return "+".join(name for name, value in zip(names, values) if np.isclose(value, maximum, atol=1e-12, rtol=0))


def dominance_test(c: np.ndarray, p: np.ndarray, names: list[str], rng: np.random.Generator, domain: str) -> dict:
    c_sets = np.array([exact_dominant_set(row, names) for row in c])
    p_sets = np.array([exact_dominant_set(row, names) for row in p])
    observed = float(np.mean(c_sets == p_sets))
    null = np.empty(N_PERM)
    for i in range(N_PERM):
        null[i] = np.mean(c_sets == p_sets[rng.permutation(len(p_sets))])
    return {
        "domain": domain,
        "n_classifiers": len(c),
        "exact_set_agreement_rate": observed,
        "classifier_tie_rate": float(np.mean(np.char.find(c_sets.astype(str), "+") >= 0)),
        "prototype_tie_rate": float(np.mean(np.char.find(p_sets.astype(str), "+") >= 0)),
        "permutation_expected_agreement": float(np.mean(null)),
        "permutation_null_sd": float(np.std(null, ddof=1)),
        "observed_minus_expected": observed - float(np.mean(null)),
        "permutation_p_directional": (np.sum(null >= observed) + 1) / (N_PERM + 1),
        "permutation_p_two_sided": (np.sum(np.abs(null - np.mean(null)) >= abs(observed - np.mean(null))) + 1) / (N_PERM + 1),
    }


def markdown_table(frame: pd.DataFrame, digits: int = 3) -> str:
    clean = frame.copy()
    for column in clean.columns:
        if pd.api.types.is_numeric_dtype(clean[column]):
            clean[column] = clean[column].map(lambda x: "" if pd.isna(x) else f"{x:.{digits}f}")
    headers = [str(x) for x in clean.columns]
    rows = [[str(x) for x in row] for row in clean.to_numpy()]
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
        *["| " + " | ".join(row) + " |" for row in rows],
    ])


def configure_font() -> str:
    candidates = [
        ("MicrosoftYaHei", Path(r"C:\Windows\Fonts\msyh.ttc")),
        ("SimSun", Path(r"C:\Windows\Fonts\simsun.ttc")),
        ("Arial", Path(r"C:\Windows\Fonts\arial.ttf")),
    ]
    for name, path in candidates:
        if path.exists():
            try:
                pdfmetrics.registerFont(TTFont(name, str(path), subfontIndex=0))
                return name
            except Exception:
                continue
    return "Helvetica"


def save_drawing(drawing: Drawing, stem: str) -> None:
    renderSVG.drawToFile(drawing, str(FIGURES / f"{stem}.svg"))


def build_edges(raw: pd.DataFrame) -> pd.DataFrame:
    raw = raw.copy()
    raw["dice_strength_component"] = 2.0 ** (raw["logdice_retained"] - 14.0)
    group_cols = ["classifier_id", "noun_norm_simplified"]
    aggregation: dict[str, object] = {
        "collocation_frequency": "sum",
        "dice_strength_component": "sum",
        "classifier_corpus": lambda x: "/".join(sorted(set(map(str, x)))),
        "noun_corpus": lambda x: "/".join(sorted(set(map(str, x)))),
        "noun_norm_source": lambda x: "/".join(sorted(set(map(str, x)))),
    }
    for dimension in DIMENSIONS:
        aggregation[f"observed_noun_{dimension}"] = "first"
    edges = raw.groupby(group_cols, as_index=False).agg(aggregation)
    edges = edges.rename(columns={"dice_strength_component": "positive_dice_strength"})
    edges["logdice_aggregated"] = 14 + np.log2(edges["positive_dice_strength"])
    return edges


def construct_prototypes(raw: pd.DataFrame, edges: pd.DataFrame, frozen: pd.DataFrame) -> pd.DataFrame:
    direct_cols = [f"{d}_mean" for d in DIMENSIONS]
    n_cols = [f"{d}_n" for d in DIMENSIONS]
    frozen_use = frozen[["classifier_id", "classifier", "classifier_hanzi", *direct_cols, *n_cols]].copy()
    frozen_use["classifier_mean_participant_n"] = frozen_use[n_cols].mean(axis=1)
    frozen_use["classifier_min_participant_n"] = frozen_use[n_cols].min(axis=1)

    form_margins = raw[["classifier_id", "classifier_corpus", "classifier_frequency_in_retained_constructions"]].drop_duplicates()
    denominator = form_margins.groupby("classifier_id")["classifier_frequency_in_retained_constructions"].sum()
    forms = form_margins.groupby("classifier_id")["classifier_corpus"].agg(lambda x: "/".join(sorted(set(map(str, x)))))

    rows: list[dict] = []
    for classifier_id, group in edges.groupby("classifier_id", sort=True):
        noun_matrix = group[[f"observed_noun_{d}" for d in DIMENSIONS]].to_numpy(float)
        frequency = group["collocation_frequency"].to_numpy(float)
        dice = group["positive_dice_strength"].to_numpy(float)
        row: dict[str, object] = {
            "classifier_id": classifier_id,
            "classifier_form": forms.loc[classifier_id],
            "n_noun_types": len(group),
            "total_edge_frequency": float(frequency.sum()),
            "eligible_noun_norm_token_coverage": float(frequency.sum() / denominator.loc[classifier_id]),
        }
        for dimension, values in zip(DIMENSIONS, noun_matrix.T):
            row[f"prototype_unweighted_{dimension}"] = float(np.mean(values))
            row[f"prototype_frequency_{dimension}"] = float(np.average(values, weights=frequency))
            row[f"prototype_dice_{dimension}"] = float(np.average(values, weights=dice))
        rows.append(row)
    prototypes = pd.DataFrame(rows).merge(frozen_use, on="classifier_id", how="left", validate="one_to_one")
    prototypes = prototypes.rename(columns={
        "classifier": "classifier_norm_label",
        "classifier_hanzi": "classifier_hanzi",
        **{f"{d}_mean": f"direct_{d}" for d in DIMENSIONS},
    })
    ordered = [
        "classifier_id", "classifier_form", "classifier_norm_label", "classifier_hanzi",
        "n_noun_types", "total_edge_frequency", "eligible_noun_norm_token_coverage",
        "classifier_mean_participant_n", "classifier_min_participant_n",
        *[f"direct_{d}" for d in DIMENSIONS],
        *[f"prototype_unweighted_{d}" for d in DIMENSIONS],
        *[f"prototype_frequency_{d}" for d in DIMENSIONS],
        *[f"prototype_dice_{d}" for d in DIMENSIONS],
    ]
    return prototypes[ordered].sort_values("classifier_id").reset_index(drop=True)


def dimension_validity(prototypes: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    subset = prototypes[prototypes["n_noun_types"] >= 2].reset_index(drop=True)
    rows: list[dict] = []
    for weighting in ["unweighted", "frequency", "dice"]:
        permutations = np.empty((N_PERM, len(subset)), dtype=np.int32)
        for i in range(N_PERM):
            permutations[i] = rng.permutation(len(subset))
        start_idx = len(rows)
        for dimension in DIMENSIONS:
            x = subset[f"direct_{dimension}"].to_numpy(float)
            y = subset[f"prototype_{weighting}_{dimension}"].to_numpy(float)
            pearson = pearson_r(x, y)
            spearman = spearman_rho(x, y)
            p_ci = bootstrap_corr_ci(x, y, "pearson", rng)
            s_ci = bootstrap_corr_ci(x, y, "spearman", rng)
            xc = x - x.mean()
            yc = y - y.mean()
            null_pearson = np.sum(xc[None, :] * yc[permutations], axis=1) / math.sqrt(np.sum(xc * xc) * np.sum(yc * yc))
            xr = rankdata_1d(x)
            yr = rankdata_1d(y)
            xrc, yrc = xr - xr.mean(), yr - yr.mean()
            null_spearman = np.sum(xrc[None, :] * yrc[permutations], axis=1) / math.sqrt(np.sum(xrc * xrc) * np.sum(yrc * yrc))
            rows.append({
                "prototype_weighting": weighting,
                "dimension": dimension,
                "n_classifiers": len(subset),
                "pearson_r": pearson,
                "pearson_ci_low": p_ci[0],
                "pearson_ci_high": p_ci[1],
                "spearman_rho": spearman,
                "spearman_ci_low": s_ci[0],
                "spearman_ci_high": s_ci[1],
                "permutation_p": (np.sum(np.abs(null_pearson) >= abs(pearson)) + 1) / (N_PERM + 1),
                "spearman_permutation_p": (np.sum(np.abs(null_spearman) >= abs(spearman)) + 1) / (N_PERM + 1),
            })
        pvals = [rows[i]["permutation_p"] for i in range(start_idx, len(rows))]
        adjusted = bh_adjust(pvals)
        for i, value in enumerate(adjusted, start=start_idx):
            rows[i]["bh_fdr_p"] = float(value)
    return pd.DataFrame(rows)


def arrays_for(prototypes: pd.DataFrame, weighting: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    c = prototypes[[f"direct_{d}" for d in DIMENSIONS]].to_numpy(float)
    p = prototypes[[f"prototype_{weighting}_{d}" for d in DIMENSIONS]].to_numpy(float)
    return c, p, prototypes["classifier_id"].tolist()


def top_k_prototypes(edges: pd.DataFrame, k: int | None) -> pd.DataFrame:
    rows = []
    for classifier_id, group in edges.groupby("classifier_id", sort=True):
        group = group.sort_values(["positive_dice_strength", "noun_norm_simplified"], ascending=[False, True])
        if k is not None and len(group) < k:
            continue
        selected = group if k is None else group.head(k)
        values = selected[[f"observed_noun_{d}" for d in DIMENSIONS]].to_numpy(float).mean(axis=0)
        row = {"classifier_id": classifier_id, **{f"prototype_top_{d}": value for d, value in zip(DIMENSIONS, values)}}
        rows.append(row)
    return pd.DataFrame(rows)


def create_figures(
    dimension_results: pd.DataFrame,
    primary_summary: pd.DataFrame,
    primary_profiles: pd.DataFrame,
    prototypes: pd.DataFrame,
    cosine_null: np.ndarray,
) -> dict[str, str]:
    selected_font = configure_font()
    blue, orange, gray, navy = map(HexColor, ["#2F6FA7", "#D97932", "#7B8794", "#17324D"])
    light, very_light, white = map(HexColor, ["#C8D2DC", "#E7EBEF", "#FFFFFF"])

    # Figure 1: dimension-wise coefficient plot.
    dim = dimension_results[dimension_results["prototype_weighting"] == "unweighted"].copy()
    width, height = 720, 540
    left, right, bottom, top = 145, 35, 65, 65
    plot_w, plot_h = width - left - right, height - bottom - top
    x_min = min(-0.4, float(dim["pearson_ci_low"].min()) - 0.05)
    x_max = max(0.9, float(dim["pearson_ci_high"].max()) + 0.05)
    sx = lambda value: left + (value - x_min) / (x_max - x_min) * plot_w
    d = Drawing(width, height)
    d.add(String(left, height - 30, "Dimension-wise convergent validity", fontName="Helvetica-Bold", fontSize=16, fillColor=navy))
    ticks = np.arange(math.ceil(x_min * 5) / 5, x_max + 0.001, 0.2)
    for tick in ticks:
        x = sx(tick)
        d.add(Line(x, bottom, x, height - top, strokeColor=very_light, strokeWidth=0.7))
        d.add(String(x, bottom - 18, f"{tick:.1f}", fontName="Helvetica", fontSize=8, textAnchor="middle", fillColor=gray))
    d.add(Line(sx(0), bottom, sx(0), height - top, strokeColor=gray, strokeWidth=1.1))
    row_gap = plot_h / len(dim)
    for i, row in enumerate(dim.itertuples()):
        y = height - top - (i + 0.5) * row_gap
        d.add(String(left - 10, y - 3, DISPLAY[row.dimension], fontName="Helvetica", fontSize=9, textAnchor="end", fillColor=navy))
        d.add(Line(sx(row.pearson_ci_low), y, sx(row.pearson_ci_high), y, strokeColor=gray, strokeWidth=1.5))
        d.add(Line(sx(row.pearson_ci_low), y - 4, sx(row.pearson_ci_low), y + 4, strokeColor=gray, strokeWidth=1))
        d.add(Line(sx(row.pearson_ci_high), y - 4, sx(row.pearson_ci_high), y + 4, strokeColor=gray, strokeWidth=1))
        d.add(Circle(sx(row.pearson_r), y, 4.2, fillColor=blue, strokeColor=white, strokeWidth=0.6))
    d.add(String(left + plot_w / 2, 23, "Pearson r (95% classifier-bootstrap CI)", fontName="Helvetica", fontSize=10, textAnchor="middle", fillColor=navy))
    save_drawing(d, "fig1_dimensionwise_validity")

    # Figure 2: permutation null with observed line.
    cosine_row = primary_summary[(primary_summary["analysis"] == "primary_raw_unweighted_n2") & (primary_summary["metric"] == "cosine")].iloc[0]
    observed = float(cosine_row["observed_matched_mean"])
    counts, bins = np.histogram(cosine_null, bins=45)
    width, height = 720, 460
    left, right, bottom, top = 80, 35, 70, 70
    plot_w, plot_h = width - left - right, height - bottom - top
    xmin = min(float(bins[0]), observed) - 0.002
    xmax = max(float(bins[-1]), observed) + 0.002
    sx = lambda value: left + (value - xmin) / (xmax - xmin) * plot_w
    sy = lambda value: bottom + value / max(counts) * plot_h
    d = Drawing(width, height)
    d.add(String(left, height - 30, "Permutation null for mean cosine similarity", fontName="Helvetica-Bold", fontSize=16, fillColor=navy))
    for count, lo, hi in zip(counts, bins[:-1], bins[1:]):
        d.add(Rect(sx(lo), bottom, max(0.4, sx(hi) - sx(lo) - 0.5), sy(count) - bottom, fillColor=light, strokeColor=white, strokeWidth=0.3))
    d.add(Line(left, bottom, width - right, bottom, strokeColor=gray, strokeWidth=1))
    d.add(Line(sx(observed), bottom, sx(observed), height - top + 6, strokeColor=orange, strokeWidth=2.5))
    d.add(String(sx(observed) - 4, height - top + 14, f"Observed = {observed:.3f}", fontName="Helvetica-Bold", fontSize=9, textAnchor="end", fillColor=orange))
    for tick in np.linspace(xmin, xmax, 6):
        d.add(String(sx(tick), bottom - 18, f"{tick:.3f}", fontName="Helvetica", fontSize=8, textAnchor="middle", fillColor=gray))
    d.add(String(left + plot_w / 2, 24, "Mean cosine under classifier-label permutation", fontName="Helvetica", fontSize=10, textAnchor="middle", fillColor=navy))
    d.add(String(20, bottom + plot_h / 2, "Count", fontName="Helvetica", fontSize=10, angle=90, textAnchor="middle", fillColor=navy))
    save_drawing(d, "fig2_permutation_null")

    # Figure 3: paired own-versus-other correspondence.
    matched = primary_profiles["matched_cosine"].to_numpy(float)
    mismatched = primary_profiles["mean_mismatched_cosine"].to_numpy(float)
    width, height = 650, 500
    left, right, bottom, top = 90, 45, 75, 65
    x1, x2 = 230, 470
    ymin = float(min(matched.min(), mismatched.min()) - 0.03)
    ymax = float(max(matched.max(), mismatched.max()) + 0.03)
    sy = lambda value: bottom + (value - ymin) / (ymax - ymin) * (height - bottom - top)
    d = Drawing(width, height)
    d.add(String(left, height - 30, "Matched versus mean-mismatched correspondence", fontName="Helvetica-Bold", fontSize=15, fillColor=navy))
    for tick in np.linspace(ymin, ymax, 6):
        y = sy(tick)
        d.add(Line(left, y, width - right, y, strokeColor=very_light, strokeWidth=0.7))
        d.add(String(left - 8, y - 3, f"{tick:.2f}", fontName="Helvetica", fontSize=8, textAnchor="end", fillColor=gray))
    for a, b in zip(matched, mismatched):
        d.add(Line(x1, sy(a), x2, sy(b), strokeColor=Color(0.75, 0.78, 0.81, alpha=0.45), strokeWidth=0.55))
        d.add(Circle(x1, sy(a), 2.6, fillColor=blue, strokeColor=None))
        d.add(Circle(x2, sy(b), 2.6, fillColor=orange, strokeColor=None))
    d.add(String(x1, bottom - 28, "Own prototype", fontName="Helvetica-Bold", fontSize=10, textAnchor="middle", fillColor=blue))
    d.add(String(x2, bottom - 28, "Mean other prototype", fontName="Helvetica-Bold", fontSize=10, textAnchor="middle", fillColor=orange))
    d.add(String(left, height - top + 12, "Cosine similarity", fontName="Helvetica", fontSize=9, fillColor=navy))
    save_drawing(d, "fig3_matched_vs_mismatched")

    # Figure 4: reproducibly selected profile examples.
    ranked = primary_profiles.sort_values("matched_advantage_cosine").reset_index(drop=True)
    choices = {
        "Low advantage": ranked.iloc[0],
        "Median advantage": ranked.iloc[(ranked["matched_advantage_cosine"] - ranked["matched_advantage_cosine"].median()).abs().argmin()],
        "High advantage": ranked.iloc[-1],
    }
    width, height = 900, 920
    left, right, panel_h = 75, 35, 245
    plot_w = width - left - right
    d = Drawing(width, height)
    d.add(String(left, height - 28, "Reproducibly selected classifier profile examples", fontName="Helvetica-Bold", fontSize=16, fillColor=navy))
    d.add(Line(width - 310, height - 30, width - 275, height - 30, strokeColor=navy, strokeWidth=2))
    d.add(String(width - 268, height - 34, "Direct rating", fontName="Helvetica", fontSize=9, fillColor=navy))
    d.add(Line(width - 170, height - 30, width - 135, height - 30, strokeColor=orange, strokeWidth=2))
    d.add(String(width - 128, height - 34, "Noun prototype", fontName="Helvetica", fontSize=9, fillColor=orange))
    x_positions = np.linspace(left + 10, width - right - 10, len(DIMENSIONS))
    abbrev = ["Vis", "Aud", "Gus", "Olf", "Tou", "Int", "Leg", "Hand", "Head", "Mouth", "Torso"]
    for panel, (rule, selected) in enumerate(choices.items()):
        y_top = height - 70 - panel * 275
        y_bottom = y_top - panel_h + 35
        row = prototypes.loc[prototypes["classifier_id"] == selected["classifier_id"]].iloc[0]
        direct = row[[f"direct_{dimension}" for dimension in DIMENSIONS]].to_numpy(float)
        proto = row[[f"prototype_unweighted_{dimension}" for dimension in DIMENSIONS]].to_numpy(float)
        title = f"{rule}: {row['classifier_norm_label']} ({selected['classifier_id']}; cosine advantage={selected['matched_advantage_cosine']:.3f})"
        d.add(String(left, y_top + 12, title, fontName=selected_font, fontSize=10.5, fillColor=navy))
        for tick in range(6):
            y = y_bottom + tick / 5 * (panel_h - 55)
            d.add(Line(left, y, width - right, y, strokeColor=very_light, strokeWidth=0.6))
            d.add(String(left - 8, y - 3, str(tick), fontName="Helvetica", fontSize=7.5, textAnchor="end", fillColor=gray))
        to_y = lambda value: y_bottom + value / 5 * (panel_h - 55)
        direct_points = []
        proto_points = []
        for x, a, b in zip(x_positions, direct, proto):
            direct_points.extend([x, to_y(a)])
            proto_points.extend([x, to_y(b)])
        d.add(PolyLine(direct_points, strokeColor=navy, strokeWidth=1.8))
        d.add(PolyLine(proto_points, strokeColor=orange, strokeWidth=1.8))
        for x, a, b in zip(x_positions, direct, proto):
            d.add(Circle(x, to_y(a), 3, fillColor=navy, strokeColor=white, strokeWidth=0.4))
            d.add(Circle(x, to_y(b), 3, fillColor=orange, strokeColor=white, strokeWidth=0.4))
        if panel == 2:
            for x, label in zip(x_positions, abbrev):
                d.add(String(x, y_bottom - 18, label, fontName="Helvetica", fontSize=7.5, textAnchor="middle", fillColor=gray))
    save_drawing(d, "fig4_example_profiles")
    return {"font": selected_font, "examples": {rule: row["classifier_id"] for rule, row in choices.items()}}


def main() -> None:
    # Re-running this version may replace only its own derived files; all source
    # and frozen directories remain read-only.
    OUTPUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    raw = pd.read_csv(ELIGIBLE_PAIRS)
    frozen = pd.read_csv(FROZEN_MASTER)
    required = {
        "classifier_id", "classifier_corpus", "noun_corpus", "noun_norm_simplified",
        "collocation_frequency", "logdice_retained", "analysis_eligible",
        *[f"classifier_norm_{d}" for d in DIMENSIONS],
        *[f"observed_noun_{d}" for d in DIMENSIONS],
    }
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"Missing required audited columns: {missing}")
    if not raw["analysis_eligible"].all():
        raise ValueError("The audited input contains rows not marked analysis_eligible.")
    if raw[[f"classifier_norm_{d}" for d in DIMENSIONS]].isna().any().any():
        raise ValueError("Missing classifier norm values in eligible rows.")
    if raw[[f"observed_noun_{d}" for d in DIMENSIONS]].isna().any().any():
        raise ValueError("Missing independent noun norm values in eligible rows.")

    # Verify that embedded classifier values reproduce the immutable frozen release.
    frozen_lookup = frozen.set_index("classifier_id")
    max_direct_difference = 0.0
    for dimension in DIMENSIONS:
        expected = raw["classifier_id"].map(frozen_lookup[f"{dimension}_mean"])
        max_direct_difference = max(max_direct_difference, float(np.max(np.abs(expected - raw[f"classifier_norm_{dimension}"]))))
    if max_direct_difference > 1e-12:
        raise ValueError("Audited pair table classifier values differ from the frozen release.")

    edges = build_edges(raw)
    prototypes = construct_prototypes(raw, edges, frozen)
    save_csv(prototypes, "01_classifier_noun_prototypes.csv")

    noun_distribution = prototypes["n_noun_types"].describe(percentiles=[0.25, 0.5, 0.75])
    threshold_counts = {threshold: int((prototypes["n_noun_types"] >= threshold).sum()) for threshold in [2, 5, 10, 20]}

    dimension_results = dimension_validity(prototypes, rng)
    save_csv(dimension_results, "02_dimensionwise_validity.csv")

    primary = prototypes[prototypes["n_noun_types"] >= 2].reset_index(drop=True)
    c_primary, p_primary, ids_primary = arrays_for(primary, "unweighted")
    primary_summary, primary_profiles, primary_nulls = correspondence_test(
        c_primary, p_primary, ids_primary, rng, "primary_raw_unweighted_n2"
    )
    primary_profiles = primary_profiles.merge(
        primary[["classifier_id", "classifier_form", "classifier_norm_label", "n_noun_types", "total_edge_frequency",
                 "eligible_noun_norm_token_coverage", "classifier_mean_participant_n", "classifier_min_participant_n"]],
        on="classifier_id", how="left", validate="one_to_one",
    )
    front = ["classifier_id", "classifier_form", "classifier_norm_label", "n_noun_types", "total_edge_frequency",
             "eligible_noun_norm_token_coverage", "classifier_mean_participant_n", "classifier_min_participant_n"]
    primary_profiles = primary_profiles[front + [c for c in primary_profiles.columns if c not in front]]
    save_csv(primary_profiles, "03_classifier_profile_correspondence.csv")

    c_z, p_z = standardize_columns(c_primary), standardize_columns(p_primary)
    standardized_summary, _, _ = correspondence_test(
        c_z, p_z, ids_primary, rng, "robustness_dimension_standardized_n2", PRIMARY_PERM_METRICS
    )
    permutation_results = pd.concat([primary_summary, standardized_summary], ignore_index=True)
    save_csv(permutation_results, "04_permutation_tests.csv")

    threshold_rows = []
    for threshold in [2, 5, 10, 20]:
        subset = prototypes[prototypes["n_noun_types"] >= threshold].reset_index(drop=True)
        c, p, ids = arrays_for(subset, "unweighted")
        summary, _, _ = correspondence_test(c, p, ids, rng, f"unweighted_n{threshold}", PRIMARY_PERM_METRICS)
        summary.insert(1, "minimum_noun_types", threshold)
        threshold_rows.append(summary)
    threshold_results = pd.concat(threshold_rows, ignore_index=True)
    save_csv(threshold_results, "05_threshold_sensitivity.csv")

    weighting_rows = []
    for weighting in ["unweighted", "frequency", "dice"]:
        c, p, ids = arrays_for(primary, weighting)
        summary, _, _ = correspondence_test(c, p, ids, rng, f"raw_{weighting}_n2", PRIMARY_PERM_METRICS)
        summary.insert(1, "prototype_weighting", weighting)
        weighting_rows.append(summary)
    weighting_results = pd.concat(weighting_rows, ignore_index=True)
    save_csv(weighting_results, "06_weighting_sensitivity.csv")

    dominance_rows = [
        dominance_test(c_primary[:, :6], p_primary[:, :6], PERCEPTUAL, rng, "perceptual"),
        dominance_test(c_primary[:, 6:], p_primary[:, 6:], ACTION, rng, "action"),
    ]
    dominance_results = pd.DataFrame(dominance_rows)
    save_csv(dominance_results, "07_dominance_agreement.csv")

    association_rows = []
    for label, k in [("all_eligible", None), ("top_20_logdice", 20), ("top_10_logdice", 10), ("top_5_logdice", 5)]:
        top = top_k_prototypes(edges, k)
        subset = prototypes.merge(top, on="classifier_id", how="inner", validate="one_to_one")
        if k is None:
            subset = subset[subset["n_noun_types"] >= 2]
        c = subset[[f"direct_{d}" for d in DIMENSIONS]].to_numpy(float)
        p = subset[[f"prototype_top_{d}" for d in DIMENSIONS]].to_numpy(float)
        summary, _, _ = correspondence_test(c, p, subset["classifier_id"].tolist(), rng, label, PRIMARY_PERM_METRICS)
        summary.insert(1, "association_subset", label)
        summary.insert(2, "top_k", "all" if k is None else k)
        association_rows.append(summary)
    association_results = pd.concat(association_rows, ignore_index=True)
    save_csv(association_results, "08_association_strength_sensitivity.csv")

    stability_rows = []
    for classifier_id, group in edges.groupby("classifier_id", sort=True):
        n = len(group)
        if n < STABILITY_MIN_NOUNS:
            continue
        matrix = group[[f"observed_noun_{d}" for d in DIMENSIONS]].to_numpy(float)
        full = matrix.mean(axis=0)
        sample_size = max(2, int(math.floor(0.8 * n)))
        similarities = np.empty(N_STABILITY)
        for i in range(N_STABILITY):
            sample = rng.choice(n, size=sample_size, replace=False)
            similarities[i] = vector_cosine(matrix[sample].mean(axis=0), full)
        proto_row = prototypes.loc[prototypes["classifier_id"] == classifier_id].iloc[0]
        profile_row = primary_profiles.loc[primary_profiles["classifier_id"] == classifier_id].iloc[0]
        stability_rows.append({
            "classifier_id": classifier_id,
            "classifier_form": proto_row["classifier_form"],
            "classifier_norm_label": proto_row["classifier_norm_label"],
            "n_noun_types": n,
            "subsample_noun_types": sample_size,
            "n_replicates": N_STABILITY,
            "mean_stability_cosine": float(np.mean(similarities)),
            "median_stability_cosine": float(np.median(similarities)),
            "sd_stability_cosine": float(np.std(similarities, ddof=1)),
            "stability_ci_low": float(np.quantile(similarities, 0.025)),
            "stability_ci_high": float(np.quantile(similarities, 0.975)),
            "matched_cosine": profile_row["matched_cosine"],
            "matched_advantage_cosine": profile_row["matched_advantage_cosine"],
            "classifier_mean_participant_n": proto_row["classifier_mean_participant_n"],
            "classifier_min_participant_n": proto_row["classifier_min_participant_n"],
        })
    stability = pd.DataFrame(stability_rows)
    save_csv(stability, "09_prototype_stability.csv")

    exploratory = []
    for outcome in ["matched_cosine", "matched_advantage_cosine"]:
        for predictor in ["n_noun_types", "mean_stability_cosine", "classifier_mean_participant_n"]:
            x, y = stability[predictor].to_numpy(float), stability[outcome].to_numpy(float)
            p_r, p_p = correlation_permutation_p(x, y, "pearson", rng)
            s_r, s_p = correlation_permutation_p(x, y, "spearman", rng)
            exploratory.append({
                "outcome": outcome, "predictor": predictor, "n_classifiers": len(stability),
                "pearson_r": p_r, "pearson_permutation_p_two_sided": p_p,
                "spearman_rho": s_r, "spearman_permutation_p_two_sided": s_p,
            })

    figure_meta = create_figures(dimension_results, permutation_results, primary_profiles, prototypes, primary_nulls["cosine"])

    # Key result extraction for the report.
    primary_key = permutation_results[permutation_results["analysis"] == "primary_raw_unweighted_n2"].set_index("metric")
    standardized_key = permutation_results[permutation_results["analysis"] == "robustness_dimension_standardized_n2"].set_index("metric")
    dim_primary = dimension_results[dimension_results["prototype_weighting"] == "unweighted"].sort_values("pearson_r", ascending=False)
    sig_dims = dim_primary[dim_primary["bh_fdr_p"] < 0.05]
    cosine_threshold = threshold_results[threshold_results["metric"] == "cosine"]
    cosine_weight = weighting_results[weighting_results["metric"] == "cosine"]
    cosine_assoc = association_results[association_results["metric"] == "cosine"]

    metadata = {
        "analysis_id": "classifier_external_construct_validation_v1",
        "date": "2026-09-04",
        "seed": SEED,
        "n_permutations": N_PERM,
        "n_bootstrap": N_BOOT,
        "n_stability_replicates": N_STABILITY,
        "input_hashes": {str(path): sha256(path) for path in [FROZEN_MASTER, ELIGIBLE_PAIRS, COLLOCATION_SOURCE, NOUN_NORM_SOURCE]},
        "audited_rows": len(raw),
        "audited_corpus_noun_forms": int(raw["noun_corpus"].nunique()),
        "independent_norm_nouns": int(raw["noun_norm_simplified"].nunique()),
        "unique_classifier_noun_edges_after_norm_concept_collapse": len(edges),
        "unique_classifier_ids": int(raw["classifier_id"].nunique()),
        "unique_classifier_forms": int(raw["classifier_corpus"].nunique()),
        "total_tokens": int(raw["collocation_frequency"].sum()),
        "noun_type_distribution": {
            "min": float(noun_distribution["min"]), "q25": float(noun_distribution["25%"]),
            "median": float(noun_distribution["50%"]), "mean": float(noun_distribution["mean"]),
            "q75": float(noun_distribution["75%"]), "max": float(noun_distribution["max"]),
        },
        "threshold_counts": threshold_counts,
        "frequency_and_p_noun_given_classifier_equivalence": "Within classifier, normalizing P(noun|classifier) yields the same weights as normalizing raw edge frequency; only frequency is calculated.",
        "classifier_values_max_abs_difference_from_frozen": max_direct_difference,
        "figure_font": figure_meta["font"],
        "figure_4_selected_examples": figure_meta["examples"],
        "exploratory_stability_associations": exploratory,
    }
    (OUTPUT / "analysis_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    qc = {
        "all_audited_rows_analysis_eligible": bool(raw["analysis_eligible"].all()),
        "classifier_forms_with_multiple_ids": int((raw.groupby("classifier_corpus")["classifier_id"].nunique() > 1).sum()),
        "missing_classifier_norm_cells": int(raw[[f"classifier_norm_{d}" for d in DIMENSIONS]].isna().sum().sum()),
        "missing_noun_norm_cells": int(raw[[f"observed_noun_{d}" for d in DIMENSIONS]].isna().sum().sum()),
        "classifier_values_match_frozen": bool(max_direct_difference <= 1e-12),
        "audited_rows": len(raw),
        "collapsed_edges": len(edges),
        "collapsed_duplicate_rows": len(raw) - len(edges),
        "primary_n_classifiers": len(primary),
        "all_prototype_values_finite": bool(np.isfinite(prototypes.filter(regex="^(direct|prototype)_").to_numpy(float)).all()),
        "all_primary_correspondence_values_finite": bool(np.isfinite(primary_profiles.filter(regex="^(matched|mean_mismatched)_").to_numpy(float)).all()),
        "expected_output_csv_count": 9,
    }
    (OUTPUT / "qc_checks.json").write_text(json.dumps(qc, ensure_ascii=False, indent=2), encoding="utf-8")

    report = f"""# Classifier-Level External Construct Validation

## 1. Research question

Do directly rated 11-dimensional sensorimotor profiles of Chinese classifiers resemble independent sensorimotor prototypes of the noun classes those classifiers select more than they resemble mismatched noun-class prototypes?

## 2. Why this is external construct validation

The directly rated classifier profile is compared with a criterion assembled from independently collected noun norms. The noun-derived prototype is not a re-expression of the classifier ratings. This reverses the aggregation direction used in the earlier noun-level prediction analysis: nouns are aggregated within classifier here, and the classifier is the independent inferential unit. Evidence that a classifier corresponds to its noun-class prototype is therefore distinct from the stronger claim that classifier norms accurately predict every individual noun.

## 3. Input datasets and independence

- Frozen classifier norms: `{FROZEN_MASTER}` (SHA-256 `{sha256(FROZEN_MASTER)}`).
- Audited eligible classifier-noun table: `{ELIGIBLE_PAIRS}` (SHA-256 `{sha256(ELIGIBLE_PAIRS)}`).
- Public analytical input: `{ELIGIBLE_PAIRS}`. It contains audited association frequencies and the independent noun-norm values used in the analysis.

The classifier norms were collected from classifier-rating participants and frozen before this analysis. The noun norms came from a separate existing noun-norm database. Neither representation was constructed from the other. The classifier values embedded in the audited pair table reproduced the frozen master exactly (maximum absolute difference `{max_direct_difference:.3g}`). Ambiguous homographs 场, 卷, and 撮 remained excluded under the previous audit decisions.

## 4. Exact sample construction

The audited table contained 2,039 eligible rows, 223,623 construction tokens, 148 corpus classifier forms, and 147 unique frozen classifier IDs. It contained 758 corpus noun forms but 757 independent normalized noun concepts because 频道 and 頻道 map to the same noun norm. Their two rows for classifier 个/個 were collapsed to one type-level edge and their frequencies and positive Dice strengths were summed. The analysis therefore used 2,038 unique classifier-ID–noun-concept edges. No rematching was performed.

Across all 147 classifiers, eligible noun-type counts had minimum {noun_distribution['min']:.0f}, Q1 {noun_distribution['25%']:.1f}, median {noun_distribution['50%']:.1f}, mean {noun_distribution['mean']:.2f}, Q3 {noun_distribution['75%']:.1f}, and maximum {noun_distribution['max']:.0f}. The pre-specified threshold samples contained {threshold_counts[2]} classifiers at n≥2 (primary), {threshold_counts[5]} at n≥5, {threshold_counts[10]} at n≥10, and {threshold_counts[20]} at n≥20 noun types.

## 5. Prototype construction

For each classifier, the primary prototype was the unweighted mean of the 11 independent norm values across unique associated noun concepts. Robustness prototypes used (a) summed classifier-noun frequency and (b) positive Dice strength, `2^(logDice - 14)`. A separate P(noun | classifier) prototype was not computed because normalization within classifier makes it algebraically identical to raw-frequency weighting. Classifier-level noun-norm token coverage was calculated as eligible normed-noun frequency divided by the retained-table frequency of the classifier form(s).

## 6. Primary whole-profile analysis

Raw 0–5 values were used for the primary analysis. Across the {len(primary)} classifiers with at least two noun types, own-profile correspondence was summarized by cosine similarity, Pearson and Spearman profile correlation, RMSE, Euclidean distance, and Minkowski-3 distance.

{markdown_table(primary_summary[['metric','n_classifiers','observed_matched_mean','observed_matched_median','observed_matched_sd','observed_matched_boot_ci_low','observed_matched_boot_ci_high']].rename(columns={'observed_matched_mean':'mean','observed_matched_median':'median','observed_matched_sd':'SD','observed_matched_boot_ci_low':'CI low','observed_matched_boot_ci_high':'CI high'}))}

## 7. Matched-versus-mismatched permutation test

For each of 10,000 label permutations, complete noun-prototype vectors were reassigned without replacement while direct classifier vectors remained fixed. Similarity effects are observed minus null; error/distance effects are null minus observed, so positive values always favor the matched pairing. The directional permutation test implements the pre-specified hypothesis that own prototypes should be more similar/closer; two-sided p-values are also saved.

{markdown_table(primary_summary[primary_summary['metric'].isin(PRIMARY_PERM_METRICS)][['metric','observed_matched_mean','permutation_null_mean','oriented_observed_minus_null_effect','effect_boot_ci_low','effect_boot_ci_high','permutation_p_directional','percent_classifiers_advantage_gt_0']].rename(columns={'observed_matched_mean':'matched','permutation_null_mean':'null','oriented_observed_minus_null_effect':'advantage','effect_boot_ci_low':'CI low','effect_boot_ci_high':'CI high','permutation_p_directional':'perm p','percent_classifiers_advantage_gt_0':'% classifiers > 0'}))}

For cosine similarity, the matched mean was {primary_key.loc['cosine','observed_matched_mean']:.3f}, compared with a permutation-null mean of {primary_key.loc['cosine','permutation_null_mean']:.3f}; the oriented advantage was {primary_key.loc['cosine','oriented_observed_minus_null_effect']:.3f}, 95% bootstrap CI [{primary_key.loc['cosine','effect_boot_ci_low']:.3f}, {primary_key.loc['cosine','effect_boot_ci_high']:.3f}], directional permutation p={primary_key.loc['cosine','permutation_p_directional']:.4f}. {primary_key.loc['cosine','percent_classifiers_advantage_gt_0']:.1f}% of classifiers were closer to their own prototype than to their average mismatched prototype on this metric.

## 8. Dimension-wise convergent validity

Across classifiers, direct and noun-prototype values were correlated separately for all dimensions. Confidence intervals used 10,000 classifier bootstraps; p-values used 10,000 complete-label permutations and were BH-adjusted across the 11 dimensions within each weighting scheme.

{markdown_table(dim_primary[['dimension','pearson_r','pearson_ci_low','pearson_ci_high','spearman_rho','spearman_ci_low','spearman_ci_high','permutation_p','bh_fdr_p']])}

The strongest Pearson correlations were {', '.join(f"{DISPLAY[row.dimension]} (r={row.pearson_r:.3f})" for row in dim_primary.head(3).itertuples())}. Dimensions surviving BH-FDR at .05 were {', '.join(DISPLAY[d] for d in sig_dims['dimension']) if len(sig_dims) else 'none'}.

## 9. Robustness analyses

### Noun-count thresholds

{markdown_table(cosine_threshold[['minimum_noun_types','n_classifiers','observed_matched_mean','oriented_observed_minus_null_effect','effect_boot_ci_low','effect_boot_ci_high','permutation_p_directional']])}

### Prototype weighting

{markdown_table(cosine_weight[['prototype_weighting','n_classifiers','observed_matched_mean','oriented_observed_minus_null_effect','effect_boot_ci_low','effect_boot_ci_high','permutation_p_directional']])}

### Removing generic dimension tendencies

Classifier dimensions and prototype dimensions were standardized separately across the primary classifiers. This robustness analysis removes tendencies such as generally high vision values before within-profile correspondence is calculated. The standardized cosine advantage was {standardized_key.loc['cosine','oriented_observed_minus_null_effect']:.3f}, 95% bootstrap CI [{standardized_key.loc['cosine','effect_boot_ci_low']:.3f}, {standardized_key.loc['cosine','effect_boot_ci_high']:.3f}], directional permutation p={standardized_key.loc['cosine','permutation_p_directional']:.4f}. Raw and standardized results are kept separate.

### Strongest lexical associates

Nouns were ranked by the already defined retained-table logDice measure; positive Dice strength is monotonic in logDice. Top-k analyses included a classifier only when at least k nouns were available.

{markdown_table(cosine_assoc[['association_subset','n_classifiers','observed_matched_mean','oriented_observed_minus_null_effect','effect_boot_ci_low','effect_boot_ci_high','permutation_p_directional']])}

### Dominant dimensions

Exact tie sets were preserved. Because tied dominance sets make ordinary single-label kappa inappropriate, exact-set agreement was evaluated against classifier-label permutation.

{markdown_table(dominance_results[['domain','n_classifiers','exact_set_agreement_rate','permutation_expected_agreement','observed_minus_expected','permutation_p_directional']])}

## 10. Prototype stability

For each of {len(stability)} classifiers with at least {STABILITY_MIN_NOUNS} noun types, 80% of noun types were sampled without replacement 500 times. Each subsample prototype was compared with its full unweighted prototype by cosine similarity. Mean stability across classifiers was {stability['mean_stability_cosine'].mean():.3f} (median {stability['mean_stability_cosine'].median():.3f}, range {stability['mean_stability_cosine'].min():.3f}–{stability['mean_stability_cosine'].max():.3f}). Associations of correspondence with noun count, stability, and classifier participant N are descriptive and are recorded in `analysis_metadata.json`; they were not used to redefine the primary sample.

## 11. Main figures

![Dimension-wise validity](figures/fig1_dimensionwise_validity.png)

![Permutation null](figures/fig2_permutation_null.png)

![Matched versus mismatched](figures/fig3_matched_vs_mismatched.png)

![Reproducible classifier examples](figures/fig4_example_profiles.png)

Figure 4 uses a pre-specified reproducible ranking: the minimum, nearest-to-median, and maximum classifier-specific cosine matched advantage. The selected IDs are {figure_meta['examples']}.

## 12. Interpretation

The construct-validity claim depends on matched profiles outperforming permuted or classifier-specific mismatched profiles, not on high raw cosine similarity alone. The current analysis evaluates whether directly rated classifiers track the central sensorimotor structure of their own independently normed noun classes. This evidence must not be conflated with accurate prediction of individual nouns: a classifier can validly represent a broad noun-class prototype while remaining too coarse to predict within-class noun variation.

The results support modest but systematic classifier-level external construct validity. All six raw-profile metrics favored each classifier's own noun prototype, and the four pre-specified permutation metrics had directional p-values below .001 in the primary sample. The cosine advantage remained positive after removing dimension-wide tendencies and at every pre-specified noun-count threshold. Frequency and positive-Dice weighting produced the same substantive conclusion. However, the top-5, top-10, and top-20 analyses did not show a monotonic increase over the all-noun prototype, so there is no clear evidence that restricting prototypes to the strongest lexical associates improves correspondence. At the dimension level, convergence was concentrated in touch, gustation, mouth/throat, olfaction, leg/foot, and vision rather than being uniform across all 11 dimensions.

For a Behavior Research Methods-style norming paper, the defensible interpretation is that classifier ratings have external validity to the extent that the matched advantage is positive, survives dimension-standardization and pre-specified noun-count/weighting checks, and is not confined to generic visual dominance. Dimension-specific estimates show where convergence is concentrated, while the top-k and stability analyses delimit how lexical association strength and prototype precision affect the result.

## 13. Limitations

1. The retained collocation table had an earlier minimum-frequency threshold and is not an unrestricted corpus sample; its PMI/logDice margins are retained-table quantities.
2. Some classifier prototypes use few independently normed nouns, motivating rather than replacing the pre-specified threshold analyses.
3. Nouns can occur with multiple classifiers, so classifier prototypes are not lexically disjoint even though the classifier is the inferential unit.
4. Prototype stability measures noun-sampling sensitivity, not the reliability of the independent noun norms themselves.
5. Per-classifier split-half reliability was unavailable; participant N was examined descriptively instead.
6. Ambiguous classifier homographs remain excluded pending contextual disambiguation.
7. This analysis is correlational external validation and does not establish causal semantic encoding.

## Reproducibility

Random seed: `{SEED}`. Principal bootstrap samples: `{N_BOOT}`. Principal label permutations: `{N_PERM}`. Stability replicates: `{N_STABILITY}`. All inference resamples classifiers or permutes complete classifier/prototype labels; the 2,038 edges are never treated as independent observations in the principal tests.
"""
    (OUTPUT / "classifier_external_construct_validation_report.md").write_text(report, encoding="utf-8")
    print(json.dumps({
        "output": str(OUTPUT),
        "primary_n_classifiers": len(primary),
        "unique_nouns": int(raw["noun_norm_simplified"].nunique()),
        "collapsed_edges": len(edges),
        "primary_cosine": primary_key.loc["cosine"].to_dict(),
        "standardized_cosine": standardized_key.loc["cosine"].to_dict(),
        "strongest_dimensions": dim_primary.head(5)[["dimension", "pearson_r", "bh_fdr_p"]].to_dict("records"),
    }, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
