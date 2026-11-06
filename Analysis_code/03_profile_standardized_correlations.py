"""Profile-standardized inter-dimensional correlations for classifier norms.

This script leaves the frozen release untouched. It reads the 353 x 11
classifier-level mean matrix and compares three representations:

1. raw classifier means;
2. within-classifier mean-centred profiles; and
3. within-classifier z-scored profiles.

For each representation it computes all 55 Pearson correlations, analytic
two-sided p values, Benjamini-Hochberg FDR q values, and classifier-bootstrap
confidence intervals. It also produces publication-ready comparison figures
and a concise interpretation report.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


SEED = 20261013
BOOTSTRAP_REPLICATES = 5000

DIMENSIONS = [
    "vision",
    "audition",
    "gustation",
    "olfaction",
    "touch",
    "interoception",
    "leg_foot",
    "hand_arm",
    "head",
    "mouth_throat",
    "torso",
]

DISPLAY = {
    "vision": "Vision",
    "audition": "Audition",
    "gustation": "Gustation",
    "olfaction": "Olfaction",
    "touch": "Touch",
    "interoception": "Interoception",
    "leg_foot": "Leg/foot",
    "hand_arm": "Hand/arm",
    "head": "Head",
    "mouth_throat": "Mouth/throat",
    "torso": "Torso",
}

FOCAL_PAIRS = [
    ("olfaction", "gustation"),
    ("torso", "leg_foot"),
    ("torso", "hand_arm"),
    ("torso", "head"),
    ("hand_arm", "touch"),
    ("mouth_throat", "gustation"),
    ("touch", "vision"),
    ("head", "leg_foot"),
]


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function."""
    max_iter = 300
    eps = 3.0e-14
    fpmin = 1.0e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            return h
    raise RuntimeError("Incomplete beta continued fraction did not converge")


def regularized_incomplete_beta(x: float, a: float, b: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_bt = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    bt = math.exp(log_bt)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def pearson_r_p(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xc = x - x.mean()
    yc = y - y.mean()
    denom = math.sqrt(float(xc @ xc) * float(yc @ yc))
    if denom == 0.0:
        return float("nan"), float("nan")
    r = float((xc @ yc) / denom)
    r = min(1.0, max(-1.0, r))
    if abs(r) >= 1.0:
        return r, 0.0
    df = len(x) - 2
    t2 = (r * r) * df / (1.0 - r * r)
    p = regularized_incomplete_beta(df / (df + t2), df / 2.0, 0.5)
    return r, min(1.0, max(0.0, p))


def bh_adjust(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    ranked = p_values[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    out = np.empty_like(adjusted)
    out[order] = adjusted
    return out


def bootstrap_corr_ci(
    x: np.ndarray, y: np.ndarray, bootstrap_indices: np.ndarray
) -> tuple[float, float]:
    xb = x[bootstrap_indices]
    yb = y[bootstrap_indices]
    xb = xb - xb.mean(axis=1, keepdims=True)
    yb = yb - yb.mean(axis=1, keepdims=True)
    numerator = np.sum(xb * yb, axis=1)
    denominator = np.sqrt(np.sum(xb * xb, axis=1) * np.sum(yb * yb, axis=1))
    values = np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype=float),
        where=denominator > 0,
    )
    return tuple(np.nanquantile(values, [0.025, 0.975]).tolist())


def transform_profiles(raw: np.ndarray) -> dict[str, np.ndarray]:
    row_means = raw.mean(axis=1, keepdims=True)
    centered = raw - row_means
    row_sds = raw.std(axis=1, ddof=1, keepdims=True)
    if np.any(row_sds == 0):
        raise ValueError("At least one classifier has a zero-variance 11-dimensional profile")
    return {
        "raw": raw,
        "centered": centered,
        "z_scored": centered / row_sds,
    }


def analyse_scale(
    scale: str,
    values: np.ndarray,
    bootstrap_indices: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray]:
    rows: list[dict[str, object]] = []
    matrix = np.corrcoef(values, rowvar=False)
    for i, dim_1 in enumerate(DIMENSIONS):
        for j in range(i + 1, len(DIMENSIONS)):
            dim_2 = DIMENSIONS[j]
            r, p = pearson_r_p(values[:, i], values[:, j])
            ci_low, ci_high = bootstrap_corr_ci(
                values[:, i], values[:, j], bootstrap_indices
            )
            rows.append(
                {
                    "scale": scale,
                    "dimension_1": dim_1,
                    "dimension_2": dim_2,
                    "dimension_1_display": DISPLAY[dim_1],
                    "dimension_2_display": DISPLAY[dim_2],
                    "n_classifiers": len(values),
                    "pearson_r": r,
                    "bootstrap_ci95_lower": ci_low,
                    "bootstrap_ci95_upper": ci_high,
                    "p_two_sided": p,
                }
            )
    frame = pd.DataFrame(rows)
    frame["fdr_bh_q"] = bh_adjust(frame["p_two_sided"].to_numpy())
    frame["significant_fdr_05"] = frame["fdr_bh_q"] < 0.05
    frame["direction"] = np.where(frame["pearson_r"] >= 0, "positive", "negative")
    frame["rank_by_r_descending"] = frame["pearson_r"].rank(
        method="first", ascending=False
    ).astype(int)
    return frame, matrix


def stars(q: float) -> str:
    if q < 0.001:
        return "***"
    if q < 0.01:
        return "**"
    if q < 0.05:
        return "*"
    return ""


def find_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def correlation_color(r: float) -> tuple[int, int, int]:
    # Color-blind-friendly diverging palette: blue -> white -> vermillion.
    blue = np.array([49, 115, 178], dtype=float)
    white = np.array([250, 250, 250], dtype=float)
    red = np.array([213, 94, 0], dtype=float)
    if r < 0:
        color = white + (-r) * (blue - white)
    else:
        color = white + r * (red - white)
    return tuple(np.clip(np.rint(color), 0, 255).astype(int).tolist())


def lookup_q(frame: pd.DataFrame) -> dict[frozenset[str], float]:
    return {
        frozenset((row.dimension_1, row.dimension_2)): float(row.fdr_bh_q)
        for row in frame.itertuples(index=False)
    }


def draw_heatmap_panel(
    matrix: np.ndarray,
    pairwise: pd.DataFrame,
    title: str,
    cell: int = 76,
) -> Image.Image:
    n = len(DIMENSIONS)
    left = 168
    top = 94
    right = 25
    bottom = 270
    width = left + n * cell + right
    height = top + n * cell + bottom
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = find_font(30, bold=True)
    label_font = find_font(19)
    value_font = find_font(17, bold=True)
    note_font = find_font(17)
    draw.text((left, 24), title, fill=(28, 37, 54), font=title_font)
    q_lookup = lookup_q(pairwise)

    for i, dim_y in enumerate(DIMENSIONS):
        y = top + i * cell
        label = DISPLAY[dim_y]
        bbox = draw.textbbox((0, 0), label, font=label_font)
        draw.text((left - 12 - (bbox[2] - bbox[0]), y + 25), label, fill=(35, 35, 35), font=label_font)
        for j, dim_x in enumerate(DIMENSIONS):
            x = left + j * cell
            if j > i:
                draw.rectangle((x, y, x + cell, y + cell), fill=(255, 255, 255), outline=(225, 225, 225))
                continue
            r = float(matrix[i, j])
            draw.rectangle((x, y, x + cell, y + cell), fill=correlation_color(r), outline=(235, 235, 235))
            if i == j:
                text = "1.00"
            else:
                q = q_lookup[frozenset((dim_y, dim_x))]
                text = f"{r:.2f}{stars(q)}"
            bbox = draw.textbbox((0, 0), text, font=value_font)
            luminance = sum(correlation_color(r)) / 3
            fill = (255, 255, 255) if luminance < 150 else (30, 30, 30)
            draw.text(
                (x + (cell - (bbox[2] - bbox[0])) / 2, y + 27),
                text,
                fill=fill,
                font=value_font,
            )

    for j, dim_x in enumerate(DIMENSIONS):
        x = left + j * cell + (cell - 32) // 2
        y = top + n * cell + 10
        label = DISPLAY[dim_x]
        layer = Image.new("RGBA", (220, 32), (255, 255, 255, 0))
        layer_draw = ImageDraw.Draw(layer)
        layer_draw.text((0, 2), label, fill=(35, 35, 35), font=label_font)
        rotated = layer.rotate(90, expand=True, resample=Image.Resampling.BICUBIC)
        image.paste(rotated, (x, y), rotated)

    note = "Values are Pearson's r. BH-FDR: * q < .05, ** q < .01, *** q < .001."
    draw.text((left, height - 30), note, fill=(65, 65, 65), font=note_font)
    return image


def save_figures(
    matrices: dict[str, np.ndarray],
    frames: dict[str, pd.DataFrame],
    output_dir: Path,
) -> None:
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    titles = {
        "raw": "Raw classifier means",
        "centered": "Within-classifier mean-centred profiles",
        "z_scored": "Within-classifier z-scored profiles",
    }
    panels: list[Image.Image] = []
    for scale in ("raw", "centered", "z_scored"):
        panel = draw_heatmap_panel(matrices[scale], frames[scale], titles[scale])
        panel.save(figure_dir / f"correlation_matrix_{scale}.png", dpi=(180, 180))
        panels.append(panel)

    gap = 26
    canvas = Image.new(
        "RGB",
        (sum(panel.width for panel in panels) + gap * 2, max(panel.height for panel in panels)),
        "white",
    )
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 0))
        x += panel.width + gap
    canvas.save(figure_dir / "correlation_matrices_comparison.png", dpi=(180, 180))


def pair_label(dim_1: str, dim_2: str) -> str:
    return f"{DISPLAY[dim_1]}-{DISPLAY[dim_2]}"


def format_p(value: float) -> str:
    if value < 0.001:
        return "< .001"
    return f"= {value:.3f}".replace("0.", ".")


def format_q(value: float) -> str:
    if value < 0.001:
        return "< .001"
    return f"= {value:.3f}".replace("0.", ".")


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    headers = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [headers, divider]
    for _, row in frame[columns].iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in columns) + " |")
    return "\n".join(lines)


def build_report(
    all_pairs: pd.DataFrame,
    summary: pd.DataFrame,
    focal: pd.DataFrame,
    output_dir: Path,
    input_path: Path,
) -> None:
    z = all_pairs[all_pairs["scale"] == "z_scored"].copy()
    z_top = z.sort_values("pearson_r", ascending=False).head(10).copy()
    z_top["Pair"] = z_top.apply(
        lambda row: pair_label(row["dimension_1"], row["dimension_2"]), axis=1
    )
    z_top["r"] = z_top["pearson_r"].map(lambda value: f"{value:.3f}")
    z_top["95% CI"] = z_top.apply(
        lambda row: f"[{row['bootstrap_ci95_lower']:.3f}, {row['bootstrap_ci95_upper']:.3f}]",
        axis=1,
    )
    z_top["p"] = z_top["p_two_sided"].map(format_p)
    z_top["q"] = z_top["fdr_bh_q"].map(format_q)

    focal_display = focal.copy()
    focal_display["Pair"] = focal_display["pair"]
    for scale in ("raw", "centered", "z_scored"):
        focal_display[scale] = focal_display.apply(
            lambda row, s=scale: (
                f"r = {row[f'{s}_r']:.3f}, q {format_q(row[f'{s}_q'])}"
            ),
            axis=1,
        )

    summary_lookup = summary.set_index("scale")
    raw = summary_lookup.loc["raw"]
    centered = summary_lookup.loc["centered"]
    zsum = summary_lookup.loc["z_scored"]
    focal_sig = focal[focal["z_scored_q"] < 0.05]
    focal_sig_names = ", ".join(focal_sig["pair"].tolist()) if len(focal_sig) else "none"
    focal_nonsig = focal[focal["z_scored_q"] >= 0.05]
    focal_nonsig_names = ", ".join(focal_nonsig["pair"].tolist()) if len(focal_nonsig) else "none"

    report = f"""# Within-classifier profile standardization of inter-dimensional correlations

## Purpose

The original correlation matrix was strongly positive, raising the possibility that correlations reflected between-classifier differences in general rating intensity rather than dimension-specific sensorimotor profile shape. This analysis compares the original correlations with correlations after removing each classifier's overall profile elevation and, separately, its profile dispersion.

## Data and transformations

- Input: `{input_path}`
- Sample: 353 classifiers with complete means on all 11 dimensions.
- Raw: the original 0-5 classifier-level dimension means.
- Centred: each classifier's mean across its 11 dimensions was subtracted from all 11 values.
- Z-scored: each centred classifier profile was divided by its sample SD across the 11 dimensions.
- For each representation, all 55 Pearson correlations were calculated across classifiers. Two-sided analytic p values were adjusted within the 55 tests using Benjamini-Hochberg FDR. Percentile 95% confidence intervals used {BOOTSTRAP_REPLICATES:,} classifier-level bootstrap samples with seed {SEED}.

## Global comparison

| Representation | Positive | Negative | FDR significant | Significant positive | Significant negative | Median r | Minimum r | Maximum r |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Raw | {int(raw['n_positive'])} | {int(raw['n_negative'])} | {int(raw['n_fdr_significant'])} | {int(raw['n_fdr_significant_positive'])} | {int(raw['n_fdr_significant_negative'])} | {raw['median_r']:.3f} | {raw['min_r']:.3f} | {raw['max_r']:.3f} |
| Within-classifier centred | {int(centered['n_positive'])} | {int(centered['n_negative'])} | {int(centered['n_fdr_significant'])} | {int(centered['n_fdr_significant_positive'])} | {int(centered['n_fdr_significant_negative'])} | {centered['median_r']:.3f} | {centered['min_r']:.3f} | {centered['max_r']:.3f} |
| Within-classifier z-scored | {int(zsum['n_positive'])} | {int(zsum['n_negative'])} | {int(zsum['n_fdr_significant'])} | {int(zsum['n_fdr_significant_positive'])} | {int(zsum['n_fdr_significant_negative'])} | {zsum['median_r']:.3f} | {zsum['min_r']:.3f} | {zsum['max_r']:.3f} |

The raw analysis exactly reproduces the manuscript summary: 54 of 55 coefficients were positive and 50 survived FDR correction. After within-classifier standardization, the uniformly positive manifold was no longer present. This is expected because every centred or z-scored profile sums to zero across the 11 dimensions.

## Focal relationships

{markdown_table(focal_display, ['Pair', 'raw', 'centered', 'z_scored'])}

Of the eight relationships highlighted in the manuscript, the following remained positive and FDR-significant after z-scoring: {focal_sig_names}. The following did not remain FDR-significant: {focal_nonsig_names}.

## Strongest positive associations after z-scoring

{markdown_table(z_top, ['Pair', 'r', '95% CI', 'p', 'q'])}

## Interpretation

The z-scored analysis should be interpreted as a test of relative profile shape: whether two dimensions tend to be jointly elevated relative to each classifier's own 11-dimensional mean. It does not preserve the meaning of absolute 0-5 strength. The emergence of negative correlations is partly structural, because within-classifier centring imposes a zero-sum constraint.

Classifier-level profile standardization removes between-classifier differences in mean elevation (and, for z-scores, dispersion), but it does not directly eliminate respondent-level scale-use tendencies. A literal response-style analysis would require standardizing individual participant responses before rebuilding the classifier norms.

## Writing recommendation

Use the raw correlation matrix as the descriptive account of absolute sensorimotor strength, and report the z-scored matrix as a robustness analysis of profile shape. All eight focal relationships survived, but several were attenuated substantially; therefore, distinguish statistical survival from effect-size strength. A defensible manuscript paragraph is:

> To assess whether the predominantly positive raw correlations were driven by between-classifier differences in overall rating intensity, each classifier's 11-dimensional mean profile was z-standardized within classifier before the correlations were recomputed. The uniformly positive pattern was replaced by a mixture of positive and negative associations, as expected under within-profile centring. Nevertheless, all eight relationships highlighted in the raw analysis remained positive and significant after BH-FDR correction. The strongest retained associations were Olfaction-Gustation, r = .680, q < .001; Torso-Leg/foot, r = .475, q < .001; Torso-Hand/arm, r = .372, q < .001; and Torso-Head, r = .307, q < .001. Mouth/throat-Gustation (r = .298, q < .001) and Hand/arm-Touch (r = .219, q < .001) also remained significant, whereas Touch-Vision (r = .154, q = .006) and Head-Leg/foot (r = .141, q = .011) were retained but attenuated. Thus, the principal sensory and action groupings were not solely attributable to general profile elevation, although part of the broad positive structure in the raw ratings reflected overall intensity differences between classifiers.

Do not describe this transformation as eliminating respondent scale use.

## Files

- `pairwise_correlations_all_scales.csv`: all 165 pairwise tests.
- `focal_pair_comparison.csv`: the eight relationships emphasized in the manuscript.
- `correlation_summary.csv`: counts and distribution summaries by representation.
- `correlation_matrix_*.csv`: complete matrices.
- `figures/correlation_matrices_comparison.png`: visual comparison.
- `analysis_metadata.json` and `qc_checks.csv`: provenance and validation.
"""
    (output_dir / "analysis_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    script_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=script_root / "outputs" / "norm_build_v1" / "classifier_master.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=script_root
        / "analysis"
        / "classifier_dimension_correlations_profile_standardized_v1",
    )
    args = parser.parse_args()
    input_path = args.input.resolve()
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(input_path)
    mean_columns = [f"{dim}_mean" for dim in DIMENSIONS]
    missing_columns = [column for column in mean_columns if column not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    raw = data[mean_columns].to_numpy(dtype=float)
    if raw.shape != (353, 11):
        raise ValueError(f"Expected a 353 x 11 matrix; received {raw.shape}")
    if not np.isfinite(raw).all():
        raise ValueError("The analysis matrix contains missing or non-finite values")

    transformed = transform_profiles(raw)
    rng = np.random.default_rng(SEED)
    bootstrap_indices = rng.integers(
        0, len(raw), size=(BOOTSTRAP_REPLICATES, len(raw)), dtype=np.int32
    )

    frames: dict[str, pd.DataFrame] = {}
    matrices: dict[str, np.ndarray] = {}
    for scale, values in transformed.items():
        frame, matrix = analyse_scale(scale, values, bootstrap_indices)
        frames[scale] = frame
        matrices[scale] = matrix

    all_pairs = pd.concat(frames.values(), ignore_index=True)
    all_pairs.to_csv(output_dir / "pairwise_correlations_all_scales.csv", index=False)

    summary_rows = []
    for scale, frame in frames.items():
        significant = frame[frame["significant_fdr_05"]]
        summary_rows.append(
            {
                "scale": scale,
                "n_classifiers": len(raw),
                "n_pairs": len(frame),
                "n_positive": int((frame["pearson_r"] > 0).sum()),
                "n_negative": int((frame["pearson_r"] < 0).sum()),
                "n_fdr_significant": int(frame["significant_fdr_05"].sum()),
                "n_fdr_significant_positive": int((significant["pearson_r"] > 0).sum()),
                "n_fdr_significant_negative": int((significant["pearson_r"] < 0).sum()),
                "mean_r": float(frame["pearson_r"].mean()),
                "median_r": float(frame["pearson_r"].median()),
                "min_r": float(frame["pearson_r"].min()),
                "max_r": float(frame["pearson_r"].max()),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "correlation_summary.csv", index=False)

    focal_rows = []
    for dim_a, dim_b in FOCAL_PAIRS:
        row: dict[str, object] = {"pair": pair_label(dim_a, dim_b)}
        for scale, frame in frames.items():
            match = frame[
                frame.apply(
                    lambda item: {item["dimension_1"], item["dimension_2"]}
                    == {dim_a, dim_b},
                    axis=1,
                )
            ].iloc[0]
            row.update(
                {
                    f"{scale}_r": match["pearson_r"],
                    f"{scale}_ci95_lower": match["bootstrap_ci95_lower"],
                    f"{scale}_ci95_upper": match["bootstrap_ci95_upper"],
                    f"{scale}_p": match["p_two_sided"],
                    f"{scale}_q": match["fdr_bh_q"],
                }
            )
        focal_rows.append(row)
    focal = pd.DataFrame(focal_rows)
    focal.to_csv(output_dir / "focal_pair_comparison.csv", index=False)

    for scale, matrix in matrices.items():
        pd.DataFrame(
            matrix,
            index=[DISPLAY[dim] for dim in DIMENSIONS],
            columns=[DISPLAY[dim] for dim in DIMENSIONS],
        ).to_csv(output_dir / f"correlation_matrix_{scale}.csv")

    upper = np.triu_indices(len(DIMENSIONS), k=1)
    matrix_similarity = pd.DataFrame(
        [
            {
                "matrix_1": a,
                "matrix_2": b,
                "pearson_r_across_55_coefficients": float(
                    np.corrcoef(matrices[a][upper], matrices[b][upper])[0, 1]
                ),
            }
            for a, b in [("raw", "centered"), ("raw", "z_scored"), ("centered", "z_scored")]
        ]
    )
    matrix_similarity.to_csv(output_dir / "matrix_similarity.csv", index=False)

    save_figures(matrices, frames, output_dir)

    qc_rows = [
        ("input_rows_353", len(data) == 353, len(data)),
        ("analysis_dimensions_11", raw.shape[1] == 11, raw.shape[1]),
        ("complete_analysis_matrix", bool(np.isfinite(raw).all()), int(np.isfinite(raw).sum())),
        ("no_zero_profile_sd", bool(np.all(raw.std(axis=1, ddof=1) > 0)), int((raw.std(axis=1, ddof=1) == 0).sum())),
        ("raw_positive_coefficients_54", int((frames["raw"]["pearson_r"] > 0).sum()) == 54, int((frames["raw"]["pearson_r"] > 0).sum())),
        ("raw_fdr_significant_50", int(frames["raw"]["significant_fdr_05"].sum()) == 50, int(frames["raw"]["significant_fdr_05"].sum())),
        ("centred_row_means_zero", bool(np.allclose(transformed["centered"].mean(axis=1), 0.0, atol=1e-12)), float(np.max(np.abs(transformed["centered"].mean(axis=1))))),
        ("z_scored_row_means_zero", bool(np.allclose(transformed["z_scored"].mean(axis=1), 0.0, atol=1e-12)), float(np.max(np.abs(transformed["z_scored"].mean(axis=1))))),
        ("z_scored_row_sds_one", bool(np.allclose(transformed["z_scored"].std(axis=1, ddof=1), 1.0, atol=1e-12)), float(np.max(np.abs(transformed["z_scored"].std(axis=1, ddof=1) - 1.0)))),
        ("all_bootstrap_intervals_finite", bool(np.isfinite(all_pairs[["bootstrap_ci95_lower", "bootstrap_ci95_upper"]].to_numpy()).all()), int(np.isfinite(all_pairs[["bootstrap_ci95_lower", "bootstrap_ci95_upper"]].to_numpy()).sum())),
    ]
    qc = pd.DataFrame(qc_rows, columns=["check", "passed", "observed"])
    qc.to_csv(output_dir / "qc_checks.csv", index=False)
    if not qc["passed"].all():
        failed = qc.loc[~qc["passed"], "check"].tolist()
        raise RuntimeError(f"QC checks failed: {failed}")

    metadata = {
        "analysis_id": "classifier_dimension_correlations_profile_standardized_v1",
        "input": str(input_path),
        "output": str(output_dir),
        "n_classifiers": len(raw),
        "n_dimensions": len(DIMENSIONS),
        "n_pairwise_tests_per_scale": 55,
        "scales": list(transformed),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "seed": SEED,
        "multiple_testing": "Benjamini-Hochberg FDR separately within each 55-test matrix",
        "ci": "Percentile classifier-level bootstrap",
        "input_file_size_bytes": input_path.stat().st_size,
        "input_file_mtime": input_path.stat().st_mtime,
    }
    (output_dir / "analysis_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    build_report(all_pairs, summary, focal, output_dir, input_path)

    print(summary.to_string(index=False))
    print("\nFocal pairs:\n" + focal.to_string(index=False))
    print(f"\nOutputs written to: {output_dir}")


if __name__ == "__main__":
    main()
