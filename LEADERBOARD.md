# BenchCAD Leaderboard

_A comprehensive, industry-standard benchmark for programmatic CAD._

**Scoring** — Execution-grounded and deterministic. Vision2Code/CodeEdit grade by voxel IoU between executed STEP solids; QA grades by symmetric ratio accuracy.

**Reproduce** — `uv run python benchcad.py --task all --num all --model <model>`

> Numbers are **re-graded by us from raw model outputs, never self-reported.** To get on the board, submit predictions per [CONTRIBUTING.md § B](CONTRIBUTING.md).

<!-- GENERATED from leaderboard.json by tools/build_leaderboard.py — do not hand-edit -->

## Vision2Code

Image → CadQuery. Four canonical orthographic views in, a CadQuery program out, re-executed and scored against the ground-truth STEP. IoU-score↑ = 64³ voxel IoU × exec% (non-executing parts count as 0) · exec% runs cleanly · total↑ composite.

| Model | Org | Released | Tested | exec | IoU-score | total |
|---|---|---|---|---|---|---|
| Gemini 3.1 Pro | Google | 2026-05 | 2026-06 | 81.5% | 0.2890 | 0.3457 |
| Gemini 3.1 Pro | Google | 2026-05 | 2026-06 | 88.5% | 0.2779 | 0.3315 |
| Claude Opus 4.7 | Anthropic | 2026-03 | 2026-06 | 96.5% | 0.2692 | 0.3238 |
| Claude Opus 4.7 | Anthropic | 2026-03 | 2026-06 | 95.5% | 0.2617 | 0.3075 |
| Claude Sonnet 4.6 | Anthropic | 2026-01 | 2026-06 | 86.5% | 0.2220 | 0.2979 |
| Claude Sonnet 4.6 | Anthropic | 2026-01 | 2026-06 | 79.5% | 0.1920 | 0.2599 |
| GPT-5.3 | OpenAI | 2026-04 | 2026-06 | 82.0% | 0.1793 | 0.2497 |
| GPT-5.3 | OpenAI | 2026-04 | 2026-06 | 81.5% | 0.1873 | 0.2465 |
| GPT-4o | OpenAI | 2024-05 | 2026-06 | 91.0% | 0.1823 | 0.2393 |
| OpenAI o3 | OpenAI | 2025-04 | 2026-06 | 54.0% | 0.1218 | 0.1978 |
| GPT-4o · | OpenAI | — | 2026-06 | 87.0% | 0.0698 | 0.1102 |
| Moonshot v1-128k | Moonshot | 2025-02 | 2026-06 | 12.5% | 0.0160 | 0.0609 |
| Moonshot v1-8k | Moonshot | 2025-02 | 2026-06 | 10.0% | 0.0127 | 0.0595 |
| Qwen3-VL-2B (baseline) | Qwen | 2025-09 | 2026-06 | 14.6% | 0.0005 | 0.0084 |
| Claude Mythos 5 † | Anthropic | — | — | — | 0.3840 | — |
| Claude Mythos Preview † | Anthropic | — | — | — | 0.3550 | — |
| Claude Opus 4.8 † | Anthropic | 2026-05 | 2026-06 | — | 0.2730 | — |

## Vision QA

Numeric geometric reasoning from multi-view renders, broken out along the four-level capability hierarchy. ±5% tolerance for ratios, exact match for integers. Same 2,400 questions as Code QA — the matched-pair gap isolates visual recognition from reasoning.

| Model | Org | Released | Tested | Holistic Visual Recognition | CAD Operation Understanding | Industrial Parametric Abstraction | Spatial Reasoning | Total |
|---|---|---|---|---|---|---|---|---|
| Gemini 3.1 Pro | Google | 2026-05 | 2026-05 | 0.75 | 0.462 | 0.536 | 0.688 | 0.587 |
| Gemini 3.1 Pro | Google | 2026-05 | 2026-05 | 0.722 | 0.426 | 0.551 | 0.669 | 0.576 |
| Claude Opus 4.7 | Anthropic | 2026-03 | 2026-05 | 0.715 | 0.485 | 0.421 | 0.614 | 0.53 |
| Claude Opus 4.7 | Anthropic | 2026-03 | 2026-05 | 0.699 | 0.464 | 0.426 | 0.668 | 0.526 |
| GPT-5.3 | OpenAI | 2026-04 | 2026-05 | 0.65 | 0.429 | 0.482 | 0.534 | 0.514 |
| GPT-5.3 | OpenAI | 2026-04 | 2026-05 | 0.636 | 0.423 | 0.488 | 0.548 | 0.513 |
| GPT-4o | OpenAI | 2024-05 | 2026-05 | 0.599 | 0.408 | 0.431 | 0.396 | 0.464 |
| Moonshot v1-8k | Moonshot | 2025-02 | 2026-05 | 0.6 | 0.246 | 0.465 | 0.181 | 0.447 |
| Moonshot v1-128k | Moonshot | 2025-02 | 2026-05 | 0.556 | 0.387 | 0.427 | 0.334 | 0.442 |
| blank-image baseline · | — | — | 2026-05 | 0.376 | 0.325 | 0.418 | 0.296 | 0.375 |
| OpenAI o3 | OpenAI | 2025-04 | 2026-05 | 0.328 | 0.188 | 0.398 | 0.56 | 0.327 |

## Code QA

The same 2,400 numeric questions as Vision QA, but conditioned on CadQuery source instead of renders. Best Code QA reaches 0.838 while best Vision QA caps at 0.587 — a ~25 pt modality gap on identical questions (the Holistic Spatial & Detailing Deficit).

| Model | Org | Released | Tested | CadQuery Code Recognition | CAD Operation Understanding | Industrial Parametric Abstraction | Spatial Reasoning | Total |
|---|---|---|---|---|---|---|---|---|
| Gemini 3.1 Pro | Google | 2026-05 | 2026-05 | 0.907 | 0.783 | 0.876 | 0.537 | 0.838 |
| Gemini 3.1 Pro | Google | 2026-05 | 2026-05 | 0.914 | 0.782 | 0.867 | 0.537 | 0.836 |
| Claude Opus 4.7 | Anthropic | 2026-03 | 2026-05 | 0.891 | 0.781 | 0.851 | 0.632 | 0.829 |
| GPT-5.3 | OpenAI | 2026-04 | 2026-05 | 0.879 | 0.805 | 0.815 | 0.731 | 0.823 |
| GPT-5.3 | OpenAI | 2026-04 | 2026-05 | 0.885 | 0.802 | 0.811 | 0.73 | 0.821 |
| Claude Opus 4.7 | Anthropic | 2026-03 | 2026-05 | 0.868 | 0.8 | 0.793 | 0.595 | 0.801 |
| GPT-4o | OpenAI | 2024-05 | 2026-05 | 0.865 | 0.593 | 0.732 | 0.688 | 0.726 |
| OpenAI o3 | OpenAI | 2025-04 | 2026-05 | 0.804 | 0.701 | 0.689 | 0.492 | 0.708 |
| Moonshot v1-128k | Moonshot | 2025-02 | 2026-05 | 0.842 | 0.551 | 0.692 | 0.792 | 0.7 |
| gpt-oss-120b | OpenAI | 2025-08 | 2026-05 | 0.79 | 0.732 | 0.656 | 0.379 | 0.689 |
| Nemotron-3 120B | NVIDIA | 2026-01 | 2026-05 | 0.771 | 0.66 | 0.661 | 0.293 | 0.671 |
| Gemma-4-31B-it | Google | 2026-02 | 2026-05 | 0.791 | 0.674 | 0.606 | 0.528 | 0.664 |
| Moonshot v1-8k | Moonshot | 2025-02 | 2026-05 | 0.772 | 0.603 | 0.555 | 0.536 | 0.62 |
| blank-code baseline · | — | — | 2026-05 | 0.04 | 0.257 | 0.29 | 0.42 | 0.223 |

## Code Edit

Given a CadQuery program and a natural-language edit instruction, output a minimally modified program matching the target. Accuracy↑ is headroom-normalised improvement: how much of the original→target IoU gap the edit closes. 748 pairs across five edit types T1–T5.

| Model | Org | Released | Tested | Thinking | Accuracy |
|---|---|---|---|---|---|
| GPT-5.3 | OpenAI | 2026-04 | 2026-05 | True | 0.865 |
| Claude Opus 4.7 | Anthropic | 2026-03 | 2026-05 | True | 0.853 |
| Gemini 3.1 Pro | Google | 2026-05 | 2026-05 | True | 0.837 |
| Claude Opus 4.7 | Anthropic | 2026-03 | 2026-05 | False | 0.811 |
| Gemini 3.1 Pro | Google | 2026-05 | 2026-05 | False | 0.795 |
| GPT-5.3 | OpenAI | 2026-04 | 2026-05 | False | 0.74 |
| OpenAI o3 | OpenAI | 2025-04 | 2026-05 | True | 0.708 |
| GPT-4o | OpenAI | 2024-05 | 2026-05 | False | 0.615 |
| Nemotron-3 120B | NVIDIA | 2026-01 | 2026-05 | False | 0.608 |
| gpt-oss-120b | OpenAI | 2025-08 | 2026-05 | False | 0.561 |
| no-change baseline · | — | — | 2026-05 | False | 0.0 |

---

⭐ BenchCAD's own model · 🔧 CAD specialist.
**† self-reported by the provider, not re-graded by us** — preview/unreleased models for which only the provider-published IoU-score is shown (no exec% / total). Every other row is re-graded by us from raw outputs.

Generated from `leaderboard.json` — `uv run python tools/build_leaderboard.py`.
