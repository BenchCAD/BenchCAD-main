# benchcad-qa

Rendered views of a mechanical part + one numeric question → a single number. The
model is shown a 2×2 composite of four diagonal views of an industry-standard
part and answers a question about a dimension, count, or ratio.

**Reward = symmetric ratio accuracy** `min(pred, gt) / max(pred, gt)` for
dimensions/ratios, exact match for counts/integers/yes-no. Deterministic — **no
judge model and no CAD execution** (so this environment is lightweight: just
`verifiers` + `datasets` + `pillow`).

- Benchmark: [BenchCAD](https://github.com/BenchCAD/BenchCAD-main) · paper [arXiv:2605.10865](https://arxiv.org/abs/2605.10865)
- Data: [`BenchCAD/BenchCAD`](https://huggingface.co/datasets/BenchCAD/BenchCAD), config `QA` (2,400 questions over 200 parts)

## Usage

```bash
uv run vf-eval benchcad-qa -n 20
```

`load_environment(num_examples=None)` — set `num_examples` to evaluate the first
N questions; omit for the full set.

## Notes

The numeric scoring is vendored from the canonical scorer
(`BenchCAD/BenchCAD-main`, `QA/scoring/qa_score.py`); that repo is the source
of truth. See also the heavier execution-based [`benchcad-vision2code`](../benchcad-vision2code)
(image → CadQuery program, scored by voxel IoU).
