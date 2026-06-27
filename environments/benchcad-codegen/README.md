# benchcad-codegen

Image of a mechanical part → a **CadQuery** program. The model is shown a 2×2
composite of four diagonal rendered views of an industry-standard mechanical part
and must write a CadQuery program that reproduces the geometry.

**Reward = voxel IoU** between the model's executed STEP solid and the
ground-truth STEP. Execution-grounded and deterministic — there is **no judge
model**. Non-executable outputs score 0.

- Benchmark: [BenchCAD](https://github.com/BenchCAD/BenchCAD-main) · paper [arXiv:2605.10865](https://arxiv.org/abs/2605.10865)
- Data: [`BenchCAD/BenchCAD`](https://huggingface.co/datasets/BenchCAD/BenchCAD), config `code_gen` (17,900 parts / 106 families / 49 standards)

## Task

| | |
|---|---|
| Input | a rendered 2×2 view composite (image) |
| Output | a CadQuery program binding the final solid to `result` |
| Reward | voxel IoU on a normalized 64³ grid, `\|A∩B\| / \|A∪B\|` |

## Usage

```bash
# install + quick eval on a few parts
uv run vf-eval benchcad-codegen -n 5

# in code
import verifiers as vf
env = vf.load_environment("benchcad-codegen", num_examples=100)
```

`load_environment(num_examples=None)` — set `num_examples` to evaluate on the
first N parts (handy for quick runs); omit for the full set.

## Notes

The CadQuery execution and voxel-IoU scoring are vendored verbatim from the
canonical scorer (`BenchCAD/BenchCAD-main`, `Vision2Code/scoring/`) so this
environment is a self-contained package; that repository remains the source of
truth. Requires the CadQuery / OCP stack (declared in `pyproject.toml`).
