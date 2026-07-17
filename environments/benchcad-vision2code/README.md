# benchcad-vision2code

Image of a mechanical part → a **CadQuery** program. The model is shown a 2×2
composite of four diagonal rendered views of an industry-standard mechanical part
and must write a CadQuery program that reproduces the geometry.

**Reward = voxel IoU** between the model's executed STEP solid and the
ground-truth STEP. Execution-grounded and deterministic — there is **no judge
model**. Non-executable outputs score 0.

- Benchmark: [BenchCAD](https://github.com/BenchCAD/BenchCAD-main) · paper [arXiv:2605.10865](https://arxiv.org/abs/2605.10865)
- Data: [`BenchCAD/BenchCAD`](https://huggingface.co/datasets/BenchCAD/BenchCAD), config `code_gen` (17,900 parts / 106 families / 47 standards)

## Task

| | |
|---|---|
| Input | a rendered 2×2 view composite (image) |
| Output | a CadQuery program binding the final solid to `result` |
| Reward | voxel IoU on a normalized 64³ grid, `\|A∩B\| / \|A∪B\|` |

## Usage

```bash
# install + quick eval on a few parts
uv run vf-eval benchcad-vision2code -n 5

# in code
import verifiers as vf
env = vf.load_environment("benchcad-vision2code", num_examples=100)
```

`load_environment(num_examples=None, exec_timeout=300)` — `num_examples` caps to
the first N parts (handy for quick runs; omit for the full set). `exec_timeout`
is the per-program CadQuery→STEP execution timeout in seconds; raise it for
slow-tessellating families on slower hardware.

Generation-side knobs — max output tokens, request timeout, reasoning effort —
are **sampling args** passed to the rollout at eval time, not part of the
environment:

```bash
uv run vf-eval benchcad-vision2code -n 5 \
  -S '{"max_tokens": 16000, "reasoning_effort": "high"}'
```

## Notes

The CadQuery execution and voxel-IoU scoring are vendored verbatim from the
canonical scorer (`BenchCAD/BenchCAD-main`, `Vision2Code/scoring/`) so this
environment is a self-contained package; that repository remains the source of
truth. Requires the CadQuery / OCP stack (declared in `pyproject.toml`).
