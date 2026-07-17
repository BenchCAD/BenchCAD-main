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
the first N parts (handy for quick runs; omit for the full set).

### Two independent timeouts — don't confuse them

**Response timeout** — how long the model is allowed to take to answer. Reasoning
models often think for a while before responding, and a short client timeout cuts
them off. This lives on the **rollout's OpenAI client, not the environment** — set
it there, generously, for slow reasoning models:

```python
from openai import AsyncOpenAI
import verifiers as vf

client = AsyncOpenAI(timeout=3600)   # 1 h — so slow reasoning isn't cut off
env = vf.load_environment("benchcad-vision2code", num_examples=100)
# ...evaluate the env with `client`. The other generation knobs — max_tokens,
# reasoning_effort — are sampling args (`vf-eval -S '{...}'`), also client-side.
```

**Execution timeout** (`exec_timeout`, default 300 s) — a *separate* knob that
only bounds the CadQuery→STEP subprocess runs used to **score** a rollout (the
model's program and the ground truth), not the model's response. It's a
`load_environment` argument:

```python
env = vf.load_environment("benchcad-vision2code", exec_timeout=600)
```

Raise it for slow-tessellating families on slower hardware.

## Notes

The CadQuery execution and voxel-IoU scoring are vendored verbatim from the
canonical scorer (`BenchCAD/BenchCAD-main`, `Vision2Code/scoring/`) so this
environment is a self-contained package; that repository remains the source of
truth. Requires the CadQuery / OCP stack (declared in `pyproject.toml`).
