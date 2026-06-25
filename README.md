# BenchCAD

Three benchmarks for evaluating LLMs on CAD code understanding. All three share one Python environment but each task has its own scripts, data, and result store.

**Scoring is execution-grounded and objective — there is no LLM judge in the loop.**
CodeGen and CodeEdit grade by *voxel IoU* between the model's executed STEP solid
and the ground-truth STEP; CodeQA grades by *symmetric ratio accuracy* on numeric
answers. Scores are deterministic and reproducible, not model-judged.

## Setup

```bash
# Python 3.11, managed by uv
uv sync

# Secrets (LLM API keys only — benchmark data is public)
cp .env.example .env
# Edit .env: paste OPENAI / ANTHROPIC / GEMINI / OPENROUTER keys
```

Real benchmark data is hosted on HuggingFace under [`BenchCAD/BenchCAD`](https://huggingface.co/datasets/BenchCAD/BenchCAD) and pulled into the gitignored `data/` folder. Each task also ships a tiny `test_data/` (≈4 records) committed to the repo for smoke tests without any HF pull.

## One-click reproduction

Smoke run all three benchmarks end-to-end (4 records each, gpt-4o):

```bash
uv sync                              # one-time: install deps
cp .env.example .env && $EDITOR .env # paste your OPENAI_API_KEY
uv run python run_all.py             # → CodeEdit + CodeGen + CodeQA
```

Pick one task instead, or switch to the full bench:

```bash
uv run python run_all.py --task codegen           # just CodeGen, smoke
uv run python run_all.py --config prod            # all three, full HF bench
uv run python run_all.py --task codeqa --plot     # plot existing results
```

`--config <name>` looks up `<Task>/configs/<name>.yaml` per task. `prod`
requires a one-time HF download per task (see each task's README).

## Tasks

| Folder | What it tests | Docs |
|---|---|---|
| `CodeEdit/` | Text instruction → modify an existing CadQuery program | [CodeEdit/README.md](CodeEdit/README.md) |
| `CodeGen/` | Generate CadQuery code from a spec or image | [CodeGen/README.md](CodeGen/README.md) |
| `CodeQA/` | Answer questions about a given CadQuery program | [CodeQA/README.md](CodeQA/README.md) |

## Repo layout

```
BenchCAD/
├── pyproject.toml          shared deps (covers all 3 tasks)
├── .env.example            template for secrets
├── run_all.py              one-click runner across all 3 tasks
├── CodeEdit/               text-edit benchmark
├── CodeGen/                code-generation benchmark (image → CadQuery code)
└── CodeQA/                 code-QA benchmark (CadQuery code → numeric answers)
```

Each task subdir is independently runnable (`cd <Task> && uv run python main.py`)
and has the same shape: `main.py`, `configs/{test,prod}.yaml`, `pipeline/`,
`scoring/`, `models/`, `test_data/`, `tools/download_*.py`, `README.md`.

## Leaderboard (frontier baselines)

Full `prod` results on the public split. Numbers are re-graded from submitted
predictions, never accepted as self-reported — see
[.github/ISSUE_TEMPLATE/model_result.md](.github/ISSUE_TEMPLATE/model_result.md)
to submit.

| Model | CodeGen (IoU) | CodeEdit (norm IoU) | CodeQA (ratio acc) |
|---|---|---|---|
| _Claude Opus 4.x_ | _TODO_ | _TODO_ | _TODO_ |
| _GPT-5.x_ | _TODO_ | _TODO_ | _TODO_ |
| _Gemini 2.x_ | _TODO_ | _TODO_ | _TODO_ |

> Baselines are being populated. To reproduce: `uv run python run_all.py --config prod`.

## License & citation

- **Code** (this repo) — [MIT](LICENSE).
- **Data** ([BenchCAD on HuggingFace](https://huggingface.co/datasets/BenchCAD/BenchCAD)) — CC-BY-4.0.

If you use BenchCAD, please cite it (see [CITATION.cff](CITATION.cff)):

```bibtex
@article{zhang2026benchcad,
  title  = {BenchCAD: Benchmarks for Evaluating LLMs on CAD Code Understanding},
  author = {Zhang, Haozhe and Li, Lei and Peng, Cheng and Chen, Hanjie},
  year   = {2026}
}
```
