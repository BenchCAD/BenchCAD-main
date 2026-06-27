<!-- Thanks for contributing to BenchCAD! -->

## What this changes

<!-- One or two sentences. Which task (Vision2Code / CodeEdit / QA) or shared infra? -->

## Type

- [ ] Bug fix (scoring, execution, data loading)
- [ ] New / updated benchmark task or data
- [ ] Tooling / CI / docs
- [ ] Model result submission (see below)

## Checklist

- [ ] `uv run ruff check .` passes
- [ ] `uv run pytest -q` passes
- [ ] If I changed scoring or execution logic, I added/updated a test that fails
      before the change and passes after
- [ ] I did **not** commit any benchmark ground-truth answers into the repo
      (held-out test answers stay out of git / public data)

## Model result submission (fill in only if submitting leaderboard numbers)

- Model + version:
- Config used (`prod` / custom):
- Link to raw predictions + logs (required — numbers are re-graded, not trusted as reported):
