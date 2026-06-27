---
name: Model result submission
about: Report a model's BenchCAD scores for the leaderboard
title: "[result] <model name>"
labels: leaderboard
---

**Model**: <name + version + provider>

**Scores** (per task, full `prod` config)

| Task | Metric | Score |
|---|---|---|
| Vision2Code | voxel IoU | |
| CodeEdit | normalized IoU | |
| QA | ratio accuracy | |

**Reproducibility**
- Config / commit SHA:
- Link to raw predictions + run logs (**required** — all leaderboard numbers are
  re-graded from submitted predictions, not accepted as self-reported):
- Any non-default harness settings:
