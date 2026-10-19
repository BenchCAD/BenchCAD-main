# tolerance_stack

Two-block axial clearance specimen with coaxial fastening bores and optional relief pockets; QA covers clearance and dimensional relationships.

Declared seeds are 0–49 for each of easy, medium and hard: 150 deterministic
CadQuery source variants. The examples use seed 0 for easy and hard. Difficulty
labels denote relative geometric feature complexity, not measured human or model
difficulty. No new dependency beyond the upstream CadQuery/OCP stack is required.

Each example has 12 numeric questions, with matching qa/<stem>.json and
examples/<stem>/qa.json. Source-operation questions are code-conditioned;
engineering conditions that cannot be inferred from the geometry are stated.
Edits use seed 0 of the indicated prototype difficulty, two per difficulty.
All edits are T1 literal replacement; no T2–T5 coverage is claimed.

Geometry is independently authored; it is not an exact reconstruction of the cited design.
Source attribution is in SOURCES.txt. The new contribution code is MIT-licensed;
see LICENSE. No manufacturing fitness, stress/contact or load validation is claimed.

The shared design module also contains the attributed MIT cycloidal-profile formula. Its upstream notice is included in this directory even when this family does not emit that template.

Validation coverage and open review items: [engineering-family validation](../../docs/contributions/engineering_families_v5.md).
