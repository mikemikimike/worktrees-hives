# Research Hive v0 experiment contract design

## Scope

GitHub #92 adds a machine-readable, pre-execution research contract. The
contract records what an experiment intends to test before code changes or runs
begin. It is a Python domain document and does not add orchestration, execution,
verification, benchmarking, result assembly, CLI behavior, or Rust code.

## Architecture

Add a `worktrees_hives.research` module following the existing findings-domain
pattern:

- a version constant checked by exact equality;
- a frozen, slotted `ResearchContract` dataclass;
- an exact four-value `ResearchOutcome` string enum containing `supported`,
  `not_supported`, `inconclusive`, and `invalid`;
- JSON parsing and serialization helpers;
- a dedicated `ResearchValidationError`.

`ResearchOutcome` is deliberately separate from `ResearchContract`. The
contract describes an experiment before execution, while an outcome belongs to
a later result document owned by subsequent Research Hive issues.

JSON is the only canonical input and output format. No dependency is added.

## Contract fields and validation

Validation distinguishes field presence from non-empty scientific content. It
does not infer experimental-policy rules beyond #92.

| Field | Presence | Validation |
| --- | --- | --- |
| `schema_version` | Required | Integer (not Boolean), exactly version 1 |
| `research_id` | Required | Non-empty string |
| `question` | Required | Non-empty string |
| `hypothesis` | Required | Non-empty string |
| `null_hypothesis` | Optional | String when present |
| `independent_variable` | Required | Non-empty string |
| `dependent_metrics` | Required and non-empty | Array of non-empty strings |
| `baselines` | Required; may be empty | Array of non-empty strings |
| `arms` | Required; may be empty | Array of non-empty strings |
| `acceptance_criteria` | Required and non-empty | Array of non-empty strings |
| `failure_criteria` | Required; may be empty | Array of non-empty strings |
| `resource_budget` | Optional | JSON object when present; no invented budget keys or limits |
| `seed_policy` | Required; object may be empty | JSON object; no invented seed strategy requirements |
| `split_policy` | Optional | JSON object when applicable; no invented split requirements |
| `repo` | Required | Non-empty string; no allowlist or repository-existence check |
| `ref` | Required | Non-empty string; no ref-existence check |
| `artifact_expectations` | Required; may be empty | Array of non-empty strings |

All modeled arrays become tuples. JSON objects retained by the model become
read-only mappings, recursively freezing nested arrays and objects. JSON numbers
must be finite. Validation rejects malformed JSON, missing required fields,
incorrect field types, invalid collection members, unsupported versions, and
non-finite numbers with actionable errors.

Empty `baselines`, `arms`, `failure_criteria`, and `artifact_expectations` are
valid because #92 requires their representation but does not require non-empty
content. `resource_budget` and `split_policy` remain optional as the issue
explicitly permits lightweight or non-dataset experiments.

## Acceptance-criteria freeze

Construction and parsing copy and freeze acceptance criteria along with all
other collections. Callers cannot mutate a validated contract through either
the source object or returned attributes. Later execution code must accept a
validated contract before starting a run; implementing that sequence is outside
the scope of GitHub #92.

## Envelope compatibility and additive fields

The research contract is a nested domain document, not a second transport
protocol. This abbreviated shape illustrates placement only; the nested object
omits required fields and is not a complete contract fixture:

```json
{
  "ok": true,
  "schema_version": 1,
  "command": "<existing or future lab command>",
  "data": {
    "research_contract": {
      "schema_version": 1,
      "research_id": "structured-contract-first-pass-evidence-v1"
    }
  },
  "error": null
}
```

The outer `schema_version` versions the Python/Rust transport envelope. The
nested `data.research_contract.schema_version` versions the research domain
document. Either can evolve independently.

Unknown top-level research-contract fields are accepted, recursively frozen,
and re-emitted during serialization. Therefore parsing and serializing a future
additive v1 document does not silently discard extension data. Known fields
remain authoritative and cannot be shadowed by extensions. Breaking changes to
known field meaning or type require a research-contract version bump.

## Fixture and tests

Add a realistic JSON fixture for a cloud coding-agent experiment. It tests
whether a frozen, structured pre-execution contract improves first-pass
evidence quality over issue text alone, without an unacceptable token-cost
increase. It includes a baseline and treatment arm, quality and cost metrics,
explicit criteria, run/time/token budgets, deterministic seeds, task splits, a
repository ref, and expected artifacts. It does not model local GPU thermals,
because the agents under study are cloud-hosted models.

Focused unit tests cover:

- dictionary and JSON round trips;
- the realistic fixture;
- exact outcome vocabulary;
- the field-presence/non-empty matrix above;
- invalid collection members and non-finite numbers, including nested values;
- deep immutability and defensive copying;
- preservation of unknown additive fields;
- separation between transport and domain schema versions.

The full relevant Python test, lint, type-check, and formatting checks run before
the implementation commit.

## Out of scope

No part of #93 or later Research Hive work is included: no role declarations,
scheduling, run lifecycle, independent verification, benchmark policy, artifact
bundle assembly, result schema, result-to-contract evaluation, or CLI command.
