# AI Coding-Agent Harness

**An autonomous verification framework that holds AI coding agents accountable.**

Rather than trust an AI agent's self-reported success, this harness compiles
every coding issue into a formal contract, enforces a bounded change scope,
verifies results through executed evidence, and actively attempts to falsify
the solution before declaring it correct.

---

## 1. Problem

AI coding agents can produce patches that *appear* successful while still
violating requirements, missing edge cases, or modifying unrelated code.
A passing test suite does not prove the task was completed correctly — the
agent may have weakened assertions, deleted inconvenient tests, or changed
files outside the scope of the issue.  Without structured verification,
"the agent said it's done" is the only signal an evaluator receives.

## 2. Solution

The harness replaces self-reported success with a pipeline of executable
evidence.  Every issue passes through a deterministic sequence of stages:

```
Issue
  → Contract           Compile the issue into structured proof obligations
  → Impact Map         Identify affected files and dependencies
  → Change Budget      Cap the number and scope of permitted modifications
  → Patch              Apply code changes (via the agent / model)
  → Verification       Multi-level executed checks (syntax → tests → contract)
  → Falsification      Actively try to break the solution with adversarial inputs
  → Repair             Classify failures and attempt targeted fixes
  → Fresh Evidence     Reject stale results; reverify after any repair
  → Final Proof        Obligation-by-obligation proof report
```

The model is **never** allowed to declare the task complete by itself.
The final status (`VERIFIED`, `FAILED`, or `UNKNOWN`) is determined solely
by executed verification evidence.

## 3. Core Differentiators

The following capabilities are each well-known in isolation.  The novelty
of this harness is their **integration into a single autonomous workflow**
where no human intervention is required between issue submission and the
final proof report.

| Capability                  | What it does |
|-----------------------------|-------------|
| **Issue-to-Contract Compiler** | Converts a natural-language issue into a structured `TaskContract` with typed proof obligations, acceptance criteria, constraints, non-goals, and a verification plan. |
| **Proof Obligations**          | Each contract produces individually trackable obligations (`PENDING → PASS / FAIL / UNKNOWN`).  No obligation is marked PASS simply because another test passed. |
| **Change Budget**              | Before any modification, the harness computes the maximum number of files and directories the agent is allowed to touch, derived from the contract's risk level and expected scope. |
| **Blast Radius Detection**     | After modification, a guard compares actual changes against the budget and flags unexpected files, suspicious test modifications, weakened assertions, and out-of-scope edits. |
| **Active Falsification**       | After verification passes, a dedicated falsifier generates bounded adversarial test cases (boundary values, null inputs, off-by-one errors, type variations) and executes them.  `NOT_FALSIFIED` means "no counterexample was found within the bounded checks performed" — never "mathematically proven correct." |
| **Evidence Graph**             | A directed graph links requirements → obligations → code locations → changes → verification checks → evidence → falsification results.  Gaps in the graph prevent a VERIFIED verdict. |
| **Evidence Freshness**         | Every piece of evidence carries deterministic file fingerprints (SHA-256).  If any file changes after evidence was collected, that evidence is marked `STALE` and the harness forces reverification. |
| **Failure-aware Repair**       | When verification fails, the harness classifies the failure type (bad patch, regression, scope violation, missing context, etc.) and selects a targeted repair strategy — not a blind retry. |
| **Safe Rollback**              | Checkpoints are captured before modification.  Rollback restores only harness-modified files; pre-existing user files are never touched. |
| **Honest UNKNOWN State**       | If the harness cannot establish success or failure — budget exhausted, oscillating fixes, stuck pattern — the outcome is `UNKNOWN`, not a false `VERIFIED`. |

## 4. Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          ORCHESTRATOR                              │
│  Autonomous loop: Contract → Execute → Verify → Falsify → Repair  │
├─────────────┬──────────────┬──────────────┬─────────────┬──────────┤
│  CONTRACT   │   CHANGE     │  VERIFIER    │ FALSIFIER   │ EVIDENCE │
│  ENGINE     │   GUARD      │  (6 levels)  │             │ GRAPH    │
│             │              │              │             │          │
│ Issue →     │ Budget →     │ L1 Syntax    │ Boundary    │ Req →    │
│ Contract →  │ Scope check →│ L2 Existing  │ Null/empty  │ Oblig →  │
│ Obligations │ Blast radius │ L3 Targeted  │ Invalid     │ Code →   │
│             │ detection    │ L4 Contract  │ Off-by-one  │ Change → │
│             │              │ L5 Regression│ Type vars   │ Check →  │
│             │              │ L6 Falsif.   │ Error paths │ Evidence │
├─────────────┴──────────────┴──────────────┴─────────────┴──────────┤
│  ROLLBACK ENGINE          │  CONTEXT MANAGER   │  METRICS/CACHE    │
│  Checkpoint / restore     │  Repo map, search  │  Dedup, budget    │
├───────────────────────────┴────────────────────┴───────────────────┤
│  TOOLS LAYER                                                       │
│  file_ops · shell · search · risk (path restrictions, redaction)   │
├────────────────────────────────────────────────────────────────────┤
│  MODEL CLIENT (Anthropic / Claude)                                 │
│  Text-only · retry with backoff · rate-limit handling              │
└────────────────────────────────────────────────────────────────────┘
```

## 5. Installation

```bash
git clone https://github.com/saarthxkk/ai_harness.git
cd ai_harness
make setup
```

`make setup` installs all Python dependencies from `requirements.txt`:

- `pyyaml` — configuration parsing
- `anthropic` — LLM API client
- `pytest` — test framework

No additional system packages are required.

## 6. Configuration

Edit `config.yaml` to set the model, iteration limits, and falsification
parameters:

```yaml
model:
  name: "claude-sonnet-4-20250514"

runtime:
  max_iterations: 10
  max_tokens: 4096

falsification:
  max_falsification_cases: 50
  max_falsification_time: 300
  max_attempts_per_obligation: 10
  command_timeout: 30
```

### API Key

The harness requires an `AI_API_KEY` environment variable.  This is the
API key for the Anthropic Claude model.

```bash
export AI_API_KEY='your-key-here'
```

**Never commit an API key to the repository.**  The `.env` file is
listed in `.gitignore`.  The harness actively redacts any API key that
appears in error messages.

## 7. Running

```bash
export AI_API_KEY='your-key-here'
make run
```

The harness presents an interactive terminal where you can enter a
coding issue in natural language and watch the full pipeline execute.

If `AI_API_KEY` is not set, the harness prints a clear error and exits
with code 1.

### Demo Mode

```bash
make demo
```

Runs a **deterministic 60-second demonstration** that executes a
complete pipeline against a real temporary repository — no API key
required.  The demo shows contract compilation, verification,
active falsification discovering a hidden counterexample, repair,
re-verification, and final `VERIFIED` status.  All test results are
produced by real `pytest` execution.

## 8. Testing

```bash
make test
```

Runs the full test suite (734 tests) with `pytest`.  All tests are
offline — no API key or network access is required for testing.

## 9. Cleaning

```bash
make clean
```

Removes `__pycache__/` and `.pytest_cache/` directories.

## 10. Example

Run `make demo` to watch the full pipeline live.  Here is the key
sequence it demonstrates:

```
ISSUE
  "Add a clamp(value, lo, hi) function to utils.py"

CONTRACT
  → OB-1: Returns lo when value < lo                       [PENDING]
  → OB-2: Returns hi when value > hi                       [PENDING]
  → OB-3: Returns value when lo ≤ value ≤ hi               [PENDING]
  → OB-4: Postcondition: lo ≤ result ≤ hi always holds     [PENDING]
  → OB-5: No regression in existing utils functions        [PENDING]

VERIFICATION
  ✓ Syntax         PASS
  ✓ Existing tests  PASS  (5 passed)
  ✓ New tests       PASS  (6 passed)
  ✓ Regression      PASS

  ┌──────────────────────────────────────────────────┐
  │  All tests pass.                                 │
  │  A standard CI pipeline would stop here.         │
  │  Our harness does not.                           │
  └──────────────────────────────────────────────────┘

ACTIVE FALSIFICATION
  ⚡ Testing postcondition: lo ≤ result ≤ hi
  ⚡ Testing: clamp(5, 10, 0)  ← lo > hi

  ✗ COUNTEREXAMPLE FOUND
    Input:    clamp(5, 10, 0)
    Expected: ValueError
    Got:      returned 10
    Violated: OB-4

REPAIR
  → Fix: add input validation (lo > hi → ValueError)

RE-VERIFICATION     ✓ PASS
RE-FALSIFICATION    ✓ No counterexample found
EVIDENCE            ✓ Fresh

FINAL STATUS: VERIFIED
  All obligations verified by executed evidence.
```

## 11. Security

The harness enforces several safety constraints:

- **No hardcoded credentials.**  The API key is read exclusively from the
  `AI_API_KEY` environment variable.  The `.env` file is gitignored.
- **Path restrictions.**  The tools layer resolves all file paths relative
  to the repository root and rejects traversal attempts (`../`).
- **Command safety.**  Shell commands are executed with timeouts and
  restricted to the repository working directory.
- **Secret redaction.**  If an API key accidentally appears in an error
  message or stack trace, it is replaced with `[REDACTED]` before output.
- **Bounded execution.**  The orchestrator enforces a maximum iteration
  count (configurable in `config.yaml`) to prevent runaway loops.
- **Rollback.**  Failed or scope-violating modifications can be restored
  to a checkpoint state.  Only harness-modified files are affected.

## 12. Limitations

This harness is a **verification framework**, not a formal proof system.
Honest limitations include:

- **Bounded falsification.**  The falsifier generates a finite set of
  adversarial test cases.  `NOT_FALSIFIED` means no counterexample was
  found within those bounds — it does not mean correctness is guaranteed.
- **Heuristic dependency analysis.**  Impact mapping uses filename,
  import, and text-based heuristics.  It may miss dynamic dependencies
  or runtime-only coupling.
- **Repository-specific test quality.**  Verification quality depends
  on the quality of the target repository's existing test suite.  A
  codebase with no tests provides weaker verification signals.
- **No mathematical proof of arbitrary software.**  The harness verifies
  by *executing* evidence, not by constructing formal mathematical proofs.
  It cannot prove the absence of all possible bugs.

## 13. Evaluation

An evaluator can reproduce the full setup and test cycle with no
team-specific commands:

```bash
git clone https://github.com/saarthxkk/ai_harness.git
cd ai_harness
make setup
make test
```

To see a deterministic end-to-end demonstration (no API key needed):

```bash
make demo
```

To run the interactive harness:

```bash
export AI_API_KEY='your-key-here'
make run
```

All 734 tests pass offline with no API key required.  The `make run`
command requires a valid `AI_API_KEY` to communicate with the Claude API.

## Project Structure

```
├── Makefile                # Build & run targets
├── config.yaml             # Model and runtime configuration
├── requirements.txt        # Python dependencies
├── .env.example            # Template for environment variables
├── src/
│   ├── main.py             # Entry point and terminal UI
│   ├── demo.py             # Deterministic hackathon demo (no API key)
│   ├── model_client.py     # LLM API client (Anthropic / Claude)
│   ├── orchestrator.py     # Central autonomous pipeline loop
│   ├── contract_engine.py  # Issue → TaskContract compiler
│   ├── verifier.py         # Multi-level verification (6 layers)
│   ├── falsifier.py        # Active falsification engine
│   ├── change_guard.py     # Change budget and blast radius guard
│   ├── rollback.py         # Checkpoint and safe rollback
│   ├── evidence_graph.py   # Evidence freshness and graph
│   ├── context_manager.py  # Repository intelligence and search
│   ├── metrics.py          # Execution metrics and caching
│   └── tools/
│       ├── file_ops.py     # File read/write operations
│       ├── shell.py        # Shell command execution
│       ├── search.py       # Code/file search
│       ├── risk.py         # Path restriction and safety
│       └── schemas.py      # Tool schemas and dispatch
└── tests/
    ├── conftest.py
    ├── test_harness.py
    ├── test_contract_engine.py
    ├── test_verifier.py
    ├── test_falsifier.py
    ├── test_change_guard.py
    ├── test_rollback.py
    ├── test_evidence_graph.py
    ├── test_orchestrator.py
    ├── test_context_manager.py
    ├── test_metrics.py
    ├── test_tools.py
    └── test_e2e.py
```

## License

MIT
