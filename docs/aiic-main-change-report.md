# AIIC branch change report vs `main`

## Compared refs

- Base branch: `origin/main` at `dc9be35` (`fix: allow Claude Code unlimited turns`)
- Feature branch head: `aiic` at `1814208` (`feat: add terminal sandbox SWE-bench support`)
- Merge base: `dc9be355da83ec88b4c89f711a2bdc7c0f23cbda`
- Working-tree diff after cleanup: `22 files changed, 7881 insertions(+), 89 deletions(-)`

> Note: local `main` is stale in this checkout, so this report uses `origin/main` as the current main reference.

## Executive summary

The cleaned `aiic` branch is focused on AIIC/StarGaze terminal sandbox support and OpenCode SWE-bench execution:

1. Adds a `terminal` sandbox provider that adapts cluster terminal sessions to the xGym sandbox interface.
2. Extends the sandbox API for long-running command execution and provider-neutral config handling needed by terminal sessions.
3. Adds SWE-bench terminal presets and direct OpenCode runners for SWE-bench Verified / SWE-bench Pro experiments.
4. Extends `mini_swe_agent_2` so SWE-bench tasks can run against terminal sandboxes and, when configured, OpenCode inside those sandboxes.
5. Adds focused unit coverage for terminal provider behavior and SWE-bench progress monitoring.

Unrelated EvalPlus MaaS replay/debug scripts, GDPVal/Bunsen/Tau2 cleanup, generic inference-provider streaming changes, and chat-transcript planning artifacts were de-scoped from the cleaned branch.

## File-level change inventory

### Sandbox API and provider infrastructure

- `nemo_gym/sandbox/api.py`
  - Adds `AsyncSandbox.exec_long()` / `Sandbox.exec_long()` path for commands whose backend can stream or heartbeat long-running terminal work.
  - Adjusts synchronous operation waiting so caller timeouts can include a grace period instead of being capped by a fixed one-hour wrapper timeout.

- `nemo_gym/sandbox/providers/base.py`
  - Adds shared provider config coercion used by concrete providers.

- `nemo_gym/sandbox/providers/apptainer/provider.py`
- `nemo_gym/sandbox/providers/opensandbox/provider.py`
  - Replace local config coercion helpers with the shared base helper. This is intentionally small and provider-neutral.

- `nemo_gym/sandbox/providers/registry.py`
  - Registers the new `terminal` provider.

### AIIC/StarGaze terminal sandbox provider

- `nemo_gym/sandbox/providers/terminal/__init__.py`
- `nemo_gym/sandbox/providers/terminal/provider.py`

The new provider implements cluster terminal sessions as xGym sandboxes:

- terminal session create/delete;
- StarGaze/AIIC auth and token helpers;
- region and config handling;
- session readiness probing;
- command execution and long command execution;
- upload/download with native and fallback paths;
- retry classification for auth, session creation, and command execution;
- process/session admission locks to reduce terminal service pressure under high concurrency;
- provider metadata/config dataclasses suitable for Hydra/YAML wiring.

### SWE-bench terminal / OpenCode adapter

- `responses_api_agents/mini_swe_agent_2/app.py`
  - Adds provider-neutral SWE-bench image-map support.
  - Adds terminal-sandbox runtime environment handling.
  - Adds optional `agent_framework: opencode` path for running OpenCode in the sandbox while preserving xGym task/result plumbing.
  - Adds patch extraction, opencode setup, infrastructure-error classification, and progress recording hooks used by direct runners.

- `responses_api_agents/mini_swe_agent_2/tests/test_app.py`
  - Adds focused coverage for image-map selection, OpenCode config generation, and terminal/OpenCode helper behavior.

- `responses_api_agents/swe_agents_terminal/__init__.py`
- `responses_api_agents/swe_agents_terminal/app.py`
  - Adds a thin `swe_agents_terminal` agent package over `MiniSWEAgent` for terminal-sandbox SWE-bench presets.

- `responses_api_agents/swe_agents_terminal/configs/swebench_terminal.yaml`
  - Generic terminal-sandbox SWE-bench preset.
  - Keeps model calls through xGym's configured model server and uses terminal only for sandbox command execution.

- `responses_api_agents/swe_agents_terminal/configs/swebench_terminal_opencode_latest.yaml`
  - OpenCode-on-terminal preset.
  - Sanitized during cleanup: no hard-coded API key and no user-specific absolute image-map path. Private launch YAML or Hydra overrides should provide private image maps and non-xGym model endpoints.

- `responses_api_agents/swe_agents_terminal/requirements.txt`
  - Adds terminal SWE-bench runner dependency surface.

### Direct evaluation helpers

- `responses_api_agents/swe_agents_terminal/direct_opencode_eval.py`
  - Direct SWE-bench Verified terminal/OpenCode runner for operational experiments outside full server startup.
  - Sanitized during cleanup: no default API key and no default private MaaS URL.

- `responses_api_agents/swe_agents_terminal/direct_opencode_swebench_pro_eval.py`
  - Direct SWE-bench Pro terminal/OpenCode runner.
  - Tracks sandbox infrastructure failures separately from valid unresolved model outcomes.
  - Defaults to repo-relative image-map and official-assets paths; private paths should be passed via CLI.
  - Sanitized during cleanup: no default private MaaS URL.

- `responses_api_agents/swe_agents_terminal/reproduce_swebench_verified.py`
  - Helper to prepare and combine SWE-bench Verified reproduction inputs/results.
  - Defaults to repo-relative image-map path; private path should be passed via CLI.

- `responses_api_agents/swe_agents_terminal/monitor_swebench_progress.py`
  - Builds progress snapshots from rollout/results directories.
  - Tracks completed, running, resolved, unresolved, and infra-failure categories.

### Tests

- `tests/unit_tests/test_terminal_provider.py`
  - Focused tests for terminal provider config, auth/client behavior, retries, file transfer, locks, and command execution behavior.

- `tests/unit_tests/test_swebench_progress_monitor.py`
  - Focused tests for SWE-bench progress aggregation.

### Documentation and generated artifacts policy

- `docs/architecture/decisions/terminal-sandbox-provider-for-swebench.md`
  - Captures the architecture boundary: terminal service is a sandbox provider, not a model or SWE-bench agent loop.
  - Documents why `mini_swe_agent_2` remains the compatibility target and how OpenCode should stay separate/configurable.

- `.gitignore`
  - Adds ignores for private terminal sandbox config and generated SWE-bench terminal rollout data/results.

## De-scoped / removed technical debt

The cleanup intentionally removes these categories from the branch narrative:

- EvalPlus MaaS/OpenCode debug artifacts:
  - `scripts/capture_opencode_chat_payload.py`
  - `scripts/replay_maas_evalplus_prompts.py`
  - `scripts/replay_maas_opencode_exact_evalplus.py`
  - `scripts/run_opencode_evalplus_prompts.py`
  - `scripts/verify_opencode_evalplus_results.py`
  - `RUN.md`

- Chat/transcript planning artifact:
  - `Opencode_swe_rl_impl.md`

- Separate AnyTerminal terminal harness that is not required for the terminal sandbox provider / OpenCode SWE-bench path:
  - `responses_api_agents/anyterminal_agent_terminal/*`

- Unrelated benchmark/server cleanup:
  - Bunsen chemistry MCQ changes
  - Tau2 prepare-utils change
  - GDPVal app/scoring cleanup
  - EvalPlus app/runner/test adjustments
  - generic inference-provider streaming changes
  - `nemo_gym/openai_utils.py` raw chat-completion helper
  - `CLAUDE.md` local environment instruction change

## Security and portability cleanup

- Removed committed `sk-1234` defaults from terminal/OpenCode configs and direct runners.
- Removed user-specific absolute image-map defaults from committed YAML and helper defaults.
- Remaining `maas.byteintl.net` references are in unit tests as example provider config assertions, not committed launch defaults.
- Private image maps, private MaaS URLs, and API keys should be supplied through environment variables, private launch YAML, or Hydra/CLI overrides.

## Known limitations / follow-ups

- The terminal provider is cluster-specific and still depends on AIIC/StarGaze service availability and credentials.
- `direct_opencode_*` runners are operational helpers, not the primary xGym server path; they should remain isolated from core library APIs.
- The `mini_swe_agent_2` OpenCode path increases the size of that agent module; a future cleanup could split OpenCode-specific helpers into a separate module while keeping public behavior unchanged.
- Full end-to-end SWE-bench validation depends on external terminal sandbox and model infrastructure; unit tests only validate local logic and provider behavior with fakes/mocks.

## Validation

Focused validation run after cleanup:

```bash
git diff --check
python -m py_compile \
  responses_api_agents/swe_agents_terminal/direct_opencode_eval.py \
  responses_api_agents/swe_agents_terminal/direct_opencode_swebench_pro_eval.py \
  responses_api_agents/swe_agents_terminal/reproduce_swebench_verified.py \
  responses_api_agents/swe_agents_terminal/monitor_swebench_progress.py
UV_PROJECT_ENVIRONMENT="$HOME/.venvs/xgym" uv run pytest \
  tests/unit_tests/test_terminal_provider.py \
  tests/unit_tests/test_swebench_progress_monitor.py \
  responses_api_agents/mini_swe_agent_2/tests/test_app.py \
  -x
```

Result:

- `git diff --check`: passed.
- `py_compile` for changed direct/helper scripts: passed.
- Focused pytest: `47 passed, 5 warnings in 9.15s`.

Warnings were existing/dependency-style warnings from `requests`, `fastapi.testclient`, and Pydantic serialization in tests; no test failed.

If a signed cleanup commit is made, re-run the same focused validation after committing.
