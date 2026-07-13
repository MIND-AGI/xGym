# Decision: integrate cluster terminal sandboxes as an xGym sandbox provider

## Status

Draft framework for the AIIC/StarGaze terminal sandbox migration.

## Context

The AIIC branch has cluster-specific sandbox work that started from a
StarGazeWorkflow-style design. That style tends to combine these concerns in one
workflow:

- terminal session creation and auth;
- agent runtime setup;
- direct model API configuration;
- SWE-bench patch collection and grading.

xGym already has separate abstractions for these concerns:

- `responses_api_agents/*` own `/run`, trajectories, rewards, and model-server
  wiring;
- `responses_api_models/*` own model endpoint access;
- `nemo_gym.sandbox` owns environment creation, command execution, and file
  transfer;
- SWE-bench grading can be reused from `mini_swe_agent_2`.

For SWE-bench Verified, the `mini_swe_agent_2` integration is the better
compatibility target: the mini-swe-agent control loop runs in the xGym
agent/Ray worker, calls xGym's registered `model_server`, and sends shell
commands into a sandbox through `MiniSWESandboxEnvironment`.

## Decision

Model AIIC/StarGaze terminal sessions as a `nemo_gym.sandbox` provider named
`terminal`. Do **not** make the terminal service a separate SWE-bench agent
pipeline. The first-class path is:

```text
xGym eval
  -> responses_api_agents/mini_swe_agent_2 /run
    -> mini-swe-agent DefaultAgent in the agent/Ray worker
      -> xGym model_server /v1
      -> MiniSWESandboxEnvironment
        -> Sandbox(provider={terminal: ...})
          -> cluster terminal session running commands in /testbed
```

The `responses_api_agents/swe_agents_terminal` package is intentionally a thin
preset around `MiniSWEAgent`; its YAML selects the terminal provider. This keeps
agent/model semantics identical to `mini_swe_agent_2` and makes terminal vs.
OpenSandbox a provider swap.

## Boundaries

### Terminal provider owns

- cluster terminal session create/delete;
- readiness probe;
- command execution;
- upload/download;
- auth/JWT/netrc/region config in the concrete client;
- provider retry/timeout semantics.

### Terminal provider must not own

- SWE-bench prompts or reward logic;
- model API endpoints or API keys;
- OpenCode/Claude Code/StarGazeWorkflow agent loops;
- dataset row interpretation beyond generic `SandboxSpec` fields.

### `mini_swe_agent_2` owns

- SWE-bench row handling;
- mini-swe-agent config construction;
- xGym `model_server` base URL selection;
- Responses API trajectory/reward response;
- generic SWE-bench image derivation and optional instance-id image mapping.

## Image mapping

Some clusters cannot use public SWE-bench Docker image refs directly. The
framework adds a provider-neutral optional image map to `mini_swe_agent_2`:

```yaml
image_map_path: /path/to/images.json
require_image_map: true
```

Accepted JSON forms:

```json
{"images": {"django__django-13410": "internal/image:id"}}
```

or a direct object:

```json
{"django__django-13410": "internal/image:id"}
```

Image precedence is:

1. row `sandbox_image`;
2. row `swebench_image`;
3. configured image map by `instance_id`;
4. row `image_name`;
5. built-in SWE-bench Docker naming rule.

Cluster-specific map paths belong in private YAML, not in code.

## OpenCode / StarGazeWorkflow compatibility

OpenCode or Claude Code running inside a terminal session can still be useful,
but it should be a separate experimental agent harness if needed. It should not
be embedded into `mini_swe_agent_2`, because that mixes agent-framework choice
with sandbox-provider choice and risks bypassing xGym `model_server`.

If an inside-sandbox agent is added later, it should still target xGym's
`model_server` endpoint rather than a direct MaaS/OpenAI URL whenever possible.

## Migration notes for the AIIC branch

1. Port the StarGaze terminal HTTP/JWT implementation into a concrete
   `TerminalServiceClient` behind `nemo_gym/sandbox/providers/terminal/provider.py`.
2. Keep the `responses_api_agents/swe_agents_terminal` app as a thin subclass of
   `MiniSWEAgent`.
3. Move StarGaze image-map paths into YAML or dataset metadata; use the generic
   `image_map_path` / `require_image_map` knobs.
4. Avoid adding `agent_framework: opencode` or direct MaaS API settings to
   `mini_swe_agent_2`; split those into a separate agent if still required.
5. Validate the provider with `tests/unit_tests/test_terminal_provider.py`, then
   run focused `mini_swe_agent_2` tests.

## Consequences

Benefits:

- preserves xGym model-server accounting and trajectory semantics;
- avoids exposing model API credentials inside the sandbox;
- makes terminal vs. OpenSandbox a YAML-level provider swap;
- keeps cluster-specific code isolated for easier review and upstreaming.

Trade-offs:

- a concrete terminal HTTP client still has to be implemented for AIIC;
- OpenCode/StarGazeWorkflow behavior is not included in the mini-swe-agent path;
- inside-sandbox agent frameworks require a separate design if they must be
  supported later.
