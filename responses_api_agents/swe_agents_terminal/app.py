# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Terminal-sandbox SWE-bench agent.

This entrypoint deliberately does not import or modify the apptainer-backed
``responses_api_agents.swe_agents`` implementation. It reuses the
provider-neutral mini-swe-agent v2 harness and selects the ``terminal`` sandbox
provider from config.
"""

from responses_api_agents.mini_swe_agent_2.app import (
    MiniSWEAgent,
    MiniSWEAgentConfig,
    MiniSWEAgentRunRequest,
    MiniSWEAgentVerifyRequest,
    MiniSWEAgentVerifyResponse,
    run_mini_swe_with_sandbox,
)


class SWETerminalAgentConfig(MiniSWEAgentConfig):
    """Config alias for the terminal sandbox SWE-bench entrypoint."""


class SWETerminalAgent(MiniSWEAgent):
    """mini-swe-agent v2 running on the StarGaze terminal sandbox provider."""

    config: SWETerminalAgentConfig


__all__ = [
    "MiniSWEAgentRunRequest",
    "MiniSWEAgentVerifyRequest",
    "MiniSWEAgentVerifyResponse",
    "SWETerminalAgent",
    "SWETerminalAgentConfig",
    "run_mini_swe_with_sandbox",
]


if __name__ == "__main__":
    SWETerminalAgent.run_webserver()
