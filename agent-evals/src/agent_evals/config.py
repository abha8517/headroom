"""Configuration surface (pydantic-settings).

Every threshold is config, never a literal in logic. Loads from defaults, env vars
(prefix ``AGENT_EVALS_``, nested delimiter ``__``), and is frozen into the RunManifest.
Example: ``AGENT_EVALS_CONCURRENCY=8``, ``AGENT_EVALS_STATS__K_RUNS=20``.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from .models import Pricing, Provider


class StatsConfig(BaseModel):
    """Statistical-design knobs (used Phase 1+)."""

    k_runs: int = Field(default=10, ge=1)
    alpha: float = Field(default=0.05, gt=0.0, lt=1.0)
    margin_ccr_pp: float = Field(default=0.0, ge=0.0)  # lossless CCR: demand near-parity
    margin_lossy_pp: float = Field(default=2.0, ge=0.0)  # disclosed tolerance for lossy modes
    bootstrap_resamples: int = Field(default=10_000, ge=100)
    seed: int = 12345


class AiderConfig(BaseModel):
    """Aider Polyglot harness knobs (Phase 1)."""

    # Checkout of github.com/Aider-AI/polyglot-benchmark (cloned, not vendored).
    exercises_dir: Path = Path(".cache/polyglot-benchmark")
    language: str = "python"  # python/go/rust/java run locally; js/cpp need Docker (excluded)
    edit_format: str = "whole"  # safe across arbitrary models
    tries: int = Field(default=2, ge=1)  # pass_rate_1 = try 1; pass_rate_2 = within 2 tries
    pytest_timeout_s: float = Field(default=180.0, gt=0.0)
    # Subset selection for cheap smokes / dev. subset_names (exact exercise dir names) wins;
    # else the first ``subset_limit`` exercises in sorted order; None limit = the full set.
    subset_limit: int | None = None
    subset_names: list[str] = Field(default_factory=list)


class ProxyLaunchConfig(BaseModel):
    """How arms.py launches and probes the Headroom proxy. No command is hardcoded in logic."""

    headroom_cmd: list[str] = Field(default_factory=lambda: ["headroom", "proxy"])
    port_range_start: int = Field(default=18800, ge=1024, le=65535)
    port_range_end: int = Field(default=18900, ge=1024, le=65535)
    readyz_path: str = "/readyz"
    stats_path: str = "/stats"
    readyz_timeout_s: float = Field(default=30.0, gt=0.0)
    poll_interval_s: float = Field(default=0.25, gt=0.0)


class Settings(BaseSettings):
    """Top-level resolved settings for an agent-evals run."""

    model_config = SettingsConfigDict(
        env_prefix="AGENT_EVALS_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    provider: Provider = Provider.ANTHROPIC
    model_snapshot: str = "claude-sonnet-4-6"
    # Default pricing is config data (overridable), not a logic constant. Pin per experiment.
    pricing: Pricing = Field(
        default_factory=lambda: Pricing(input_usd_per_1m=3.0, output_usd_per_1m=15.0)
    )
    anthropic_base_url: str = "https://api.anthropic.com"
    openai_base_url: str = "https://api.openai.com/v1"
    concurrency: int = Field(default=4, ge=1)
    # Per-cell rollout timeout (seconds). Frozen into the run for reproducibility; the
    # orchestrator uses this unless an explicit override is passed to its constructor.
    cell_timeout_s: float = Field(default=1800.0, gt=0.0)
    run_dir: Path = Path("./runs")
    log_level: str = "INFO"
    log_json: bool = True
    # Path to a .env carrying provider keys for live runs. None => python-dotenv searches upward
    # (finds the repo-root .env). Keys themselves are never stored in Settings.
    dotenv_path: Path | None = None
    stats: StatsConfig = Field(default_factory=StatsConfig)
    proxy: ProxyLaunchConfig = Field(default_factory=ProxyLaunchConfig)
    aider: AiderConfig = Field(default_factory=AiderConfig)

    def litellm_model_name(self) -> str:
        """The litellm-routed model name for the configured provider (e.g. ``anthropic/<snap>``)."""

        prefix = "anthropic" if self.provider is Provider.ANTHROPIC else "openai"
        return f"{prefix}/{self.model_snapshot}"
