"""Deployment files stay in sync with the settings and keep their safety properties."""

import re
from pathlib import Path

import pytest

from app.config import Settings

ROOT = Path(__file__).resolve().parent.parent
DOCKERFILE = (ROOT / "Dockerfile").read_text(encoding="utf-8")
ENV_EXAMPLE = (ROOT / ".env.example").read_text(encoding="utf-8")
SETTING_NAMES = {name.upper() for name in Settings.model_fields}


def env_example_keys(include_commented: bool) -> set[str]:
    pattern = r"^#?\s*([A-Z_]+)=" if include_commented else r"^([A-Z_]+)="
    return set(re.findall(pattern, ENV_EXAMPLE, flags=re.MULTILINE))


def render_service() -> dict:
    yaml = pytest.importorskip("yaml")
    blueprint = yaml.safe_load((ROOT / "render.yaml").read_text(encoding="utf-8"))
    (service,) = blueprint["services"]
    return service


def test_env_example_lists_every_setting() -> None:
    assert env_example_keys(include_commented=True) == SETTING_NAMES


def test_env_example_has_no_key_value() -> None:
    assert re.search(r"^OPENAI_API_KEY=$", ENV_EXAMPLE, flags=re.MULTILINE)
    assert "sk-" not in ENV_EXAMPLE


def test_env_example_loads_with_defaults() -> None:
    loaded = Settings(_env_file=ROOT / ".env.example")  # type: ignore[call-arg]
    defaults = Settings(_env_file=None)
    for name in Settings.model_fields:
        if name not in ("openai_api_key", "cache_dir"):
            assert getattr(loaded, name) == getattr(defaults, name), name


def test_dockerfile_runs_as_non_root() -> None:
    assert "useradd --create-home --uid 1000 appuser" in DOCKERFILE
    assert re.search(r"^USER appuser$", DOCKERFILE, flags=re.MULTILINE)


def test_dockerfile_is_production_with_one_worker_on_port_env() -> None:
    assert "ENVIRONMENT=production" in DOCKERFILE
    command = DOCKERFILE[DOCKERFILE.index("\nCMD ") :]
    for part in ('${PORT:-8000}', "--workers 1", "--no-proxy-headers", "exec uvicorn app.main:app"):
        assert part in command, part


def test_dockerfile_has_a_healthcheck_on_health() -> None:
    assert re.search(r"^HEALTHCHECK .*\\\n\s+CMD python .*/health", DOCKERFILE, flags=re.MULTILINE)


def test_render_blueprint() -> None:
    service = render_service()
    assert service["runtime"] == "docker" and service["healthCheckPath"] == "/health"
    env = {item["key"]: item for item in service["envVars"]}
    assert env["OPENAI_API_KEY"] == {"key": "OPENAI_API_KEY", "sync": False}
    assert env["ENVIRONMENT"]["value"] == "production"
    assert env["TRUSTED_PROXY_HOPS"]["value"] == "1"
    assert set(env) <= SETTING_NAMES


def test_render_values_are_valid_settings() -> None:
    values = {item["key"].lower(): item["value"] for item in render_service()["envVars"] if "value" in item}
    loaded = Settings(_env_file=None, **values)  # type: ignore[arg-type]
    assert loaded.environment == "production"
    assert loaded.frame_ancestors == "'none'"
