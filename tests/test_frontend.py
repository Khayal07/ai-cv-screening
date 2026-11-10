from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.main import app
from app.samples import load_samples, make_label


@pytest.fixture
def client() -> Iterator[TestClient]:
    yield TestClient(app)
    app.dependency_overrides.clear()


def use_samples_dir(directory: Path) -> None:
    app.dependency_overrides[get_settings] = lambda: Settings(
        _env_file=None, samples_dir=directory
    )


def test_index_serves_html(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "<title>" in response.text


def test_stylesheet_is_served(client: TestClient) -> None:
    response = client.get("/static/style.css")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/css")


def test_script_is_served(client: TestClient) -> None:
    response = client.get("/static/app.js")
    assert response.status_code == 200
    assert "javascript" in response.headers["content-type"]


def test_unknown_static_file_is_404(client: TestClient) -> None:
    assert client.get("/static/missing.js").status_code == 404


def test_api_routes_unchanged(client: TestClient) -> None:
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/health", "/match", "/match/pdf", "/samples"} <= set(paths)
    assert "/" not in paths


def test_samples_returns_synthetic_data(client: TestClient) -> None:
    body = client.get("/samples").json()
    assert len(body["cvs"]) >= 3
    assert len(body["jobs"]) >= 2
    for sample in body["cvs"] + body["jobs"]:
        assert sample["id"] and sample["label"] and sample["text"].strip()
    assert all(sample["id"].startswith("cv_") for sample in body["cvs"])
    assert all(sample["id"].startswith("jd_") for sample in body["jobs"])


def test_samples_ignores_other_files(client: TestClient, tmp_path: Path) -> None:
    (tmp_path / "cv_one.txt").write_text("cv text", encoding="utf-8")
    (tmp_path / "jd_role.txt").write_text("job text", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("ignore", encoding="utf-8")
    (tmp_path / "cv_two.md").write_text("ignore", encoding="utf-8")
    use_samples_dir(tmp_path)
    body = client.get("/samples").json()
    assert [s["id"] for s in body["cvs"]] == ["cv_one"]
    assert [s["id"] for s in body["jobs"]] == ["jd_role"]
    assert body["cvs"][0]["text"] == "cv text"


def test_samples_empty_directory(client: TestClient, tmp_path: Path) -> None:
    use_samples_dir(tmp_path)
    assert client.get("/samples").json() == {"cvs": [], "jobs": []}


def test_samples_missing_directory(tmp_path: Path) -> None:
    result = load_samples(tmp_path / "does-not-exist")
    assert result.cvs == [] and result.jobs == []


@pytest.mark.parametrize(
    ("stem", "prefix", "expected"),
    [("cv_backend_senior", "cv_", "Backend senior"), ("jd_data_analyst", "jd_", "Data analyst")],
)
def test_make_label(stem: str, prefix: str, expected: str) -> None:
    assert make_label(stem, prefix) == expected


STATIC = Path(__file__).resolve().parent.parent / "app" / "static"


def read_static(name: str) -> str:
    return (STATIC / name).read_text(encoding="utf-8")


def test_index_links_assets() -> None:
    html = read_static("index.html")
    assert '<link rel="stylesheet" href="/static/style.css">' in html
    assert '<script src="/static/app.js" defer></script>' in html


@pytest.mark.parametrize("field_id", ["cv-file", "cv-text", "job-text"])
def test_inputs_have_labels(field_id: str) -> None:
    assert f'for="{field_id}"' in read_static("index.html")


@pytest.mark.parametrize(
    "hook",
    [
        'role="tablist"',
        'aria-live="polite"',
        'role="alert"',
        'id="theme-toggle"',
        'id="gauge-arc"',
        'id="breakdown"',
        'id="matched-skills"',
        'id="missing-skills"',
        'id="reasons"',
        'id="warnings"',
        'id="warnings-block"',
        'id="disclaimer"',
    ],
)
def test_index_has_ui_hooks(hook: str) -> None:
    assert hook in read_static("index.html")


def test_styles_support_both_themes_and_reduced_motion() -> None:
    css = read_static("style.css")
    assert "prefers-color-scheme: dark" in css
    assert ':root[data-theme="dark"]' in css
    assert "prefers-reduced-motion" in css
    assert ":focus-visible" in css


@pytest.mark.parametrize(
    "sink", ["innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval("]
)
def test_script_never_uses_html_string_sinks(sink: str) -> None:
    assert sink not in read_static("app.js")


def test_script_maps_every_error_status() -> None:
    js = read_static("app.js")
    for marker in ("413:", "415:", "502:", "503:", "status === 422", '"no extractable text"',
                   '"invalid PDF file"', "TypeError"):
        assert marker in js, marker


def test_script_calls_both_match_endpoints_and_samples() -> None:
    js = read_static("app.js")
    for endpoint in ('"/match"', '"/match/pdf"', '"/samples"'):
        assert endpoint in js


def test_warnings_block_is_a_hidden_note_by_default() -> None:
    html = read_static("index.html")
    start = html.index('id="warnings-block"')
    tag = html[html.rindex("<", 0, start) : html.index(">", start)]
    assert 'role="note"' in tag and "hidden" in tag
    assert html.index('id="warnings-block"') < html.index('id="breakdown"')


def test_script_renders_warnings_and_accepts_results_without_them() -> None:
    js = read_static("app.js")
    assert "renderWarnings(result.warnings || [])" in js
    assert "result.warnings === undefined || Array.isArray(result.warnings)" in js
    assert '$("warnings-block").hidden = warnings.length === 0' in js


def test_frontend_handles_not_a_cv_with_reasons() -> None:
    js = read_static("app.js")
    assert '"does not look like a CV"' in js
    assert "error-reasons" in js
    assert 'id="error-reasons"' in read_static("index.html")
