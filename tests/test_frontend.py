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


def test_frontend_explains_rate_limits() -> None:
    js = read_static("app.js")
    assert "429:" in js and "status === 429" in js
    assert '"Too many requests."' in js


def test_frontend_explains_the_daily_limit() -> None:
    js = read_static("app.js")
    assert "status === 503" in js and '"Daily demo limit"' in js
    assert "dailyLimit" in js


def test_form_has_numbered_sections_and_live_counters() -> None:
    html = read_static("index.html")
    for hook in ('id="cv-count"', 'id="job-count"', 'id="cv-samples"', 'id="job-samples"',
                 'class="section-num">1<', 'class="section-num">2<', 'id="privacy-notice"'):
        assert hook in html, hook
    js = read_static("app.js")
    assert 'updateCounter(e.target, "job-count")' in js
    assert "${count} / 50,000" in js


def test_sample_cvs_render_as_cards_with_a_pressed_state() -> None:
    js = read_static("app.js")
    assert '"Synthetic CV"' in js and 'className: "sample-card"' in js
    assert '"aria-pressed"' in js


# ---------- results panel ----------


def test_preview_card_lists_what_the_result_contains() -> None:
    html = read_static("index.html")
    assert "What you&#39;ll get" in html or "What you'll get" in html
    assert 'class="gauge gauge-placeholder"' in html and 'class="checklist"' in html
    js = read_static("app.js")
    assert '"What you\'ll get" : "Your result"' in js


def test_verdict_bands_match_the_evaluation_bands() -> None:
    js = read_static("app.js")
    assert "score >= 80" in js and "score >= 50" in js
    for label in ('"Strong match"', '"Partial match"', '"Weak match"'):
        assert label in js
    assert '$("verdict").setAttribute("data-range", range.key)' in js


def test_results_scroll_into_view_and_focus_the_heading() -> None:
    js = read_static("app.js")
    assert '$("results-heading").focus({ preventScroll: true })' in js
    assert '$("results-panel").scrollIntoView(' in js


def test_evidence_parsing_matches_the_scoring_reason_wording() -> None:
    """app.js reads evidence levels from the reasons; the wording must match scoring.py."""
    from app.models import CVProfile, JobRequirements
    from app.scoring import score_match
    from tests.conftest import listed, role, used

    cv = CVProfile(
        skills=used("PostgreSQL", "Docker") + listed("Redis"),
        roles=[role(5, "PostgreSQL")],
    )
    job = JobRequirements(required_skills=["SQL", "Docker", "Redis", "AWS"])
    reason = score_match(cv, job).reasons[0]
    assert reason == (
        "Required skills: 3 of 4 matched (SQL via PostgreSQL; shown only in projects: Docker; "
        "listed without evidence: Redis; missing: AWS)."
    )
    js = read_static("app.js")
    for marker in ('"shown only in projects: "', '"listed without evidence: "', '"missing: "', '" via "',
                   "Required skills|Nice-to-have skills"):
        assert marker in js, marker


# ---------- header navigation and evaluation band ----------

REPO_URL = "https://github.com/Khayal07/ai-cv-screening"


def test_nav_anchors_point_at_page_sections() -> None:
    html = read_static("index.html")
    for anchor in ("how", "eval"):
        assert f'href="#{anchor}"' in html and f'id="{anchor}"' in html, anchor
    assert f'href="{REPO_URL}"' in html


def test_eval_band_links_to_the_readme_method() -> None:
    html = read_static("index.html")
    assert f'href="{REPO_URL}#evaluation"' in html
    readme = (Path(__file__).resolve().parent.parent / "README.md").read_text(encoding="utf-8")
    assert "\n## Evaluation\n" in readme


def test_eval_band_numbers_match_the_replayed_evaluation() -> None:
    from app.eval_clients import replay_client
    from app.evaluation import check_rankings, load_labels, run_case

    labels = load_labels()
    results = [run_case(case, replay_client(case), labels.today) for case in labels.cases]
    checks = check_rankings(labels.rankings, results)
    passed = sum(result.passed for result in results)
    ranked = sum(check.passed for check in checks)
    html = read_static("index.html")
    assert f'id="eval-cases" class="stat-value">{passed} / {len(results)}<' in html
    assert f'id="eval-rankings" class="stat-value">{ranked} / {len(checks)}<' in html
