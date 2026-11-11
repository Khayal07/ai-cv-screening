# CV Screening

A FastAPI service that compares a CV with a job description and returns an
**explainable match score** (0–100) with a per-component breakdown and plain-language reasons.

> The service does **not** make hiring decisions. It returns a score and reasons only,
> as decision support for a human reviewer.

![Web interface showing a match score of 90 with breakdown bars, matched and missing skills, reasons and the disclaimer](docs/screenshot.png)

*Web interface with a synthetic CV and job description. The screenshot was taken with a stubbed model client.*

## How it works

```
CV (text or PDF) ──► PDF text extraction ──► document check ──► personal data removal ──► structured extraction ─┐
                                                                                                                 ├─► deterministic scoring ──► score + reasons + warnings
Job description ──────────────────────────────────────────────────────────────────────► structured extraction ─┘
```

1. **Parsing.** PDFs are read with `pypdf` as text only, so embedded images such as photos are never processed.
2. **Document check.** `app/parsing/cv_check.py` rejects text that does not look like a CV, before any model call (HTTP 422 with a list of reasons). It counts dated ranges (`2019 - 2022`, `Jan 2020 - Present`), education words, word count and job-posting phrases (`we are looking for`, `nice to have`, a bare `Requirements:` heading, ...). Text is rejected when it has fewer than 40 words, 3 or more posting phrases, 2 or more posting phrases and no dated ranges, or neither dated ranges nor education details. A job description pasted as the CV is rejected.
3. **Anonymization.** Before any text leaves the service, `app/parsing/anonymize.py` removes the name, date of birth and age, gender, pronouns, photo references, marital and family details, nationality, and contact details (email, phone, URLs).
4. **Extraction.** A language model (OpenAI API, model set by `LLM_MODEL`) turns the CV and the job description into structured data: skills, dated roles with the skills used in each, education level and languages. Years of experience are computed from the role dates in plain Python, not read by the model. The schemas contain no personal fields. Output is validated with Pydantic; invalid output is retried once.
5. **Scoring.** Plain Python, with no model involved, so the score is reproducible and unit-tested:

| Component           | Weight | Rule                                             |
|---------------------|-------:|--------------------------------------------------|
| Required skills     | 50     | credit per required skill / number required      |
| Nice-to-have skills | 15     | credit per optional skill / number optional      |
| Experience          | 20     | `min(relevant_years / required_years, 1)`        |
| Education           | 10     | full if the level is met, partial if below       |
| Languages           | 5      | share of required languages found                |

Skills are normalized (case, whitespace, aliases such as `JS → JavaScript` and `k8s → Kubernetes`).
A job skill also counts as matched when its word tokens appear as a contiguous sequence inside a CV skill:
CV `SQL query optimisation` satisfies job `SQL`. This works in one direction only, so a short CV skill never
satisfies a longer job skill (CV `Data` does not satisfy job `Data analysis`). Matching uses whole tokens, never
substrings, so `Java` never matches `JavaScript`, `Go` never matches `MongoDB`, and `C` never matches `C++`, `C#`
or `Objective-C`. When the CV wording differs, the reason names it, for example `SQL via SQL query optimisation`.
A small one-way implication map (`IMPLIES` in `app/scoring.py`) lets a specific CV skill satisfy a general job skill:
`PostgreSQL → SQL`, `FastAPI → Python`, `React → JavaScript`. It never works in reverse (CV `SQL` does not satisfy job
`PostgreSQL`) and never chains (`Next.js` satisfies `React`, not `JavaScript`). The reason reads `SQL via PostgreSQL`.
Component points stay as floats; the final score is rounded once to an integer.

**Evidence.** The extraction marks each CV skill as used (named in a job, project or education entry) or only
listed (skills section or summary). A skill shown in use earns full credit; a listed-only skill earns half, and the
reason names it: `listed without evidence: Redis`. A list of keywords therefore cannot reach a high score.

**Experience.** Years are computed from the dated roles, not taken from phrases like "5+ years":
- Dates are `YYYY-MM`, `YYYY` (counted from June) or `present`. An end date in the future counts as today.
- Overlapping roles are merged, so parallel jobs are not counted twice.
- A role is relevant when it used at least one of the job's skills, with the same token and implication rules as
  skill matching. Only relevant years count toward the minimum. If the job names no skills, every role counts.
- The reason shows both numbers: `Experience: 3.5 relevant years of 5.0 total (required 4).`

**Warnings.** Some findings do not change the score but should be checked by the reviewer. They are returned in
`warnings` and shown above the breakdown:
- the CV shares at least 25% of its word sequences (word 3-grams) with the job description, which suggests text was
  copied from the posting (`app/parsing/overlap.py`; the highest value among the synthetic CV-job pairs is 6.7%)
- roles with no listed skills, which are not counted as relevant experience
- roles with unreadable dates, which are ignored

## Fairness and limits

- Name, age, gender and photo are removed before extraction and are never part of the scoring input.
- Scoring is rule-based and transparent: every point can be traced to a line in `reasons`.
- Skill matching works on word tokens, aliases and the one-way implication map only. Related skills
  (for example Django vs FastAPI) do not count as matches.
- Whether a skill was used and which skills belong to a role are judged by the model, so they can be wrong.
  The reasons and warnings make each of these judgments visible to the reviewer.
- The document check and the overlap warning are heuristics. They catch pasted postings and keyword lists,
  not every kind of misleading CV.
- All CVs in this repository are synthetic (`data/synthetic/`). Do not commit real CVs.

## Cost control

- Every model call goes through the `LLMClient` interface and is cached on disk (`CACHE_DIR`, default `.cache/llm`).
- The cache key covers the model name, the prompt's own version (`CV_PROMPT_VERSION` or `JOB_PROMPT_VERSION`), the system prompt and the input. Bumping one prompt's version invalidates only that prompt's entries.
- Inputs are truncated to 12,000 characters before they are sent.

## Setup

Requires Python 3.12.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
cp .env.example .env               # then set OPENAI_API_KEY
```

| Variable         | Default       | Description                     |
|------------------|---------------|---------------------------------|
| `OPENAI_API_KEY` | (empty)       | API key; required for `/match*` |
| `LLM_MODEL`      | `gpt-4o-mini` | Model used for extraction       |
| `CACHE_DIR`      | `.cache/llm`  | Disk cache for model responses  |

## Run

```bash
uvicorn app.main:app --reload
```

- Web interface: http://127.0.0.1:8000
- Interactive API docs: http://127.0.0.1:8000/docs

## Web interface

The page is plain HTML, CSS and JavaScript served by the same FastAPI app (`app/static/`).
There is no build step and it makes no external requests.

1. Open http://127.0.0.1:8000.
2. Add the CV in one of two ways:
   - **Upload PDF**: drop a PDF on the upload area or choose a file (text-based PDF, up to 5 MB).
   - **Paste text**: paste the CV text.
3. Paste the job description.
   To try it quickly, select one of the **Sample CVs** and **Sample jobs**. They are loaded from `data/synthetic/` through `GET /samples`.
4. Select **Score match**.

The result panel shows:

- a score gauge from 0 to 100: green for 70 and above, amber for 40–69, red below 40
- one bar per scoring component, with points earned out of the component weight
- **Check these**: warnings such as text copied from the job description or roles without skills (shown only when present)
- matched skills (green) and missing required skills (red)
- the reasons behind every point, and the decision-support disclaimer

If something goes wrong, the panel says what happened and what to do next, for example:
- text that does not look like a CV, such as a pasted job posting (the reasons are listed)
- a scanned PDF with no text: paste the CV text instead
- a file that is too large
- a model or provider error
- no API key configured
- the server can't be reached

Other details:
- The page follows the system light or dark setting and has a theme toggle; the choice is remembered in the browser.
- It works on mobile and can be used with the keyboard alone.
- All text from the API is inserted as plain text, never as HTML.

### Examples

Text CV (the synthetic samples, wrapped into a JSON body):

```bash
python -c "import json; print(json.dumps({'cv_text': open('data/synthetic/cv_backend_senior.txt').read(), 'job_text': open('data/synthetic/jd_backend_engineer.txt').read()}))" > request.json
curl -s http://127.0.0.1:8000/match -H "Content-Type: application/json" -d @request.json
```

PDF CV:

```bash
curl -s http://127.0.0.1:8000/match/pdf \
  -F "cv_file=@cv.pdf;type=application/pdf" \
  -F "job_text=<data/synthetic/jd_backend_engineer.txt"
```

Example response (shortened). The CV lists Redis without using it, and one role names no skills:

```json
{
  "score": 79,
  "breakdown": [
    {"name": "required_skills", "weight": 50.0, "ratio": 0.875, "points": 43.75,
     "detail": "Required skills: 4 of 4 matched (SQL via PostgreSQL; listed without evidence: Redis)."},
    {"name": "experience", "weight": 20.0, "ratio": 1.0, "points": 20.0,
     "detail": "Experience: 5.7 relevant years of 7.8 total (required 4)."}
  ],
  "matched_skills": ["Python", "SQL", "Docker", "Redis"],
  "missing_skills": [],
  "reasons": [
    "Required skills: 4 of 4 matched (SQL via PostgreSQL; listed without evidence: Redis).",
    "Nice-to-have skills: 0 of 1 matched (missing: Kubernetes).",
    "Experience: 5.7 relevant years of 7.8 total (required 4).",
    "Education: bachelor meets the required bachelor.",
    "Languages: 1 of 1 matched."
  ],
  "warnings": ["1 role(s) have no listed skills and were not counted as relevant experience."],
  "disclaimer": "This score is decision support only. ..."
}
```

A pasted job description is rejected before any model call:

```json
{
  "detail": "does not look like a CV",
  "reasons": [
    "It contains several job-posting phrases (\"we are looking for\", \"nice to have\", \"requirements\")."
  ]
}
```

### Endpoints and errors

| Method | Path         | Body                                            |
|--------|--------------|-------------------------------------------------|
| GET    | `/`          | web interface (HTML)                            |
| GET    | `/samples`   | none; returns synthetic sample CVs and jobs     |
| GET    | `/health`    | none                                            |
| POST   | `/match`     | JSON `{cv_text, job_text}`                      |
| POST   | `/match/pdf` | multipart `cv_file` (PDF, ≤ 5 MB) + `job_text`  |

| Status | Meaning                                                          |
|--------|------------------------------------------------------------------|
| 413    | PDF larger than 5 MB                                             |
| 415    | uploaded file is not a PDF                                       |
| 422    | invalid input, invalid PDF, `no extractable text` (scanned), or `does not look like a CV` (with `reasons`) |
| 502    | model output still invalid after one retry, or provider error    |
| 503    | `OPENAI_API_KEY` not configured                                  |

## Tests

```bash
pytest
```

Tests use a fake model client and block all external network access, so they cost nothing and run offline.

## Docker

```bash
docker build -t cv-screening .
docker run -p 8000:8000 --env-file .env cv-screening
```

## Project layout

```
app/
  main.py          API routes and error mapping
  service.py       pipeline: anonymize → extract → score
  extraction.py    prompts, per-prompt versions, validation, retry
  scoring.py       deterministic weighted scoring
  experience.py    date math for roles: parsing, merging overlaps, years
  models.py        Pydantic models
  samples.py       read-only loader for the synthetic samples
  config.py        settings from .env
  llm/             LLMClient interface, OpenAI client, disk cache
  parsing/         PDF text extraction, anonymization, document check, text overlap
  static/          web interface: index.html, style.css, app.js
data/synthetic/    synthetic CVs and job descriptions
docs/              README screenshot
tests/             unit, API, frontend and end-to-end tests
```
