# CV Screening

A FastAPI service that compares a CV with a job description and returns an
**explainable match score** (0–100) with a per-component breakdown and plain-language reasons.

> The service does **not** make hiring decisions. It returns a score and reasons only,
> as decision support for a human reviewer.

**Live demo:** <https://cv-screening-lpi1.onrender.com/>. It runs on free hosting, so the first load can take
30-50 seconds while the server wakes up. It is a demo only: do not upload real CVs. Use the sample CVs or a
made-up one.

![Web interface: the CV and job form with sample cards and the privacy note, and a result of 72 (partial match) with a warning box, score breakdown bars, matched skills with their evidence, missing skills, reasons and the disclaimer, above the evaluation band](docs/screenshot.png)

*Web interface with a synthetic CV and job description. The screenshot was taken locally with a stand-in model
client, so no real model calls were made.*

## How it works

```
CV (text or PDF) ──► PDF text extraction ──► document check ──► personal data removal ──► structured extraction ─┐
                                                                                                                 ├─► deterministic scoring ──► score + reasons + warnings
Job description ──────────────────────────────────────────────────────────────────────► structured extraction ─┘
```

1. **Parsing.** PDFs are read with `pypdf` as text only, so embedded images such as photos are never processed.
2. **Document check.** `app/parsing/cv_check.py` rejects text that does not look like a CV, before any model call (HTTP 422 with a list of reasons). It counts dated ranges (`2019 - 2022`, `Jan 2020 - Present`), education words, word count and job-posting phrases (`we are looking for`, `nice to have`, a bare `Requirements:` heading, ...). Text is rejected when it has fewer than 40 words, 3 or more posting phrases, 2 or more posting phrases and no dated ranges, or neither dated ranges nor education details. A job description pasted as the CV is rejected.
3. **Anonymization.** Before any text leaves the service, `app/parsing/anonymize.py` removes the name, date of birth and age, gender, pronouns, photo references, marital and family details, nationality, and contact details (email, phone, URLs). It also removes referees (`References: ...` lines and the block under a `References` heading) and a signer's name after a sign-off such as `Kind regards,`.
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
An either-or requirement such as `Tableau or Power BI` is one job skill, satisfied by its best-matching
alternative (`Tableau or Power BI via Tableau`). A job skill that is a spoken language (a nice-to-have `Spanish`)
is also matched against the CV's languages.
Component points stay as floats; the final score is rounded once to an integer.

**Evidence.** The extraction marks each CV skill as used (named in a job, project or education entry) or only
listed (skills section or summary). Credit per skill:

| How the CV shows the skill                     | Credit | Reason note                       |
|------------------------------------------------|--------|-----------------------------------|
| used in a job role                             | 1.0    | (none)                            |
| used only in projects or coursework, no role   | 0.75   | `shown only in projects: Tableau` |
| only listed in a skills section or summary     | 0.5    | `listed without evidence: Redis`  |

A list of keywords therefore cannot reach a high score.

**Experience.** Years are computed from the dated roles, not taken from phrases like "5+ years":
- Dates are `YYYY-MM`, `YYYY` (counted from June) or `present`. An end date in the future counts as today.
- Overlapping roles are merged, so parallel jobs are not counted twice.
- A role is relevant when it used at least one required skill, or at least two of the job's skills, with the same
  token and implication rules as skill matching. Only relevant years count toward the minimum. If the job names no
  skills, every role counts.
- Projects and coursework are not roles: they add skill evidence but no years.
- The reason shows both numbers: `Experience: 3.5 relevant years of 5.0 total (required 4).`

**Warnings.** Some findings do not change the score but should be checked by the reviewer. They are returned in
`warnings` and shown above the breakdown:
- the CV shares at least 25% of its word sequences (word 3-grams) with the job description, which suggests text was
  copied from the posting (`app/parsing/overlap.py`; the highest value among the synthetic CV-job pairs is 6.7%).
  When this fires, skills that appear only in copied lines get no credit, and a second warning names them
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
- Anonymization is rule-based and conservative: a name in free text without a label, title, reference heading or
  sign-off (for example "worked with Jane on ...") is not removed.
- All CVs in this repository are synthetic (`data/synthetic/`). Do not commit real CVs.

## Privacy

This is a demo. Do not submit real CVs: use the synthetic samples or a made-up CV. The page says the same under
the Score match button.

- **What is sent out.** The CV text, after anonymization (see [How it works](#how-it-works)), and the job
  description are sent to the OpenAI API for extraction. OpenAI's
  [API data policy](https://platform.openai.com/docs/guides/your-data) applies to that text.
- **What is stored.** Only the extraction cache in `CACHE_DIR`: one JSON file per prompt, named by a SHA-256 hash,
  holding the model's structured output (skills, role dates, education level, languages, job requirements). The CV
  text, the uploaded PDF and the job text are never written to disk. The daily usage file holds a single number.
- **What is logged.** The app itself logs nothing about requests. The server's access log (uvicorn, and the hosting
  platform) records method, path, status and client IP, not request bodies. A test fails if app code calls
  `print` or `logging`.
- **Error responses never echo the submitted text.** Validation errors (422) list only the field, message and type,
  without FastAPI's usual `input` value.
- **Client IPs** are kept in memory for rate limiting, for at most 24 hours, and are never written to disk.
- **Limits.** Anonymization is rule-based (see [Fairness and limits](#fairness-and-limits)), and the job
  description is not anonymized. Deleting `CACHE_DIR` removes everything the server has stored.

## Security

- **Headers on every response** (`app/security.py`): a Content Security Policy that allows only the app's own
  scripts, styles and requests (`script-src 'self'; style-src 'self'`, no inline script or style, images also from
  `data:` for the favicon), `X-Content-Type-Options: nosniff`, `Referrer-Policy: no-referrer`, a
  `Permissions-Policy` that turns off camera, microphone and location, and in production
  `Strict-Transport-Security`. Tests check that the page has no inline script, `style` attribute or event handler.
- **Request size limit:** bodies over 6 MB get 413, both when `Content-Length` says so and when a streamed body
  grows past it.
- **Production mode** (`ENVIRONMENT=production`, the default in the Docker image): the OpenAPI schema and the docs
  pages are not served.
- **No stack traces:** an unhandled error returns `{"detail": "internal error"}` with status 500, and only the
  exception type is logged.

## Cost control

- Every model call goes through the `LLMClient` interface and is cached on disk (`CACHE_DIR`, default `.cache/llm`).
- The cache key covers the model name, the prompt's own version (`CV_PROMPT_VERSION` or `JOB_PROMPT_VERSION`), the system prompt and the input. Bumping one prompt's version invalidates only that prompt's entries.
- Inputs are truncated to 12,000 characters before they are sent.
- `POST /match` and `POST /match/pdf` are rate limited per client IP (`RATE_LIMIT_PER_MINUTE`, `RATE_LIMIT_PER_DAY`).
  The counts live in memory, so run one worker per instance. Behind a proxy, set `TRUSTED_PROXY_HOPS` to the number
  of proxies: the client IP is that many entries from the right of `X-Forwarded-For`, so entries a client adds on
  the left are ignored.
- A global cap, `DAILY_MODEL_CALL_LIMIT` (default 100), limits real model calls per UTC day for all clients together.
  It sits inside the cache, so cache hits do not count; failed calls and retries do. The count is kept in
  `CACHE_DIR/usage/YYYY-MM-DD.json`, so a restart does not reset it. When the cap is reached, matches return 503
  until 00:00 UTC. A match makes at most 2 calls (plus 1 retry each on invalid output).

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
| `RATE_LIMIT_PER_MINUTE` | `5`    | Match requests per client IP per minute (0 = off) |
| `RATE_LIMIT_PER_DAY`    | `30`   | Match requests per client IP per rolling 24 hours (0 = off) |
| `DAILY_MODEL_CALL_LIMIT` | `100` | Real model calls per UTC day across all clients; cache hits are free (0 = no calls) |
| `TRUSTED_PROXY_HOPS`    | `0`    | Proxies in front of the app; the client IP is read from `X-Forwarded-For` only when this is above 0 |
| `ENVIRONMENT`           | `development` | `production` turns off `/docs`, `/redoc` and `/openapi.json` and adds HSTS |
| `FRAME_ANCESTORS`       | `'none'` | CSP `frame-ancestors`: who may embed the page in a frame |

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
   To try it quickly, select one of the cards under **Try a sample CV** and a chip under **Try a sample job**. They are loaded from `data/synthetic/` through `GET /samples`.
4. Select **Score match**.

The result panel shows:

- a score gauge from 0 to 100 with a written label: **Strong match** for 80 and above (green), **Partial match**
  for 50–79 (amber), **Weak match** below 50 (red). These are the same bands the evaluation uses.
- one bar per scoring component, with points earned out of the component weight
- **Check these**: warnings such as text copied from the job description or roles without skills (shown only when present)
- matched skills (green), each marked with its evidence (used in a role, shown only in projects, or listed only),
  and missing required skills (red)
- the reasons behind every point, what was removed before scoring, and the decision-support disclaimer

Below the form, an evaluation band shows the replayed evaluation result (see [Evaluation](#evaluation)).

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
| 429    | too many match requests from this client; `Retry-After` gives the wait in seconds |
| 502    | model output still invalid after one retry, or provider error    |
| 503    | `OPENAI_API_KEY` not configured, or the daily model call cap is reached (`Retry-After` until 00:00 UTC) |

## Tests

```bash
pytest
```

Tests use a fake model client and block all external network access, so they cost nothing and run offline.

## Evaluation

`data/eval/` holds 28 synthetic CVs and `labels.json`: 34 labeled cases scored against the two sample job
descriptions (`data/synthetic/jd_backend_engineer.txt`, `jd_data_analyst.txt`).

**Method**
- **Labels first.** The CVs and every expected outcome (score range or 422 rejection, plus expected warnings) were
  committed before any model output existed. Each category has a default band: strong 80-100, partial 50-79,
  weak 0-45, junior with projects only 25-65, career changer 15-55, keyword stuffing 25-49. A case may narrow its
  band but never widen it, and no label was changed to make a case pass.
- **Ranking rules, per job.** Every strong CV must beat every partial one. Every partial CV must beat every weak
  or keyword-stuffed one. Every strong CV must beat the strong CVs written for the other job (`cross_job`).
- **Recorded once.** Real extractions were recorded with `--real --record` into `data/eval/extractions/`. Each
  file stores only the model output, the prompt version and a hash of the input: no CV text and no personal data.
- **Fixed date.** `labels.json` sets `today` to 2026-11-01, so roles that run to "present" do not drift.
- **Replay with 0 calls.** `tests/test_eval.py` and the script replay the recordings through the full pipeline
  (document check, anonymization, extraction validation, scoring). A recording whose prompt version or input
  text has changed is reported as stale instead of being used silently.

**How to rerun**

```bash
python scripts/run_eval.py                                  # replay, 0 model calls
python scripts/run_eval.py --markdown                       # the table below
python scripts/run_eval.py --real --max-calls 40            # real model, through the disk cache
python scripts/run_eval.py --real --record --max-calls 40   # re-record after a prompt change
```

With `--real`, the script first counts the cache misses and stops without any call if they exceed
`--max-calls`. Exit codes: 0 all pass, 1 a case or ranking fails, 2 the run was stopped.

**Results** (gpt-4o-mini; CV prompt v4, job prompt v5): **33 of 34 cases pass (97%); 6 of 6 ranking checks
pass.**

| Case | Category | Expected | Actual | Result |
|---|---|---|---|---|
| be_strong_1@jd_backend_engineer | strong | 85-100 | 100 | PASS |
| be_strong_2@jd_backend_engineer | strong | 80-100 | 90 | PASS |
| be_strong_3@jd_backend_engineer | strong | 80-100 | 90 | PASS |
| be_partial_1@jd_backend_engineer | partial | 50-79 | 71 | PASS |
| be_partial_2@jd_backend_engineer | partial | 50-79 | 74 | PASS |
| be_weak_1@jd_backend_engineer | weak | 0-45 | 15 | PASS |
| be_weak_2@jd_backend_engineer | weak | 0-30 | 15 | PASS |
| be_junior_projects@jd_backend_engineer | junior_projects | 25-65 | 41 | PASS |
| be_career_changer@jd_backend_engineer | career_changer | 15-55 +no_skill_roles | 39 +no_skill_roles | PASS |
| be_stuffed@jd_backend_engineer | stuffed | 25-49 | 48 | PASS |
| be_copied_reqs@jd_backend_engineer | copied_requirements | 45-79 +overlap | 68 +overlap | PASS |
| da_strong_1@jd_data_analyst | strong | 85-100 | 100 | PASS |
| da_strong_2@jd_data_analyst | strong | 80-100 | 90 | PASS |
| da_strong_3@jd_data_analyst | strong | 80-100 | 95 | PASS |
| da_partial_1@jd_data_analyst | partial | 50-79 | 60 | PASS |
| da_partial_2@jd_data_analyst | partial | 50-79 | 68 | PASS |
| da_weak_1@jd_data_analyst | weak | 0-30 | 20 | PASS |
| da_weak_2@jd_data_analyst | weak | 0-45 | 18 | PASS |
| da_junior_projects@jd_data_analyst | junior_projects | 25-65 | 60 | PASS |
| da_career_changer@jd_data_analyst | career_changer | 15-55 | 60 +no_skill_roles | FAIL |
| da_stuffed@jd_data_analyst | stuffed | 25-49 | 45 +no_skill_roles | PASS |
| da_copied_reqs@jd_data_analyst | copied_requirements | 45-79 +overlap | 68 +overlap | PASS |
| be_strong_1@jd_data_analyst | cross_job | 10-60 | 57 | PASS |
| be_strong_2@jd_data_analyst | cross_job | 10-60 | 57 | PASS |
| be_strong_3@jd_data_analyst | cross_job | 10-60 | 57 | PASS |
| da_strong_1@jd_backend_engineer | cross_job | 10-55 | 52 | PASS |
| da_strong_2@jd_backend_engineer | cross_job | 10-55 | 52 | PASS |
| da_strong_3@jd_backend_engineer | cross_job | 10-55 | 44 | PASS |
| x_pasted_backend@jd_backend_engineer | pasted_job | 422 | 422 | PASS |
| x_pasted_analyst_reqs@jd_data_analyst | pasted_job | 422 | 422 | PASS |
| x_short_backend@jd_backend_engineer | very_short | 422 | 422 | PASS |
| x_short_analyst@jd_data_analyst | very_short | 422 | 422 | PASS |
| x_cover_letter@jd_data_analyst | non_cv | 422 | 422 | PASS |
| x_blog_post@jd_backend_engineer | non_cv | 422 | 422 | PASS |

`+overlap` and `+no_skill_roles` are warnings: expected ones must appear, and an unexpected overlap warning fails
the case.

**Rankings**

| Job | Check | Result |
|---|---|---|
| Backend engineer | lowest strong 90 > highest partial 74 | PASS |
| Backend engineer | lowest partial 71 > highest weak or stuffed 48 | PASS |
| Backend engineer | lowest strong 90 > highest cross-job 52 | PASS |
| Data analyst | lowest strong 90 > highest partial 68 | PASS |
| Data analyst | lowest partial 60 > highest weak or stuffed 45 | PASS |
| Data analyst | lowest strong 90 > highest cross-job 57 | PASS |

**What the evaluation found and fixed** (the first recorded run passed 27 of 34)
- Pasted job text: a reworded posting and a bare requirements block are rejected with 422 before any model call.
- Cover letter: it passed the document check and scored 68. Letter signals (salutation, "I am writing to apply",
  sign-off) now reject it.
- Birth date leak: "Born 1990" reached the model. Birth details are now removed with the other personal data.
- Phone redaction ate dates: "2023.01 - 2023.12" and ISO ranges such as "2019-03 - 2021-07" were deleted as phone
  numbers. Date ranges are now kept, and dotted ones are normalised to `YYYY-MM`.
- Either-or requirements: "a BI tool such as Tableau or Power BI" was read as two required skills. It is now one
  item that either tool satisfies.
- Copied requirements: skills that appear only in text copied from the job now get no credit, with a warning that
  names them.

**Known limitations**
- **Career changer with a shared generic skill.** A role that used one required skill counts in full as relevant
  experience, so a teacher who kept grades in Excel gets 12.4 relevant years for the data analyst job
  (`da_career_changer`: 60, label 15-55). Requiring two job skills per role fixes this case but breaks a genuine
  marketing analyst (`da_partial_1`), so the case stays as a known failure, kept visible as a strict `xfail`.
- Synthetic data only: the CVs are fictional, clean and well structured; real CVs are messier.
- Small sample: 28 CVs, 2 jobs, 34 cases. The pass rate is a sanity check, not a measure of accuracy.
- The labels were written by the author of the scoring rules. Committing them before recording limits this bias
  but does not remove it.
- Thresholds (overlap 25%, copied line 50%, document-check rules, credit weights) were tuned on few files.
- English CVs and job descriptions only.

## Docker

```bash
docker build -t cv-screening .
docker run -p 8000:8000 --env-file .env cv-screening
```

The image runs as a non-root user (`appuser`, uid 1000), sets `ENVIRONMENT=production`, listens on `PORT`
(default 8000), and has a `HEALTHCHECK` on `/health`. It starts one uvicorn worker, because the rate limits and
the daily cap are counted in memory and in one usage file. With `--env-file .env`, the `ENVIRONMENT` line in
`.env` (development) overrides the image default, so set it to `production` there to test production mode.

## Deploy

Before the first deploy:
1. **Set a monthly spend limit for the OpenAI project** whose key you use (OpenAI dashboard: project limits /
   billing). The app's own limits protect the budget, but the provider limit is the last line of defence.
2. Use a key that belongs to that project only, so you can revoke it without affecting anything else.
3. **Never put the key in the repository**, a Dockerfile, `render.yaml` or a build argument. Set it only in the
   platform's secret settings.

### Render (recommended)

1. Push this repository to GitHub.
2. In Render: **New > Blueprint**, then pick the repository. Render reads `render.yaml`: a Docker web service on
   the free plan with a health check on `/health`, `ENVIRONMENT=production`, `TRUSTED_PROXY_HOPS=1`, and the
   default limits.
3. When Render asks for `OPENAI_API_KEY` (it is declared with `sync: false`, so it has no value in the file),
   paste the key. Render stores it as a secret environment variable.
4. Deploy, then work through the [after-deploy checklist](#after-deploy-checklist).

Notes:
- Render sets `PORT` itself; the image reads it.
- The free plan has no persistent disk and sleeps when idle. A restart or redeploy empties the extraction cache
  and resets the daily usage file, so the daily cap restarts too; per-IP limits also reset. The OpenAI spend
  limit is what holds across restarts.
- Keep one instance. Each instance counts its own limits.

### Hugging Face Spaces (alternative)

1. Create a Space with the **Docker** SDK and push this repository to it.
2. The Space's `README.md` needs this header. Keep it in the Space repository only; the GitHub README stays as it
   is:

   ```yaml
   ---
   title: CV Screening
   sdk: docker
   app_port: 8000
   ---
   ```

3. In the Space settings, add `OPENAI_API_KEY` as a **secret**, and these as variables:
   `TRUSTED_PROXY_HOPS=1` and `FRAME_ANCESTORS=https://huggingface.co` (the Space page shows the app in a frame on
   huggingface.co; with the default `'none'` the frame stays blank).
4. Free Spaces also restart without persistent storage, with the same effect on the cache and the daily cap as on
   Render.

### After-deploy checklist

Replace `https://APP` with the service URL. Steps 1-5 make no model calls; step 6 makes up to 2.

1. **Health:** `curl -i https://APP/health` returns 200 `{"status":"ok"}` with `content-security-policy`,
   `x-content-type-options: nosniff`, `referrer-policy: no-referrer` and `strict-transport-security` headers.
2. **Docs are off:** `/docs`, `/redoc` and `/openapi.json` each return 404.
3. **Page:** open `https://APP/` with the browser's developer tools open. The console shows no CSP errors, the
   favicon and the privacy notice appear, the theme toggle works, and the sample buttons fill the inputs.
4. **Request size:** a body over 6 MB returns 413:
   `head -c 7000000 /dev/zero | curl -s -o /dev/null -w '%{http_code}\n' -X POST -H 'content-type: application/json' --data-binary @- https://APP/match`.
5. **Rate limit:** send 6 invalid requests within a minute; the first 5 return 422 (or 503 if the key is missing)
   and the 6th returns 429 with a `Retry-After` header:
   `for i in 1 2 3 4 5 6; do curl -s -o /dev/null -w '%{http_code} ' -X POST -H 'content-type: application/json' -d '{"cv_text":"","job_text":""}' https://APP/match; done`.
   Note that these count toward this IP's limits for the next minute and day.
6. **One real match:** in the page, pick a sample CV and a sample job and select **Score match**. Expect a score,
   reasons and the disclaimer. This is up to 2 model calls on a fresh cache; repeating it is free.
7. **Client IP behind the proxy:** after step 5 has blocked your connection, run one request from a different
   network (for example a phone on mobile data). It must not get 429. If it does, all visitors share one bucket,
   and `TRUSTED_PROXY_HOPS` is wrong for this platform (try 2, then check again). Then, still blocked on the
   first network, add a fake header: `curl -s -o /dev/null -w '%{http_code}\n' -X POST -H 'X-Forwarded-For: 203.0.113.9' -H 'content-type: application/json' -d '{"cv_text":"","job_text":""}' https://APP/match`.
   It must still return 429. If it does not, the hop count is too high and clients can choose their own IP.
8. **Logs:** the platform logs show method, path, status and IP only, never CV text.
9. **Budget:** the OpenAI project has a monthly spend limit, and its usage page shows the calls from step 6.

## Project layout

```
app/
  main.py          app factory, API routes, error mapping
  security.py      security headers, request size limit, 500 without details
  ratelimit.py     per-client request limits, client IP behind proxies
  service.py       pipeline: check → anonymize → extract → score
  extraction.py    prompts, per-prompt versions, validation, retry
  scoring.py       deterministic weighted scoring
  experience.py    date math for roles: parsing, merging overlaps, years
  models.py        Pydantic models
  samples.py       read-only loader for the synthetic samples
  config.py        settings from .env
  evaluation.py    evaluation labels, case checks, rankings, report table
  eval_clients.py  replay, call budget and recording clients for the evaluation
  llm/             LLMClient interface, OpenAI client, disk cache, daily call cap
  parsing/         PDF text, anonymization, date ranges, document check, text overlap
  static/          web interface: index.html, style.css, theme.js, app.js
data/synthetic/    synthetic CVs and job descriptions
data/eval/         evaluation CVs, labels.json, recorded extractions
docs/              README screenshot
scripts/           run_eval.py
tests/             unit, API, frontend, end-to-end and evaluation tests
Dockerfile         production image (non-root, PORT, healthcheck, one worker)
render.yaml        Render Blueprint
```
