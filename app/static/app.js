"use strict";

// All dynamic text is inserted with textContent / text nodes only.
// Never use HTML-string sinks in this file; a test enforces this.

(() => {
  const MAX_PDF_BYTES = 5 * 1024 * 1024;
  const MAX_TEXT_CHARS = 50000;
  const GAUGE_CIRCUMFERENCE = 326.73;
  const COUNT_UP_MS = 900;

  const COMPONENT_LABELS = {
    required_skills: "Required skills",
    nice_to_have_skills: "Nice-to-have skills",
    experience: "Experience",
    education: "Education",
    languages: "Languages",
  };

  const ERRORS = {
    noText: {
      title: "No text found in this PDF",
      message: "It looks like a scanned image. Switch to Paste text and paste the CV instead.",
    },
    invalidPdf: {
      title: "This PDF could not be read",
      message: "The file may be damaged. Export it again, or paste the CV text instead.",
    },
    notCv: {
      title: "This does not look like a CV",
      message: "It reads like a job posting or has no work history or education. Use the candidate's CV instead.",
    },
    validation: {
      title: "Some input is missing or too long",
      message: "Fill in both the CV and the job description. Each can be up to 50,000 characters.",
    },
    413: {
      title: "The PDF is too large",
      message: "Upload a PDF of 5 MB or less, or paste the CV text instead.",
    },
    415: {
      title: "This file is not a PDF",
      message: "Upload a PDF file, or paste the CV text instead.",
    },
    502: {
      title: "The model service returned an invalid response",
      message: "Nothing was scored. Try again in a moment.",
    },
    503: {
      title: "Scoring is not configured",
      message: "The server has no API key. Set OPENAI_API_KEY in .env and restart the server.",
    },
    network: {
      title: "Could not reach the server",
      message: "Check that the server is running and your connection works, then try again.",
    },
    unexpected: {
      title: "Unexpected response",
      message: "The server sent a response this page cannot read. Try again.",
    },
  };

  const $ = (id) => document.getElementById(id);
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  const darkScheme = window.matchMedia("(prefers-color-scheme: dark)");
  const state = { mode: "pdf", file: null, controller: null };

  // ---------- DOM helper ----------

  function el(tag, props = {}, ...children) {
    const node = document.createElement(tag);
    for (const [key, value] of Object.entries(props)) {
      if (key === "className") node.className = value;
      else node.setAttribute(key, String(value));
    }
    for (const child of children) {
      node.append(child instanceof Node ? child : document.createTextNode(String(child)));
    }
    return node;
  }

  // ---------- Theme ----------

  function currentTheme() {
    const explicit = document.documentElement.getAttribute("data-theme");
    if (explicit === "light" || explicit === "dark") return explicit;
    return darkScheme.matches ? "dark" : "light";
  }

  function syncThemeButton() {
    $("theme-toggle").setAttribute("aria-pressed", String(currentTheme() === "dark"));
  }

  function toggleTheme() {
    const next = currentTheme() === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", next);
    try {
      localStorage.setItem("theme", next);
    } catch (e) {
      // Storage can be unavailable (private mode); the toggle still works for this visit.
    }
    syncThemeButton();
  }

  // ---------- Tabs ----------

  const TABS = [
    { mode: "pdf", tab: "tab-pdf", panel: "panel-pdf" },
    { mode: "text", tab: "tab-text", panel: "panel-text" },
  ];

  function selectTab(mode, moveFocus = false) {
    state.mode = mode;
    for (const item of TABS) {
      const active = item.mode === mode;
      const tab = $(item.tab);
      tab.setAttribute("aria-selected", String(active));
      tab.tabIndex = active ? 0 : -1;
      $(item.panel).hidden = !active;
      if (active && moveFocus) tab.focus();
    }
    hideFormError();
  }

  function onTabKeydown(event) {
    const keys = ["ArrowLeft", "ArrowRight", "Home", "End"];
    if (!keys.includes(event.key)) return;
    event.preventDefault();
    const next = event.key === "Home" ? "pdf" : event.key === "End" ? "text"
      : state.mode === "pdf" ? "text" : "pdf";
    selectTab(next, true);
  }

  // ---------- File input ----------

  function formatSize(bytes) {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  }

  function fileProblem(file) {
    const isPdf = file.type === "application/pdf" || /\.pdf$/i.test(file.name);
    if (!isPdf) return ERRORS[415];
    if (file.size > MAX_PDF_BYTES) return ERRORS[413];
    return null;
  }

  function setFile(file) {
    if (!file) return;
    const problem = fileProblem(file);
    if (problem) {
      clearFile();
      showFormError(`${problem.title}. ${problem.message}`);
      return;
    }
    state.file = file;
    $("file-name").textContent = file.name;
    $("file-size").textContent = formatSize(file.size);
    $("file-info").hidden = false;
    hideFormError();
  }

  function clearFile() {
    state.file = null;
    $("cv-file").value = "";
    $("file-info").hidden = true;
  }

  function bindDropzone() {
    const zone = $("dropzone");
    const setActive = (on) => zone.classList.toggle("is-dragover", on);
    for (const type of ["dragenter", "dragover"]) {
      zone.addEventListener(type, (event) => {
        event.preventDefault();
        setActive(true);
      });
    }
    zone.addEventListener("dragleave", (event) => {
      if (!zone.contains(event.relatedTarget)) setActive(false);
    });
    zone.addEventListener("drop", (event) => {
      event.preventDefault();
      setActive(false);
      setFile(event.dataTransfer && event.dataTransfer.files[0]);
    });
    // Stop the browser from opening a file dropped next to the zone.
    window.addEventListener("dragover", (event) => event.preventDefault());
    window.addEventListener("drop", (event) => event.preventDefault());
  }

  // ---------- Counters ----------

  function updateCounter(textarea, counterId) {
    const count = textarea.value.length.toLocaleString("en-US");
    $(counterId).textContent = `${count} / 50,000`;
  }

  function setText(textarea, counterId, value) {
    textarea.value = value;
    updateCounter(textarea, counterId);
  }

  // ---------- Samples ----------

  async function loadSamples() {
    try {
      const response = await fetch("/samples");
      if (!response.ok) return;
      const data = await response.json();
      renderSamples("cv", data.cvs, "Load sample CV", (sample) => {
        setText($("cv-text"), "cv-count", sample.text);
        selectTab("text");
      });
      renderSamples("job", data.jobs, "Load sample job", (sample) => {
        setText($("job-text"), "job-count", sample.text);
      });
    } catch (e) {
      // Samples are optional; the form works without them.
    }
  }

  function renderSamples(kind, samples, actionLabel, onPick) {
    if (!Array.isArray(samples) || samples.length === 0) return;
    const buttons = samples.map((sample) => {
      const button = el("button", {
        type: "button",
        className: "sample-button",
        "aria-label": `${actionLabel}: ${sample.label}`,
      }, sample.label);
      button.addEventListener("click", () => onPick(sample));
      return button;
    });
    $(`${kind}-samples`).replaceChildren(...buttons);
    $(`${kind}-samples-group`).hidden = false;
  }

  // ---------- Form validation and errors ----------

  function formProblem() {
    const cvText = $("cv-text").value.trim();
    const jobText = $("job-text").value.trim();
    if (state.mode === "pdf" && !state.file) return "Choose a PDF file, or switch to Paste text.";
    if (state.mode === "text" && !cvText) return "Paste the CV text.";
    if (!jobText) return "Paste the job description.";
    if (cvText.length > MAX_TEXT_CHARS || jobText.length > MAX_TEXT_CHARS) {
      return "Each text can be up to 50,000 characters.";
    }
    return null;
  }

  function showFormError(message) {
    const box = $("form-error");
    box.textContent = message;
    box.hidden = false;
  }

  function hideFormError() {
    const box = $("form-error");
    box.hidden = true;
    box.textContent = "";
  }

  function describeError(status, detail) {
    if (status === 422) {
      if (detail === "no extractable text") return ERRORS.noText;
      if (detail === "invalid PDF file") return ERRORS.invalidPdf;
      if (detail === "does not look like a CV") return ERRORS.notCv;
      return ERRORS.validation;
    }
    return ERRORS[status] || {
      title: "Scoring failed",
      message: `The server responded with status ${status}. Try again.`,
    };
  }

  // ---------- Requests ----------

  function buildRequest() {
    const jobText = $("job-text").value.trim();
    if (state.mode === "pdf") {
      const form = new FormData();
      form.append("cv_file", state.file, state.file.name);
      form.append("job_text", jobText);
      return { url: "/match/pdf", options: { method: "POST", body: form } };
    }
    return {
      url: "/match",
      options: {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ cv_text: $("cv-text").value.trim(), job_text: jobText }),
      },
    };
  }

  async function readJson(response) {
    try {
      return await response.json();
    } catch (e) {
      return null;
    }
  }

  async function onSubmit(event) {
    event.preventDefault();
    hideFormError();
    const problem = formProblem();
    if (problem) {
      showFormError(problem);
      return;
    }
    if (state.controller) state.controller.abort();
    const controller = new AbortController();
    state.controller = controller;
    setLoading(true);
    try {
      const { url, options } = buildRequest();
      const response = await fetch(url, { ...options, signal: controller.signal });
      const body = await readJson(response);
      if (!response.ok) {
        showError(describeError(response.status, body && body.detail), body && body.reasons);
      }
      else renderResult(body);
    } catch (error) {
      if (error.name === "AbortError") return;
      showError(error instanceof TypeError ? ERRORS.network : ERRORS.unexpected);
    } finally {
      if (state.controller === controller) {
        state.controller = null;
        setLoading(false);
      }
    }
  }

  // ---------- Result states ----------

  const STATES = ["state-empty", "state-loading", "state-error", "state-results"];

  function showState(id) {
    for (const stateId of STATES) $(stateId).hidden = stateId !== id;
  }

  function setLoading(on) {
    const button = $("submit");
    button.disabled = on;
    button.classList.toggle("is-loading", on);
    button.querySelector(".button-label").textContent = on ? "Scoring…" : "Score match";
    $("results-panel").setAttribute("aria-busy", String(on));
    if (on) showState("state-loading");
  }

  function showError(error, reasons) {
    $("error-title").textContent = error.title;
    $("error-message").textContent = error.message;
    const list = Array.isArray(reasons) ? reasons.map(String) : [];
    renderList("error-reasons", list);
    $("error-reasons").hidden = list.length === 0;
    showState("state-error");
  }

  function isValidResult(result) {
    return Boolean(result)
      && Number.isFinite(result.score)
      && Array.isArray(result.breakdown)
      && Array.isArray(result.matched_skills)
      && Array.isArray(result.missing_skills)
      && Array.isArray(result.reasons);
  }

  function scoreRange(score) {
    if (score >= 70) return { key: "strong", label: "Strong match" };
    if (score >= 40) return { key: "partial", label: "Partial match" };
    return { key: "weak", label: "Weak match" };
  }

  function renderResult(result) {
    if (!isValidResult(result)) {
      showError(ERRORS.unexpected);
      return;
    }
    const score = Math.max(0, Math.min(100, Math.round(result.score)));
    renderVerdict(score, result.missing_skills.length);
    renderBreakdown(result.breakdown);
    renderChips("matched-skills", result.matched_skills, "No listed skills matched.");
    renderChips("missing-skills", result.missing_skills, "None. Every required skill was found.");
    renderList("reasons", result.reasons);
    $("disclaimer").textContent = String(result.disclaimer || "");
    showState("state-results");
    $("results-heading").focus();
    animateResult(score);
  }

  function renderVerdict(score, missingCount) {
    const range = scoreRange(score);
    $("gauge").setAttribute("data-range", range.key);
    $("verdict").textContent = range.label;
    $("score-text").textContent = `Score ${score} out of 100. ${range.label}.`;
    $("summary").textContent = missingCount === 0
      ? `Score ${score} of 100. All required skills were found.`
      : `Score ${score} of 100. ${missingCount} required skill${missingCount === 1 ? " is" : "s are"} missing.`;
  }

  function formatPoints(value) {
    const number = Number(value) || 0;
    return Number.isInteger(number) ? String(number) : number.toFixed(1);
  }

  function renderBreakdown(items) {
    const rows = items.map((item) => {
      const label = COMPONENT_LABELS[item.name] || String(item.name);
      const ratio = Math.max(0, Math.min(1, Number(item.ratio) || 0));
      const percent = Math.round(ratio * 100);
      const fill = el("div", { className: "bar-fill", "data-ratio": ratio });
      return el("li", { className: "breakdown-item" },
        el("div", { className: "breakdown-head" },
          el("span", { className: "breakdown-name" }, label),
          el("span", { className: "breakdown-points" },
            `${formatPoints(item.points)} / ${formatPoints(item.weight)} pts`)),
        el("div", {
          className: "bar-track",
          role: "meter",
          "aria-label": label,
          "aria-valuemin": 0,
          "aria-valuemax": 100,
          "aria-valuenow": percent,
          "aria-valuetext": `${percent}%`,
        }, fill));
    });
    $("breakdown").replaceChildren(...rows);
  }

  function renderChips(listId, skills, emptyText) {
    const items = skills.length
      ? skills.map((skill) => el("li", { className: "chip" }, String(skill)))
      : [el("li", { className: "chip-empty" }, emptyText)];
    $(listId).replaceChildren(...items);
  }

  function renderList(listId, lines) {
    $(listId).replaceChildren(...lines.map((line) => el("li", {}, String(line))));
  }

  // ---------- Animation ----------

  function animateResult(score) {
    const arc = $("gauge-arc");
    const fills = document.querySelectorAll("#breakdown .bar-fill");
    const targetOffset = GAUGE_CIRCUMFERENCE * (1 - score / 100);
    if (reducedMotion.matches) {
      arc.style.strokeDashoffset = String(targetOffset);
      fills.forEach((fill) => { fill.style.transform = `scaleX(${fill.dataset.ratio})`; });
      $("gauge-value").textContent = String(score);
      return;
    }
    arc.style.strokeDashoffset = String(GAUGE_CIRCUMFERENCE);
    void arc.getBoundingClientRect(); // commit the start position before animating
    requestAnimationFrame(() => {
      arc.style.strokeDashoffset = String(targetOffset);
      fills.forEach((fill) => { fill.style.transform = `scaleX(${fill.dataset.ratio})`; });
    });
    countUp(score);
  }

  function countUp(target) {
    const output = $("gauge-value");
    const start = performance.now();
    const step = (now) => {
      const progress = Math.min(1, (now - start) / COUNT_UP_MS);
      const eased = 1 - Math.pow(1 - progress, 3);
      output.textContent = String(Math.round(target * eased));
      if (progress < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }

  // ---------- Init ----------

  function init() {
    syncThemeButton();
    $("theme-toggle").addEventListener("click", toggleTheme);
    darkScheme.addEventListener("change", syncThemeButton);

    for (const item of TABS) {
      const tab = $(item.tab);
      tab.addEventListener("click", () => selectTab(item.mode));
      tab.addEventListener("keydown", onTabKeydown);
    }

    $("cv-file").addEventListener("change", (event) => setFile(event.target.files[0]));
    $("file-remove").addEventListener("click", () => {
      clearFile();
      $("cv-file").focus();
    });
    bindDropzone();

    $("cv-text").addEventListener("input", (e) => updateCounter(e.target, "cv-count"));
    $("job-text").addEventListener("input", (e) => updateCounter(e.target, "job-count"));
    $("match-form").addEventListener("submit", onSubmit);

    loadSamples();
  }

  init();
})();
