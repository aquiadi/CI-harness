"use strict";

// The answer arrives with [chunk#id] markers in it and a separate list of
// citations flagged valid or not. Rendering the markers as buttons that scroll
// to the passage is the whole point of the interface: a citation the reader
// cannot check is decoration.
const CITE = /\[chunk#([^\]\s]+)\]/g;

const $ = (id) => document.getElementById(id);
const el = (tag, cls, text) => {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
};

function ms(seconds) {
  const value = seconds * 1000;
  return value >= 1000 ? `${(value / 1000).toFixed(2)} s` : `${Math.round(value)} ms`;
}

function usd(value) {
  if (!value) return "$0";
  return value < 0.01 ? `$${value.toFixed(6)}` : `$${value.toFixed(4)}`;
}

function renderAnswer(text, validById) {
  const target = $("answer");
  target.replaceChildren();
  let last = 0;
  for (const match of text.matchAll(CITE)) {
    if (match.index > last) target.append(text.slice(last, match.index));
    const id = match[1];
    const ref = el("button", "cite-ref", `[${id}]`);
    ref.type = "button";
    if (validById.get(id) === false) ref.classList.add("invalid");
    ref.title = validById.get(id) === false
      ? "This citation does not match anything that was retrieved"
      : "Show the cited passage";
    ref.addEventListener("click", () => {
      const card = document.querySelector(`[data-chunk="${CSS.escape(id)}"]`);
      if (card) {
        card.scrollIntoView({ behavior: "smooth", block: "center" });
        card.animate(
          [{ background: "var(--accent-soft)" }, { background: "var(--panel)" }],
          { duration: 900 },
        );
      }
    });
    target.append(ref);
    last = match.index + match[0].length;
  }
  target.append(text.slice(last));
}

function renderCitations(citations, chunks) {
  const byId = new Map(chunks.map((chunk) => [chunk.chunk_id, chunk]));
  const list = $("citations");
  list.replaceChildren();

  if (!citations.length) {
    $("cite-hint").textContent =
      "This answer cited nothing. For a regulatory question that is itself a finding.";
    return;
  }
  const bad = citations.filter((citation) => !citation.valid).length;
  $("cite-hint").textContent = bad
    ? `${citations.length} cited, ${bad} not matching anything retrieved.`
    : `${citations.length} cited, all matching retrieved passages.`;

  for (const citation of citations) {
    const chunk = byId.get(citation.chunk_id);
    const item = el("li", citation.valid ? "" : "invalid");
    item.dataset.chunk = citation.chunk_id;

    const top = el("div", "cite-top");
    top.append(el("span", "cite-id", citation.chunk_id));
    if (chunk) {
      const where = chunk.section ? `${chunk.doc_title} - ${chunk.section}` : chunk.doc_title;
      top.append(el("span", "cite-doc", where));
    }
    top.append(
      el("span", `flag ${citation.valid ? "ok" : "bad"}`,
        citation.valid ? "verified" : "unsupported"),
    );
    item.append(top);
    item.append(
      el("p", "cite-text",
        chunk ? chunk.text : "This id was cited but never retrieved, so there is no passage to show."),
    );
    list.append(item);
  }
}

function renderTrace(retrieval) {
  $("trace-count").textContent = `- ${retrieval.chunks.length} chunks, ${retrieval.retriever}, ${ms(retrieval.latency_s)}`;
  const list = $("trace");
  list.replaceChildren();
  for (const chunk of retrieval.chunks) {
    const item = el("li");
    const where = chunk.section ? `${chunk.doc_title} - ${chunk.section}` : chunk.doc_title;
    item.append(el("span", "cite-id", chunk.chunk_id), document.createTextNode(` ${where}`));
    item.append(el("span", "score", chunk.score.toFixed(4)));
    list.append(item);
  }
}

function renderProvenance(provenance, cost) {
  const rows = {
    corpus: provenance.corpus,
    "corpus hash": provenance.corpus_hash.slice(0, 12),
    "index hash": provenance.index_hash.slice(0, 12),
    generator: provenance.generator_model,
    "prompt hash": provenance.generator_prompt_hash.slice(0, 12),
    "config hash": provenance.config_hash.slice(0, 12),
    replayed: String(provenance.replayed),
    "context tokens": String(cost.context_tokens),
    "measured cost": usd(cost.usd),
    "projected cost": usd(cost.projected_usd),
  };
  const target = $("prov");
  target.replaceChildren();
  for (const [key, value] of Object.entries(rows)) {
    target.append(el("dt", "", key), el("dd", "", value));
  }
}

function renderBadges(payload) {
  const badges = $("badges");
  badges.replaceChildren();
  for (const label of [
    `${ms(payload.latency.total_s)} total`,
    `${ms(payload.retrieval.latency_s)} retrieval`,
    `${payload.cost.context_tokens} ctx tokens`,
    usd(payload.cost.usd || payload.cost.projected_usd),
  ]) {
    badges.append(el("span", "badge", label));
  }
}

function showError(message, detail) {
  const box = $("error");
  box.replaceChildren(document.createTextNode(message));
  if (detail) box.append(el("code", "", detail));
  box.hidden = false;
  $("result").hidden = true;
}

async function ask(question, k) {
  const button = $("go");
  button.disabled = true;
  button.textContent = "Asking";
  $("error").hidden = true;

  try {
    const response = await fetch("/query", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(k ? { question, k } : { question }),
    });
    if (!response.ok) {
      let detail = "";
      try {
        detail = (await response.json()).detail || "";
      } catch { /* a non-JSON error body is still worth reporting by status */ }
      const label = response.status === 429
        ? "Rate limited. Wait a moment and try again."
        : response.status === 401 || response.status === 403
          ? "Not authorised. This deployment requires an API key."
          : `The service returned HTTP ${response.status}.`;
      showError(label, detail);
      return;
    }
    const payload = await response.json();
    const validById = new Map(payload.citations.map((c) => [c.chunk_id, c.valid]));
    renderAnswer(payload.answer, validById);
    renderCitations(payload.citations, payload.retrieval.chunks);
    renderTrace(payload.retrieval);
    renderProvenance(payload.provenance, payload.cost);
    renderBadges(payload);
    $("result").hidden = false;
  } catch (error) {
    showError("Could not reach the service.", String(error));
  } finally {
    button.disabled = false;
    button.textContent = "Ask";
  }
}

$("ask").addEventListener("submit", (event) => {
  event.preventDefault();
  const question = $("q").value.trim();
  if (question) ask(question, Number($("k").value) || null);
});

$("q").addEventListener("keydown", (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
    $("ask").requestSubmit();
  }
});

$("examples").addEventListener("click", (event) => {
  const button = event.target.closest("button[data-q]");
  if (!button) return;
  $("q").value = button.dataset.q;
  $("ask").requestSubmit();
});

(async function boot() {
  const status = $("status");
  try {
    const health = await fetch("/health");
    if (!health.ok) throw new Error(`HTTP ${health.status}`);
    const info = await health.json();
    status.textContent = `${info.corpus} - ${info.chunks} chunks - ${info.retriever} - ${info.generator_model}`;
    status.className = "status ok";
    $("examples").hidden = false;
  } catch (error) {
    status.textContent = "service unavailable";
    status.className = "status down";
    showError("The service is not answering /health. It may still be building its index.", String(error));
  }
})();
