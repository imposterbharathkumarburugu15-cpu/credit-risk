// Dashboard logic. Talks to the FastAPI backend under /api.

const API = "/api";

const MODEL_COLORS = {
  logreg: "#4e79a7",
  svm: "#f28e2b",
  tree: "#59a14f",
  forest: "#b07aa1",
  xgboost: "#e15759",
};

let costChart = null;
let prChart = null;
let treeRules = [];          // kept around so we can re-shade rows when costs change
let latestComparison = null;
let latestScoreResult = null;

// ---------- small helpers ----------

function esc(text) {
  const div = document.createElement("div");
  div.textContent = String(text);
  return div.innerHTML;
}

async function getJSON(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url} failed (${res.status})`);
  return res.json();
}

function readCosts() {
  const val = (id) => Math.max(0, parseFloat(document.getElementById(id).value) || 0);
  return { fn: val("cost-fn"), fp: val("cost-fp"), tp: val("cost-tp"), tn: val("cost-tn") };
}

function costQuery(c) {
  return `cost_fn=${c.fn}&cost_fp=${c.fp}&cost_tp=${c.tp}&cost_tn=${c.tn}`;
}

const pct = (x) => (x * 100).toFixed(1) + "%";
const num = (x) => Number(x).toLocaleString(undefined, { maximumFractionDigits: 0 });

// ---------- model comparison table ----------

function renderComparison(data) {
  latestComparison = data;
  const rows = data.models;
  const lowest = Math.min(...rows.map((r) => r.at_best.total_cost));

  let html = `<tr>
    <th>Model</th><th>ROC-AUC</th><th>PR-AUC</th><th>Threshold</th>
    <th>Recall</th><th>Precision</th><th>Missed (FN)</th><th>False alarms (FP)</th>
    <th>Cost @ threshold</th><th>Cost @ 0.5</th><th>Saving</th></tr>`;

  for (const r of rows) {
    const b = r.at_best;
    html += `<tr class="${b.total_cost === lowest ? "best" : ""}">
      <td>${esc(r.label)}</td>
      <td>${r.roc_auc.toFixed(3)}</td>
      <td>${r.pr_auc.toFixed(3)}</td>
      <td>${r.best_threshold.toFixed(2)}</td>
      <td>${pct(b.recall)}</td>
      <td>${pct(b.precision)}</td>
      <td>${b.confusion.fn}</td>
      <td>${b.confusion.fp}</td>
      <td>${num(b.total_cost)}</td>
      <td>${num(r.cost_at_half)}</td>
      <td>${r.saving_pct}%</td>
    </tr>`;
  }
  document.getElementById("compare-table").innerHTML = html;

  const base = data.baselines;
  document.getElementById("baselines").textContent =
    `For reference, on the same ${num(data.test_size)} test customers: approving everyone would cost ` +
    `${num(base.approve_everyone)} and flagging everyone would cost ${num(base.flag_everyone)}. ` +
    `Green row = lowest total cost.`;

  shadeTreeRules();
}

// ---------- charts ----------

function renderCostChart(data) {
  const datasets = Object.entries(data.curves).map(([name, c]) => ({
    label: `${c.label} (best ${c.best_threshold.toFixed(2)})`,
    data: c.cost,
    borderColor: MODEL_COLORS[name],
    borderWidth: name === "xgboost" ? 3 : 1.6,
    pointRadius: 0,
    tension: 0.2,
  }));

  if (costChart) costChart.destroy();
  costChart = new Chart(document.getElementById("cost-chart"), {
    type: "line",
    data: { labels: data.thresholds, datasets },
    options: {
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      scales: {
        x: { title: { display: true, text: "Decision threshold" }, ticks: { maxTicksLimit: 10 } },
        y: { title: { display: true, text: "Total cost" } },
      },
      plugins: { legend: { position: "bottom", labels: { boxWidth: 14 } } },
    },
  });
}

function renderPRChart(data) {
  const datasets = Object.entries(data.curves).map(([name, c]) => ({
    label: `${c.label} (AP ${c.average_precision.toFixed(3)})`,
    data: c.points,
    borderColor: MODEL_COLORS[name],
    borderWidth: name === "xgboost" ? 3 : 1.6,
    pointRadius: 0,
    showLine: true,
  }));
  datasets.push({
    label: "No skill (default rate)",
    data: [{ x: 0, y: data.base_rate }, { x: 1, y: data.base_rate }],
    borderColor: "#999",
    borderDash: [6, 6],
    borderWidth: 1.2,
    pointRadius: 0,
    showLine: true,
  });

  if (prChart) prChart.destroy();
  prChart = new Chart(document.getElementById("pr-chart"), {
    type: "scatter",
    data: { datasets },
    options: {
      maintainAspectRatio: false,
      scales: {
        x: { min: 0, max: 1, title: { display: true, text: "Recall" } },
        y: { min: 0, max: 1, title: { display: true, text: "Precision" } },
      },
      plugins: { legend: { position: "bottom", labels: { boxWidth: 14 } } },
    },
  });
}

// ---------- decision rules ----------

function renderRules(data) {
  treeRules = data.tree_rules;
  shadeTreeRules();

  document.getElementById("logreg-table").innerHTML =
    "<tr><th>Feature</th><th>Coefficient</th></tr>" +
    data.logreg_top
      .map((r) => `<tr><td>${esc(r.feature)}</td><td>${r.coef > 0 ? "+" : ""}${r.coef}</td></tr>`)
      .join("");

  document.getElementById("xgb-table").innerHTML =
    "<tr><th>Feature</th><th>Importance</th></tr>" +
    data.xgb_importance
      .map((r) => `<tr><td>${esc(r.feature)}</td><td>${r.importance.toFixed(3)}</td></tr>`)
      .join("");
}

function shadeTreeRules() {
  if (!treeRules.length) return;
  // a leaf is flagged when its default rate reaches the tree's cost-optimal threshold
  const treeRow = latestComparison && latestComparison.models.find((m) => m.model === "tree");
  const cutoff = treeRow ? treeRow.best_threshold : null;

  let html = "<tr><th>If all of these hold</th><th>Default rate</th><th>Customers</th><th>Action</th></tr>";
  for (const r of treeRules) {
    const flagged = cutoff !== null && r.default_rate >= cutoff;
    html += `<tr class="${flagged ? "flagged" : ""}">
      <td class="cond">${r.conditions.map(esc).join("<br>AND ")}</td>
      <td>${pct(r.default_rate)}</td>
      <td>${r.customers}</td>
      <td>${flagged ? "Flag" : "Approve"}</td>
    </tr>`;
  }
  document.getElementById("tree-rules").innerHTML = html;
}

// ---------- customer form ----------

const PROFILE_FIELDS = [
  { name: "LIMIT_BAL", label: "Credit limit", type: "number", value: 50000 },
  { name: "AGE", label: "Age", type: "number", value: 35 },
  { name: "SEX", label: "Sex", type: "select", options: { 1: "Male", 2: "Female" }, value: 2 },
  {
    name: "EDUCATION", label: "Education", type: "select", value: 2,
    options: { 1: "Graduate school", 2: "University", 3: "High school", 4: "Other" },
  },
  { name: "MARRIAGE", label: "Marital status", type: "select", options: { 1: "Married", 2: "Single", 3: "Other" }, value: 2 },
];

const MONTH_ROWS = [
  { title: "Repayment status", names: ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"], value: 0 },
  { title: "Bill amount", names: ["BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6"], value: 20000 },
  { title: "Payment made", names: ["PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"], value: 1500 },
];

function buildForm() {
  const profile = document.getElementById("profile-fields");
  profile.innerHTML = PROFILE_FIELDS.map((f) => {
    if (f.type === "select") {
      const opts = Object.entries(f.options)
        .map(([v, text]) => `<option value="${v}" ${Number(v) === f.value ? "selected" : ""}>${text}</option>`)
        .join("");
      return `<label>${f.label}<select id="f-${f.name}">${opts}</select></label>`;
    }
    return `<label>${f.label}<input type="number" id="f-${f.name}" value="${f.value}"></label>`;
  }).join("");

  let html = "<tr><th></th>" +
    ["Month 1 (latest)", "Month 2", "Month 3", "Month 4", "Month 5", "Month 6"].map((m) => `<th>${m}</th>`).join("") +
    "</tr>";
  for (const row of MONTH_ROWS) {
    html += `<tr><th>${row.title}</th>` +
      row.names.map((n) => `<td><input type="number" id="f-${n}" value="${row.value}"></td>`).join("") +
      "</tr>";
  }
  document.getElementById("months-table").innerHTML = html;
}

function readCustomer() {
  const customer = {};
  const allNames = [...PROFILE_FIELDS.map((f) => f.name), ...MONTH_ROWS.flatMap((r) => r.names)];
  for (const name of allNames) {
    customer[name] = Number(document.getElementById(`f-${name}`).value);
  }
  return customer;
}

function fillCustomer(values) {
  for (const [name, value] of Object.entries(values)) {
    const el = document.getElementById(`f-${name}`);
    if (el) el.value = value;
  }
}

// ---------- scoring ----------

function renderResult(r) {
  const box = document.getElementById("result");
  box.classList.remove("empty");

  const maxAbs = Math.max(...r.reasons.map((x) => Math.abs(x.contribution)), 0.01);
  const reasons = r.reasons.map((x) => {
    const width = (Math.abs(x.contribution) / maxAbs) * 100;
    return `<li>
      <span>${esc(x.label)} = <b>${x.value}</b> &mdash; ${x.direction}</span>
      <div class="bar-track"><div class="bar ${x.contribution > 0 ? "up" : "down"}" style="width:${width}%"></div></div>
    </li>`;
  }).join("");

  box.innerHTML = `
    <div><span class="score-big">${pct(r.probability)}</span><span class="chip ${r.band}">${r.band}</span></div>
    <p><b>${esc(r.decision)}</b></p>
    <div class="meter">
      <div class="fill" style="width:${Math.min(r.probability * 100, 100)}%"></div>
      <div class="tick" style="left:${r.threshold * 100}%"></div>
    </div>
    <div class="meter-label">Estimated default probability. Black tick = cost-optimal threshold (${r.threshold.toFixed(2)}).</div>
    <h3>Why this score</h3>
    <ul class="reasons">${reasons}</ul>
    <div class="meter-label">Red pushes the risk up, green pushes it down (from ${esc(r.model)}).</div>`;
}

async function scoreCustomer() {
  const box = document.getElementById("result");
  try {
    const res = await fetch(`${API}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ customer: readCustomer(), costs: readCosts() }),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(JSON.stringify(err.detail));
    }
    const data = await res.json();
    // Normalize probability field between /api/predict variants
    if (data.probability === undefined && data.default_probability !== undefined) {
      data.probability = data.default_probability;
    }
    latestScoreResult = data;
    renderResult(data);
  } catch (e) {
    box.classList.remove("empty");
    box.innerHTML = `<p class="error">Could not score: ${esc(e.message)}</p>`;
  }
}

async function sendNotificationEmail() {
  const statusEl = document.getElementById("email-status-msg");
  const recipient = document.getElementById("email-recipient").value.trim();
  const customerName = document.getElementById("email-customer-name").value.trim() || "Customer";
  const eventType = document.getElementById("email-event-type").value;
  const employeeKey = document.getElementById("email-employee-key").value.trim();

  if (eventType !== "high_risk_review" && !recipient) {
    statusEl.textContent = "Please enter a recipient email.";
    statusEl.className = "hint error";
    return;
  }

  statusEl.textContent = "Sending...";
  statusEl.className = "hint";

  const band = latestScoreResult ? (latestScoreResult.band || latestScoreResult.risk_classification || "MEDIUM") : "MEDIUM";
  const decisionText = band === "HIGH" ? "DECLINED" : "APPROVED";

  try {
    const headers = { "Content-Type": "application/json" };
    if (employeeKey) headers["X-Employee-Key"] = employeeKey;

    const res = await fetch(`${API}/notifications/send`, {
      method: "POST",
      headers,
      body: JSON.stringify({
        event_type: eventType,
        recipient_email: recipient || null,
        customer_name: customerName,
        risk_classification: band,
        status: band === "HIGH" ? "UNDER_REVIEW" : "SUBMITTED",
        decision: decisionText,
      }),
    });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.detail || `HTTP ${res.status}`);
    }
    statusEl.textContent = `${data.status}: ${data.message}`;
    statusEl.className = data.success ? "hint success-msg" : "hint error";
  } catch (e) {
    statusEl.textContent = `Error: ${e.message}`;
    statusEl.className = "hint error";
  }
}

// ---------- wiring ----------

async function refreshAnalysis() {
  const q = costQuery(readCosts());
  const [comparison, curves] = await Promise.all([
    getJSON(`${API}/models/comparison?${q}`),
    getJSON(`${API}/cost-curves?${q}`),
  ]);
  renderComparison(comparison);
  renderCostChart(curves);
}

async function init() {
  buildForm();

  document.getElementById("apply-costs").addEventListener("click", refreshAnalysis);
  document.getElementById("score-btn").addEventListener("click", scoreCustomer);
  document.getElementById("send-email-btn").addEventListener("click", sendNotificationEmail);
  document.querySelectorAll("[data-sample]").forEach((btn) => {
    btn.addEventListener("click", async () => {
      fillCustomer(await getJSON(`${API}/sample/${btn.dataset.sample}`));
      scoreCustomer();
    });
  });

  try {
    const [pr, rules] = await Promise.all([getJSON(`${API}/pr-curves`), getJSON(`${API}/rules`)]);
    renderPRChart(pr);
    renderRules(rules);
    await refreshAnalysis();
  } catch (e) {
    document.querySelector("main").insertAdjacentHTML(
      "afterbegin",
      `<p class="error">Couldn't reach the API: ${esc(e.message)}. Is the backend running?</p>`
    );
  }
}

init();
