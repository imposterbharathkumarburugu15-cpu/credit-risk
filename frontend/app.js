const API = "/api";

const VIEW_META = {
  dashboard: ["Abhaya Risk Dashboard", "Cost-sensitive credit default intelligence & collections portal"],
  single: ["Single Applicant Assessment", "View applicant's credit score, payment history and loan eligibility"],
  bulk: ["Bulk Applicant Processing", "Upload applicant CSV, predict default risk and manage collections"],
  applicants: ["Applicants Directory", "Browse demo and uploaded applicants scored by XGBoost"],
  emails: ["Automated Emails", "Send Abhaya workflow notifications and monitor SMTP delivery"],
  reports: ["Model & Cost Reports", "Compare models, tune FN/FP costs and inspect decision rules"],
  settings: ["Portal Settings", "Configure Employee API Key and inspect MySQL, ML & SMTP health"],
};

const MODEL_COLORS = {
  logreg: "#60a5fa",
  svm: "#f59e0b",
  tree: "#34d399",
  forest: "#c084fc",
  xgboost: "#f4cd68",
};

const EDU_LABELS = { 1: "Graduate School", 2: "University", 3: "High School", 4: "Other" };
const MONTHS_6 = ["Month -1", "Month -2", "Month -3", "Month -4", "Month -5", "Month -6"];

const state = {
  applicants: [],
  rawCsvRecords: [],
  bulkTab: "default",
  bulkPage: 1,
  bulkPageSize: 5,
  bulkSelectedIds: new Set(),
  activeBulkId: null,
  emailTab: "pending",
  emailSelectedIds: new Set(),
  emailLogs: [],
  employeeKey: sessionStorage.getItem("abhaya_emp_key") || "",
  charts: { single: null, singleUtil: null, cost: null, pr: null, dashRisk: null, dashExposure: null, dashScoreDist: null },
};

function showToast(msg, kind = "info") {
  const wrap = document.getElementById("toast-container");
  if (!wrap) return;
  const el = document.createElement("div");
  el.className = `toast ${kind}`;
  el.textContent = msg;
  wrap.appendChild(el);
  setTimeout(() => el.remove(), 3600);
}

// ---------- Helpers ----------
function esc(v) {
  const d = document.createElement("div");
  d.textContent = v == null ? "" : String(v);
  return d.innerHTML;
}

function fmtINR(n) {
  const val = Number(n || 0);
  return "₹ " + val.toLocaleString("en-IN", { maximumFractionDigits: 0 });
}

function probToScore(prob) {
  const p = Math.min(1, Math.max(0, Number(prob || 0)));
  return Math.round(300 + (1 - p) * 600);
}

function riskPillClass(risk) {
  const r = String(risk || "").toUpperCase();
  if (r === "HIGH") return "red";
  if (r === "MEDIUM") return "amber";
  return "green";
}

function setGauge(arcId, scoreId, score, dotId = null) {
  const arc = document.getElementById(arcId);
  const label = document.getElementById(scoreId);
  if (!arc || !label) return;
  const clamped = Math.min(900, Math.max(300, Number(score || 300)));
  const ratio = (clamped - 300) / 600;
  const totalLen = 276;
  arc.style.strokeDashoffset = String(totalLen * (1 - ratio));
  label.textContent = String(clamped);
  if (dotId) {
    const dot = document.getElementById(dotId);
    if (dot) {
      const angle = Math.PI * (1 - ratio);
      dot.setAttribute("cx", (110 + 88 * Math.cos(angle)).toFixed(1));
      dot.setAttribute("cy", (110 - 88 * Math.sin(angle)).toFixed(1));
    }
  }
}

async function apiGet(path, headers = {}) {
  const res = await fetch(`${API}${path}`, { headers });
  if (!res.ok) throw new Error(`${path} failed (${res.status})`);
  return res.json();
}

async function apiPost(path, body, headers = {}) {
  const res = await fetch(`${API}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...headers },
    body: JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || `${path} failed (${res.status})`);
  return data;
}

// ---------- Navigation ----------
function switchView(viewName) {
  document.querySelectorAll(".view").forEach((v) => v.classList.remove("active"));
  document.querySelectorAll(".nav-item").forEach((b) => b.classList.remove("active"));
  const target = document.getElementById(`view-${viewName}`);
  if (target) target.classList.add("active");
  const navBtn = document.querySelector(`.nav-item[data-view="${viewName}"]`);
  if (navBtn) navBtn.classList.add("active");

  const [title, sub] = VIEW_META[viewName] || VIEW_META.dashboard;
  document.getElementById("page-title").textContent = title;
  document.getElementById("page-subtitle").textContent = sub;

  if (viewName === "reports") {
    if (!state.charts.cost) loadReports();
    else {
      state.charts.cost.resize();
      if (state.charts.pr) state.charts.pr.resize();
    }
  }
  if (viewName === "emails") {
    renderEmailView();
  }
  if (viewName === "applicants") {
    renderDirectory();
  }
}

// ---------- CSV Parser ----------
function parseCSV(text) {
  const cleanText = String(text || "").replace(/^\uFEFF/, "");
  const lines = cleanText.replace(/\r\n/g, "\n").replace(/\r/g, "\n").split("\n").filter((l) => l.trim());
  if (lines.length < 2) return [];

  const splitRow = (line) => {
    const out = [];
    let cur = "";
    let inQuotes = false;
    for (let i = 0; i < line.length; i++) {
      const ch = line[i];
      if (ch === '"') {
        if (inQuotes && line[i + 1] === '"') { cur += '"'; i++; }
        else inQuotes = !inQuotes;
      } else if (ch === "," && !inQuotes) {
        out.push(cur.trim());
        cur = "";
      } else {
        cur += ch;
      }
    }
    out.push(cur.trim());
    return out;
  };

  const headers = splitRow(lines[0]);
  const textCols = new Set(["record_id", "customer_id", "id", "customer_name", "name", "email", "email_address", "application_id", "app_id", "due_date"]);
  const rows = [];

  for (let i = 1; i < lines.length && i <= 10000; i++) {
    const vals = splitRow(lines[i]);
    if (vals.length === 1 && !vals[0]) continue;
    const obj = {};
    headers.forEach((h, idx) => {
      const raw = vals[idx] ?? "";
      if (textCols.has(h)) {
        obj[h] = raw;
      } else if (raw !== "" && !Number.isNaN(Number(raw))) {
        obj[h] = Number(raw);
      } else {
        obj[h] = raw;
      }
    });
    rows.push(obj);
  }
  if (lines.length - 1 > 10000) {
    showToast(`CSV truncated to first 10,000 rows (${lines.length - 1} total in file).`, "info");
  }
  return rows;
}

function normalizeApplicant(item, idx = 0) {
  const fs = item.features_summary || {};
  const bills = Array.isArray(fs.bills) ? fs.bills : [0, 0, 0, 0, 0, 0];
  const payments = Array.isArray(fs.payments) ? fs.payments : [0, 0, 0, 0, 0, 0];
  const payStatuses = Array.isArray(fs.pay_statuses) ? fs.pay_statuses : [fs.pay_0 ?? 0, 0, 0, 0, 0, 0];
  const prob = Number(item.default_probability || 0);
  const dueAmt = fs.due_amount != null && fs.due_amount !== ""
    ? Number(fs.due_amount)
    : Math.max(0, Number(bills[0] || 0));

  return {
    customer_id: String(item.customer_id || `CUST-${idx + 1}`),
    customer_name: String(item.customer_name || `Applicant #${idx + 1}`),
    email: String(item.email || ""),
    application_id: String(item.application_id || `APP-${idx + 1001}`),
    default_probability: prob,
    score: probToScore(prob),
    risk_classification: String(item.risk_classification || "LOW").toUpperCase(),
    flagged: Boolean(item.flagged),
    suggested_action: item.suggested_action || "",
    due_amount: dueAmt,
    due_date: fs.due_date || "05 Dec 2026",
    limit_bal: Number(fs.limit_bal || 0),
    age: Number(fs.age || 0),
    sex: Number(fs.sex || 2),
    education: Number(fs.education || 2),
    marriage: Number(fs.marriage || 1),
    bills,
    payments,
    pay_statuses: payStatuses,
  };
}

// ---------- Dashboard (Employer Portfolio View) ----------
function renderDashboard() {
  const total = state.applicants.length;
  const highApps = state.applicants.filter((a) => a.flagged || a.risk_classification === "HIGH");
  const highC = state.applicants.filter((a) => a.risk_classification === "HIGH").length;
  const medC = state.applicants.filter((a) => a.risk_classification === "MEDIUM").length;
  const lowC = state.applicants.filter((a) => a.risk_classification === "LOW").length;

  const totalDue = state.applicants.reduce((s, a) => s + Number(a.due_amount || 0), 0);
  const exposureRisk = highApps.reduce((s, a) => s + Number(a.due_amount || 0), 0);
  const expectedLoss = state.applicants.reduce((s, a) => s + Number(a.default_probability || 0) * Number(a.due_amount || 0), 0);
  const avgScore = total ? Math.round(state.applicants.reduce((s, a) => s + Number(a.score || 0), 0) / total) : 0;
  const expShare = totalDue > 0 ? ((exposureRisk / totalDue) * 100).toFixed(1) : "0.0";

  document.getElementById("dash-exposure-risk").textContent = fmtINR(exposureRisk);
  document.getElementById("dash-exposure-share").textContent = `${expShare}% of total portfolio due`;
  document.getElementById("dash-expected-loss").textContent = fmtINR(expectedLoss);
  document.getElementById("dash-total-due-meta").textContent = `Total portfolio due: ${fmtINR(totalDue)}`;
  document.getElementById("dash-avg-score").textContent = total ? `${avgScore} / 900` : "--";
  document.getElementById("dash-default-rate").textContent = `${highC} high · ${medC} med · ${lowC} low`;
  document.getElementById("dash-mix-label").textContent = `${total} Applicants`;
  document.getElementById("notif-count").textContent = String(highApps.length);

  // Employer Analytics Strip
  const avgUtil = total
    ? state.applicants.reduce((s, a) => s + (a.limit_bal > 0 ? (Math.max(0, a.bills[0] || 0) / a.limit_bal) * 100 : 0), 0) / total
    : 0;
  const onTimeTotal = state.applicants.reduce((s, a) => s + a.pay_statuses.filter((p) => Number(p) <= 0).length, 0);
  const onTimeDenom = state.applicants.reduce((s, a) => s + a.pay_statuses.length, 0) || 1;
  const severeLate = state.applicants.filter((a) => a.pay_statuses.some((p) => Number(p) >= 2)).length;
  document.getElementById("dash-avg-util").textContent = `${avgUtil.toFixed(1)}%`;
  document.getElementById("dash-ontime-rate").textContent = `${((onTimeTotal / onTimeDenom) * 100).toFixed(1)}%`;
  document.getElementById("dash-severe-late").textContent = String(severeLate);
  document.getElementById("dash-recovery-target").textContent = fmtINR(exposureRisk * 0.65);

  // Sort by highest exposure * default risk for employer collections priority
  const priorityList = [...state.applicants].sort((a, b) => (b.default_probability * b.due_amount) - (a.default_probability * a.due_amount));
  const tbody = document.querySelector("#dash-snapshot-table tbody");
  tbody.innerHTML = priorityList.slice(0, 5).map((a) => `
    <tr data-open-single="${esc(a.customer_id)}" data-open-email="${esc(a.email)}" tabindex="0">
      <td>${esc(a.customer_id)}</td>
      <td>${esc(a.customer_name)}</td>
      <td><strong>${a.score}</strong></td>
      <td>${fmtINR(a.due_amount)}</td>
      <td><span class="pill ${riskPillClass(a.risk_classification)}">${esc(a.risk_classification)}</span></td>
    </tr>
  `).join("") || `<tr><td colspan="5" class="hint">No applicants loaded yet.</td></tr>`;

  tbody.querySelectorAll("tr[data-open-single]").forEach((tr) => {
    const open = () => {
      const targetKey = tr.dataset.openEmail || tr.dataset.openSingle;
      document.getElementById("single-id-input").value = targetKey;
      switchView("single");
      loadSingleById(targetKey);
    };
    tr.addEventListener("click", open);
    tr.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); open(); } });
  });

  // Portfolio Risk Mix Donut Chart
  const riskCanvas = document.getElementById("dash-risk-chart");
  if (riskCanvas) {
    if (state.charts.dashRisk) state.charts.dashRisk.destroy();
    state.charts.dashRisk = new Chart(riskCanvas, {
      type: "doughnut",
      data: {
        labels: ["High", "Medium", "Low"],
        datasets: [{
          data: total ? [highC, medC, lowC] : [0, 0, 1],
          backgroundColor: total ? ["#ef4444", "#f59e0b", "#22c55e"] : ["#262a35", "#262a35", "#262a35"],
          borderWidth: 0,
        }],
      },
      options: {
        maintainAspectRatio: false,
        cutout: "68%",
        plugins: {
          legend: { position: "bottom", labels: { color: "#9ca3af", boxWidth: 10, font: { size: 11 } } },
        },
      },
    });
  }

  // Exposure by Risk Tier Bar Chart
  const expCanvas = document.getElementById("dash-exposure-chart");
  if (expCanvas) {
    const sumTier = (tier) => state.applicants.filter((a) => a.risk_classification === tier).reduce((s, a) => s + Number(a.due_amount || 0), 0);
    if (state.charts.dashExposure) state.charts.dashExposure.destroy();
    state.charts.dashExposure = new Chart(expCanvas, {
      type: "bar",
      data: {
        labels: ["High Risk", "Medium Risk", "Low Risk"],
        datasets: [{
          label: "Exposure (₹)",
          data: [sumTier("HIGH"), sumTier("MEDIUM"), sumTier("LOW")],
          backgroundColor: ["rgba(239,68,68,0.8)", "rgba(245,158,11,0.8)", "rgba(34,197,94,0.8)"],
          borderRadius: 6,
        }],
      },
      options: {
        maintainAspectRatio: false,
        scales: {
          x: { ticks: { color: "#9ca3af", font: { size: 11 } }, grid: { display: false } },
          y: { ticks: { color: "#9ca3af", font: { size: 10 } }, grid: { color: "rgba(255,255,255,0.06)" } },
        },
        plugins: { legend: { display: false } },
      },
    });
  }

  // Credit Score Distribution Histogram
  const distCanvas = document.getElementById("dash-score-dist-chart");
  if (distCanvas) {
    const buckets = [0, 0, 0, 0, 0];
    state.applicants.forEach((a) => {
      if (a.score < 450) buckets[0]++;
      else if (a.score < 550) buckets[1]++;
      else if (a.score < 650) buckets[2]++;
      else if (a.score < 750) buckets[3]++;
      else buckets[4]++;
    });
    if (state.charts.dashScoreDist) state.charts.dashScoreDist.destroy();
    state.charts.dashScoreDist = new Chart(distCanvas, {
      type: "bar",
      data: {
        labels: ["300-449", "450-549", "550-649", "650-749", "750-900"],
        datasets: [{
          label: "Applicants",
          data: buckets,
          backgroundColor: "#f4cd68",
          borderRadius: 5,
        }],
      },
      options: {
        maintainAspectRatio: false,
        scales: {
          x: { ticks: { color: "#9ca3af", font: { size: 10 } }, grid: { display: false } },
          y: { ticks: { color: "#9ca3af", stepSize: 1, font: { size: 10 } }, grid: { color: "rgba(255,255,255,0.06)" } },
        },
        plugins: { legend: { display: false } },
      },
    });
  }
}

// ---------- Single Applicant Assessment ----------
async function loadSingleById(idOrKind) {
  const query = String(idOrKind || "").trim();
  if (!query) return;

  let app = state.applicants.find(
    (a) =>
      (a.email && a.email.toLowerCase() === query.toLowerCase()) ||
      a.customer_id.toLowerCase() === query.toLowerCase()
  );

  let reasons = [];
  if (app) {
    // Score via /api/predict to get real TreeSHAP reasons
    try {
      const payload = {
        LIMIT_BAL: app.limit_bal || 50000,
        SEX: app.sex || 2,
        EDUCATION: app.education || 2,
        MARRIAGE: app.marriage || 1,
        AGE: app.age || 30,
        PAY_0: app.pay_statuses[0] ?? 0,
        PAY_2: app.pay_statuses[1] ?? 0,
        PAY_3: app.pay_statuses[2] ?? 0,
        PAY_4: app.pay_statuses[3] ?? 0,
        PAY_5: app.pay_statuses[4] ?? 0,
        PAY_6: app.pay_statuses[5] ?? 0,
        BILL_AMT1: app.bills[0] ?? 0,
        BILL_AMT2: app.bills[1] ?? 0,
        BILL_AMT3: app.bills[2] ?? 0,
        BILL_AMT4: app.bills[3] ?? 0,
        BILL_AMT5: app.bills[4] ?? 0,
        BILL_AMT6: app.bills[5] ?? 0,
        PAY_AMT1: app.payments[0] ?? 0,
        PAY_AMT2: app.payments[1] ?? 0,
        PAY_AMT3: app.payments[2] ?? 0,
        PAY_AMT4: app.payments[3] ?? 0,
        PAY_AMT5: app.payments[4] ?? 0,
        PAY_AMT6: app.payments[5] ?? 0,
      };
      const pred = await apiPost("/predict", payload);
      reasons = pred.reasons || [];
      app.default_probability = pred.default_probability;
      app.score = probToScore(pred.default_probability);
      app.risk_classification = pred.risk_classification;
      app.flagged = Boolean(pred.flagged);
      app.suggested_action = pred.suggested_action;
    } catch (_) {}
  } else if (["low", "medium", "high"].includes(query.toLowerCase())) {
    const kind = query.toLowerCase();
    try {
      const raw = await apiGet(`/sample/${kind}`);
      const pred = await apiPost("/predict", raw);
      reasons = pred.reasons || [];
      app = normalizeApplicant({
        customer_id: query.toUpperCase(),
        customer_name: `${kind.charAt(0).toUpperCase() + kind.slice(1)} Risk Sample`,
        email: "",
        default_probability: pred.default_probability,
        risk_classification: pred.risk_classification,
        suggested_action: pred.suggested_action,
        flagged: pred.flagged,
        features_summary: {
          limit_bal: raw.LIMIT_BAL,
          age: raw.AGE,
          sex: raw.SEX,
          education: raw.EDUCATION,
          marriage: raw.MARRIAGE,
          bills: [raw.BILL_AMT1, raw.BILL_AMT2, raw.BILL_AMT3, raw.BILL_AMT4, raw.BILL_AMT5, raw.BILL_AMT6],
          payments: [raw.PAY_AMT1, raw.PAY_AMT2, raw.PAY_AMT3, raw.PAY_AMT4, raw.PAY_AMT5, raw.PAY_AMT6],
          pay_statuses: [raw.PAY_0, raw.PAY_2, raw.PAY_3, raw.PAY_4, raw.PAY_5, raw.PAY_6],
        },
      });
    } catch (err) {
      showToast(`Could not load sample: ${err.message}`, "error");
      return;
    }
  } else {
    showToast(`Applicant "${query}" not found. Enter an email from loaded/demo records or click Low/Medium/High.`, "error");
    return;
  }

  renderSingleApplicant(app, reasons);
}

function renderSingleApplicant(app, reasons = []) {
  setGauge("single-gauge-arc", "single-score", app.score, "single-gauge-dot");
  const badge = document.getElementById("single-credit-badge");
  if (app.score >= 700) {
    badge.className = "pill green";
    badge.textContent = "Good Credit";
  } else if (app.score >= 560) {
    badge.className = "pill amber";
    badge.textContent = "Fair Credit";
  } else {
    badge.className = "pill red";
    badge.textContent = "High Risk";
  }

  state.currentSingleApp = app;
  document.getElementById("sd-email").textContent = app.email || "Not provided";
  document.getElementById("sd-name").textContent = app.customer_name;
  document.getElementById("sd-age").textContent = app.age ? `${app.age} Years` : "--";
  document.getElementById("sd-edu").textContent = EDU_LABELS[app.education] || "University";
  document.getElementById("sd-limit").textContent = fmtINR(app.limit_bal);

  const approvalProb = Math.max(0, 1 - app.default_probability);
  const currentLimit = Number(app.limit_bal || 0);
  let recLimit = currentLimit;
  let limitLabel = "Eligible Credit Limit";
  let payLabel = "Recommended Tenure";
  let payValue = "60 Months";
  let limitNote = "Strong credit score — eligible for full limit or expansion.";

  if (app.risk_classification === "HIGH" || app.score < 560) {
    recLimit = Math.round(currentLimit * 0.5);
    const reducedEmi = app.due_amount > 0 ? Math.max(1000, Math.round(app.due_amount / 6)) : 0;
    limitLabel = "Reduced Credit Limit (-50%)";
    payLabel = "Reduced Monthly Payment";
    payValue = reducedEmi > 0 ? `${fmtINR(reducedEmi)} / mo` : "₹ 0 (No Due)";
    limitNote = `Low credit score (${app.score}/900): reduce credit limit from ${fmtINR(currentLimit)} to ${fmtINR(recLimit)} and lower monthly payment to prevent next-cycle default.`;
  } else if (app.risk_classification === "MEDIUM" || app.score < 700) {
    recLimit = Math.round(currentLimit * 0.8);
    const reducedEmi = app.due_amount > 0 ? Math.max(1500, Math.round(app.due_amount / 4)) : 0;
    limitLabel = "Capped Credit Limit (-20%)";
    payLabel = "Adjusted Monthly Payment";
    payValue = reducedEmi > 0 ? `${fmtINR(reducedEmi)} / mo` : "₹ 0 (No Due)";
    limitNote = `Moderate risk (${app.score}/900): cap credit limit at ${fmtINR(recLimit)} and monitor monthly payments.`;
  } else {
    recLimit = Math.round(currentLimit * (0.8 + approvalProb * 0.4));
  }

  document.getElementById("single-elig-label").textContent = limitLabel;
  document.getElementById("single-pay-label").textContent = payLabel;
  document.getElementById("single-elig-amount").textContent = fmtINR(recLimit);
  document.getElementById("single-elig-tenure").textContent = payValue;
  document.getElementById("single-limit-note").textContent = limitNote;

  const pays = app.payments.map((x) => Number(x || 0));
  const bills = app.bills.map((x) => Math.max(0, Number(x || 0)));
  const totalPaid = pays.reduce((a, b) => a + b, 0);
  const avgPay = totalPaid / (pays.length || 1);
  const variance = pays.reduce((acc, p) => acc + Math.pow(p - avgPay, 2), 0) / (pays.length || 1);
  const stdDev = Math.sqrt(variance);
  const onTimeCount = app.pay_statuses.filter((s) => Number(s) <= 0).length;
  const onTimePct = ((onTimeCount / (app.pay_statuses.length || 1)) * 100).toFixed(1) + "%";

  document.getElementById("si-avg-pay").textContent = fmtINR(avgPay);
  document.getElementById("si-ontime").textContent = onTimePct;
  document.getElementById("si-total-paid").textContent = fmtINR(totalPaid);
  document.getElementById("si-total-due").textContent = fmtINR(app.due_amount);
  document.getElementById("si-std").textContent = fmtINR(stdDev);

  // Past Payments History table
  const histBody = document.querySelector("#single-history-table tbody");
  histBody.innerHTML = MONTHS_6.map((m, i) => {
    const code = Number(app.pay_statuses[i] ?? 0);
    const paid = code <= 0;
    return `<tr>
      <td>${m}</td>
      <td>${fmtINR(bills[i])}</td>
      <td>${fmtINR(pays[i])}</td>
      <td>${code}</td>
      <td><span class="pill ${paid ? "green" : "red"}">${paid ? "Paid / Current" : `${code}M Late`}</span></td>
    </tr>`;
  }).join("");

  // Risk & Recommendation
  const riskPill = document.getElementById("sr-risk-pill");
  riskPill.className = `pill ${riskPillClass(app.risk_classification)}`;
  riskPill.textContent = `${app.risk_classification} Risk`;
  document.getElementById("sr-approval-prob").textContent = `${(approvalProb * 100).toFixed(0)}%`;
  document.getElementById("sr-recommendation").textContent =
    app.suggested_action || `Predicted default probability is ${(app.default_probability * 100).toFixed(1)}%.`;

  const shapEl = document.getElementById("sr-shap-reasons");
  shapEl.innerHTML = reasons.slice(0, 4).map((r) =>
    `<li>${esc(r.feature)}: <strong>${esc(r.direction)}</strong> (impact ${Number(r.shap_value).toFixed(3)})</li>`
  ).join("");

  // Payment Behavior Chart
  const ctx = document.getElementById("single-payment-chart");
  if (state.charts.single) state.charts.single.destroy();
  state.charts.single = new Chart(ctx, {
    type: "bar",
    data: {
      labels: [...MONTHS_6].reverse(),
      datasets: [
        { label: "Paid Amount", data: [...pays].reverse(), backgroundColor: "#e6b447", borderRadius: 4 },
        { label: "Due Amount", data: [...bills].reverse(), backgroundColor: "#333846", borderRadius: 4 },
      ],
    },
    options: {
      maintainAspectRatio: false,
      scales: {
        x: { ticks: { color: "#9ca3af" }, grid: { display: false } },
        y: { ticks: { color: "#9ca3af" }, grid: { color: "rgba(255,255,255,0.06)" } },
      },
      plugins: { legend: { display: false } },
    },
  });

  // Utilization Trend Chart
  const limit = Math.max(1, Number(app.limit_bal || 1));
  const utils = bills.map((b) => Number(((b / limit) * 100).toFixed(1)));
  document.getElementById("single-util-badge").textContent = `${utils[0].toFixed(1)}% Latest`;
  const uCtx = document.getElementById("single-util-chart");
  if (uCtx) {
    if (state.charts.singleUtil) state.charts.singleUtil.destroy();
    state.charts.singleUtil = new Chart(uCtx, {
      type: "line",
      data: {
        labels: [...MONTHS_6].reverse(),
        datasets: [{
          label: "Utilization %",
          data: [...utils].reverse(),
          borderColor: "#f4cd68",
          backgroundColor: "rgba(244, 205, 104, 0.14)",
          fill: true,
          tension: 0.3,
          pointRadius: 3,
        }],
      },
      options: {
        maintainAspectRatio: false,
        scales: {
          x: { ticks: { color: "#9ca3af" }, grid: { display: false } },
          y: { ticks: { color: "#9ca3af" }, grid: { color: "rgba(255,255,255,0.06)" } },
        },
        plugins: { legend: { display: false } },
      },
    });
  }

  populateRaw23AndRender46(app);
}

const PAY_KEYS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"];

function populateRaw23AndRender46(app) {
  document.getElementById("in-LIMIT_BAL").value = app.limit_bal || 50000;
  document.getElementById("in-SEX").value = app.sex || 2;
  document.getElementById("in-EDUCATION").value = app.education || 2;
  document.getElementById("in-MARRIAGE").value = app.marriage || 1;
  document.getElementById("in-AGE").value = app.age || 30;

  const tbody = document.getElementById("raw-months-inputs");
  if (tbody) {
    tbody.innerHTML = MONTHS_6.map((m, i) => `
      <tr>
        <td>${m}</td>
        <td><input type="number" class="input compact" id="in-${PAY_KEYS[i]}" value="${Number(app.pay_statuses[i] ?? 0)}" min="-2" max="8"></td>
        <td><input type="number" class="input compact" id="in-BILL_AMT${i + 1}" value="${Number(app.bills[i] ?? 0)}"></td>
        <td><input type="number" class="input compact" id="in-PAY_AMT${i + 1}" value="${Number(app.payments[i] ?? 0)}" min="0"></td>
      </tr>
    `).join("");
  }

  render46Features(app);
}

function readRaw23FromForm() {
  const numVal = (id) => Number(document.getElementById(id)?.value || 0);
  const raw = {
    LIMIT_BAL: numVal("in-LIMIT_BAL") || 50000,
    SEX: numVal("in-SEX") || 2,
    EDUCATION: numVal("in-EDUCATION") || 2,
    MARRIAGE: numVal("in-MARRIAGE") || 1,
    AGE: numVal("in-AGE") || 30,
  };
  PAY_KEYS.forEach((k, i) => {
    raw[k] = numVal(`in-${k}`);
    raw[`BILL_AMT${i + 1}`] = numVal(`in-BILL_AMT${i + 1}`);
    raw[`PAY_AMT${i + 1}`] = numVal(`in-PAY_AMT${i + 1}`);
  });
  return raw;
}

function slope6(arrOldToNew) {
  const n = arrOldToNew.length;
  const meanX = (n - 1) / 2;
  const meanY = arrOldToNew.reduce((a, b) => a + b, 0) / n;
  let num = 0;
  let den = 0;
  arrOldToNew.forEach((y, x) => {
    num += (x - meanX) * (y - meanY);
    den += (x - meanX) ** 2;
  });
  return den ? num / den : 0;
}

function render46Features(app) {
  const grid = document.getElementById("features-46-grid");
  if (!grid) return;

  const limit = Math.max(1, Number(app.limit_bal || 1));
  const bills = app.bills.map(Number);
  const pays = app.payments.map(Number);
  const statuses = app.pay_statuses.map(Number);
  const utils = bills.map((b) => b / limit);
  const delays = statuses.map((s) => Math.max(0, s));

  const mean = (a) => a.reduce((x, y) => x + y, 0) / a.length;
  const std = (a) => { const m = mean(a); return Math.sqrt(mean(a.map((v) => (v - m) ** 2))); };

  let streak = 0;
  for (const s of statuses) { if (s > 0) streak++; else break; }

  const totalBill = bills.reduce((a, b) => a + Math.max(0, b), 0);
  const totalPay = pays.reduce((a, b) => a + b, 0);

  const pairs = [
    ["LIMIT_BAL", fmtINR(app.limit_bal)], ["SEX", app.sex], ["EDUCATION", app.education], ["MARRIAGE", app.marriage], ["AGE", app.age],
    ...PAY_KEYS.map((k, i) => [k, statuses[i]]),
    ...bills.map((b, i) => [`BILL_AMT${i + 1}`, fmtINR(b)]),
    ...pays.map((p, i) => [`PAY_AMT${i + 1}`, fmtINR(p)]),
    ["utilization_latest", (utils[0] * 100).toFixed(1) + "%"],
    ["utilization_mean_6m", (mean(utils) * 100).toFixed(1) + "%"],
    ["utilization_max_6m", (Math.max(...utils) * 100).toFixed(1) + "%"],
    ["utilization_std_6m", (std(utils) * 100).toFixed(1) + "%"],
    ["utilization_trend_6m", slope6([...utils].reverse()).toFixed(3)],
    ["avg_bill_6m", fmtINR(mean(bills))],
    ["max_bill_6m", fmtINR(Math.max(...bills))],
    ["bill_volatility_6m", fmtINR(std(bills))],
    ["bill_trend_slope_6m", Math.round(slope6([...bills].reverse()))],
    ["total_payment_6m", fmtINR(totalPay)],
    ["avg_payment_6m", fmtINR(mean(pays))],
    ["payment_volatility_6m", fmtINR(std(pays))],
    ["payment_trend_slope_6m", Math.round(slope6([...pays].reverse()))],
    ["payment_active_months_6m", pays.filter((p) => p > 0).length],
    ["late_month_count_6m", statuses.filter((s) => s > 0).length],
    ["severe_late_month_count_6m", statuses.filter((s) => s >= 2).length],
    ["recent_late_month_count_3m", statuses.slice(0, 3).filter((s) => s > 0).length],
    ["max_delay_severity_6m", Math.max(...delays)],
    ["avg_delay_severity_6m", mean(delays).toFixed(2)],
    ["consecutive_recent_late_months", streak],
    ["credit_score_300_900", app.score],
    ["payment_to_bill_ratio", totalBill > 0 ? ((totalPay / totalBill) * 100).toFixed(1) + "%" : "100%"],
    ["credit_headroom", fmtINR(Math.max(0, limit - Math.max(0, bills[0] || 0)))],
  ];

  grid.innerHTML = pairs.map(([k, v]) => `<div class="feat-chip"><span>${esc(k)}</span><strong>${esc(v)}</strong></div>`).join("");
}

// ---------- Bulk Processing ----------
function setStepper(stepNum) {
  document.querySelectorAll("#bulk-stepper .step").forEach((s) => {
    s.classList.toggle("active", Number(s.dataset.step) <= stepNum);
  });
}

async function runBulkPrediction(records, sourceName = "uploaded.csv") {
  if (!records.length) {
    document.getElementById("bulk-status-text").textContent = "No valid rows found in CSV.";
    showToast("No valid rows found in CSV.", "error");
    return;
  }
  const runBtn = document.getElementById("run-bulk-btn");
  if (runBtn) {
    runBtn.disabled = true;
    runBtn.textContent = "Scoring...";
  }
  setStepper(2);
  document.getElementById("bulk-status-text").textContent = `Scoring ${records.length} rows with XGBoost...`;

  try {
    const res = await apiPost("/predict/batch", { records });
    const defList = (res.table_predicted_default || []).map((r, i) => normalizeApplicant(r, i));
    const nonDefList = (res.table_predicted_non_default || []).map((r, i) => normalizeApplicant(r, defList.length + i));
    state.applicants = [...defList, ...nonDefList];
    state.bulkSelectedIds.clear();
    state.bulkTab = "default";
    document.querySelectorAll("[data-bulk-tab]").forEach((b) => {
      b.classList.toggle("active", b.dataset.bulkTab === "default");
    });
    state.bulkPage = 1;
    state.activeBulkId = state.applicants[0]?.customer_id || null;

    document.getElementById("csv-file-name").textContent = sourceName;
    document.getElementById("csv-file-meta").textContent = `${state.applicants.length} rows scored`;
    const skippedMsg = res.skipped_count ? ` (${res.skipped_count} invalid rows skipped)` : "";
    const imputedMsg = res.imputed_count ? ` · ${res.imputed_count} rows had null values — imputed with dataset defaults` : "";
    document.getElementById("bulk-status-text").textContent = `Completed: ${defList.length} predicted default, ${nonDefList.length} no default${skippedMsg}${imputedMsg}.`;

    setStepper(3);
    renderBulkView();
    renderDashboard();
    showToast(`Scored ${state.applicants.length} applicants (${defList.length} default, ${nonDefList.length} no default)${skippedMsg}.`, res.skipped_count ? "info" : "success");
    if (res.imputed_count) {
      showToast(`⚠️ ${res.imputed_count} row(s) had null/missing values — imputed with dataset defaults before scoring.`, "info");
    }
  } catch (err) {
    document.getElementById("bulk-status-text").textContent = `Batch scoring failed: ${err.message}`;
    showToast(`Batch scoring failed: ${err.message}`, "error");
  } finally {
    if (runBtn) {
      runBtn.disabled = false;
      runBtn.textContent = "▶ Start Prediction";
    }
  }
}

function getBulkFilteredList() {
  const isDefaultTab = state.bulkTab === "default";
  const q = (document.getElementById("bulk-search-input").value || "").trim().toLowerCase();
  return state.applicants.filter((a) => {
    const isDef = a.flagged || a.risk_classification === "HIGH";
    if (isDefaultTab !== isDef) return false;
    if (!q) return true;
    return a.customer_id.toLowerCase().includes(q) || a.customer_name.toLowerCase().includes(q);
  });
}

function renderBulkView() {
  const total = state.applicants.length;
  const defCount = state.applicants.filter((a) => a.flagged || a.risk_classification === "HIGH").length;
  const nonDefCount = total - defCount;
  const defPct = total ? ((defCount / total) * 100).toFixed(1) : "0";
  const nonDefPct = total ? ((nonDefCount / total) * 100).toFixed(1) : "0";

  document.getElementById("bulk-total-count").textContent = String(total);
  document.getElementById("bulk-default-count").innerHTML = `${defCount} <span class="stat-pct">(${defPct}%)</span>`;
  document.getElementById("bulk-nodefault-count").innerHTML = `${nonDefCount} <span class="stat-pct">(${nonDefPct}%)</span>`;
  document.getElementById("tab-default-btn").textContent = `Default Applicants (${defCount})`;
  document.getElementById("tab-nodefault-btn").textContent = `No Default Applicants (${nonDefCount})`;
  document.getElementById("bulk-target-selected").textContent = `Selected Applicants (${state.bulkSelectedIds.size})`;
  document.getElementById("bulk-target-default").textContent = `All Default (${defCount})`;
  document.getElementById("bulk-target-nodefault").textContent = `All No Default (${nonDefCount})`;

  const list = getBulkFilteredList();
  const totalPages = Math.max(1, Math.ceil(list.length / state.bulkPageSize));
  if (state.bulkPage > totalPages) state.bulkPage = totalPages;
  const start = (state.bulkPage - 1) * state.bulkPageSize;
  const pageItems = list.slice(start, start + state.bulkPageSize);

  const tbody = document.querySelector("#bulk-table tbody");
  tbody.innerHTML = pageItems.map((a) => {
    const checked = state.bulkSelectedIds.has(a.customer_id) ? "checked" : "";
    const activeRow = state.activeBulkId === a.customer_id ? "selected-row" : "";
    return `<tr class="${activeRow}" data-id="${esc(a.customer_id)}">
      <td><input type="checkbox" class="bulk-row-cb" data-id="${esc(a.customer_id)}" ${checked}></td>
      <td>${esc(a.customer_id)}</td>
      <td>${esc(a.customer_name)}</td>
      <td><span class="pill ${riskPillClass(a.risk_classification)}">${a.score}</span></td>
      <td>${fmtINR(a.due_amount)}</td>
      <td>${esc(a.due_date)}</td>
    </tr>`;
  }).join("") || `<tr><td colspan="6" class="hint">No applicants in this category.</td></tr>`;

  tbody.querySelectorAll("tr[data-id]").forEach((tr) => {
    tr.addEventListener("click", (e) => {
      if (e.target.type === "checkbox") return;
      state.activeBulkId = tr.dataset.id;
      renderBulkView();
    });
  });

  tbody.querySelectorAll(".bulk-row-cb").forEach((cb) => {
    cb.addEventListener("change", () => {
      if (cb.checked) state.bulkSelectedIds.add(cb.dataset.id);
      else state.bulkSelectedIds.delete(cb.dataset.id);
      document.getElementById("bulk-target-selected").textContent = `Selected Applicants (${state.bulkSelectedIds.size})`;
    });
  });

  // Windowed Pagination (prevents 2000+ DOM buttons on 10k-row CSVs)
  const pagEl = document.getElementById("bulk-pagination");
  pagEl.innerHTML = "";
  const pages = [];
  for (let p = 1; p <= totalPages; p++) {
    if (p === 1 || p === totalPages || Math.abs(p - state.bulkPage) <= 1) {
      pages.push(p);
    } else if (pages[pages.length - 1] !== "...") {
      pages.push("...");
    }
  }
  pages.forEach((p) => {
    if (p === "...") {
      const span = document.createElement("span");
      span.className = "hint";
      span.style.padding = "4px 6px";
      span.textContent = "…";
      pagEl.appendChild(span);
      return;
    }
    const btn = document.createElement("button");
    btn.className = `page-btn ${p === state.bulkPage ? "active" : ""}`;
    btn.textContent = String(p);
    btn.addEventListener("click", () => { state.bulkPage = p; renderBulkView(); });
    pagEl.appendChild(btn);
  });

  // Sync Select-All checkbox for current filtered list
  const selectAllCb = document.getElementById("bulk-select-all");
  if (selectAllCb) {
    selectAllCb.checked = list.length > 0 && list.every((a) => state.bulkSelectedIds.has(a.customer_id));
  }

  // Right-hand details panel
  const active = state.applicants.find((a) => a.customer_id === state.activeBulkId) || pageItems[0];
  if (active) renderBulkDetailPanel(active);
  else resetBulkDetailPanel();
}

function resetBulkDetailPanel() {
  document.getElementById("bulk-detail-title").textContent = "Select an applicant";
  const pill = document.getElementById("bulk-detail-pill");
  pill.className = "pill muted";
  pill.textContent = "--";
  setGauge("bulk-gauge-arc", "bulk-detail-score", 300);
  document.getElementById("bulk-detail-score").textContent = "--";
  document.getElementById("bd-risk").className = "pill muted";
  document.getElementById("bd-risk").textContent = "--";
  document.getElementById("bd-due").textContent = "--";
  document.getElementById("bd-date").textContent = "--";
  document.getElementById("bd-prob").textContent = "--";
  document.getElementById("bd-limit-action").textContent = "--";
  document.querySelector("#bulk-tx-table tbody").innerHTML = "";
}

function renderBulkDetailPanel(a) {
  document.getElementById("bulk-detail-title").textContent = `${a.customer_id} – ${a.customer_name}`;
  const pill = document.getElementById("bulk-detail-pill");
  pill.className = `pill ${riskPillClass(a.risk_classification)}`;
  pill.textContent = a.flagged || a.risk_classification === "HIGH" ? "Predicted Default" : "No Default";

  setGauge("bulk-gauge-arc", "bulk-detail-score", a.score);
  const riskEl = document.getElementById("bd-risk");
  riskEl.className = `pill ${riskPillClass(a.risk_classification)}`;
  riskEl.textContent = `${a.risk_classification} Risk`;
  document.getElementById("bd-due").textContent = fmtINR(a.due_amount);
  document.getElementById("bd-date").textContent = a.due_date;
  document.getElementById("bd-prob").textContent = `${(a.default_probability * 100).toFixed(1)}%`;
  const lim = Number(a.limit_bal || 0);
  const limAction = a.risk_classification === "HIGH"
    ? `Reduce -50% (${fmtINR(lim * 0.5)})`
    : a.risk_classification === "MEDIUM"
      ? `Cap -20% (${fmtINR(lim * 0.8)})`
      : `Maintain (${fmtINR(lim)})`;
  document.getElementById("bd-limit-action").textContent = limAction;

  const txBody = document.querySelector("#bulk-tx-table tbody");
  txBody.innerHTML = MONTHS_6.map((m, i) => {
    const code = Number(a.pay_statuses[i] ?? 0);
    const paid = code <= 0;
    return `<tr>
      <td>${m}</td>
      <td>Payment</td>
      <td>${fmtINR(a.payments[i] || 0)}</td>
      <td><span class="pill ${paid ? "green" : "red"}">${paid ? "Paid" : `${code}M Late`}</span></td>
    </tr>`;
  }).join("");
}

// ---------- Directory ----------
function renderDirectory() {
  const q = (document.getElementById("dir-search").value || "").trim().toLowerCase();
  const rf = document.getElementById("dir-risk-filter").value;
  const rows = state.applicants.filter((a) => {
    if (rf !== "ALL" && a.risk_classification !== rf) return false;
    if (!q) return true;
    return a.customer_id.toLowerCase().includes(q) || a.customer_name.toLowerCase().includes(q) || a.email.toLowerCase().includes(q);
  });

  const tbody = document.querySelector("#dir-table tbody");
  tbody.innerHTML = rows.map((a) => `
    <tr>
      <td>${esc(a.customer_id)}</td>
      <td>${esc(a.customer_name)}</td>
      <td>${esc(a.email || "—")}</td>
      <td><strong>${a.score}</strong></td>
      <td>${(a.default_probability * 100).toFixed(1)}%</td>
      <td><span class="pill ${riskPillClass(a.risk_classification)}">${esc(a.risk_classification)}</span></td>
      <td>
        <button class="btn ghost small" data-assess="${esc(a.email || a.customer_id)}">Assess</button>
      </td>
    </tr>
  `).join("") || `<tr><td colspan="7" class="hint">No matching applicants.</td></tr>`;

  tbody.querySelectorAll("[data-assess]").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.getElementById("single-id-input").value = btn.dataset.assess;
      switchView("single");
      loadSingleById(btn.dataset.assess);
    });
  });
}

// ---------- Automated Emails ----------
function getEmailFilteredRows() {
  const rf = document.getElementById("email-risk-filter").value;
  const q = (document.getElementById("email-search-input").value || "").trim().toLowerCase();
  return state.applicants.filter((a) => {
    if (!(a.email && a.email.includes("@"))) return false;
    if (rf !== "ALL" && a.risk_classification !== rf) return false;
    if (!q) return true;
    return a.customer_id.toLowerCase().includes(q) || a.customer_name.toLowerCase().includes(q);
  });
}

function renderEmailView() {
  const eligible = state.applicants.filter((a) => a.email && a.email.includes("@"));
  const highRisk = eligible.filter((a) => a.risk_classification === "HIGH" || a.flagged);
  document.getElementById("tab-pending-emails").textContent = `Pending Reminders (${eligible.length})`;
  document.getElementById("tab-sent-emails").textContent = `Sent Emails (${state.emailLogs.length})`;
  document.getElementById("ep-target-selected").textContent = `Selected (${state.emailSelectedIds.size})`;
  document.getElementById("ep-target-high").textContent = `All High Risk (${highRisk.length})`;
  document.getElementById("ep-target-all").textContent = `All Eligible (${eligible.length})`;

  const pendingSec = document.getElementById("email-pending-section");
  const logsCard = document.getElementById("email-logs-card");
  if (pendingSec && logsCard) {
    pendingSec.style.display = state.emailTab === "pending" ? "block" : "none";
    logsCard.style.display = state.emailTab === "sent" ? "block" : "none";
  }

  const rows = getEmailFilteredRows();
  const selectAllEmailCb = document.getElementById("email-select-all");
  if (selectAllEmailCb) {
    selectAllEmailCb.checked = rows.length > 0 && rows.every((a) => state.emailSelectedIds.has(a.customer_id));
  }

  const tbody = document.querySelector("#email-applicants-table tbody");
  tbody.innerHTML = rows.map((a) => {
    const checked = state.emailSelectedIds.has(a.customer_id) ? "checked" : "";
    return `<tr>
      <td><input type="checkbox" class="email-row-cb" data-id="${esc(a.customer_id)}" ${checked}></td>
      <td>${esc(a.customer_id)}</td>
      <td>${esc(a.customer_name)}</td>
      <td>${fmtINR(a.due_amount)}</td>
      <td>${esc(a.due_date)}</td>
      <td><span class="pill ${riskPillClass(a.risk_classification)}">${esc(a.risk_classification)}</span></td>
    </tr>`;
  }).join("") || `<tr><td colspan="6" class="hint">No applicants with valid email addresses.</td></tr>`;

  tbody.querySelectorAll(".email-row-cb").forEach((cb) => {
    cb.addEventListener("change", () => {
      if (cb.checked) state.emailSelectedIds.add(cb.dataset.id);
      else state.emailSelectedIds.delete(cb.dataset.id);
      document.getElementById("ep-target-selected").textContent = `Selected (${state.emailSelectedIds.size})`;
      renderEmailPreview();
    });
  });

  renderEmailPreview();
  renderEmailLogs();
}

function renderEmailPreview() {
  const firstId = Array.from(state.emailSelectedIds)[0];
  const app = state.applicants.find((a) => a.customer_id === firstId)
    || state.applicants.find((a) => a.email)
    || { customer_name: "Applicant", application_id: "APP-001", due_amount: 15000, due_date: "05 Dec 2026", risk_classification: "HIGH" };

  const tpl = document.getElementById("email-template-select").value;
  const safeName = esc(app.customer_name || "Customer");
  const safeAppId = esc(app.application_id || "N/A");
  const safeRisk = esc(app.risk_classification || "HIGH");

  let subject = `Application Received — Reference ${app.application_id || "N/A"}`;
  let greeting = `Dear <strong>${safeName}</strong>,`;
  let body = `Thank you for submitting your credit application (Reference: <strong>${safeAppId}</strong>) with Abhaya. Our underwriting and risk review team has received your submission and will notify you promptly as your application progresses.`;

  if (tpl === "missing_documents") {
    subject = `Action Required: Supporting Documents Needed (${app.application_id || "N/A"})`;
    body = `To continue processing your credit application (Reference: <strong>${safeAppId}</strong>), please provide the following supporting documents:<ul><li>Proof of identity</li><li>Recent income verification</li></ul>Kindly upload the requested items through your Abhaya portal at your earliest convenience.`;
  } else if (tpl === "status_change") {
    subject = `Application Status Update (${app.application_id || "N/A"}): UNDER_REVIEW`;
    body = `The status of your credit application (Reference: <strong>${safeAppId}</strong>) has been updated to: <strong>UNDER_REVIEW</strong>. Please sign in to your Abhaya portal to review the latest details and any next steps.`;
  } else if (tpl === "final_decision") {
    subject = `Final Decision Notice — Application ${app.application_id || "N/A"}: REVIEWED`;
    body = `A final decision has been recorded for your credit application (Reference: <strong>${safeAppId}</strong>): <strong>REVIEWED</strong>. Please sign in to your Abhaya portal to view the complete decision summary.`;
  } else if (tpl === "credit_limit_adjustment") {
    subject = `Important Account Notice: Credit Limit Adjustment (${app.application_id})`;
    body = `Following a recent review of your account (<strong>${esc(app.application_id)}</strong>), we noted that your credit score is currently below our standard benchmark. To help keep your repayments manageable, we are <strong>temporarily decreasing your credit limit</strong>. Please continue making on-time payments and keeping utilization low to improve your credit score for future limit reviews.`;
  } else if (tpl === "high_risk_review") {
    subject = `[Review Required] High-Risk Application ${app.application_id || "N/A"}`;
    greeting = `Dear <strong>Risk Review Team</strong>,`;
    body = `Application <strong>${safeAppId}</strong> has been classified as <strong>${safeRisk}</strong> risk and requires manual underwriting review. Please open the Abhaya internal review queue to inspect the application and record your assessment.`;
  }

  document.getElementById("email-preview-box").innerHTML = `
    <div class="email-preview-subject">Subject: ${esc(subject)}</div>
    <p>${greeting}</p>
    <div style="margin:10px 0;">${body}</div>
    <p>Sincerely,<br><strong>Abhaya Risk &amp; Collections Team</strong></p>
  `;
}

function renderEmailLogs() {
  const tbody = document.querySelector("#email-log-table tbody");
  tbody.innerHTML = state.emailLogs.map((log) => `
    <tr>
      <td>${esc(String(log.created_at || "").slice(0, 19).replace("T", " "))}</td>
      <td>${esc(log.recipient || "—")}</td>
      <td>${esc(log.event_type || "—")}</td>
      <td>${esc(log.message || "")}</td>
      <td><span class="pill ${log.success ? "green" : "red"}">${esc(log.status || (log.success ? "sent" : "failed"))}</span></td>
    </tr>
  `).join("") || `<tr><td colspan="5" class="hint">No emails sent yet.</td></tr>`;
}

async function sendBatchEmails(targets, eventType, feedbackId) {
  const fb = document.getElementById(feedbackId);
  if (!state.employeeKey) {
    fb.textContent = "Please set your Employee API Key in Settings first (must match EMPLOYEE_API_KEY in .env).";
    return;
  }
  const withEmail = targets.filter((a) => a.email && a.email.includes("@"));
  if (!withEmail.length) {
    fb.textContent = "No selected applicants have a valid email address.";
    return;
  }

  fb.textContent = `Sending ${withEmail.length} email(s)...`;
  let sent = 0;
  let suppressed = 0;
  let failed = 0;
  let lastErr = "";
  for (const a of withEmail) {
    try {
      const payload = {
        event_type: eventType,
        application_id: a.application_id,
        customer_name: a.customer_name,
        risk_classification: a.risk_classification,
        missing_documents: ["Proof of identity", "Recent income verification"],
      };
      if (eventType !== "high_risk_review") {
        payload.recipient_email = a.email;
      }
      const res = await apiPost("/notifications/send", payload, { "X-Employee-Key": state.employeeKey });
      if (res.success) sent++;
      else if (res.status === "suppressed") suppressed++;
      else { failed++; lastErr = res.message || "Delivery failed"; }
    } catch (err) {
      failed++;
      lastErr = err.message;
    }
  }
  setStepper(4);
  const summary = sent === 0 && suppressed === 0 && lastErr
    ? `Error: ${lastErr}`
    : `Done: ${sent} sent, ${suppressed} already sent (suppressed), ${failed} failed.`;
  fb.textContent = summary;
  showToast(summary, sent > 0 ? "success" : (suppressed > 0 && failed === 0 ? "info" : "error"));
  await loadEmailLogs();
  renderEmailView();
}

async function loadEmailLogs() {
  if (!state.employeeKey) return false;
  try {
    const res = await apiGet("/email/logs?limit=30", { "X-Employee-Key": state.employeeKey });
    state.emailLogs = res.logs || [];
    return true;
  } catch (_) {
    return false;
  }
}

// ---------- Reports ----------
async function loadReports() {
  const fn = Math.max(0, parseFloat(document.getElementById("cost-fn").value) || 0);
  const fp = Math.max(0, parseFloat(document.getElementById("cost-fp").value) || 0);
  const tp = Math.max(0, parseFloat(document.getElementById("cost-tp").value) || 0);
  const tn = Math.max(0, parseFloat(document.getElementById("cost-tn").value) || 0);
  const qs = `cost_fn=${fn}&cost_fp=${fp}&cost_tp=${tp}&cost_tn=${tn}`;

  try {
    const [comp, costC, prC, rules] = await Promise.all([
      apiGet(`/models/comparison?${qs}`),
      apiGet(`/cost-curves?${qs}`),
      apiGet("/pr-curves"),
      apiGet("/rules"),
    ]);

    const rows = comp.models || [];
    const lowest = rows.length ? Math.min(...rows.map((r) => r.at_best.total_cost)) : 0;
    document.getElementById("compare-table").innerHTML =
      `<thead><tr><th>Model</th><th>ROC-AUC</th><th>PR-AUC</th><th>Threshold</th><th>Recall</th><th>Precision</th><th>Cost @ Best</th><th>Saving</th></tr></thead><tbody>` +
      rows.map((r) => `
        <tr class="${r.at_best.total_cost === lowest ? "selected-row" : ""}">
          <td><strong>${esc(r.label)}</strong></td>
          <td>${r.roc_auc.toFixed(3)}</td>
          <td>${r.pr_auc.toFixed(3)}</td>
          <td>${r.best_threshold.toFixed(2)}</td>
          <td>${(r.at_best.recall * 100).toFixed(1)}%</td>
          <td>${(r.at_best.precision * 100).toFixed(1)}%</td>
          <td>${Math.round(r.at_best.total_cost).toLocaleString()}</td>
          <td><span class="pill green">${r.saving_pct}%</span></td>
        </tr>
      `).join("") + `</tbody>`;

    if (comp.baselines) {
      document.getElementById("baselines").textContent =
        `Test set (${Number(comp.test_size || 0).toLocaleString()} rows) baselines: Approve everyone cost = ${Math.round(comp.baselines.approve_everyone).toLocaleString()}, Flag everyone cost = ${Math.round(comp.baselines.flag_everyone).toLocaleString()}.`;
    }

    const xgb = rows.find((r) => r.name === "xgboost") || rows[0];
    if (xgb) {
      document.getElementById("dash-threshold").textContent = xgb.best_threshold.toFixed(2);
      document.getElementById("dash-saving").textContent = `ROC-AUC ${xgb.roc_auc.toFixed(3)} · ${xgb.saving_pct}% cost saving`;
    }

    if (state.charts.cost) state.charts.cost.destroy();
    state.charts.cost = new Chart(document.getElementById("cost-chart"), {
      type: "line",
      data: {
        labels: costC.thresholds,
        datasets: Object.entries(costC.curves).map(([name, c]) => ({
          label: `${c.label} (${c.best_threshold.toFixed(2)})`,
          data: c.cost,
          borderColor: MODEL_COLORS[name] || "#ccc",
          borderWidth: name === "xgboost" ? 3 : 1.5,
          pointRadius: 0,
        })),
      },
      options: { maintainAspectRatio: false, plugins: { legend: { labels: { color: "#cbd5e1" } } } },
    });

    if (state.charts.pr) state.charts.pr.destroy();
    state.charts.pr = new Chart(document.getElementById("pr-chart"), {
      type: "line",
      data: {
        datasets: Object.entries(prC.curves).map(([name, c]) => ({
          label: `${c.label} (AP ${c.average_precision.toFixed(3)})`,
          data: c.points,
          borderColor: MODEL_COLORS[name] || "#ccc",
          borderWidth: name === "xgboost" ? 3 : 1.5,
          pointRadius: 0,
          showLine: true,
        })),
      },
      options: {
        maintainAspectRatio: false,
        scales: { x: { type: "linear", min: 0, max: 1 }, y: { min: 0, max: 1 } },
        plugins: { legend: { labels: { color: "#cbd5e1" } } },
      },
    });

    document.getElementById("policy-text").textContent = rules.policy || "";
    document.getElementById("tree-rules-table").innerHTML =
      `<thead><tr><th>Rule</th><th>Samples</th><th>Default Rate</th></tr></thead><tbody>` +
      (rules.tree_rules || []).slice(0, 8).map((r) => `
        <tr><td>${esc(r.rule)}</td><td>${r.samples}</td><td>${(r.default_rate * 100).toFixed(1)}%</td></tr>
      `).join("") + `</tbody>`;
  } catch (err) {
    console.error("Failed to load reports:", err);
    showToast(`Failed to load model reports: ${err.message}`, "error");
  }
}

// ---------- Health & Settings ----------
async function refreshHealth() {
  try {
    const [h, em] = await Promise.all([apiGet("/health"), apiGet("/email/status")]);
    document.getElementById("set-api-status").textContent = h.status === "ok" ? "Online" : "Offline";
    document.getElementById("set-model-status").textContent = h.model_loaded ? `Loaded (${h.model_version})` : "Not loaded";
    document.getElementById("set-mysql-status").textContent = h.mysql_connected ? "Connected" : "Offline (in-memory fallback)";
    document.getElementById("set-smtp-status").textContent = em.configured ? "Configured" : em.message;
    document.getElementById("set-smtp-host").textContent = `${em.smtp_host}:${em.smtp_port}`;

    document.getElementById("dash-system-status").textContent = h.model_loaded ? "Online" : "Degraded";
    document.getElementById("dash-db-status").textContent =
      `MySQL: ${h.mysql_connected ? "Connected" : "Fallback"} · SMTP: ${em.configured ? "Ready" : "Not configured"}`;
  } catch (err) {
    document.getElementById("dash-system-status").textContent = "Offline";
    document.getElementById("set-api-status").textContent = err.message;
  }
}

// ---------- Boot & Event Wiring ----------
async function init() {
  const sidebarEl = document.querySelector(".sidebar");
  const mobileBtn = document.getElementById("mobile-menu-btn");
  if (mobileBtn && sidebarEl) {
    mobileBtn.addEventListener("click", () => sidebarEl.classList.toggle("open"));
  }

  document.querySelectorAll(".nav-item").forEach((btn) => {
    btn.addEventListener("click", () => {
      switchView(btn.dataset.view);
      if (sidebarEl) sidebarEl.classList.remove("open");
    });
  });
  document.querySelectorAll("[data-goto]").forEach((btn) => {
    btn.addEventListener("click", () => switchView(btn.dataset.goto));
  });
  const empPill = document.getElementById("employee-pill");
  empPill.addEventListener("click", () => switchView("settings"));
  empPill.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); switchView("settings"); } });
  document.getElementById("notif-btn").addEventListener("click", () => switchView("emails"));

  document.getElementById("logout-btn").addEventListener("click", () => {
    state.employeeKey = "";
    sessionStorage.removeItem("abhaya_emp_key");
    document.getElementById("settings-emp-key").value = "";
    state.emailLogs = [];
    renderEmailLogs();
    showToast("Logged out of employee session.", "info");
    switchView("settings");
  });

  // Settings key
  const keyInput = document.getElementById("settings-emp-key");
  const empLabel = document.getElementById("employee-label");
  keyInput.value = state.employeeKey;
  if (state.employeeKey) empLabel.textContent = "Employee (Key Set)";

  document.getElementById("save-emp-key-btn").addEventListener("click", async () => {
    state.employeeKey = keyInput.value.trim();
    if (!state.employeeKey) {
      sessionStorage.removeItem("abhaya_emp_key");
      empLabel.textContent = "Employee";
      document.getElementById("settings-key-msg").textContent = "Employee API key cleared.";
      showToast("Employee key cleared.", "info");
      state.emailLogs = [];
      renderEmailLogs();
      return;
    }
    const ok = await loadEmailLogs();
    if (ok) {
      sessionStorage.setItem("abhaya_emp_key", state.employeeKey);
      empLabel.textContent = "Employee (Verified)";
      document.getElementById("settings-key-msg").textContent = "Employee API key verified and saved.";
      showToast("Employee key verified.", "success");
      renderEmailLogs();
    } else {
      sessionStorage.removeItem("abhaya_emp_key");
      state.employeeKey = "";
      empLabel.textContent = "Employee";
      document.getElementById("settings-key-msg").textContent = "Invalid Employee API key (401 Unauthorized).";
      showToast("Invalid Employee API key.", "error");
    }
  });
  document.getElementById("refresh-health-btn").addEventListener("click", refreshHealth);

  // Single Applicant
  document.getElementById("fetch-single-btn").addEventListener("click", () => {
    loadSingleById(document.getElementById("single-id-input").value);
  });
  document.getElementById("single-id-input").addEventListener("keydown", (e) => {
    if (e.key === "Enter") loadSingleById(e.target.value);
  });
  document.getElementById("single-send-email-btn").addEventListener("click", async () => {
    const typedEmail = document.getElementById("single-id-input").value.trim();
    const app = state.currentSingleApp || { customer_name: "Applicant", application_id: "APP-SINGLE", risk_classification: "MEDIUM" };
    const recipient = (typedEmail.includes("@") ? typedEmail : app.email) || "";
    if (!recipient.includes("@")) {
      showToast("Enter a valid email address in Applicant Email first.", "error");
      return;
    }
    if (!state.employeeKey) {
      showToast("Set your Employee API Key in Settings first.", "error");
      return;
    }
    try {
      const isLowScore = app.risk_classification === "HIGH" || app.score < 560;
      const res = await apiPost(
        "/notifications/send",
        {
          event_type: isLowScore ? "credit_limit_adjustment" : "application_submitted",
          recipient_email: recipient,
          application_id: app.application_id || "APP-SINGLE",
          customer_name: app.customer_name || "Applicant",
          risk_classification: app.risk_classification || "MEDIUM",
        },
        { "X-Employee-Key": state.employeeKey }
      );
      showToast(res.success ? `Email sent to ${recipient}` : `Email status: ${res.message}`, res.success ? "success" : "info");
      await loadEmailLogs();
    } catch (err) {
      showToast(`Email failed: ${err.message}`, "error");
    }
  });
  document.querySelectorAll("[data-sample-kind]").forEach((b) => {
    b.addEventListener("click", () => {
      document.getElementById("single-id-input").value = b.dataset.sampleKind;
      loadSingleById(b.dataset.sampleKind);
    });
  });

  const scoreCustom23 = async (rawObj) => {
    try {
      const pred = await apiPost("/predict", rawObj);
      const app = normalizeApplicant({
        customer_id: rawObj.customer_id || rawObj.record_id || "CUSTOM-23",
        customer_name: rawObj.customer_name || "Custom Applicant",
        email: rawObj.email || "",
        default_probability: pred.default_probability,
        risk_classification: pred.risk_classification,
        suggested_action: pred.suggested_action,
        flagged: pred.flagged,
        features_summary: {
          limit_bal: rawObj.LIMIT_BAL,
          age: rawObj.AGE,
          sex: rawObj.SEX,
          education: rawObj.EDUCATION,
          marriage: rawObj.MARRIAGE,
          bills: [1, 2, 3, 4, 5, 6].map((i) => rawObj[`BILL_AMT${i}`] || 0),
          payments: [1, 2, 3, 4, 5, 6].map((i) => rawObj[`PAY_AMT${i}`] || 0),
          pay_statuses: PAY_KEYS.map((k) => rawObj[k] || 0),
        },
      });
      document.getElementById("single-id-input").value = app.email || app.customer_id;
      renderSingleApplicant(app, pred.reasons || []);
      showToast(`Scored all 46 features: ${app.risk_classification} Risk (${app.score}/900)`, "success");
    } catch (err) {
      showToast(`Scoring failed: ${err.message}`, "error");
    }
  };

  document.getElementById("score-custom-23-btn").addEventListener("click", () => {
    scoreCustom23(readRaw23FromForm());
  });

  document.getElementById("single-csv-input").addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (ev) => {
      const rows = parseCSV(String(ev.target.result || ""));
      if (!rows.length) {
        showToast("CSV file has no data rows.", "error");
        return;
      }
      scoreCustom23(rows[0]);
      e.target.value = "";
    };
    reader.readAsText(file);
  });

  // CSV Upload & Drag/Drop
  const dropzone = document.getElementById("csv-dropzone");
  const fileInput = document.getElementById("csv-file-input");
  const handleFile = (file) => {
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (e) => {
      const parsed = parseCSV(String(e.target.result || ""));
      state.rawCsvRecords = parsed;
      document.getElementById("csv-file-name").textContent = file.name;
      document.getElementById("csv-file-meta").textContent = `${parsed.length} rows ready · ${(file.size / 1024).toFixed(1)} KB`;
      document.getElementById("bulk-status-text").textContent = `Uploaded ${file.name} (${parsed.length} rows). Click ▶ Start Prediction to split Default & No Default.`;
      setStepper(1);
      showToast(`Uploaded ${file.name} (${parsed.length} rows). Click ▶ Start Prediction.`, "info");
    };
    reader.readAsText(file);
  };

  fileInput.addEventListener("change", (e) => handleFile(e.target.files[0]));
  dropzone.addEventListener("dragover", (e) => { e.preventDefault(); dropzone.classList.add("dragover"); });
  dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    handleFile(e.dataTransfer.files[0]);
  });

  document.getElementById("load-sample-csv-btn").addEventListener("click", async () => {
    try {
      const res = await apiGet("/demo-applicants?limit=15");
      state.applicants = (res.records || []).map((r, i) => normalizeApplicant(r, i));
      state.activeBulkId = state.applicants[0]?.customer_id || null;
      document.getElementById("csv-file-name").textContent = "sample_applications_with_emails.csv";
      document.getElementById("csv-file-meta").textContent = `${state.applicants.length} rows loaded`;
      setStepper(3);
      renderBulkView();
      renderDashboard();
      showToast(`Loaded ${state.applicants.length} sample CSV rows.`, "success");
    } catch (err) {
      showToast(`Failed to load sample CSV: ${err.message}`, "error");
    }
  });

  document.getElementById("clear-csv-btn").addEventListener("click", () => {
    state.applicants = [];
    state.rawCsvRecords = [];
    state.bulkSelectedIds.clear();
    fileInput.value = "";
    document.getElementById("csv-file-name").textContent = "No file selected";
    document.getElementById("csv-file-meta").textContent = "0 rows";
    document.getElementById("bulk-status-text").textContent = "Upload a CSV file or click Load Sample CSV.";
    setStepper(1);
    renderBulkView();
    renderDashboard();
  });

  document.getElementById("run-bulk-btn").addEventListener("click", () => {
    if (state.rawCsvRecords.length) {
      runBulkPrediction(state.rawCsvRecords, document.getElementById("csv-file-name").textContent);
    } else {
      showToast("Upload a CSV file or click Load Sample CSV first.", "info");
    }
  });

  document.getElementById("export-bulk-btn").addEventListener("click", () => {
    if (!state.applicants.length) {
      showToast("No scored applicants to export.", "error");
      return;
    }
    const csvCell = (val) => `"${String(val == null ? "" : val).replace(/"/g, '""')}"`;
    const headers = ["customer_id", "customer_name", "email", "credit_score", "default_probability", "risk_classification", "due_amount", "due_date"];
    const csvRows = [headers.join(",")].concat(
      state.applicants.map((a) => [
        csvCell(a.customer_id),
        csvCell(a.customer_name),
        csvCell(a.email),
        csvCell(a.score),
        csvCell(a.default_probability.toFixed(4)),
        csvCell(a.risk_classification),
        csvCell(a.due_amount),
        csvCell(a.due_date),
      ].join(","))
    );
    const blob = new Blob([csvRows.join("\n")], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "abhaya_scored_applicants.csv";
    link.click();
    URL.revokeObjectURL(url);
    showToast(`Exported ${state.applicants.length} scored rows to CSV.`, "success");
  });

  document.querySelectorAll("[data-bulk-tab]").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll("[data-bulk-tab]").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      state.bulkTab = btn.dataset.bulkTab;
      state.bulkPage = 1;
      renderBulkView();
    });
  });

  document.getElementById("bulk-search-input").addEventListener("input", () => { state.bulkPage = 1; renderBulkView(); });
  document.getElementById("bulk-page-size").addEventListener("change", (e) => {
    state.bulkPageSize = Number(e.target.value) || 5;
    state.bulkPage = 1;
    renderBulkView();
  });
  document.getElementById("bulk-select-all").addEventListener("change", (e) => {
    const list = getBulkFilteredList();
    list.forEach((a) => e.target.checked ? state.bulkSelectedIds.add(a.customer_id) : state.bulkSelectedIds.delete(a.customer_id));
    renderBulkView();
  });

  document.getElementById("bulk-send-email-btn").addEventListener("click", () => {
    const mode = document.querySelector('input[name="bulk-email-target"]:checked').value;
    let targets = [];
    if (mode === "selected") targets = state.applicants.filter((a) => state.bulkSelectedIds.has(a.customer_id));
    else if (mode === "default") targets = state.applicants.filter((a) => a.flagged || a.risk_classification === "HIGH");
    else targets = state.applicants.filter((a) => !(a.flagged || a.risk_classification === "HIGH"));
    sendBatchEmails(targets, mode === "default" ? "credit_limit_adjustment" : "application_submitted", "bulk-email-feedback");
  });

  // Directory filters
  document.getElementById("dir-search").addEventListener("input", renderDirectory);
  document.getElementById("dir-risk-filter").addEventListener("change", renderDirectory);

  // Email view filters & actions
  document.querySelectorAll("[data-email-tab]").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll("[data-email-tab]").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      state.emailTab = btn.dataset.emailTab;
      renderEmailView();
    });
  });
  document.getElementById("email-risk-filter").addEventListener("change", renderEmailView);
  document.getElementById("email-search-input").addEventListener("input", renderEmailView);
  document.getElementById("email-template-select").addEventListener("change", renderEmailPreview);
  document.getElementById("email-select-all").addEventListener("change", (e) => {
    getEmailFilteredRows().forEach((a) =>
      e.target.checked ? state.emailSelectedIds.add(a.customer_id) : state.emailSelectedIds.delete(a.customer_id)
    );
    renderEmailView();
  });
  document.getElementById("email-page-send-btn").addEventListener("click", () => {
    const mode = document.querySelector('input[name="email-page-target"]:checked').value;
    const eligible = state.applicants.filter((a) => a.email && a.email.includes("@"));
    let targets = [];
    if (mode === "selected") targets = eligible.filter((a) => state.emailSelectedIds.has(a.customer_id));
    else if (mode === "high") targets = eligible.filter((a) => a.risk_classification === "HIGH" || a.flagged);
    else targets = eligible;
    const tpl = document.getElementById("email-template-select").value;
    sendBatchEmails(targets, tpl, "email-page-feedback");
  });
  document.getElementById("refresh-logs-btn").addEventListener("click", async () => {
    await loadEmailLogs();
    renderEmailLogs();
  });

  // Reports & Chatbot
  document.getElementById("apply-costs-btn").addEventListener("click", loadReports);
  const sendChat = async () => {
    const input = document.getElementById("chat-input");
    const msg = input.value.trim();
    if (!msg) return;
    const replyEl = document.getElementById("chat-reply");
    replyEl.classList.remove("hidden");
    replyEl.textContent = "Thinking...";
    try {
      const res = await apiPost("/chatbot", { message: msg });
      replyEl.textContent = res.reply;
    } catch (err) {
      replyEl.textContent = err.message;
    }
  };
  document.getElementById("chat-send-btn").addEventListener("click", sendChat);
  document.getElementById("chat-input").addEventListener("keydown", (e) => {
    if (e.key === "Enter") sendChat();
  });

  const assistPanel = document.getElementById("assistant-panel");
  document.getElementById("assistant-toggle-btn").addEventListener("click", () => {
    assistPanel.classList.toggle("hidden");
    if (!assistPanel.classList.contains("hidden")) {
      document.getElementById("chat-input").focus();
    }
  });
  document.getElementById("assistant-close-btn").addEventListener("click", () => {
    assistPanel.classList.add("hidden");
  });
  document.querySelectorAll("[data-chat-q]").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.getElementById("chat-input").value = btn.dataset.chatQ;
      sendChat();
    });
  });

  // Initial data load (5 real demo rows from sample CSV)
  await Promise.all([
    refreshHealth(),
    loadEmailLogs(),
    (async () => {
      try {
        const res = await apiGet("/demo-applicants?limit=5");
        state.applicants = (res.records || []).map((r, i) => normalizeApplicant(r, i));
        state.activeBulkId = state.applicants[0]?.customer_id || null;
      } catch (_) {}
    })(),
  ]);

  renderDashboard();
  renderBulkView();
  if (state.applicants[0]) {
    const initialKey = state.applicants[0].email || state.applicants[0].customer_id;
    document.getElementById("single-id-input").value = initialKey;
    loadSingleById(initialKey);
  }
  loadReports();
}

init();
