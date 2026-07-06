/* ==========================================================================
   backend/static/js/app.js
   Wires the dashboard UI to the Flask REST API.
   No frameworks - plain fetch() + DOM updates, kept deliberately simple
   so a student can read this top to bottom and understand the whole flow.
   ========================================================================== */

const API_BASE = "/api";

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

function $(id) {
  return document.getElementById(id);
}

function showToast(message, isError = true) {
  const toast = $("toast");
  toast.textContent = message;
  toast.style.borderLeftColor = isError ? "var(--accent-red)" : "var(--accent-teal)";
  toast.classList.add("visible");
  setTimeout(() => toast.classList.remove("visible"), 4000);
}

async function apiFetch(path, options = {}) {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(data.error || `Request failed (${res.status})`);
  }
  return data;
}

function formatTimestamp(iso) {
  const d = new Date(iso);
  return d.toLocaleString(undefined, {
    month: "short", day: "numeric", hour: "2-digit", minute: "2-digit", second: "2-digit",
  });
}

// ---------------------------------------------------------------------------
// Live clock in the top bar
// ---------------------------------------------------------------------------

function tickClock() {
  $("clock").textContent = new Date().toLocaleTimeString();
}
setInterval(tickClock, 1000);
tickClock();

// ---------------------------------------------------------------------------
// Health check -> model status chip
// ---------------------------------------------------------------------------

async function checkHealth() {
  const chip = $("modelStatusChip");
  try {
    const res = await fetch("/health");
    const data = await res.json();
    if (data.model_ready) {
      chip.textContent = "Model Ready";
      chip.classList.remove("offline");
    } else {
      chip.textContent = "Model Not Trained";
      chip.classList.add("offline");
    }
  } catch (e) {
    chip.textContent = "Server Unreachable";
    chip.classList.add("offline");
  }
}

// ---------------------------------------------------------------------------
// Advanced V1-V28 fields (built dynamically to avoid 28 lines of HTML)
// ---------------------------------------------------------------------------

function buildAdvancedFields() {
  const container = $("advancedFields");
  let html = "";
  for (let i = 1; i <= 28; i++) {
    html += `
      <div class="form-group" style="margin-bottom: 6px;">
        <label for="v${i}">V${i}</label>
        <input type="number" step="any" id="v${i}" placeholder="0.0" />
      </div>`;
  }
  container.innerHTML = html;
}
buildAdvancedFields();

$("advancedToggle").addEventListener("click", () => {
  const el = $("advancedFields");
  el.classList.toggle("open");
});

// ---------------------------------------------------------------------------
// Collect form data into the API payload shape
// ---------------------------------------------------------------------------

function collectPayload() {
  const payload = {
    amount: parseFloat($("amount").value),
  };
  const timeVal = $("time").value;
  if (timeVal !== "") {
    payload.time = parseFloat(timeVal);
  } else {
    // Default to "now" expressed as seconds-since-midnight, a reasonable
    // stand-in when the user doesn't supply a raw dataset-style Time value.
    const now = new Date();
    payload.time = now.getHours() * 3600 + now.getMinutes() * 60 + now.getSeconds();
  }

  for (let i = 1; i <= 28; i++) {
    const el = $(`v${i}`);
    if (el && el.value !== "") {
      payload[`V${i}`] = parseFloat(el.value);
    }
  }
  return payload;
}

// ---------------------------------------------------------------------------
// Risk gauge rendering (the signature visual element)
// ---------------------------------------------------------------------------

function updateGauge(riskScore) {
  // Map risk score [0,1] to needle rotation across a 180-degree sweep,
  // from -90deg (far left, low risk) to +90deg (far right, high risk).
  const angle = -90 + riskScore * 180;
  $("gaugeNeedle").style.transform = `rotate(${angle}deg)`;

  const scoreEl = $("gaugeScore");
  scoreEl.textContent = riskScore.toFixed(2);
  if (riskScore >= 0.7) {
    scoreEl.style.color = "var(--accent-red)";
  } else if (riskScore >= 0.4) {
    scoreEl.style.color = "var(--accent-amber)";
  } else {
    scoreEl.style.color = "var(--accent-teal)";
  }
}

// ---------------------------------------------------------------------------
// Submit transaction for prediction
// ---------------------------------------------------------------------------

let lastPayload = null;

$("txnForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const submitBtn = $("submitBtn");
  submitBtn.disabled = true;
  submitBtn.innerHTML = `<span class="loading-spinner"></span> Analyzing…`;

  try {
    const payload = collectPayload();
    lastPayload = payload;
    const result = await apiFetch("/predict", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    renderResult(result);
    await Promise.all([loadStats(), loadHistory()]);
  } catch (err) {
    showToast(err.message);
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = "Analyze Transaction";
  }
});

function renderResult(result) {
  $("resultEmpty").style.display = "none";
  $("resultContent").style.display = "block";

  updateGauge(result.risk_score);

  const banner = $("verdictBanner");
  banner.textContent = result.is_fraud
    ? "⚠ Likely Fraudulent Transaction"
    : "✓ Transaction Looks Legitimate";
  banner.className = "verdict-banner " + (result.is_fraud ? "fraud" : "legit");

  $("metaRiskTier").textContent = result.risk_label;
  $("metaModel").textContent = result.model_used;
  $("resultModelTag").textContent = `via ${result.model_used}`;

  // Reset any previous SHAP explanation when a new prediction comes in
  $("shapContainer").style.display = "none";
}

// ---------------------------------------------------------------------------
// Random sample button - convenience for demoing without real card data
// ---------------------------------------------------------------------------

$("randomBtn").addEventListener("click", () => {
  // Occasionally generate values that look "fraud-like" (large amount,
  // unusual hour) so the demo can show both verdicts, not just legit ones.
  const isSuspiciousSample = Math.random() < 0.4;
  if (isSuspiciousSample) {
    $("amount").value = (Math.random() * 2000 + 800).toFixed(2);
    $("time").value = Math.floor(Math.random() * 10800); // late-night seconds
  } else {
    $("amount").value = (Math.random() * 150 + 5).toFixed(2);
    $("time").value = Math.floor(Math.random() * 50000 + 30000);
  }
  $("txnForm").requestSubmit();
});

// ---------------------------------------------------------------------------
// SHAP explanation
// ---------------------------------------------------------------------------

$("explainBtn").addEventListener("click", async () => {
  if (!lastPayload) return;
  const btn = $("explainBtn");
  btn.disabled = true;
  btn.innerHTML = `<span class="loading-spinner"></span> Explaining…`;

  try {
    const explanation = await apiFetch("/explain", {
      method: "POST",
      body: JSON.stringify(lastPayload),
    });
    renderShap(explanation);
  } catch (err) {
    showToast(err.message);
  } finally {
    btn.disabled = false;
    btn.textContent = "Explain this prediction (SHAP)";
  }
});

function renderShap(explanation) {
  const container = $("shapContainer");
  const list = $("shapList");
  const messageEl = $("shapMessage");
  container.style.display = "block";
  messageEl.textContent = explanation.message;

  if (!explanation.available || explanation.top_features.length === 0) {
    list.innerHTML = "";
    return;
  }

  const maxAbs = Math.max(...explanation.top_features.map((f) => f.abs_value), 0.0001);

  list.innerHTML = explanation.top_features
    .map((f) => {
      const widthPct = (f.abs_value / maxAbs) * 100;
      const direction = f.shap_value >= 0 ? "positive" : "negative";
      return `
        <div class="shap-row">
          <span class="shap-feature" title="${f.feature}">${f.feature}</span>
          <div class="shap-bar-track">
            <div class="shap-bar-fill ${direction}" style="width:${widthPct}%;"></div>
          </div>
          <span class="shap-value">${f.shap_value.toFixed(3)}</span>
        </div>`;
    })
    .join("");
}

// ---------------------------------------------------------------------------
// Dashboard stats
// ---------------------------------------------------------------------------

async function loadStats() {
  try {
    const stats = await apiFetch("/stats");
    $("statTotal").textContent = stats.total_predictions;
    $("statFraud").textContent = stats.fraud_detected;
    $("statLegit").textContent = stats.legit_count;
    $("statRate").textContent = `${stats.fraud_rate_pct}%`;
  } catch (err) {
    console.error("Failed to load stats:", err);
  }
}

// ---------------------------------------------------------------------------
// Fraud history table
// ---------------------------------------------------------------------------

async function loadHistory() {
  const tbody = $("historyBody");
  try {
    const data = await apiFetch("/history?limit=25");
    if (!data.history || data.history.length === 0) {
      tbody.innerHTML = `<tr><td colspan="6" class="text-faint">No predictions yet.</td></tr>`;
      return;
    }

    tbody.innerHTML = data.history
      .map((r) => {
        const verdictBadge = r.is_fraud
          ? `<span class="badge fraud">FRAUD</span>`
          : `<span class="badge legit">LEGIT</span>`;
        const tierClass = r.risk_label.toLowerCase();
        return `
          <tr>
            <td>${formatTimestamp(r.timestamp)}</td>
            <td>${Number(r.amount).toFixed(2)}</td>
            <td>${verdictBadge}</td>
            <td>${Number(r.risk_score).toFixed(3)}</td>
            <td><span class="badge ${tierClass}">${r.risk_label}</span></td>
            <td>${r.model_used}</td>
          </tr>`;
      })
      .join("");
  } catch (err) {
    tbody.innerHTML = `<tr><td colspan="6" class="text-faint">Failed to load history.</td></tr>`;
  }
}

$("refreshHistoryBtn").addEventListener("click", () => {
  loadHistory();
  loadStats();
});

// ---------------------------------------------------------------------------
// EDA / evaluation charts viewer
// ---------------------------------------------------------------------------

const CHART_LABELS = {
  class_distribution: "Class Balance",
  amount_distribution: "Amount Distribution",
  correlation_heatmap: "Feature Correlation",
  time_vs_amount: "Time vs Amount",
  model_comparison: "Model Comparison",
  roc_curves: "ROC Curves",
};

async function loadCharts() {
  try {
    const data = await apiFetch("/charts");
    const tabsEl = $("chartTabs");
    const displayEl = $("chartDisplay");

    if (!data.charts || data.charts.length === 0) {
      displayEl.innerHTML = `<span class="text-faint">No charts found. Run the training script first.</span>`;
      return;
    }

    function showChart(filename) {
      displayEl.innerHTML = `<img src="/static/charts/${filename}" alt="${filename}" />`;
      document.querySelectorAll(".chart-tab").forEach((t) => {
        t.classList.toggle("active", t.dataset.file === filename);
      });
    }

    tabsEl.innerHTML = data.charts
      .map((f) => {
        const key = f.replace(".png", "");
        const label = CHART_LABELS[key] || key.replace(/_/g, " ");
        return `<button class="chart-tab" data-file="${f}">${label}</button>`;
      })
      .join("");

    tabsEl.querySelectorAll(".chart-tab").forEach((tab) => {
      tab.addEventListener("click", () => showChart(tab.dataset.file));
    });

    showChart(data.charts[0]);
  } catch (err) {
    $("chartDisplay").innerHTML = `<span class="text-faint">Failed to load charts.</span>`;
  }
}

// ---------------------------------------------------------------------------
// Initial load
// ---------------------------------------------------------------------------

checkHealth();
loadStats();
loadHistory();
loadCharts();
