/**
 * Queens Solver — content script
 *
 * Handles three messages from the popup:
 *   { action: "check" }  → replies { connected: bool, model_loaded: bool }
 *   { action: "solve" }  → extracts puzzle, calls local server, clicks cells
 *   { action: "reset" }  → clicks every placed queen to toggle it off (clear board)
 *
 * On page load the script pings the server and updates the extension badge so
 * the user can confirm the solver is ready before navigating to the puzzle.
 */

const SERVER_URL = "http://localhost:8000";
const CLICK_DELAY_MS = 40;
const HEATMAP_PAUSE_MS = 150; // pause after showing heatmap before placing queens

// ---------------------------------------------------------------------------
// Badge — shows server status on the extension icon without opening the popup
// ---------------------------------------------------------------------------

function updateBadge(connected, modelLoaded) {
  try {
    if (!connected) {
      chrome.runtime.sendMessage({ action: "setBadge", text: "OFF", color: "#ef4444" });
    } else if (!modelLoaded) {
      chrome.runtime.sendMessage({ action: "setBadge", text: "...", color: "#f59e0b" });
    } else {
      chrome.runtime.sendMessage({ action: "setBadge", text: "ON", color: "#22c55e" });
    }
  } catch {
    // Extension context invalidated after a reload — stop pinging
    clearInterval(pingInterval);
  }
}

function pingServer() {
  fetch(`${SERVER_URL}/health`)
    .then((r) => r.json())
    .then((data) => updateBadge(true, data.model_loaded))
    .catch(() => updateBadge(false, false));
}

pingServer();
const pingInterval = setInterval(pingServer, 10_000);

// ---------------------------------------------------------------------------
// Activation overlay
// ---------------------------------------------------------------------------

const CANVAS_PX = 110;

// Diverging colormaps: value 0 → low colour, 0.5 → white, 1 → high colour
const COLORMAPS = {
  rdbu_r: { low: [33, 102, 172],  high: [178, 24, 43]  }, // blue → red  (L)
  puor_r: { low: [230, 97, 1],    high: [94, 60, 153]   }, // orange → purple (H)
};

function _interpolateColor(v, low, high) {
  const w = [255, 255, 255];
  let r, g, b;
  if (v < 0.5) {
    const t = v * 2;
    r = Math.round(low[0] + (w[0] - low[0]) * t);
    g = Math.round(low[1] + (w[1] - low[1]) * t);
    b = Math.round(low[2] + (w[2] - low[2]) * t);
  } else {
    const t = (v - 0.5) * 2;
    r = Math.round(w[0] + (high[0] - w[0]) * t);
    g = Math.round(w[1] + (high[1] - w[1]) * t);
    b = Math.round(w[2] + (high[2] - w[2]) * t);
  }
  return [r, g, b];
}

function paintHeatmap(canvas, heatmap, cmapKey) {
  if (!heatmap || heatmap.length === 0) return;
  const { low, high } = COLORMAPS[cmapKey];
  const n = heatmap.length;
  const ctx = canvas.getContext("2d");
  const img = ctx.createImageData(CANVAS_PX, CANVAS_PX);
  const cellPx = CANVAS_PX / n;

  for (let row = 0; row < n; row++) {
    for (let col = 0; col < n; col++) {
      const v = Math.max(0, Math.min(1, heatmap[row][col]));
      const [r, g, b] = _interpolateColor(v, low, high);
      const x0 = Math.floor(col * cellPx), x1 = Math.floor((col + 1) * cellPx);
      const y0 = Math.floor(row * cellPx), y1 = Math.floor((row + 1) * cellPx);
      for (let py = y0; py < y1; py++) {
        for (let px = x0; px < x1; px++) {
          const i = (py * CANVAS_PX + px) * 4;
          img.data[i] = r; img.data[i + 1] = g; img.data[i + 2] = b; img.data[i + 3] = 255;
        }
      }
    }
  }
  ctx.putImageData(img, 0, 0);

  // Grid lines
  ctx.strokeStyle = "rgba(0,0,0,0.25)";
  ctx.lineWidth = 0.5;
  for (let i = 0; i <= n; i++) {
    const p = Math.round(i * cellPx);
    ctx.beginPath(); ctx.moveTo(p, 0); ctx.lineTo(p, CANVAS_PX); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, p); ctx.lineTo(CANVAS_PX, p); ctx.stroke();
  }
}

function getOrCreateOverlay() {
  let el = document.getElementById("queens-solver-overlay");
  if (el) return el;

  el = document.createElement("div");
  el.id = "queens-solver-overlay";
  el.style.cssText = [
    "position:fixed", "right:16px", "top:50%", "transform:translateY(-50%)",
    `width:${CANVAS_PX + 32}px`,
    "background:rgba(12,12,18,0.93)",
    "border:1px solid rgba(255,255,255,0.10)",
    "border-radius:10px", "padding:14px",
    "z-index:99999",
    "font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif",
    "box-shadow:0 4px 28px rgba(0,0,0,0.5)",
  ].join(";");

  el.innerHTML = `
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
      <span style="color:#fff;font-size:11px;font-weight:700;letter-spacing:0.6px;text-transform:uppercase;opacity:0.9;">Queens Solver</span>
      <button id="queens-overlay-close" style="background:none;border:none;color:rgba(255,255,255,0.35);cursor:pointer;font-size:16px;padding:0;line-height:1;">&times;</button>
    </div>
    <div style="margin-bottom:10px;">
      <div style="color:rgba(255,255,255,0.4);font-size:9px;letter-spacing:0.5px;text-transform:uppercase;margin-bottom:4px;">L Activations</div>
      <canvas id="queens-l-canvas" width="${CANVAS_PX}" height="${CANVAS_PX}"
        style="display:block;border-radius:4px;width:${CANVAS_PX}px;height:${CANVAS_PX}px;"></canvas>
    </div>
    <div style="margin-bottom:10px;">
      <div style="color:rgba(255,255,255,0.4);font-size:9px;letter-spacing:0.5px;text-transform:uppercase;margin-bottom:4px;">H Activations</div>
      <canvas id="queens-h-activation-canvas" width="${CANVAS_PX}" height="${CANVAS_PX}"
        style="display:block;border-radius:4px;width:${CANVAS_PX}px;height:${CANVAS_PX}px;"></canvas>
    </div>
    <div>
      <div style="color:rgba(255,255,255,0.4);font-size:9px;letter-spacing:0.5px;text-transform:uppercase;margin-bottom:4px;">H Attention</div>
      <canvas id="queens-h-canvas" width="${CANVAS_PX}" height="${CANVAS_PX}"
        style="display:block;border-radius:4px;width:${CANVAS_PX}px;height:${CANVAS_PX}px;"></canvas>
    </div>
  `;

  document.body.appendChild(el);
  document.getElementById("queens-overlay-close").addEventListener("click", () => el.remove());
  return el;
}

function updateOverlayHeatmaps(lHeatmap, hActivationHeatmap, hHeatmap) {
  const overlay = getOrCreateOverlay();
  const lCanvas = overlay.querySelector("#queens-l-canvas");
  const hActCanvas = overlay.querySelector("#queens-h-activation-canvas");
  const hCanvas = overlay.querySelector("#queens-h-canvas");
  if (lCanvas && lHeatmap?.length)            paintHeatmap(lCanvas,    lHeatmap,            "rdbu_r");
  if (hActCanvas && hActivationHeatmap?.length) paintHeatmap(hActCanvas, hActivationHeatmap,  "puor_r");
  if (hCanvas && hHeatmap?.length)            paintHeatmap(hCanvas,    hHeatmap,            "puor_r");
}

// ---------------------------------------------------------------------------
// Puzzle extraction
// ---------------------------------------------------------------------------

function extractPuzzle() {
  const grid = document.querySelector('[data-testid="interactive-grid"]');
  if (!grid) return null;

  const cells = Array.from(grid.querySelectorAll('[data-testid^="cell-"]'));
  if (cells.length === 0) return null;

  const gridSize = (() => {
    const style = grid.getAttribute("style") || "";
    const cssMatch = style.match(/--[\w-]+:\s*(\d+)/g)
      ?.map((m) => parseInt(m.match(/(\d+)$/)[1], 10))
      .find((n) => n * n === cells.length);
    if (cssMatch) return cssMatch;
    const sqrt = Math.round(Math.sqrt(cells.length));
    return sqrt * sqrt === cells.length ? sqrt : null;
  })();

  if (!gridSize) return null;

  const colorMap = {};
  let colorCount = 0;
  const region = Array.from({ length: gridSize }, () => Array(gridSize).fill(0));

  for (const cell of cells) {
    const label = cell.getAttribute("aria-label") || "";
    const colorMatch = label.match(/color (.+?),\s*row/i);
    const rowMatch = label.match(/row (\d+)/i);
    const colMatch = label.match(/column (\d+)/i);
    if (!colorMatch || !rowMatch || !colMatch) continue;

    const color = colorMatch[1].trim();
    const row = parseInt(rowMatch[1], 10) - 1;
    const col = parseInt(colMatch[1], 10) - 1;

    if (!(color in colorMap)) colorMap[color] = colorCount++;
    region[row][col] = colorMap[color];
  }

  return { region, gridSize };
}

// ---------------------------------------------------------------------------
// Click helpers
// ---------------------------------------------------------------------------

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function simulateDoubleClick(element) {
  element.focus();
  const rect = element.getBoundingClientRect();
  const x = rect.left + rect.width / 2;
  const y = rect.top + rect.height / 2;
  const opts = { bubbles: true, cancelable: true, clientX: x, clientY: y };

  // First click
  element.dispatchEvent(new PointerEvent("pointerdown", { ...opts, isPrimary: true }));
  element.dispatchEvent(new MouseEvent("mousedown", opts));
  element.dispatchEvent(new PointerEvent("pointerup", { ...opts, isPrimary: true }));
  element.dispatchEvent(new MouseEvent("mouseup", opts));
  element.dispatchEvent(new MouseEvent("click", opts));

  await sleep(60);

  // Second click + dblclick
  element.dispatchEvent(new PointerEvent("pointerdown", { ...opts, isPrimary: true }));
  element.dispatchEvent(new MouseEvent("mousedown", opts));
  element.dispatchEvent(new PointerEvent("pointerup", { ...opts, isPrimary: true }));
  element.dispatchEvent(new MouseEvent("mouseup", opts));
  element.dispatchEvent(new MouseEvent("click", { ...opts, detail: 2 }));
  element.dispatchEvent(new MouseEvent("dblclick", { ...opts, detail: 2 }));
}

async function resetBoard() {
  const grid = document.querySelector('[data-testid="interactive-grid"]');
  if (!grid) return 0;

  const placed = Array.from(
    grid.querySelectorAll('[data-testid^="cell-"]')
  ).filter((c) => (c.getAttribute("aria-label") || "").toLowerCase().includes("queen"));

  for (const cell of placed) {
    simulateDoubleClick(cell);
    await sleep(CLICK_DELAY_MS);
  }
  return placed.length;
}

// ---------------------------------------------------------------------------
// Main solve flow
// ---------------------------------------------------------------------------

async function runSolve(batchPlacement = true) {
  const puzzle = extractPuzzle();
  if (!puzzle) {
    return {
      success: false,
      error: "Could not find the Queens game board. Make sure the puzzle is visible on the page.",
    };
  }

  let response;
  try {
    response = await fetch(`${SERVER_URL}/solve`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ region: puzzle.region, batch_placement: batchPlacement }),
    });
  } catch {
    return {
      success: false,
      error:
        "Cannot reach the local solver server. Start it with:\n" +
        "uvicorn server.app:app",
    };
  }

  if (!response.ok) {
    const detail = await response.text();
    return { success: false, error: `Server error ${response.status}: ${detail}` };
  }

  const data = await response.json();
  const grid = document.querySelector('[data-testid="interactive-grid"]');
  let totalQueens = 0;

  for (const step of data.steps) {
    // Show heatmaps for this forward pass, then place its queens
    updateOverlayHeatmaps(step.l_heatmap, step.h_activation_heatmap, step.h_heatmap);
    await sleep(HEATMAP_PAUSE_MS);

    for (const [row, col] of step.queens) {
      const cellIdx = row * puzzle.gridSize + col;
      const cell = grid?.querySelector(`[data-testid="cell-${cellIdx}"]`);
      if (cell) {
        await simulateDoubleClick(cell);
        await sleep(CLICK_DELAY_MS);
        totalQueens++;
      }
    }
  }

  return { success: true, count: totalQueens };
}

// ---------------------------------------------------------------------------
// Message listener
// ---------------------------------------------------------------------------

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message.action === "check") {
    fetch(`${SERVER_URL}/health`)
      .then((r) => r.json())
      .then((data) => sendResponse({ connected: true, model_loaded: data.model_loaded }))
      .catch(() => sendResponse({ connected: false, model_loaded: false }));
    return true;
  }

  if (message.action === "solve") {
    runSolve(message.batchPlacement !== false).then(sendResponse);
    return true;
  }

  if (message.action === "reset") {
    resetBoard().then((count) => sendResponse({ success: true, count }));
    return true;
  }
});
