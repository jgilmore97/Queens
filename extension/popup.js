const solveBtn = document.getElementById("solve-btn");
const resetBtn = document.getElementById("reset-btn");
const statusDot = document.getElementById("status-dot");
const statusText = document.getElementById("status-text");
const messageEl = document.getElementById("message");

function setMessage(text, type = "") {
  messageEl.textContent = text;
  messageEl.className = type;
}

function setStatus(connected, modelLoaded) {
  if (!connected) {
    statusDot.className = "err";
    statusText.textContent = "Server not running";
    solveBtn.disabled = true;
    resetBtn.disabled = true;
  } else if (!modelLoaded) {
    statusDot.className = "err";
    statusText.textContent = "Server up — model not loaded";
    solveBtn.disabled = true;
    resetBtn.disabled = true;
  } else {
    statusDot.className = "ok";
    statusText.textContent = "Server ready";
    solveBtn.disabled = false;
    resetBtn.disabled = false;
  }
}

async function sendToContentScript(action, extra = {}) {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  return new Promise((resolve) => {
    chrome.tabs.sendMessage(tab.id, { action, ...extra }, resolve);
  });
}

// Check server status on open
sendToContentScript("check").then((res) => {
  if (!res) {
    setStatus(false, false);
    setMessage("Could not reach the content script.\nMake sure you're on linkedin.com.");
    return;
  }
  setStatus(res.connected, res.model_loaded);
  if (!res.connected) {
    setMessage(
      "Start the server:\nQUEENS_CHECKPOINT=<path> uvicorn server.app:app"
    );
  }
});

solveBtn.addEventListener("click", async () => {
  solveBtn.disabled = true;
  resetBtn.disabled = true;
  setMessage("Extracting puzzle…");

  const batchPlacement = document.getElementById("batch-toggle").checked;
  const res = await sendToContentScript("solve", { batchPlacement });

  solveBtn.disabled = false;
  resetBtn.disabled = false;

  if (!res) {
    setMessage("No response from page. Reload and try again.", "error");
  } else if (res.success) {
    setMessage(`Placed ${res.count} queens.`, "success");
  } else {
    setMessage(res.error, "error");
  }
});

resetBtn.addEventListener("click", async () => {
  solveBtn.disabled = true;
  resetBtn.disabled = true;
  setMessage("Resetting board…");

  const res = await sendToContentScript("reset");

  solveBtn.disabled = false;
  resetBtn.disabled = false;

  if (res && res.success) {
    const noun = res.count === 1 ? "queen" : "queens";
    setMessage(`Removed ${res.count} ${noun}.`, "success");
  } else {
    setMessage("Could not reset board.", "error");
  }
});
