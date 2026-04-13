/**
 * Queens Solver — service worker
 *
 * Receives badge update requests from the content script and applies them,
 * since content scripts cannot call chrome.action APIs directly.
 */

chrome.runtime.onMessage.addListener((message) => {
  if (message.action === "setBadge") {
    chrome.action.setBadgeText({ text: message.text });
    chrome.action.setBadgeBackgroundColor({ color: message.color });
  }
});
