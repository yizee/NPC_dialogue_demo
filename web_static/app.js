const state = {
  npcs: [],
  selectedNpcId: null,
  sessionId: null,
  playerId: "demo_player",
};

const elements = {
  npcList: document.getElementById("npcList"),
  modeSelect: document.getElementById("modeSelect"),
  chatLog: document.getElementById("chatLog"),
  messageInput: document.getElementById("messageInput"),
  sendButton: document.getElementById("sendButton"),
  chatForm: document.getElementById("chatForm"),
  npcName: document.getElementById("npcName"),
  npcTitle: document.getElementById("npcTitle"),
  npcPersona: document.getElementById("npcPersona"),
  sessionBadge: document.getElementById("sessionBadge"),
  intentStatus: document.getElementById("intentStatus"),
  ragStatus: document.getElementById("ragStatus"),
  toolStatus: document.getElementById("toolStatus"),
  memoryStatus: document.getElementById("memoryStatus"),
  fallbackStatus: document.getElementById("fallbackStatus"),
  errorBox: document.getElementById("errorBox"),
};

async function loadNpcs() {
  const response = await fetch("/api/npcs");
  const body = await response.json();
  state.npcs = body.npcs;
  renderNpcList();
  if (state.npcs.length > 0) {
    await selectNpc(state.npcs[0].id);
  }
}

function renderNpcList() {
  elements.npcList.innerHTML = "";
  state.npcs.forEach((npc) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `npc-button${npc.id === state.selectedNpcId ? " active" : ""}`;
    button.textContent = `${npc.name} - ${npc.title}`;
    button.addEventListener("click", () => selectNpc(npc.id));
    elements.npcList.appendChild(button);
  });
}

async function selectNpc(npcId) {
  clearError();
  state.selectedNpcId = npcId;
  renderNpcList();

  const response = await fetch("/api/sessions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      npc_id: npcId,
      player_id: state.playerId,
      mode: elements.modeSelect.value,
    }),
  });
  const body = await response.json();
  if (!response.ok) {
    showError(body.detail || "Failed to create session");
    return;
  }

  state.sessionId = body.session_id;
  elements.npcName.textContent = body.npc.name;
  elements.npcTitle.textContent = body.npc.title;
  elements.npcPersona.textContent = body.npc.personality;
  elements.sessionBadge.textContent = body.session_id;
  elements.chatLog.innerHTML = "";
  appendMessage("npc", body.npc.greeting);
  resetStatus();
}

async function sendMessage(message) {
  appendMessage("player", message);
  elements.messageInput.value = "";
  elements.sendButton.disabled = true;
  clearError();

  const response = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      session_id: state.sessionId,
      npc_id: state.selectedNpcId,
      player_id: state.playerId,
      message,
      mode: elements.modeSelect.value,
    }),
  });
  const body = await response.json();
  elements.sendButton.disabled = false;

  if (!response.ok) {
    showError(body.detail || "Message failed");
    return;
  }

  appendMessage("npc", body.reply);
  renderMetadata(body.metadata);
}

function appendMessage(role, text) {
  const message = document.createElement("div");
  message.className = `message ${role}`;
  message.textContent = text;
  elements.chatLog.appendChild(message);
  elements.chatLog.scrollTop = elements.chatLog.scrollHeight;
}

function renderMetadata(metadata) {
  elements.intentStatus.textContent = metadata.intent || "-";
  elements.ragStatus.textContent = metadata.rag_used ? "Used" : "No";
  elements.toolStatus.textContent = metadata.tool_used ? "Used" : "No";
  elements.memoryStatus.textContent = metadata.memory_recorded ? `Turn ${metadata.turn_count}` : "Not recorded";
  elements.fallbackStatus.textContent = metadata.fallback || "None";
}

function resetStatus() {
  elements.intentStatus.textContent = "-";
  elements.ragStatus.textContent = "-";
  elements.toolStatus.textContent = "-";
  elements.memoryStatus.textContent = "-";
  elements.fallbackStatus.textContent = "None";
}

function showError(message) {
  elements.errorBox.textContent = message;
  elements.errorBox.hidden = false;
}

function clearError() {
  elements.errorBox.textContent = "";
  elements.errorBox.hidden = true;
}

elements.chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = elements.messageInput.value.trim();
  if (!message) {
    showError("Message cannot be empty");
    return;
  }
  await sendMessage(message);
});

elements.modeSelect.addEventListener("change", async () => {
  if (state.selectedNpcId) {
    await selectNpc(state.selectedNpcId);
  }
});

loadNpcs().catch((error) => {
  showError(error.message);
});
