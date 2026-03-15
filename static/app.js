const state = {
  sessionId: null,
  busy: false,
  health: null,
};

const els = {
  sessionId: document.querySelector("#session-id"),
  docList: document.querySelector("#document-list"),
  pathInput: document.querySelector("#path-input"),
  fileInput: document.querySelector("#file-input"),
  chatLog: document.querySelector("#chat-log"),
  chatInput: document.querySelector("#chat-input"),
  chatForm: document.querySelector("#chat-form"),
  status: document.querySelector("#status"),
  loadSamples: document.querySelector("#load-samples"),
  loadPaths: document.querySelector("#load-paths"),
  uploadDocs: document.querySelector("#upload-docs"),
  resetHistory: document.querySelector("#reset-history"),
  clearSession: document.querySelector("#clear-session"),
  healthGrid: document.querySelector("#health-grid"),
  resetCache: document.querySelector("#reset-cache"),
  runEval: document.querySelector("#run-eval"),
  evalSummary: document.querySelector("#eval-summary"),
  evalCases: document.querySelector("#eval-cases"),
};

async function request(url, options = {}) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.detail || "请求失败");
  }
  return data;
}

function setStatus(message) {
  els.status.textContent = message || "";
}

function setBusy(nextBusy) {
  state.busy = nextBusy;
  const buttons = document.querySelectorAll("button");
  buttons.forEach((button) => {
    button.disabled = nextBusy;
  });
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function formatScore(score) {
  return Number(score || 0).toFixed(4);
}

function formatBoolean(value, truthy = "已启用", falsy = "未启用") {
  return value ? truthy : falsy;
}

function renderDocuments(documents) {
  if (!documents.length) {
    els.docList.innerHTML = '<div class="doc-card"><span>当前还没有知识库文档。</span></div>';
    return;
  }

  els.docList.innerHTML = documents
    .map(
      (doc) => `
        <div class="doc-card">
          <strong>${escapeHtml(doc.source_name)}</strong>
          <span>类型: ${escapeHtml(doc.source_type)} | chunks: ${doc.chunk_count}</span>
          <span>${escapeHtml(doc.source_path)}</span>
        </div>
      `
    )
    .join("");
}

function renderHealth(data) {
  state.health = data;
  const cacheStats = data.cache_stats || {};
  const retrievalLabel = data.reranking_enabled
    ? "Dense + BM25 + Rerank"
    : "Dense + BM25";

  const cards = [
    {
      label: "模型提供方",
      value: `${data.provider} / ${data.model}`,
      meta: data.base_url || "默认网关",
    },
    {
      label: "Embedding",
      value: data.embedding_model,
      meta: retrievalLabel,
    },
    {
      label: "Reranking",
      value: formatBoolean(data.reranking_enabled, "已启用", "未启用"),
      meta: data.reranker_model || "当前未使用 Cross-Encoder",
    },
    {
      label: "LLM Cache",
      value: formatBoolean(data.cache_enabled, "已启用", "未启用"),
      meta: data.cache_enabled
        ? `hits ${cacheStats.hits || 0} / misses ${cacheStats.misses || 0} / entries ${cacheStats.entries || 0}`
        : "LangChain InMemoryCache 未开启",
    },
    {
      label: "API Key",
      value: formatBoolean(data.api_key_configured, "已配置", "未配置"),
      meta: data.api_key_configured ? "可以直接发起问答和 Benchmark" : "请先配置模型 API Key",
    },
  ];

  els.healthGrid.innerHTML = cards
    .map(
      (card) => `
        <div class="metric-card">
          <span class="metric-label">${escapeHtml(card.label)}</span>
          <strong class="metric-value">${escapeHtml(card.value)}</strong>
          <span class="metric-meta">${escapeHtml(card.meta)}</span>
        </div>
      `
    )
    .join("");
}

function buildMessageHtml(role, content, extra = {}) {
  const citations = (extra.citations || [])
    .map(
      (item) =>
        `<span class="chip">${escapeHtml(item.source_name)} / ${escapeHtml(item.segment_label)}</span>`
    )
    .join("");

  const sourceDocuments = (extra.source_documents || [])
    .map(
      (doc) => `
        <div class="source-item">
          <strong>${escapeHtml(doc.source_name)} / ${escapeHtml(doc.segment_label)}</strong>
          <span>score: ${formatScore(doc.score)}</span>
          <span>${escapeHtml(doc.content)}</span>
        </div>
      `
    )
    .join("");

  const rewritten = extra.rewritten_question
    ? `<div class="meta-row"><span class="chip">检索改写: ${escapeHtml(extra.rewritten_question)}</span></div>`
    : "";

  const grounded = role === "assistant" && typeof extra.grounded === "boolean"
    ? `<div class="meta-row"><span class="chip">${extra.grounded ? "已命中证据" : "证据不足"}</span></div>`
    : "";

  return `
    <div class="role">${role === "user" ? "User" : "Assistant"}</div>
    <div class="content">${escapeHtml(content)}</div>
    ${grounded}
    ${rewritten}
    ${citations ? `<div class="citation-row">${citations}</div>` : ""}
    ${sourceDocuments ? `<div class="source-row">${sourceDocuments}</div>` : ""}
  `;
}

function renderMessage(role, content, extra = {}) {
  const wrapper = document.createElement("div");
  wrapper.className = `message ${role}`;
  wrapper.innerHTML = buildMessageHtml(role, content, extra);
  els.chatLog.appendChild(wrapper);
  els.chatLog.scrollTop = els.chatLog.scrollHeight;
  return wrapper;
}

function updateMessage(wrapper, role, content, extra = {}) {
  wrapper.className = `message ${role}`;
  wrapper.innerHTML = buildMessageHtml(role, content, extra);
  els.chatLog.scrollTop = els.chatLog.scrollHeight;
}

function renderEvaluation(data) {
  if (!data.summary_metrics.length) {
    els.evalSummary.innerHTML = '<div class="empty-state">没有可展示的评估指标。</div>';
  } else {
    els.evalSummary.innerHTML = data.summary_metrics
      .map(
        (metric) => `
          <div class="metric-card compact">
            <span class="metric-label">${escapeHtml(metric.label)}</span>
            <strong class="metric-value">${formatScore(metric.score)}</strong>
            <span class="metric-meta">${escapeHtml(metric.name)}</span>
          </div>
        `
      )
      .join("");
  }

  els.evalCases.innerHTML = data.cases
    .map((item) => {
      const metrics = item.metrics
        .map(
          (metric) => `
            <span class="score-pill ${metric.score >= 0.75 ? "good" : metric.score >= 0.5 ? "mid" : "low"}">
              ${escapeHtml(metric.label)} ${formatScore(metric.score)}
            </span>
          `
        )
        .join("");

      const citations = item.citations
        .map(
          (citation) =>
            `<span class="chip">${escapeHtml(citation.source_name)} / ${escapeHtml(
              citation.segment_label
            )}</span>`
        )
        .join("");

      return `
        <article class="eval-card">
          <div class="eval-header">
            <strong>${escapeHtml(item.question)}</strong>
            <span class="score-pill ${item.grounded ? "good" : "low"}">
              ${item.grounded ? "grounded" : "ungrounded"}
            </span>
          </div>
          <p><b>系统回答：</b>${escapeHtml(item.answer)}</p>
          <p><b>参考答案：</b>${escapeHtml(item.reference_answer)}</p>
          <p><b>检索改写：</b>${escapeHtml(item.rewritten_question)}</p>
          <div class="score-row">${metrics}</div>
          ${citations ? `<div class="citation-row">${citations}</div>` : ""}
        </article>
      `;
    })
    .join("");
}

async function ensureSession() {
  const cached = window.localStorage.getItem("rag-session-id");
  if (cached) {
    state.sessionId = cached;
    els.sessionId.textContent = cached;
    await refreshDocuments();
    return;
  }

  const data = await request("/api/session", { method: "POST" });
  state.sessionId = data.session_id;
  window.localStorage.setItem("rag-session-id", data.session_id);
  els.sessionId.textContent = data.session_id;
}

async function refreshDocuments() {
  if (!state.sessionId) return;
  const data = await request(`/api/sessions/${state.sessionId}/documents`);
  if (data.session_id !== state.sessionId) {
    state.sessionId = data.session_id;
    window.localStorage.setItem("rag-session-id", data.session_id);
    els.sessionId.textContent = data.session_id;
  }
  renderDocuments(data.documents);
}

async function refreshHealth() {
  const data = await request("/api/health");
  renderHealth(data);
}

async function ingestSamples() {
  setBusy(true);
  setStatus("正在加载内置业务样例知识库...");
  try {
    await request("/api/documents/sample", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: state.sessionId }),
    });
    await refreshDocuments();
    setStatus("样例知识库已加载，现在可以直接提问或运行 Benchmark。");
  } catch (error) {
    setStatus(error.message);
  } finally {
    setBusy(false);
  }
}

async function ingestPaths() {
  const paths = els.pathInput.value
    .split("\n")
    .map((item) => item.trim())
    .filter(Boolean);
  if (!paths.length) {
    setStatus("请先输入至少一个文档路径。");
    return;
  }

  setBusy(true);
  setStatus("正在按路径建立知识库...");
  try {
    const data = await request("/api/documents/path", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: state.sessionId, paths }),
    });
    await refreshDocuments();
    setStatus(`已导入 ${data.documents.length} 份文档。`);
  } catch (error) {
    setStatus(error.message);
  } finally {
    setBusy(false);
  }
}

async function uploadDocuments() {
  const files = Array.from(els.fileInput.files || []);
  if (!files.length) {
    setStatus("请选择要上传的文档。");
    return;
  }

  setBusy(true);
  setStatus("正在上传并切块向量化...");
  try {
    const form = new FormData();
    form.append("session_id", state.sessionId);
    files.forEach((file) => form.append("files", file));
    const data = await request("/api/documents/upload", {
      method: "POST",
      body: form,
    });
    await refreshDocuments();
    els.fileInput.value = "";
    setStatus(`已上传 ${data.documents.length} 份文档。`);
  } catch (error) {
    setStatus(error.message);
  } finally {
    setBusy(false);
  }
}

async function sendQuestion(event) {
  event.preventDefault();
  const question = els.chatInput.value.trim();
  if (!question) {
    setStatus("请输入问题。");
    return;
  }

  renderMessage("user", question);
  els.chatInput.value = "";
  setBusy(true);
  setStatus("正在进行检索和回答...");
  const assistantMessage = renderMessage("assistant", "正在检索相关片段...");

  try {
    const data = await streamChat(question, assistantMessage);
    await refreshHealth();
    setStatus(data.grounded ? "回答完成。" : "回答完成，但证据不足时已自动降级为“我不知道”。");
  } catch (error) {
    updateMessage(assistantMessage, "assistant", `失败: ${error.message}`);
    setStatus(error.message);
  } finally {
    setBusy(false);
  }
}

async function streamChat(question, assistantMessage) {
  const response = await fetch("/api/chat/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: state.sessionId, question }),
  });

  if (!response.ok) {
    const data = await response.json();
    throw new Error(data.detail || "请求失败");
  }
  if (!response.body) {
    throw new Error("浏览器未收到流式响应。");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";
  let content = "";
  let finalPayload = null;
  let assistantState = {
    grounded: null,
    rewritten_question: "",
    citations: [],
    source_documents: [],
  };

  const applyEvent = (eventName, payload) => {
    if (eventName === "meta") {
      assistantState = { ...assistantState, ...payload };
      if (!content) {
        updateMessage(assistantMessage, "assistant", "正在生成回答...", assistantState);
      }
      return;
    }

    if (eventName === "delta") {
      content += payload.text || "";
      updateMessage(assistantMessage, "assistant", content || "正在生成回答...", assistantState);
      return;
    }

    if (eventName === "final") {
      finalPayload = payload;
      assistantState = payload;
      content = payload.answer || content;
      updateMessage(assistantMessage, "assistant", content, assistantState);
      return;
    }

    if (eventName === "error") {
      throw new Error(payload.detail || "流式响应失败");
    }
  };

  const flushBuffer = () => {
    while (buffer.includes("\n\n")) {
      const boundary = buffer.indexOf("\n\n");
      const rawEvent = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      if (!rawEvent.trim()) {
        continue;
      }

      let eventName = "message";
      const dataLines = [];
      rawEvent.split("\n").forEach((line) => {
        if (line.startsWith("event:")) {
          eventName = line.slice(6).trim();
          return;
        }
        if (line.startsWith("data:")) {
          dataLines.push(line.slice(5).trim());
        }
      });
      if (!dataLines.length) {
        continue;
      }
      applyEvent(eventName, JSON.parse(dataLines.join("\n")));
    }
  };

  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
    flushBuffer();
    if (done) {
      break;
    }
  }

  if (buffer.trim()) {
    buffer += "\n\n";
    flushBuffer();
  }

  if (!finalPayload) {
    throw new Error("流式响应提前结束，未收到最终结果。");
  }
  return finalPayload;
}

async function runEvaluation() {
  setBusy(true);
  setStatus("正在运行 RAGAS Benchmark，这一步会额外调用评估模型...");
  try {
    const data = await request("/api/evaluate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: state.sessionId }),
    });
    renderEvaluation(data);
    await refreshHealth();
    setStatus(`Benchmark 已完成，共评估 ${data.sample_count} 个问题。`);
  } catch (error) {
    setStatus(error.message);
  } finally {
    setBusy(false);
  }
}

async function resetCache() {
  setBusy(true);
  setStatus("正在清空 LLM Cache...");
  try {
    await request("/api/cache/reset", { method: "POST" });
    await refreshHealth();
    setStatus("LLM Cache 已清空。");
  } catch (error) {
    setStatus(error.message);
  } finally {
    setBusy(false);
  }
}

async function resetSession(clearDocuments) {
  setBusy(true);
  setStatus(clearDocuments ? "正在清空会话和知识库..." : "正在清空会话历史...");
  try {
    const data = await request("/api/session/reset", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: state.sessionId, clear_documents: clearDocuments }),
    });
    els.chatLog.innerHTML = "";
    renderDocuments(data.documents);
    setStatus(clearDocuments ? "会话和知识库已清空。" : "会话历史已清空。");
  } catch (error) {
    setStatus(error.message);
  } finally {
    setBusy(false);
  }
}

async function bootstrap() {
  try {
    await ensureSession();
    await refreshHealth();
    setStatus("会话已创建，可以先加载样例知识库再开始提问。");
  } catch (error) {
    setStatus(error.message);
  }
}

els.loadSamples.addEventListener("click", ingestSamples);
els.loadPaths.addEventListener("click", ingestPaths);
els.uploadDocs.addEventListener("click", uploadDocuments);
els.chatForm.addEventListener("submit", sendQuestion);
els.resetHistory.addEventListener("click", () => resetSession(false));
els.clearSession.addEventListener("click", () => resetSession(true));
els.resetCache.addEventListener("click", resetCache);
els.runEval.addEventListener("click", runEvaluation);

bootstrap();
