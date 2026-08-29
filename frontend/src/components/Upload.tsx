import { useRef, useState } from "react";
import { ingestFile, ingestText } from "../api";
import type { IngestResult } from "../types";
import {
  CATEGORY_COLORS,
  categoryLabel,
  confidenceColor,
  formatDateTime,
} from "../common";

export default function Upload() {
  const [tab, setTab] = useState<"file" | "text">("file");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [result, setResult] = useState<IngestResult | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  const [text, setText] = useState("");
  const [filename, setFilename] = useState("");

  async function handleFile(file: File) {
    setErr(null);
    setBusy(true);
    setResult(null);
    try {
      const r = await ingestFile(file);
      setResult(r);
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function handleText() {
    if (!text.trim()) {
      setErr("请输入文本内容");
      return;
    }
    setErr(null);
    setBusy(true);
    setResult(null);
    try {
      const r = await ingestText({ content: text, filename: filename || undefined });
      setResult(r);
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="page">
      <h1>导入通知</h1>
      <p className="page-sub">
        支持图片（海报/截图）、PDF（课程通知/作业要求）、文本。系统会自动识别、抽取关键信息并生成待办。
      </p>

      <div className="tabs">
        <button className={tab === "file" ? "active" : ""} onClick={() => setTab("file")}>
          上传文件
        </button>
        <button className={tab === "text" ? "active" : ""} onClick={() => setTab("text")}>
          粘贴文本
        </button>
      </div>

      {tab === "file" && (
        <div
          className={`dropzone ${dragOver ? "over" : ""}`}
          onClick={() => fileRef.current?.click()}
          onDragOver={(e) => {
            e.preventDefault();
            setDragOver(true);
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragOver(false);
            const f = e.dataTransfer.files?.[0];
            if (f) handleFile(f);
          }}
        >
          <div className="dropzone-icon">📥</div>
          <div>点击或拖拽文件到此处</div>
          <div className="muted">图片 / PDF / 文本，单文件</div>
          <input
            ref={fileRef}
            type="file"
            hidden
            accept="image/*,application/pdf,.txt,.md,.text"
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) handleFile(f);
            }}
          />
        </div>
      )}

      {tab === "text" && (
        <div className="card">
          <label className="field-label">文本内容</label>
          <textarea
            className="textarea"
            rows={10}
            placeholder="把通知原文粘贴到这里，例如：&#10;各位同学：&#10;《数据结构》课程调整至周五下午2:30，地点改为第三教学楼A305，请相互转告。"
            value={text}
            onChange={(e) => setText(e.target.value)}
          />
          <label className="field-label">文件名（可选）</label>
          <input
            className="input"
            placeholder="例如：数据结构调课通知"
            value={filename}
            onChange={(e) => setFilename(e.target.value)}
          />
          <button className="btn primary" disabled={busy} onClick={handleText}>
            {busy ? "处理中…" : "提交并解析"}
          </button>
        </div>
      )}

      {busy && <div className="card muted">正在解析与抽取，请稍候…</div>}
      {err && <div className="card error">❌ {err}</div>}
      {result && <ResultView r={result} />}
    </div>
  );
}

function ResultView({ r }: { r: IngestResult }) {
  const doc = r.document;
  return (
    <div className="result">
      <div className="card">
        <div className="card-title">文档：{doc.filename}</div>
        <div className="muted">
          类型 {doc.kind} · {doc.size_bytes} 字节 · 状态 {doc.status}
          {doc.sha256 && (
            <>
              {" "}
              · sha256 {doc.sha256.slice(0, 12)}…
            </>
          )}
        </div>

        {r.duplicate && (
          <div className="banner warn">
            ⚠️ 检测到重复通知（与 #{r.duplicate_of_id} 相似度 {r.dedup_score?.toFixed(2)}），已跳过建库。
          </div>
        )}
        {r.needs_review && (
          <div className="banner info">
            🔎 抽取置信度偏低，建议到「通知复核」中人工确认。
          </div>
        )}
        {r.warnings.length > 0 && (
          <ul className="warn-list">
            {r.warnings.map((w, i) => (
              <li key={i}>⚠️ {w}</li>
            ))}
          </ul>
        )}
      </div>

      {r.notice && (
        <div className="card">
          <div className="card-title">
            <span
              className="badge"
              style={{ background: CATEGORY_COLORS[r.notice.category] ?? "#6b7280" }}
            >
              {categoryLabel(r.notice.category)}
            </span>
            {r.notice.title}
          </div>
          <div className="kv">
            <span>发布方</span>
            <b>{r.notice.issuer ?? "—"}</b>
          </div>
          <div className="kv">
            <span>地点</span>
            <b>{r.notice.location ?? "—"}</b>
          </div>
          <div className="kv">
            <span>截止</span>
            <b>{formatDateTime(r.notice.deadline)}</b>
          </div>
          <div className="kv">
            <span>时间</span>
            <b>{formatDateTime(r.notice.event_time)}</b>
          </div>
          <div className="kv">
            <span>置信度</span>
            <b style={{ color: confidenceColor(r.notice.confidence) }}>
              {(r.notice.confidence * 100).toFixed(0)}%
            </b>
          </div>
          {r.notice.summary && <div className="summary">{r.notice.summary}</div>}
        </div>
      )}

      {r.tasks.length > 0 && (
        <div className="card">
          <div className="card-title">已生成待办（{r.tasks.length}）</div>
          <ul className="task-mini">
            {r.tasks.map((t) => (
              <li key={t.id}>
                <span className="dot-p" /> {t.title}
                <span className="muted"> · 截止 {formatDateTime(t.due_at)}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {r.trace.length > 0 && (
        <details className="card">
          <summary className="card-title">处理轨迹（Pipeline Trace）</summary>
          <ol className="trace">
            {r.trace.map((step, i) => (
              <li key={i}>
                <code>{JSON.stringify(step)}</code>
              </li>
            ))}
          </ol>
        </details>
      )}
    </div>
  );
}
