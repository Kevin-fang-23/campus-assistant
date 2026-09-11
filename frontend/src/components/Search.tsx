import { useState } from "react";
import { askQA, search } from "../api";
import type { QAOut, SearchOut } from "../types";
import { CATEGORY_COLORS, categoryLabel, formatDateTime } from "../common";

const EXAMPLES = [
  "调课通知",
  "周五下午的讲座",
  "数据结构作业截止",
  "宿舍报修",
  "奖学金申请",
];

type Mode = "search" | "qa";

/** 命中来源标签：让"这条是靠语义还是关键词召回"可见。 */
const MATCH_LABELS: Record<string, { text: string; color: string }> = {
  both: { text: "混合", color: "#2563eb" },
  vector: { text: "语义", color: "#7c3aed" },
  bm25: { text: "关键词", color: "#0d9488" },
};

/** 把回答里的 [1] [2] 引用标记渲染成高亮标签，便于与下方引用列表对照。 */
function renderAnswer(text: string) {
  return text.split(/(\[\d+\])/g).map((part, i) =>
    /^\[\d+\]$/.test(part) ? (
      <span key={i} className="cite-mark">
        {part}
      </span>
    ) : (
      <span key={i}>{part}</span>
    )
  );
}

export default function Search() {
  const [mode, setMode] = useState<Mode>("search");
  const [q, setQ] = useState("");
  const [topK, setTopK] = useState(5);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [hits, setHits] = useState<SearchOut | null>(null);
  const [qa, setQa] = useState<QAOut | null>(null);

  function switchMode(next: Mode) {
    setMode(next);
    // 切换模式时清掉上一次结果，避免把检索结果当问答结果展示
    setHits(null);
    setQa(null);
    setErr(null);
  }

  async function run(query: string) {
    if (!query.trim()) return;
    setErr(null);
    setBusy(true);
    setHits(null);
    setQa(null);
    try {
      if (mode === "qa") {
        setQa(await askQA(query, topK));
      } else {
        setHits(await search(query, topK));
      }
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="page">
      <h1>知识库问答</h1>
      <p className="page-sub">
        先向量召回相关通知，再由大模型据此作答并标注引用来源；
        模型不可用时自动降级为抽取式回答，功能不中断。
      </p>

      <div className="tabs">
        <button className={mode === "qa" ? "active" : ""} onClick={() => switchMode("qa")}>
          💬 智能问答
        </button>
        <button
          className={mode === "search" ? "active" : ""}
          onClick={() => switchMode("search")}
        >
          🔍 语义检索
        </button>
      </div>

      <div className="search-bar">
        <input
          className="input grow"
          placeholder={
            mode === "qa"
              ? "用一句话提问，例如：操作系统作业什么时候截止？"
              : "输入自然语言查询，例如：下周的作业什么时候交？"
          }
          value={q}
          onChange={(e) => setQ(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && run(q)}
        />
        <label className="field-label inline">
          Top
          <input
            type="number"
            className="input num"
            min={1}
            max={20}
            value={topK}
            onChange={(e) => setTopK(Number(e.target.value))}
          />
        </label>
        <button className="btn primary" disabled={busy} onClick={() => run(q)}>
          {busy ? (mode === "qa" ? "生成中…" : "检索中…") : mode === "qa" ? "提问" : "检索"}
        </button>
      </div>

      <div className="examples">
        试试：
        {EXAMPLES.map((ex) => (
          <button
            key={ex}
            className="chip"
            onClick={() => {
              setQ(ex);
              run(ex);
            }}
          >
            {ex}
          </button>
        ))}
      </div>

      {err && <div className="card error">{err}</div>}

      {/* ---------- 问答结果 ---------- */}
      {qa && (
        <>
          <div className="card">
            <div className="card-title">
              回答
              {qa.degraded ? (
                <span className="flag sm">抽取式</span>
              ) : (
                <span className="badge sm" style={{ background: "#16a34a" }}>
                  AI 生成
                </span>
              )}
              <span className="muted" style={{ fontWeight: 400 }}>
                · 引擎 {qa.backend}
              </span>
            </div>

            {qa.degraded && (
              <div className="banner info">
                生成式回答当前不可用（未配置模型密钥或调用失败），已自动降级为
                <strong>抽取式回答</strong>：直接给出最相关通知的原文片段。检索与引用不受影响。
              </div>
            )}

            <div className="answer">{renderAnswer(qa.answer)}</div>
          </div>

          {qa.citations.length > 0 && (
            <div className="card">
              <div className="card-title">
                引用来源
                <span className="muted" style={{ fontWeight: 400 }}>
                  · 共 {qa.citations.length} 条
                </span>
              </div>
              <ol className="citations">
                {qa.citations.map((c, i) => (
                  <li key={c.notice_id} className="citation">
                    <div className="citation-head">
                      <span className="cite-mark">[{i + 1}]</span>
                      <span className="citation-title">{c.title}</span>
                      <span className="score">{(c.score * 100).toFixed(0)}%</span>
                    </div>
                    <div className="citation-snippet">{c.snippet}</div>
                    <div className="hit-meta">
                      <span className="muted">通知 #{c.notice_id}</span>
                    </div>
                  </li>
                ))}
              </ol>
            </div>
          )}
        </>
      )}

      {/* ---------- 检索结果 ---------- */}
      {hits && (
        <div className="card">
          <div className="card-title">
            检索结果
            <span className="muted"> · 引擎 {hits.backend}</span>
          </div>
          {hits.hits.length === 0 && <div className="muted">没有匹配的历史通知。</div>}
          <ul className="hits">
            {hits.hits.map((h) => (
              <li key={h.notice_id} className="hit">
                <div className="hit-head">
                  <span
                    className="badge sm"
                    style={{ background: CATEGORY_COLORS[h.category] ?? "#6b7280" }}
                  >
                    {categoryLabel(h.category)}
                  </span>
                  {h.match && MATCH_LABELS[h.match] && (
                    <span
                      className="badge sm"
                      title={
                        h.score_vector != null || h.score_bm25 != null
                          ? `语义 ${h.score_vector ?? "—"} · 关键词 ${h.score_bm25 ?? "—"}`
                          : undefined
                      }
                      style={{ background: MATCH_LABELS[h.match].color }}
                    >
                      {MATCH_LABELS[h.match].text}
                    </span>
                  )}
                  <span className="hit-title">{h.title}</span>
                  <span className="score">{(h.score * 100).toFixed(0)}%</span>
                </div>
                {h.summary && <div className="hit-summary">{h.summary}</div>}
                <div className="hit-meta">
                  <span>🕒 {formatDateTime(h.deadline)}</span>
                  <span className="muted">通知 #{h.notice_id}</span>
                </div>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
