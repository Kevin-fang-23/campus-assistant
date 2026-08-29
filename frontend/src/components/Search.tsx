import { useState } from "react";
import { search } from "../api";
import type { SearchOut } from "../types";
import { CATEGORY_COLORS, categoryLabel, formatDateTime } from "../common";

const EXAMPLES = [
  "调课通知",
  "周五下午的讲座",
  "数据结构作业截止",
  "宿舍报修",
  "奖学金申请",
];

export default function Search() {
  const [q, setQ] = useState("");
  const [topK, setTopK] = useState(5);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [res, setRes] = useState<SearchOut | null>(null);

  async function run(query: string) {
    if (!query.trim()) return;
    setErr(null);
    setBusy(true);
    setRes(null);
    try {
      setRes(await search(query, topK));
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="page">
      <h1>语义检索</h1>
      <p className="page-sub">
        基于向量（FAISS / 余弦相似度）在已入库的历史通知中检索，无需精确关键词。
      </p>

      <div className="search-bar">
        <input
          className="input grow"
          placeholder="输入自然语言查询，例如：下周的作业什么时候交？"
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
          {busy ? "检索中…" : "检索"}
        </button>
      </div>

      <div className="examples">
        试试：
        {EXAMPLES.map((ex) => (
          <button key={ex} className="chip" onClick={() => { setQ(ex); run(ex); }}>
            {ex}
          </button>
        ))}
      </div>

      {err && <div className="card error">{err}</div>}
      {res && (
        <div className="card">
          <div className="card-title">
            检索结果
            <span className="muted"> · 引擎 {res.backend}</span>
          </div>
          {res.hits.length === 0 && <div className="muted">没有匹配的历史通知。</div>}
          <ul className="hits">
            {res.hits.map((h) => (
              <li key={h.notice_id} className="hit">
                <div className="hit-head">
                  <span
                    className="badge sm"
                    style={{ background: CATEGORY_COLORS[h.category] ?? "#6b7280" }}
                  >
                    {categoryLabel(h.category)}
                  </span>
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
