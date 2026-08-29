import { useEffect, useState } from "react";
import { stats } from "../api";
import type { StatsOut } from "../types";

export default function Dashboard({ onNavigate }: { onNavigate: (v: string) => void }) {
  const [data, setData] = useState<StatsOut | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    stats().then(setData).catch((e) => setErr(e.message));
  }, []);

  if (err) return <div className="card error">加载统计失败：{err}</div>;
  if (!data) return <div className="card muted">加载中…</div>;

  const cards: { label: string; value: number; accent: string; to?: string }[] = [
    { label: "文档", value: data.documents, accent: "#2563eb" },
    { label: "通知", value: data.notices, accent: "#7c3aed" },
    { label: "待办(待处理)", value: data.tasks_todo, accent: "#d97706", to: "tasks" },
    { label: "进行中", value: data.tasks_doing, accent: "#0891b2" },
    { label: "已完成", value: data.tasks_done, accent: "#16a34a" },
    { label: "已逾期", value: data.tasks_overdue, accent: "#dc2626", to: "tasks" },
    { label: "待复核", value: data.needs_review, accent: "#db2777", to: "notices" },
  ];

  const maxCat = Math.max(1, ...Object.values(data.by_category));

  return (
    <div className="page">
      <h1>概览</h1>
      <p className="page-sub">多模态校园通知识别 → 关键信息抽取 → 待办生成 → 任务追踪。</p>

      <div className="stat-grid">
        {cards.map((c) => (
          <button
            key={c.label}
            className={`stat-card ${c.to ? "clickable" : ""}`}
            style={{ borderTopColor: c.accent }}
            onClick={() => c.to && onNavigate(c.to)}
          >
            <div className="stat-value" style={{ color: c.accent }}>
              {c.value}
            </div>
            <div className="stat-label">{c.label}</div>
          </button>
        ))}
      </div>

      <div className="card">
        <div className="card-title">通知类别分布</div>
        {Object.keys(data.by_category).length === 0 ? (
          <div className="muted">暂无数据，去「导入」上传一份通知吧。</div>
        ) : (
          <div className="bars">
            {Object.entries(data.by_category).map(([k, v]) => (
              <div className="bar-row" key={k}>
                <span className="bar-label">{k}</span>
                <div className="bar-track">
                  <div className="bar-fill" style={{ width: `${(v / maxCat) * 100}%` }} />
                </div>
                <span className="bar-value">{v}</span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
