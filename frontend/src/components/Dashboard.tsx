import { useEffect, useState } from "react";
import { stats } from "../api";
import type { StatsOut } from "../types";

function Bars({ title, data, empty }: { title: string; data: Record<string, number>; empty: string }) {
  const max = Math.max(1, ...Object.values(data));
  return (
    <div className="card">
      <div className="card-title">{title}</div>
      {Object.keys(data).length === 0 ? (
        <div className="muted">{empty}</div>
      ) : (
        <div className="bars">
          {Object.entries(data).map(([k, v]) => (
            <div className="bar-row" key={k}>
              <span className="bar-label">{k}</span>
              <div className="bar-track">
                <div className="bar-fill" style={{ width: `${(v / max) * 100}%` }} />
              </div>
              <span className="bar-value">{v}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default function Dashboard({ onNavigate }: { onNavigate: (v: string) => void }) {
  const [data, setData] = useState<StatsOut | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    const load = () => stats().then(setData).catch((e) => setErr(e.message));
    load();
    // 统计只在挂载时拉一次会过期：导入/看板操作后回到本页若不再挂载
    // （或开着另一个标签页），数字就失去时效。轮询 + 聚焦刷新兜底。
    const timer = setInterval(load, 15000);
    const onFocus = () => load();
    window.addEventListener("focus", onFocus);
    return () => {
      clearInterval(timer);
      window.removeEventListener("focus", onFocus);
    };
  }, []);

  if (err) return <div className="card error">加载统计失败：{err}</div>;
  if (!data) return <div className="card muted">加载中…</div>;

  const cards: { label: string; value: number; accent: string; to?: string }[] = [
    // 通知与待办是 1:N（一条通知可生成多条待办，手动待办不挂通知），
    // 展示总数卡片，避免「待处理+进行中+已完成」被误读为统计错误。
    { label: "待办总数", value: data.tasks_total, accent: "#475569", to: "tasks" },
    { label: "待办(待处理)", value: data.tasks_todo, accent: "#d97706", to: "tasks" },
    { label: "进行中", value: data.tasks_doing, accent: "#0891b2" },
    { label: "已完成", value: data.tasks_done, accent: "#16a34a" },
    { label: "已逾期", value: data.tasks_overdue, accent: "#dc2626", to: "tasks" },
    { label: "待复核", value: data.needs_review, accent: "#db2777", to: "notices" },
  ];

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

      <Bars
        title="任务类别分布（不含已归档）"
        data={data.tasks_by_category ?? {}}
        empty="暂无待办，去「导入」上传通知或在「待办看板」手动新建。"
      />
    </div>
  );
}
