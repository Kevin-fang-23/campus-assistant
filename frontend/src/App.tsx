import { useEffect, useState } from "react";
import type { Health } from "./types";
import { health } from "./api";
import Dashboard from "./components/Dashboard";
import Upload from "./components/Upload";
import Notices from "./components/Notices";
import Tasks from "./components/Tasks";
import Search from "./components/Search";

type View = "dashboard" | "upload" | "notices" | "tasks" | "search";

const NAV: { key: View; label: string; icon: string }[] = [
  { key: "dashboard", label: "概览", icon: "📊" },
  { key: "upload", label: "导入", icon: "📥" },
  { key: "notices", label: "通知复核", icon: "📋" },
  { key: "tasks", label: "待办看板", icon: "✅" },
  { key: "search", label: "知识库问答", icon: "🔍" },
];

export default function App() {
  const [view, setView] = useState<View>("dashboard");
  const [health_, setHealth] = useState<Health | null>(null);

  useEffect(() => {
    health()
      .then(setHealth)
      .catch(() =>
        setHealth({
          status: "error",
          app: "校园事务智能助手",
          pipeline_engine: "?",
          vlm: { provider: "?", model: "?", mock: true },
          ocr: "?",
          vector: { backend: "?", indexed: 0 },
          database: "?",
        })
      );
  }, []);

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-logo">🎓</div>
          <div className="brand-text">
            <div className="brand-title">校园事务助手</div>
            <div className="brand-sub">多模态 · 抽取 · 待办</div>
          </div>
        </div>
        <nav>
          {NAV.map((n) => (
            <button
              key={n.key}
              className={`nav-item ${view === n.key ? "active" : ""}`}
              onClick={() => setView(n.key)}
            >
              <span className="nav-icon">{n.icon}</span>
              <span>{n.label}</span>
            </button>
          ))}
        </nav>
        <div className="sidebar-foot">
          {health_ ? (
            <div className={`health ${health_.status === "ok" ? "ok" : "err"}`}>
              <span className="dot" />
              {health_.status === "ok" ? "后端已连接" : "后端不可用"}
            </div>
          ) : (
            <div className="health pending">
              <span className="dot" />连接中…
            </div>
          )}
          {health_ && health_.status === "ok" && (
            <div className="health-meta">
              {health_.vlm.provider} · {health_.ocr} · {health_.vector.backend}
            </div>
          )}
        </div>
      </aside>

      <main className="content">
        {view === "dashboard" && <Dashboard onNavigate={(v) => setView(v as View)} />}
        {view === "upload" && <Upload />}
        {view === "notices" && <Notices />}
        {view === "tasks" && <Tasks />}
        {view === "search" && <Search />}
      </main>
    </div>
  );
}
