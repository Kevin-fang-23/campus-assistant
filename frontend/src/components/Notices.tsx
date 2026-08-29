import { useEffect, useState } from "react";
import { listNotices, updateNotice } from "../api";
import type { NoticeOut } from "../types";
import {
  CATEGORIES,
  CATEGORY_COLORS,
  categoryLabel,
  confidenceColor,
  formatDateTime,
  toLocalInputValue,
} from "../common";

export default function Notices() {
  const [items, setItems] = useState<NoticeOut[]>([]);
  const [err, setErr] = useState<string | null>(null);
  const [cat, setCat] = useState<string>("");
  const [reviewOnly, setReviewOnly] = useState(false);
  const [editingId, setEditingId] = useState<number | null>(null);
  const [saving, setSaving] = useState(false);

  async function load() {
    setErr(null);
    try {
      const data = await listNotices({
        category: cat || undefined,
        needs_review: reviewOnly ? true : undefined,
      });
      setItems(data);
    } catch (e) {
      setErr((e as Error).message);
    }
  }

  useEffect(() => {
    load();
  }, [cat, reviewOnly]);

  return (
    <div className="page">
      <h1>通知复核</h1>
      <p className="page-sub">AI 抽取结果可在此人工修正，保存后系统会自动按新信息重算待办。</p>

      <div className="filters">
        <select className="input" value={cat} onChange={(e) => setCat(e.target.value)}>
          <option value="">全部类别</option>
          {CATEGORIES.map((c) => (
            <option key={c} value={c}>
              {categoryLabel(c)}
            </option>
          ))}
        </select>
        <label className="checkbox">
          <input
            type="checkbox"
            checked={reviewOnly}
            onChange={(e) => setReviewOnly(e.target.checked)}
          />
          仅看待复核
        </label>
        <button className="btn" onClick={load}>
          刷新
        </button>
      </div>

      {err && <div className="card error">{err}</div>}
      {!err && items.length === 0 && <div className="card muted">暂无通知。</div>}

      <div className="notice-list">
        {items.map((n) =>
          editingId === n.id ? (
            <EditCard
              key={n.id}
              notice={n}
              saving={saving}
              onCancel={() => setEditingId(null)}
              onSaved={async () => {
                setEditingId(null);
                setSaving(false);
                await load();
              }}
              setSaving={setSaving}
            />
          ) : (
            <div key={n.id} className={`card notice ${n.needs_review ? "review" : ""}`}>
              <div className="notice-head">
                <span
                  className="badge"
                  style={{ background: CATEGORY_COLORS[n.category] ?? "#6b7280" }}
                >
                  {categoryLabel(n.category)}
                </span>
                <span className="notice-title">{n.title}</span>
                {n.needs_review && <span className="flag">待复核</span>}
              </div>
              <div className="notice-meta">
                <span>📍 {n.location ?? "—"}</span>
                <span>🏛 {n.issuer ?? "—"}</span>
                <span>⏰ 截止 {formatDateTime(n.deadline)}</span>
                <span>📅 时间 {formatDateTime(n.event_time)}</span>
                <span style={{ color: confidenceColor(n.confidence) }}>
                  置信 {(n.confidence * 100).toFixed(0)}%
                </span>
              </div>
              {n.summary && <div className="summary">{n.summary}</div>}
              <div className="notice-foot">
                <span className="muted">#{n.id} · 关联待办 {n.tasks.length}</span>
                <button className="btn small" onClick={() => setEditingId(n.id)}>
                  人工修正
                </button>
              </div>
            </div>
          )
        )}
      </div>
    </div>
  );
}

interface EditProps {
  notice: NoticeOut;
  saving: boolean;
  setSaving: (b: boolean) => void;
  onCancel: () => void;
  onSaved: () => void;
}

function EditCard({ notice, saving, setSaving, onCancel, onSaved }: EditProps) {
  const [form, setForm] = useState({
    title: notice.title,
    category: notice.category,
    issuer: notice.issuer ?? "",
    location: notice.location ?? "",
    course: notice.course ?? "",
    summary: notice.summary ?? "",
    deadline: toLocalInputValue(notice.deadline),
    event_time: toLocalInputValue(notice.event_time),
    contacts: notice.contacts.join(", "),
    tags: notice.tags.join(", "),
  });
  const [err, setErr] = useState<string | null>(null);

  function set<K extends keyof typeof form>(k: K, v: string) {
    setForm((f) => ({ ...f, [k]: v }));
  }

  async function save() {
    setErr(null);
    setSaving(true);
    try {
      const patch = {
        title: form.title.trim() || undefined,
        category: form.category,
        issuer: form.issuer.trim() || undefined,
        location: form.location.trim() || undefined,
        course: form.course.trim() || undefined,
        summary: form.summary.trim() || undefined,
        deadline: form.deadline ? new Date(form.deadline).toISOString() : null,
        event_time: form.event_time ? new Date(form.event_time).toISOString() : null,
        contacts: form.contacts
          .split(",")
          .map((s) => s.trim())
          .filter(Boolean),
        tags: form.tags
          .split(",")
          .map((s) => s.trim())
          .filter(Boolean),
        reviewed: true,
      };
      await updateNotice(notice.id, patch);
      await onSaved();
    } catch (e) {
      setErr((e as Error).message);
      setSaving(false);
    }
  }

  return (
    <div className="card edit">
      <div className="card-title">修正 #{notice.id}</div>
      <div className="grid2">
        <div>
          <label className="field-label">标题</label>
          <input className="input" value={form.title} onChange={(e) => set("title", e.target.value)} />
        </div>
        <div>
          <label className="field-label">类别</label>
          <select
            className="input"
            value={form.category}
            onChange={(e) => set("category", e.target.value)}
          >
            {CATEGORIES.map((c) => (
              <option key={c} value={c}>
                {categoryLabel(c)}
              </option>
            ))}
          </select>
        </div>
        <div>
          <label className="field-label">发布方</label>
          <input className="input" value={form.issuer} onChange={(e) => set("issuer", e.target.value)} />
        </div>
        <div>
          <label className="field-label">地点</label>
          <input className="input" value={form.location} onChange={(e) => set("location", e.target.value)} />
        </div>
        <div>
          <label className="field-label">课程</label>
          <input className="input" value={form.course} onChange={(e) => set("course", e.target.value)} />
        </div>
        <div>
          <label className="field-label">截止时间</label>
          <input
            type="datetime-local"
            className="input"
            value={form.deadline}
            onChange={(e) => set("deadline", e.target.value)}
          />
        </div>
        <div>
          <label className="field-label">活动时间</label>
          <input
            type="datetime-local"
            className="input"
            value={form.event_time}
            onChange={(e) => set("event_time", e.target.value)}
          />
        </div>
        <div>
          <label className="field-label">联系人（逗号分隔）</label>
          <input
            className="input"
            value={form.contacts}
            onChange={(e) => set("contacts", e.target.value)}
          />
        </div>
      </div>
      <label className="field-label">摘要</label>
      <textarea
        className="textarea"
        rows={3}
        value={form.summary}
        onChange={(e) => set("summary", e.target.value)}
      />
      <label className="field-label">标签（逗号分隔）</label>
      <input className="input" value={form.tags} onChange={(e) => set("tags", e.target.value)} />

      {err && <div className="error">{err}</div>}
      <div className="actions">
        <button className="btn primary" disabled={saving} onClick={save}>
          {saving ? "保存中…" : "保存并重算待办"}
        </button>
        <button className="btn" disabled={saving} onClick={onCancel}>
          取消
        </button>
      </div>
    </div>
  );
}
