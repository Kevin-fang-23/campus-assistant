import { useEffect, useMemo, useState } from "react";
import { createTask, deleteTask, listTasks, updateTask } from "../api";
import type { TaskCreateIn, TaskOut, TaskUpdateIn } from "../types";

type TaskForm = TaskCreateIn & { id?: number };
import {
  CATEGORIES,
  CATEGORY_COLORS,
  categoryLabel,
  formatDateTime,
  isOverdue,
  PRIORITY_COLORS,
  PRIORITY_LABELS,
  STATUS_COLORS,
  STATUS_LABELS,
  STATUS_ORDER,
  toApiDateTime,
  toLocalInputValue,
} from "../common";

export default function Tasks() {
  const [tasks, setTasks] = useState<TaskOut[]>([]);
  const [err, setErr] = useState<string | null>(null);
  const [modal, setModal] = useState<null | { id: number | null; task?: TaskOut }>(null);

  async function load() {
    setErr(null);
    try {
      setTasks(await listTasks());
    } catch (e) {
      setErr((e as Error).message);
    }
  }
  useEffect(() => {
    load();
  }, []);

  const grouped = useMemo(() => {
    const g: Record<string, TaskOut[]> = { todo: [], doing: [], done: [], archived: [] };
    for (const t of tasks) (g[t.status] ??= []).push(t);
    return g;
  }, [tasks]);

  async function changeStatus(t: TaskOut, status: string) {
    try {
      await updateTask(t.id, { status: status as TaskUpdateIn["status"] });
      await load();
    } catch (e) {
      setErr((e as Error).message);
    }
  }

  async function remove(t: TaskOut) {
    if (!confirm(`确认删除待办「${t.title}」？`)) return;
    try {
      await deleteTask(t.id);
      await load();
    } catch (e) {
      setErr((e as Error).message);
    }
  }

  return (
    <div className="page">
      <div className="page-head">
        <div>
          <h1>待办看板</h1>
          <p className="page-sub">按状态流转，逾期自动标红。</p>
        </div>
        <button className="btn primary" onClick={() => setModal({ id: null })}>
          + 新建待办
        </button>
      </div>

      {err && <div className="card error">{err}</div>}

      <div className="board">
        {STATUS_ORDER.map((st) => (
          <div className="column" key={st}>
            <div className="column-head" style={{ borderTopColor: STATUS_COLORS[st] }}>
              <span>{STATUS_LABELS[st]}</span>
              <span className="count">{grouped[st]?.length ?? 0}</span>
            </div>
            <div className="column-body">
              {(grouped[st] ?? []).map((t) => (
                <div key={t.id} className={`tcard ${isOverdue(t.due_at, t.status) ? "overdue" : ""}`}>
                  <div className="tcard-top">
                    <span
                      className="badge sm"
                      style={{ background: CATEGORY_COLORS[t.category] ?? "#6b7280" }}
                    >
                      {categoryLabel(t.category)}
                    </span>
                    <span className="prio" style={{ color: PRIORITY_COLORS[t.priority] }}>
                      {PRIORITY_LABELS[t.priority]}优先
                    </span>
                  </div>
                  <div className="tcard-title">{t.title}</div>
                  {t.detail && <div className="tcard-detail">{t.detail}</div>}
                  <div className="tcard-meta">
                    <span>🕒 {formatDateTime(t.due_at)}</span>
                    {t.needs_review && <span className="flag sm">待复核</span>}
                  </div>
                  <div className="tcard-actions">
                    <select
                      className="input sm"
                      value={t.status}
                      onChange={(e) => changeStatus(t, e.target.value)}
                    >
                      {STATUS_ORDER.map((s) => (
                        <option key={s} value={s}>
                          {STATUS_LABELS[s]}
                        </option>
                      ))}
                    </select>
                    <button className="btn small" onClick={() => setModal({ id: t.id, task: t })}>
                      编辑
                    </button>
                    <button className="btn small danger" onClick={() => remove(t)}>
                      删
                    </button>
                  </div>
                </div>
              ))}
              {(grouped[st] ?? []).length === 0 && <div className="muted empty">—</div>}
            </div>
          </div>
        ))}
      </div>

      {modal && (
        <TaskModal
          initial={modal.task}
          onClose={() => setModal(null)}
          onSaved={async () => {
            setModal(null);
            await load();
          }}
        />
      )}
    </div>
  );
}

function TaskModal({
  initial,
  onClose,
  onSaved,
}: {
  initial?: TaskOut;
  onClose: () => void;
  onSaved: () => void;
}) {
  const [form, setForm] = useState<TaskForm>({
    id: initial?.id,
    title: initial?.title ?? "",
    detail: initial?.detail ?? "",
    category: initial?.category ?? "other",
    due_at: initial?.due_at ? toLocalInputValue(initial.due_at) : "",
    remind_at: initial?.remind_at ? toLocalInputValue(initial.remind_at) : "",
    priority: initial?.priority ?? 2,
    notice_id: initial?.notice_id ?? null,
  });
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  function set<K extends keyof TaskForm>(k: K, v: TaskForm[K]) {
    setForm((f: TaskForm) => ({ ...f, [k]: v }));
  }

  async function submit() {
    setErr(null);
    setBusy(true);
    const payload = {
      title: form.title.trim(),
      detail: form.detail?.trim() || null,
      category: form.category,
        due_at: toApiDateTime(form.due_at),
        remind_at: toApiDateTime(form.remind_at),
      priority: Number(form.priority),
      notice_id: form.notice_id ?? null,
    };
    try {
      if (initial) {
        const patch: TaskUpdateIn = {
          title: payload.title,
          detail: payload.detail,
          due_at: payload.due_at,
          remind_at: payload.remind_at,
          priority: payload.priority,
        };
        await updateTask(initial.id, patch);
      } else {
        await createTask(payload);
      }
      onSaved();
    } catch (e) {
      setErr((e as Error).message);
      setBusy(false);
    }
  }

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="card-title">{initial ? "编辑待办" : "新建待办"}</div>
        <label className="field-label">标题 *</label>
        <input className="input" value={form.title} onChange={(e) => set("title", e.target.value)} />
        <label className="field-label">详情</label>
        <textarea
          className="textarea"
          rows={2}
          value={form.detail ?? ""}
          onChange={(e) => set("detail", e.target.value)}
        />
        <div className="grid2">
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
            <label className="field-label">优先级</label>
            <select
              className="input"
              value={form.priority}
              onChange={(e) => set("priority", Number(e.target.value))}
            >
              <option value={1}>高</option>
              <option value={2}>中</option>
              <option value={3}>低</option>
            </select>
          </div>
          <div>
            <label className="field-label">截止</label>
            <input
              type="datetime-local"
              className="input"
              value={form.due_at ?? ""}
              onChange={(e) => set("due_at", e.target.value)}
            />
          </div>
          <div>
            <label className="field-label">提醒时间</label>
            <input
              type="datetime-local"
              className="input"
              value={form.remind_at ?? ""}
              onChange={(e) => set("remind_at", e.target.value)}
            />
          </div>
        </div>
        {err && <div className="error">{err}</div>}
        <div className="actions">
          <button className="btn primary" disabled={busy} onClick={submit}>
            {busy ? "保存中…" : "保存"}
          </button>
          <button className="btn" disabled={busy} onClick={onClose}>
            取消
          </button>
        </div>
      </div>
    </div>
  );
}
