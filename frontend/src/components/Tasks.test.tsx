import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { createTask, deleteTask, listTasks, updateTask } from "../api";
import type { TaskOut } from "../types";
import Tasks from "./Tasks";

// 与 Search.test.tsx 同一策略：组件只依赖 ../api，整体 mock 掉。
vi.mock("../api");

const mockedListTasks = vi.mocked(listTasks);
const mockedUpdateTask = vi.mocked(updateTask);
const mockedCreateTask = vi.mocked(createTask);
const mockedDeleteTask = vi.mocked(deleteTask);

function makeTask(over: Partial<TaskOut> = {}): TaskOut {
  return {
    id: 1,
    notice_id: null,
    title: "完成操作系统预习报告",
    detail: null,
    category: "homework",
    due_at: null,
    remind_at: null,
    priority: 2,
    status: "todo",
    source: "auto",
    needs_review: false,
    completed_at: null,
    created_at: "2026-09-10T09:00:00",
    updated_at: "2026-09-10T09:00:00",
    ...over,
  };
}

describe("Tasks 看板", () => {
  beforeEach(() => {
    mockedListTasks.mockReset();
    mockedUpdateTask.mockReset();
    mockedCreateTask.mockReset();
    mockedDeleteTask.mockReset();
    mockedListTasks.mockResolvedValue([]);
  });

  it("按状态分列展示，列头计数与实际卡片一致", async () => {
    mockedListTasks.mockResolvedValue([
      makeTask({ id: 1, title: "待办甲", status: "todo" }),
      makeTask({ id: 2, title: "进行中乙", status: "doing" }),
      makeTask({ id: 3, title: "已完成丙", status: "done" }),
    ]);
    const { container } = render(<Tasks />);

    expect(await screen.findByText("待办甲")).toBeInTheDocument();
    expect(screen.getByText("进行中乙")).toBeInTheDocument();
    expect(screen.getByText("已完成丙")).toBeInTheDocument();
    // 四列（含空列「已归档」）都在；列头计数 = 1/1/1/0。
    // 注意「待办/已归档」同时出现在卡片 <select> 的选项里，必须按列头容器断言。
    const heads = container.querySelectorAll(".column-head");
    expect(heads).toHaveLength(4);
    const counts = [...container.querySelectorAll(".column-head .count")].map(
      (e) => e.textContent
    );
    expect(counts).toEqual(["1", "1", "1", "0"]);
  });

  it("已过期的未完成任务带 overdue 标记，已完成的不标红", async () => {
    const past = new Date(Date.now() - 3_600_000).toISOString();
    mockedListTasks.mockResolvedValue([
      makeTask({ id: 1, title: "逾期未交", status: "todo", due_at: past }),
      makeTask({ id: 2, title: "早已完成", status: "done", due_at: past }),
    ]);
    const { container } = render(<Tasks />);
    await screen.findByText("逾期未交");

    const cards = container.querySelectorAll(".tcard");
    expect(cards).toHaveLength(2);
    // isOverdue 的状态豁免逻辑在界面上的体现：done 不标红
    expect(cards[0].className).toContain("overdue");
    expect(cards[1].className).not.toContain("overdue");
  });

  it("切换状态调用 updateTask 并刷新列表", async () => {
    mockedListTasks.mockResolvedValue([makeTask({ id: 7, title: "要流转的任务" })]);
    mockedUpdateTask.mockResolvedValue({ ...makeTask({ id: 7 }), events: [] });
    const user = userEvent.setup();
    render(<Tasks />);

    const select = (await screen.findByDisplayValue("待办")) as HTMLSelectElement;
    await user.selectOptions(select, "doing");

    expect(mockedUpdateTask).toHaveBeenCalledWith(7, { status: "doing" });
    // 流转后重新拉取列表（初始 1 次 + 刷新 1 次）
    expect(mockedListTasks).toHaveBeenCalledTimes(2);
  });

  it("删除需二次确认；取消则不发请求", async () => {
    // jsdom 默认不实现 window.confirm（undefined），先打桩再交互
    const confirmSpy = vi.fn(() => false);
    window.confirm = confirmSpy;
    mockedListTasks.mockResolvedValue([makeTask({ id: 9, title: "待删除" })]);
    const user = userEvent.setup();
    render(<Tasks />);

    await user.click(await screen.findByRole("button", { name: "删" }));
    expect(confirmSpy).toHaveBeenCalled();
    expect(mockedDeleteTask).not.toHaveBeenCalled();

    // 确认后真正删除并刷新
    confirmSpy.mockReturnValue(true);
    mockedDeleteTask.mockResolvedValue(undefined);
    await user.click(screen.getByRole("button", { name: "删" }));
    expect(mockedDeleteTask).toHaveBeenCalledWith(9);
  });

  it("新建待办：截止时间以 naive 本地时间提交（不含 Z，回归 500 缺陷）", async () => {
    mockedCreateTask.mockResolvedValue(makeTask({ id: 100 }));
    const user = userEvent.setup();
    render(<Tasks />);

    await user.click(screen.getByRole("button", { name: "+ 新建待办" }));
    expect(screen.getByText("新建待办")).toBeInTheDocument();

    // 标题是第一个 .input（模态框内）
    const inputs = document.querySelectorAll(".modal .input");
    fireEvent.change(inputs[0], { target: { value: "复习线性代数" } });
    // 「截止」是 datetime-local 输入
    const dueInput = document.querySelector('.modal input[type="datetime-local"]');
    expect(dueInput).not.toBeNull();
    fireEvent.change(dueInput!, { target: { value: "2026-10-01T16:12" } });

    await user.click(screen.getByRole("button", { name: "保存" }));

    expect(mockedCreateTask).toHaveBeenCalledTimes(1);
    const payload = mockedCreateTask.mock.calls[0][0];
    expect(payload.title).toBe("复习线性代数");
    // 关键契约：toApiDateTime 补秒且绝不带 Z（带 Z 会让后端 500，见 common.tsx 注释）
    expect(payload.due_at).toBe("2026-10-01T16:12:00");
  });

  it("编辑待办走 updateTask，且模态框标题为「编辑待办」", async () => {
    mockedListTasks.mockResolvedValue([makeTask({ id: 5, title: "原标题", detail: "原详情" })]);
    mockedUpdateTask.mockResolvedValue({ ...makeTask({ id: 5 }), events: [] });
    const user = userEvent.setup();
    render(<Tasks />);

    await user.click(await screen.findByRole("button", { name: "编辑" }));
    expect(screen.getByText("编辑待办")).toBeInTheDocument();

    const titleInput = document.querySelector(".modal .input") as HTMLInputElement;
    expect(titleInput.value).toBe("原标题");
    fireEvent.change(titleInput, { target: { value: "改后的标题" } });
    await user.click(screen.getByRole("button", { name: "保存" }));

    expect(mockedUpdateTask).toHaveBeenCalledWith(5, expect.objectContaining({ title: "改后的标题" }));
  });

  it("列表接口报错时展示错误卡片而不是空白", async () => {
    mockedListTasks.mockRejectedValue(new Error("500 Internal Server Error"));
    render(<Tasks />);
    expect(await screen.findByText("500 Internal Server Error")).toBeInTheDocument();
  });
});
