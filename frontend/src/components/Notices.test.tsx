import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { listNotices, updateNotice } from "../api";
import type { NoticeOut } from "../types";
import Notices from "./Notices";

vi.mock("../api");

const mockedListNotices = vi.mocked(listNotices);
const mockedUpdateNotice = vi.mocked(updateNotice);

function makeNotice(over: Partial<NoticeOut> = {}): NoticeOut {
  return {
    id: 1,
    document_id: 1,
    category: "course_notice",
    title: "《操作系统》调课通知",
    summary: "本周五课程调整至第三教学楼。",
    issuer: "教务处",
    location: "教三 A305",
    course: "操作系统",
    event_time: null,
    deadline: null,
    contacts: [],
    tags: [],
    extra: {},
    confidence: 0.92,
    needs_review: false,
    reviewed: false,
    duplicate_of_id: null,
    dedup_score: null,
    created_at: "2026-09-10T09:00:00",
    tasks: [],
    ...over,
  };
}

describe("Notices 通知复核", () => {
  beforeEach(() => {
    mockedListNotices.mockReset();
    mockedUpdateNotice.mockReset();
    mockedListNotices.mockResolvedValue([]);
  });

  it("渲染通知卡片：标题、类别中文标签、置信度百分比", async () => {
    mockedListNotices.mockResolvedValue([makeNotice()]);
    render(<Notices />);

    expect(await screen.findByText("《操作系统》调课通知")).toBeInTheDocument();
    // 「课程通知」既出现在徽章也出现在筛选下拉选项里，用 getAllByText 断言
    expect(screen.getAllByText("课程通知").length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText("置信 92%")).toBeInTheDocument();
  });

  it("空列表展示「暂无通知」占位", async () => {
    render(<Notices />);
    expect(await screen.findByText("暂无通知。")).toBeInTheDocument();
  });

  it("needs_review 的通知带「待复核」标记与高亮样式", async () => {
    mockedListNotices.mockResolvedValue([makeNotice({ needs_review: true })]);
    const { container } = render(<Notices />);
    await screen.findByText("《操作系统》调课通知");

    expect(screen.getByText("待复核")).toBeInTheDocument();
    expect(container.querySelector(".notice.review")).not.toBeNull();
  });

  it("类别筛选与「仅看待复核」会把参数传给接口", async () => {
    const user = userEvent.setup();
    render(<Notices />);
    await screen.findByText("暂无通知。");

    // 选类别 → 触发带 category 的请求
    await user.selectOptions(screen.getByDisplayValue("全部类别"), "homework");
    expect(mockedListNotices).toHaveBeenLastCalledWith(
      expect.objectContaining({ category: "homework" })
    );

    // 勾「仅看待复核」→ needs_review=true
    await user.click(screen.getByLabelText("仅看待复核"));
    expect(mockedListNotices).toHaveBeenLastCalledWith(
      expect.objectContaining({ needs_review: true })
    );
  });

  it("人工修正：保存时提交 reviewed=true，时间保持 naive（不含 Z）", async () => {
    mockedListNotices.mockResolvedValue([
      makeNotice({ id: 3, deadline: "2026-10-01T16:12:00" }),
    ]);
    mockedUpdateNotice.mockResolvedValue(makeNotice({ id: 3 }));
    const user = userEvent.setup();
    render(<Notices />);

    await user.click(await screen.findByRole("button", { name: "人工修正" }));
    expect(screen.getByText("修正 #3")).toBeInTheDocument();

    // 截止时间回显为 datetime-local 值（去掉秒）
    const deadlineInput = document.querySelector(
      '.edit input[type="datetime-local"]'
    ) as HTMLInputElement;
    expect(deadlineInput.value).toBe("2026-10-01T16:12");

    // 改标题后保存
    const titleInput = document.querySelector(".edit .input") as HTMLInputElement;
    fireEvent.change(titleInput, { target: { value: "修正后的标题" } });
    await user.click(screen.getByRole("button", { name: "保存并重算待办" }));

    expect(mockedUpdateNotice).toHaveBeenCalledTimes(1);
    const [id, patch] = mockedUpdateNotice.mock.calls[0];
    expect(id).toBe(3);
    expect(patch.title).toBe("修正后的标题");
    expect(patch.reviewed).toBe(true);
    // 时间原样回传、绝不带 Z（带 Z 曾导致保存稳定 500，见 common.tsx）
    expect(patch.deadline).toBe("2026-10-01T16:12:00");
  });

  it("清空可选字段提交 null，使后端真正清空该字段", async () => {
    // 回归：曾用 `|| undefined` 表示空值，JSON.stringify 会把 undefined 键
    // 整个丢掉，后端 exclude_unset 看不到 → 旧值保留，清空地点/发布方永远不生效。
    mockedListNotices.mockResolvedValue([makeNotice({ id: 5, issuer: "教务处" })]);
    mockedUpdateNotice.mockResolvedValue(makeNotice({ id: 5 }));
    const user = userEvent.setup();
    render(<Notices />);

    await user.click(await screen.findByRole("button", { name: "人工修正" }));
    await user.clear(screen.getByDisplayValue("教务处"));
    await user.click(screen.getByRole("button", { name: "保存并重算待办" }));

    const [, patch] = mockedUpdateNotice.mock.calls[0];
    expect(patch.issuer).toBeNull();
  });

  it("保存失败时展示错误并停留在编辑态", async () => {
    mockedListNotices.mockResolvedValue([makeNotice({ id: 4 })]);
    mockedUpdateNotice.mockRejectedValue(new Error("422 字段校验失败"));
    const user = userEvent.setup();
    render(<Notices />);

    await user.click(await screen.findByRole("button", { name: "人工修正" }));
    await user.click(screen.getByRole("button", { name: "保存并重算待办" }));

    expect(await screen.findByText("422 字段校验失败")).toBeInTheDocument();
    // 仍在编辑态（修正卡片还在）
    expect(screen.getByText("修正 #4")).toBeInTheDocument();
  });

  it("列表接口报错时展示错误卡片", async () => {
    mockedListNotices.mockRejectedValue(new Error("500 Internal Server Error"));
    render(<Notices />);
    expect(await screen.findByText("500 Internal Server Error")).toBeInTheDocument();
  });
});
