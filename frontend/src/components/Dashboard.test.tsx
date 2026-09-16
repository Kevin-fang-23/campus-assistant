import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { stats } from "../api";
import type { StatsOut } from "../types";
import Dashboard from "./Dashboard";

vi.mock("../api");

const mockedStats = vi.mocked(stats);

function makeStats(over: Partial<StatsOut> = {}): StatsOut {
  return {
    documents: 3,
    notices: 5,
    tasks_total: 4,
    tasks_todo: 2,
    tasks_doing: 1,
    tasks_done: 1,
    tasks_overdue: 1,
    needs_review: 2,
    by_category: { 课程通知: 3, 作业要求: 2 },
    ...over,
  };
}

describe("Dashboard 概览", () => {
  beforeEach(() => {
    mockedStats.mockReset();
  });

  it("加载中先展示占位，数据到达后渲染统计卡片", async () => {
    mockedStats.mockResolvedValue(makeStats());
    render(<Dashboard onNavigate={() => {}} />);

    expect(screen.getByText("加载中…")).toBeInTheDocument();
    expect(await screen.findByText("概览")).toBeInTheDocument();
    // 关键数字可见
    expect(screen.getByText("5")).toBeInTheDocument(); // 通知
    expect(screen.getByText("待复核")).toBeInTheDocument();
  });

  it("接口报错时展示错误卡片而不是空白", async () => {
    mockedStats.mockRejectedValue(new Error("500 Internal Server Error"));
    render(<Dashboard onNavigate={() => {}} />);

    expect(await screen.findByText(/加载统计失败/)).toBeInTheDocument();
  });

  it("类别分布按条目渲染条形图；无数据时引导去导入", async () => {
    mockedStats.mockResolvedValue(makeStats());
    render(<Dashboard onNavigate={() => {}} />);

    expect(await screen.findByText("课程通知")).toBeInTheDocument();
    expect(screen.getByText("作业要求")).toBeInTheDocument();

    // 空分布的另一分支
    mockedStats.mockResolvedValue(makeStats({ by_category: {} }));
    render(<Dashboard onNavigate={() => {}} />);
    expect(await screen.findAllByText(/暂无数据/)).not.toHaveLength(0);
  });

  it("可点击卡片触发页面跳转（待办/逾期 → tasks，待复核 → notices）", async () => {
    mockedStats.mockResolvedValue(makeStats());
    const onNavigate = vi.fn();
    const user = userEvent.setup();
    render(<Dashboard onNavigate={onNavigate} />);

    await user.click(await screen.findByText("待办(待处理)"));
    expect(onNavigate).toHaveBeenCalledWith("tasks");

    await user.click(screen.getByText("已逾期"));
    expect(onNavigate).toHaveBeenCalledTimes(2);

    await user.click(screen.getByText("待复核"));
    expect(onNavigate).toHaveBeenLastCalledWith("notices");
  });
});
