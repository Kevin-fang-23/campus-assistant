import { describe, expect, it } from "vitest";
import {
  CATEGORIES,
  CATEGORY_LABELS,
  categoryLabel,
  confidenceColor,
  formatDateTime,
  isOverdue,
  STATUS_LABELS,
  statusLabel,
  toLocalInputValue,
} from "./common";

describe("formatDateTime", () => {
  it("空值统一显示为破折号", () => {
    expect(formatDateTime(null)).toBe("—");
    expect(formatDateTime(undefined)).toBe("—");
    expect(formatDateTime("")).toBe("—");
  });

  it("无法解析的字符串原样返回（不显示 Invalid Date）", () => {
    expect(formatDateTime("不是时间")).toBe("不是时间");
  });

  it("按 月/日/时/分 补零", () => {
    // 用本地时间构造，避免时区影响
    expect(formatDateTime("2026-09-08T09:05:00")).toBe("2026-09-08 09:05");
    expect(formatDateTime("2026-12-31T23:59:00")).toBe("2026-12-31 23:59");
  });
});

describe("toLocalInputValue", () => {
  it("空值与非法值都返回空串（datetime-local 不接受非法值）", () => {
    expect(toLocalInputValue(null)).toBe("");
    expect(toLocalInputValue("乱写")).toBe("");
  });

  it("输出 datetime-local 需要的 yyyy-MM-ddTHH:mm", () => {
    expect(toLocalInputValue("2026-09-08T09:05:00")).toBe("2026-09-08T09:05");
  });
});

describe("isOverdue", () => {
  const past = new Date(Date.now() - 60_000).toISOString();
  const future = new Date(Date.now() + 3_600_000).toISOString();

  it("无截止时间不算逾期", () => {
    expect(isOverdue(null)).toBe(false);
    expect(isOverdue(undefined)).toBe(false);
  });

  it("已完成/已归档的任务不算逾期（即使时间已过）", () => {
    expect(isOverdue(past, "done")).toBe(false);
    expect(isOverdue(past, "archived")).toBe(false);
    // 进行中的仍应判为逾期
    expect(isOverdue(past, "doing")).toBe(true);
  });

  it("按当前时间比较", () => {
    expect(isOverdue(past)).toBe(true);
    expect(isOverdue(future)).toBe(false);
  });

  it("非法时间不算逾期（否则会把脏数据显示成红色）", () => {
    expect(isOverdue("not-a-date")).toBe(false);
  });
});

describe("分类 / 状态标签", () => {
  it("已知值映射为中文", () => {
    expect(categoryLabel("course_notice")).toBe("课程通知");
    expect(statusLabel("doing")).toBe("进行中");
  });

  it("未知值原样返回，而不是显示 undefined", () => {
    expect(categoryLabel("unknown_cat")).toBe("unknown_cat");
    expect(statusLabel("unknown_status")).toBe("unknown_status");
  });

  it("每个可选分类都有对应标签（漏配会让界面出现英文枚举）", () => {
    for (const c of CATEGORIES) {
      expect(CATEGORY_LABELS[c]).toBeTruthy();
    }
    for (const s of Object.keys(STATUS_LABELS)) {
      expect(STATUS_LABELS[s as keyof typeof STATUS_LABELS]).toBeTruthy();
    }
  });
});

describe("confidenceColor", () => {
  it("按 0.8 / 0.55 两个阈值分档，边界值归入高/中档", () => {
    expect(confidenceColor(0.8)).toBe("#16a34a");
    expect(confidenceColor(0.95)).toBe("#16a34a");
    expect(confidenceColor(0.55)).toBe("#d97706");
    expect(confidenceColor(0.79)).toBe("#d97706");
    expect(confidenceColor(0.54)).toBe("#dc2626");
    expect(confidenceColor(0)).toBe("#dc2626");
  });
});
