import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ingestFile, ingestText } from "../api";
import type { IngestResult } from "../types";
import Upload from "./Upload";

vi.mock("../api");

const mockedIngestText = vi.mocked(ingestText);
const mockedIngestFile = vi.mocked(ingestFile);

function makeResult(over: Partial<IngestResult> = {}): IngestResult {
  return {
    document: {
      id: 1,
      filename: "调课通知.txt",
      kind: "text",
      mime_type: "text/plain",
      sha256: "abcdef1234567890",
      size_bytes: 128,
      status: "parsed",
      raw_text: null,
      parse_meta: {},
      error: null,
      created_at: "2026-09-10T09:00:00",
    },
    notice: {
      id: 1,
      document_id: 1,
      category: "course_notice",
      title: "《数据结构》调课通知",
      summary: "课程调整至周五下午。",
      issuer: "教务处",
      location: "教三 A305",
      course: "数据结构",
      event_time: null,
      deadline: null,
      contacts: [],
      tags: [],
      extra: {},
      confidence: 0.9,
      needs_review: false,
      reviewed: false,
      duplicate_of_id: null,
      dedup_score: null,
      created_at: "2026-09-10T09:00:00",
      tasks: [],
    },
    tasks: [
      {
        id: 1,
        notice_id: 1,
        title: "周五按时上课",
        detail: null,
        category: "course_notice",
        due_at: "2026-09-12T14:30:00",
        remind_at: null,
        priority: 2,
        status: "todo",
        source: "auto",
        needs_review: false,
        completed_at: null,
        created_at: "2026-09-10T09:00:00",
        updated_at: "2026-09-10T09:00:00",
      },
    ],
    duplicate: false,
    duplicate_of_id: null,
    dedup_score: null,
    needs_review: false,
    trace: [],
    warnings: [],
    ...over,
  };
}

describe("Upload 导入", () => {
  beforeEach(() => {
    mockedIngestText.mockReset();
    mockedIngestFile.mockReset();
  });

  it("文本为空时点提交给出提示，不发请求", async () => {
    const user = userEvent.setup();
    render(<Upload />);

    await user.click(screen.getByRole("button", { name: "粘贴文本" }));
    await user.click(screen.getByRole("button", { name: "提交并解析" }));

    expect(await screen.findByText(/请输入文本内容/)).toBeInTheDocument();
    expect(mockedIngestText).not.toHaveBeenCalled();
  });

  it("粘贴文本提交成功后展示文档、通知与已生成待办", async () => {
    mockedIngestText.mockResolvedValue(makeResult());
    const user = userEvent.setup();
    render(<Upload />);

    await user.click(screen.getByRole("button", { name: "粘贴文本" }));
    await user.type(
      screen.getByPlaceholderText(/把通知原文粘贴到这里/),
      "《数据结构》课程调整至周五下午2:30。"
    );
    await user.click(screen.getByRole("button", { name: "提交并解析" }));

    // 文档信息
    expect(await screen.findByText(/文档：调课通知.txt/)).toBeInTheDocument();
    // 抽取出的通知
    expect(screen.getByText("《数据结构》调课通知")).toBeInTheDocument();
    expect(screen.getByText("教务处")).toBeInTheDocument();
    // 自动生成的待办
    expect(screen.getByText(/已生成待办（1）/)).toBeInTheDocument();
    expect(screen.getByText("周五按时上课")).toBeInTheDocument();

    expect(mockedIngestText).toHaveBeenCalledWith(
      expect.objectContaining({ content: expect.stringContaining("数据结构") })
    );
  });

  it("检测到重复通知时展示警示横幅", async () => {
    mockedIngestText.mockResolvedValue(
      makeResult({ duplicate: true, duplicate_of_id: 7, dedup_score: 0.97 })
    );
    const user = userEvent.setup();
    render(<Upload />);

    await user.click(screen.getByRole("button", { name: "粘贴文本" }));
    await user.type(screen.getByPlaceholderText(/把通知原文粘贴到这里/), "重复的通知内容");
    await user.click(screen.getByRole("button", { name: "提交并解析" }));

    expect(await screen.findByText(/检测到重复通知/)).toBeInTheDocument();
    expect(screen.getByText(/与 #7 相似度 0.97/)).toBeInTheDocument();
  });

  it("置信度偏低时提示去复核", async () => {
    mockedIngestText.mockResolvedValue(makeResult({ needs_review: true }));
    const user = userEvent.setup();
    render(<Upload />);

    await user.click(screen.getByRole("button", { name: "粘贴文本" }));
    await user.type(screen.getByPlaceholderText(/把通知原文粘贴到这里/), "模糊的通知");
    await user.click(screen.getByRole("button", { name: "提交并解析" }));

    expect(await screen.findByText(/抽取置信度偏低/)).toBeInTheDocument();
  });

  it("解析接口报错时展示错误卡片", async () => {
    mockedIngestText.mockRejectedValue(new Error("文件超过 20MB 限制"));
    const user = userEvent.setup();
    render(<Upload />);

    await user.click(screen.getByRole("button", { name: "粘贴文本" }));
    await user.type(screen.getByPlaceholderText(/把通知原文粘贴到这里/), "任意内容");
    await user.click(screen.getByRole("button", { name: "提交并解析" }));

    expect(await screen.findByText(/文件超过 20MB 限制/)).toBeInTheDocument();
  });

  it("文件上传：选择文件后调用 ingestFile 并展示结果", async () => {
    mockedIngestFile.mockResolvedValue(makeResult());
    render(<Upload />);

    // 隐藏的 file input：直接触发 change（点击会打开系统对话框，测试里不可用）
    const file = new File(["通知内容"], "调课通知.txt", { type: "text/plain" });
    const input = document.querySelector('input[type="file"]') as HTMLInputElement;
    fireEvent.change(input, { target: { files: [file] } });

    expect(await screen.findByText(/文档：调课通知.txt/)).toBeInTheDocument();
    expect(mockedIngestFile).toHaveBeenCalledWith(file);
  });
});
