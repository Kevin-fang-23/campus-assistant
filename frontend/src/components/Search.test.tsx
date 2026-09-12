import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { askQA, search } from "../api";
import type { QAOut, SearchHit, SearchOut } from "../types";
import Search from "./Search";

// 组件只依赖 ../api 这一个外部模块，整体 mock 掉即可，无需请求层。
vi.mock("../api");

const mockedSearch = vi.mocked(search);
const mockedAskQA = vi.mocked(askQA);

function makeHit(over: Partial<SearchHit> = {}): SearchHit {
  return {
    notice_id: 1,
    score: 0.99,
    title: "关于开展校园创客季工作坊的通知",
    category: "activity_poster",
    summary: "为提升同学们的动手实践能力，校园创客社团举办本次工作坊。",
    deadline: null,
    ...over,
  };
}

function makeSearchOut(hits: SearchHit[], filtered?: boolean): SearchOut {
  return { query: "q", backend: "faiss+dashscope", hits, filtered };
}

/** 填入查询并点「检索」。 */
async function runSearch(user: ReturnType<typeof userEvent.setup>) {
  await user.type(screen.getByRole("textbox"), "工作坊报名什么时候截止");
  await user.click(screen.getByRole("button", { name: "检索" }));
}

describe("Search 组件", () => {
  beforeEach(() => {
    mockedSearch.mockReset();
    mockedAskQA.mockReset();
  });

  it("展示检索结果的标题与命中来源徽章", async () => {
    mockedSearch.mockResolvedValueOnce(
      makeSearchOut([makeHit({ match: "both", snippet: "报名截止时间：10 月 10 日 18:00。" })])
    );
    const user = userEvent.setup();
    render(<Search />);
    await runSearch(user);

    expect(await screen.findByText("关于开展校园创客季工作坊的通知")).toBeInTheDocument();
    expect(screen.getByText("混合")).toBeInTheDocument();

    // 三种命中来源都要有对应中文标签
    mockedSearch.mockResolvedValueOnce(
      makeSearchOut([
        makeHit({ notice_id: 2, title: "仅语义命中", match: "vector" }),
        makeHit({ notice_id: 3, title: "仅关键词命中", match: "bm25" }),
      ])
    );
    await user.click(screen.getByRole("button", { name: "检索" }));
    expect(await screen.findByText("语义")).toBeInTheDocument();
    expect(screen.getByText("关键词")).toBeInTheDocument();
  });

  it("snippet 优先展示在独立区块，summary 降级为带「摘要」标签的次要行", async () => {
    mockedSearch.mockResolvedValueOnce(
      makeSearchOut([
        makeHit({ snippet: "报名截止时间：10 月 10 日 18:00，逾期不再受理。" }),
      ])
    );
    const user = userEvent.setup();
    const { container } = render(<Search />);
    await runSearch(user);

    const snippetEl = await screen.findByText("报名截止时间：10 月 10 日 18:00，逾期不再受理。");
    expect(snippetEl.className).toContain("hit-snippet");
    // summary 与 snippet 不同，应同时展示，且带「摘要」前缀以示区分
    expect(screen.getByText("摘要")).toBeInTheDocument();
    expect(container.querySelector(".hit-summary")).not.toBeNull();
  });

  it("snippet 与 summary 相同时不重复渲染两次（短通知会整篇返回）", async () => {
    const same = "宿舍报修：3 号楼 412 室水管漏水。";
    mockedSearch.mockResolvedValueOnce(
      makeSearchOut([makeHit({ snippet: same, summary: same })])
    );
    const user = userEvent.setup();
    const { container } = render(<Search />);
    await runSearch(user);

    expect(await screen.findByText(same)).toBeInTheDocument();
    // 关键回归点：同一段文字只应出现一次，否则卡片会出现两条重复内容
    expect(screen.getAllByText(same)).toHaveLength(1);
    expect(container.querySelector(".hit-summary")).toBeNull();
  });

  it("缺失 snippet 时回退展示 summary（老数据/无原文的通知）", async () => {
    mockedSearch.mockResolvedValueOnce(
      makeSearchOut([makeHit({ snippet: null, summary: "这是一条没有原文的旧通知摘要。" })])
    );
    const user = userEvent.setup();
    const { container } = render(<Search />);
    await runSearch(user);

    const summaryEl = await screen.findByText("这是一条没有原文的旧通知摘要。");
    expect(summaryEl.className).toContain("hit-summary");
    expect(container.querySelector(".hit-snippet")).toBeNull();
  });

  it("检索接口报错时展示错误卡片，而不是静默空白", async () => {
    mockedSearch.mockRejectedValueOnce(new Error("500 Internal Server Error"));
    const user = userEvent.setup();
    render(<Search />);
    await runSearch(user);

    const err = await screen.findByText("500 Internal Server Error");
    expect(err.className).toContain("error");
  });

  it("切换模式会清掉上一次结果（避免把检索结果当问答结果展示）", async () => {
    mockedSearch.mockResolvedValueOnce(makeSearchOut([makeHit({ snippet: "片段内容" })]));
    const user = userEvent.setup();
    render(<Search />);
    await runSearch(user);
    expect(await screen.findByText("片段内容")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /智能问答/ }));
    expect(screen.queryByText("片段内容")).not.toBeInTheDocument();
  });

  // filtered 字段的唯一用途就是区分下面两种空态 —— 因此两条都要钉住。
  it("filtered=true 时提示「未找到相关内容」（有候选但都不相关）", async () => {
    mockedSearch.mockResolvedValueOnce(makeSearchOut([], true));
    const user = userEvent.setup();
    render(<Search />);
    await runSearch(user);

    expect(await screen.findByText(/未找到相关内容/)).toBeInTheDocument();
    expect(screen.queryByText("没有匹配的历史通知。")).not.toBeInTheDocument();
  });

  it("filtered=false 时提示「没有匹配的历史通知」（库里确实没有）", async () => {
    mockedSearch.mockResolvedValueOnce(makeSearchOut([], false));
    const user = userEvent.setup();
    render(<Search />);
    await runSearch(user);

    expect(await screen.findByText("没有匹配的历史通知。")).toBeInTheDocument();
    expect(screen.queryByText(/未找到相关内容/)).not.toBeInTheDocument();
  });

  it("filtered 缺失时按「库内没有」处理，不误报阈值过滤", async () => {
    // 后端字段是可选新增的：老版本接口 / 阈值关闭时都不会返回它。
    // 默认按 false 处理，避免把"属实没有"错说成"被阈值拦掉"。
    mockedSearch.mockResolvedValueOnce(makeSearchOut([], undefined));
    const user = userEvent.setup();
    render(<Search />);
    await runSearch(user);

    expect(await screen.findByText("没有匹配的历史通知。")).toBeInTheDocument();
  });
});

describe("Search 组件的问答模式", () => {
  beforeEach(() => {
    mockedSearch.mockReset();
    mockedAskQA.mockReset();
  });

  it("把回答里的 [1] 渲染成可高亮标签，与引用列表对照", async () => {
    const out: QAOut = {
      query: "工作坊报名什么时候截止",
      answer: "报名截止时间为 10 月 10 日 18:00 [1]。",
      backend: "faiss+dashscope",
      degraded: false,
      citations: [
        {
          notice_id: 10,
          title: "关于开展校园创客季工作坊的通知",
          snippet: "报名截止时间：10 月 10 日 18:00，逾期不再受理。",
          score: 0.99,
        },
      ],
    };
    mockedAskQA.mockResolvedValueOnce(out);

    const user = userEvent.setup();
    const { container } = render(<Search />);
    await user.click(screen.getByRole("button", { name: /智能问答/ }));
    await user.type(screen.getByRole("textbox"), "工作坊报名什么时候截止");
    await user.click(screen.getByRole("button", { name: "提问" }));

    // 答案中的 [1] 应被包进高亮标签
    const marks = await screen.findAllByText("[1]");
    expect(marks.length).toBeGreaterThanOrEqual(1);
    expect(container.querySelector(".cite-mark")).not.toBeNull();
    // 生成式回答展示绿色「AI 生成」徽章
    expect(screen.getByText("AI 生成")).toBeInTheDocument();
    // 引用片段可溯源
    expect(screen.getByText("报名截止时间：10 月 10 日 18:00，逾期不再受理。")).toBeInTheDocument();
  });

  it("降级回答展示「抽取式」标记与说明 banner", async () => {
    const degradedOut: QAOut = {
      query: "宿舍报修怎么申请",
      answer: "宿舍报修：3 号楼 412 室水管漏水。",
      backend: "numpy+local_hash",
      degraded: true,
      citations: [],
    };
    mockedAskQA.mockResolvedValueOnce(degradedOut);

    const user = userEvent.setup();
    const { container } = render(<Search />);
    await user.click(screen.getByRole("button", { name: /智能问答/ }));
    await user.type(screen.getByRole("textbox"), "宿舍报修怎么申请");
    await user.click(screen.getByRole("button", { name: "提问" }));

    expect(await screen.findByText("抽取式")).toBeInTheDocument();
    expect(container.querySelector(".banner")).not.toBeNull();
    expect(screen.queryByText("AI 生成")).not.toBeInTheDocument();
  });
});
