# S6 + 收尾（issuer 补词）基线（规则抽取层，mock 后端）

> 锁定时间：2026-09-08（S6 RC1–RC5 修复 + issuer 边界/信任分级修复落地后）
> 对应代码 commit：本仓库当前未提交修改（基线即 working tree），规则层 `app/services/rule_extract.py` 与 `app/services/datetime_utils.py`
> 配套基线：`BASELINE_BEFORE.md`（before）、`BASELINE_MOCK.md`（S4 锁定的快照）、本文件（after）

## 1. 评测口径
- 数据集：40 条 `eval/cases.jsonl`（homework=9 / course_notice=9 / activity_poster=11 / repair=6 / other=5）
- 配置：`VLM_PROVIDER=mock EMBEDDING_PROVIDER=local_hash`，即「纯规则路径」
- 指标：`classification_acc` + 6 字段 P/R/F1 + 微平均
- 确定性：连续两次运行 headline 数字完全一致（100% / 微平均 0.98 / 0.98 / 0.98，TP=116 / FP=2 / FN=2）

## 2. S6 + issuer 收尾后的指标对比

| 维度           | S4 before（mock） | S6 after（mock） | issuer 收尾后  | Δ（vs before） |
|----------------|-------------------|------------------|----------------|----------------|
| 分类准确率     | 92.5% (37/40)     | 100% (40/40)     | **100% (40/40)** | +7.5pt         |
| micro-P        | 0.78              | 0.98             | **0.98**         | +0.20          |
| micro-R        | 0.82              | 0.97             | **0.98**         | +0.16          |
| micro-F1       | 0.80              | 0.97             | **0.98**         | **+0.18**      |
| category F1    | 1.00              | 1.00             | 1.00             | 持平           |
| location F1    | 0.66              | 1.00             | **1.00**         | +0.34          |
| course F1      | 0.90              | 1.00             | **1.00**         | +0.10          |
| event_time F1  | 0.78              | 0.92             | **0.92**         | +0.14          |
| deadline F1    | 0.94              | 0.97             | **0.97**         | +0.03          |
| issuer F1      | 0.57              | 0.83             | **1.00**         | **+0.43**      |
| 残余 miss 数   | 37                | 5                | **3**            | -34            |

> VLM 基线仍为 `BASELINE_VLM.md` 中的 100% classification / 0.89 micro-F1；S6 主要服务「VLM 失败或被禁用时的兜底层」。

## 3. 5 类根因 → 修复要点对照表

| 编号 | 根因摘要                              | 修复点（code 位置）                                          | 修复策略                                                          | 对应字段   |
|------|---------------------------------------|-------------------------------------------------------------|-------------------------------------------------------------------|------------|
| RC1  | 地点字段吞掉标题/书名号/单位/数字残渣 | `rule_extract.py` `RE_LOC_LABEL`/`RE_LOC_GUESS`/`RE_ROOM`/`RE_LOC_NOISE`/`RE_LOC_STOPWORDS` | 收紧引导词集合 + 新增独立「房间号」回退 + 末位 stopword 拒绝 | location   |
| RC2  | 「今晚/明晚/…前」相对日期 + 截止语义  | `datetime_utils.py` `rel_map`/`rel_hour`/`RE_RELDAY`/`RE_DEADLINE_FRONT`/`pick_times` | 新增 `今晚/明晚` 映射；将 `^\s*前` 改为 `[日号天…]\s*前` + `.search` | event_time, deadline |
| RC3  | 课程字段吞「《》」包裹的活动标题       | `rule_extract.py` `RE_COURSE_GUESS`/`RE_COURSE_BOOK` | 课程前缀放宽 0–14；活动类跳过 `RE_COURSE_BOOK`               | course      |
| RC4  | 否定语境误识别「关空调→报修」等       | `rule_extract.py` `_keyword_present`/`_NOT_NEGATION`/`_overlaps` | 否定窗口 6→10 字符；先剥离含「关」但非否定的词；issuer==location 自动清 issuer | category, issuer |
| RC5  | 「原定/推迟/本周X」错配行为时间       | `datetime_utils.py` `pick_times` 重写 | event 只在显式事件命中处拾取；deadline 优先 deadline-kind → rescheduled → max；homework/repair 取 max 作截止 | event_time, deadline |
| **+issuer 收尾** | 单行海报地点吞掉「主办:X」；显式 issuer 与重叠地点被无差别误杀 | `rule_extract.py` `_FIELD_BOUND` 前瞻 + `issuer_labeled` 信任分级 | 懒匹配 + 字段边界前瞻；显式 `主办：X` 不再因与地点重叠被清空 | location, issuer |

## 4. 残余 3 条 miss 的根因（已知局限）

| ID     | 字段      | 期望                                  | 实际                                | 根因                                                              | 建议处理                  |
|--------|-----------|---------------------------------------|-------------------------------------|-------------------------------------------------------------------|---------------------------|
| act_02 | deadline  | 2027-01-05T23:59（跨年）              | None                                | 跨年语义需结合年份推断，规则层未做                               | 交给 VLM（dashscope 已能正确解析） |
| act_10 | event_time| 2026-10-15T07:30                      | 2026-10-15T19:30                    | 「7:30」被规则映射到 19:30（晨 7 误为晚 7）                     | 引入「上午/下午/早上/晚上」显式修饰 |
| rp_03  | event_time| None                                  | 2026-08-25T14:00                    | 「下午」被额外解释成事件时间字段（应归属 deadline 解释成「尽早」） | 修复优先级或加 allow_empty 上下文词 |

> 相比 S6 收尾的 5 条 miss，act_03/act_08 的 issuer 问题已通过「字段边界前瞻 + 显式标签信任分级」消解（issuer F1 0.83→1.00）。剩余 3 条全部属于「规则层边界语义」缺陷；dashScope VLM 路径目前 100% 分类、仅在时间字段上有 24 个 over-extraction miss（详 `BASELINE_VLM.md`）。整体策略：**时间语义走 VLM、字段过抽由规则收口**。

## 5. 测试回归
- `pytest -q` → **49 passed, 1 warning**（StarletteDeprecationWarning，与本次无关）
- 关键回归用例覆盖：base_time 透传、DeadlineFront、`_NOT_NEGATION` 剥离、issuer==location 清理、stopword 拒绝、跨类去歧义、QA 降级路径、QA LLM 路径（MockTransport 注入）、QA 入参校验

## 6. 结论
- S6 + issuer 收尾已完成：规则层分类从 92.5% → 100%、micro-F1 从 0.80 → 0.98，issuer F1 从 0.57 → 1.00，残余 3 条全部属于「非 RC1–RC5 主线根因」的边缘语义。
- 整套修复在 **deterministic mock** 路径上完全可复现。
- 与 S4 锁定的 `BASELINE_MOCK.md` 相比，所有受 RC1–RC5 影响的字段（location/course/event_time/deadline/issuer）均显著上升。
- S8 已落地：`/api/qa` 真实接入 qwen-plus，复用 `OpenAICompatClient` 零新封装；接口契约、降级行为、错误处理均与设计一致。

---  
_Last refreshed: 2026-09-08 · generated by S6 closeout_
