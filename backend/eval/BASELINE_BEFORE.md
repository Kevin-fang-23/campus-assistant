# S4 · before 基线快照（改动前的正式参照）

锁定时间：2026-09-08 17:45
用途：S6 逐类修复后，所有对比都以本快照为准。

---

## 1. 基线标识

| 项 | 值 |
|---|---|
| git commit | `74e1232` feat: 校园事务智能助手 MVP |
| 工作区状态 | 14 处未提交改动（含本次 API 接入与评测集扩充） |
| 评测集 `cases.jsonl` | SHA256 `5babf64dc683…`（40 条，锁定后不再增删） |
| 评测脚本 `run_eval.py` | SHA256 `48ecff82415d…` |

S6 将改动的文件（改动前指纹，用于确认"到底改了谁"）：

| 文件 | SHA256 前 12 位 |
|---|---|
| `backend/app/services/rule_extract.py` | `9a620dac21e4` |
| `backend/app/services/datetime_utils.py` | `3fa473aa0b27` |
| `backend/app/providers/vlm.py` | `b76edec23d2b` |
| `backend/app/config.py` | `9f0624bce81f` |

---

## 2. 覆盖范围

- 样本：**40 条**，全部 `input_type=text`
- 类别：activity_poster 11 / course_notice 9 / homework 9 / repair 6 / other 5
- 难度：easy 14 / medium 14 / hard 12
- 场景标签：18 类（典型 18 / 易错 9 / 边界 2）
- **未覆盖**：图片与 PDF 输入（S5 未执行）、去重专项（无 `dedup_pairs`）

---

## 3. 判定标准（必须一致，否则数字不可比）

| 维度 | 规则 |
|---|---|
| 参与评分字段 | `category` `course` `location` `issuer` `event_time` `deadline`（6 项） |
| 不参与评分 | `title` `summary`（生成式，难以客观判定） |
| 时间字段 | 容差 **±60 秒** |
| 文本字段 | 归一化（去空白标点、全角转半角、小写）后**双向包含匹配** |
| 分类字段 | 精确匹配 |
| FP/FN 判定 | 期望非空+实际非空且不匹配 → 同时计 FP 与 FN；期望空+实际非空 → FP；期望非空+实际空 → FN |

---

## 4. 两条基线（用途不同，都必须保留）

### 4.1 规则基线 —— **主基线，S6 逐类对比用**

文件：`BASELINE_MOCK.md`
复现命令（确定性、零 API 消耗、约 2 秒）：

```bash
cd campus-assistant/backend
VLM_PROVIDER=mock EMBEDDING_PROVIDER=local_hash python -m eval.run_eval
```

| 指标 | 值 |
|---|---|
| 分类准确率 | **92.5%（37/40）** |
| category F1 | 0.93 |
| course | 0.80 |
| location | 0.77 |
| issuer | 0.74 |
| event_time | 0.72 |
| deadline | 0.69 |
| **微平均 F1** | **0.80**（P 0.78 / R 0.82） |
| 未命中 | **37 项** |

**确定性已验证**：连续两次运行输出完全一致。
✅ 这是唯一能做严格 before/after 对比的基线，S6 每修完一类都跑它。

### 4.2 VLM 基线 —— 当前真实配置（`.env` 中 `VLM_PROVIDER=dashscope`）

文件：`BASELINE_VLM.md`（已存在，**不得覆盖**）
复现命令（约 2.5 分钟，消耗约 80 次真实调用）：

```bash
cd campus-assistant/backend
python -m eval.run_eval
```

| 指标 | 值 |
|---|---|
| 分类准确率 | **100%（40/40）** |
| **微平均 F1** | **0.89**（P 0.83 / R 0.97） |
| 未命中 | 24 项 |

⚠️ **非确定性**：LLM 采样（temperature=0.1）会导致结果有小幅漂移，
数字对比时应视为 ±2~3 个百分点，不宜做逐项严格比对。

---

## 5. 为什么需要两条

规则层(`rule_extract`)在 VLM 路径下**仍在运行**，并且是 VLM 返回空值时的
兜底来源（`OpenAICompatVLM.extract` 里 `merged = dict(rule_data)` 后按字段覆盖）。
因此：

- 修规则层的 RC1~RC5 **同时**会改善 VLM 基线的表现（消除兜底污染）
- 但分类类问题（RC4）在 VLM 路径下已不显现（分类已 100%）

所以：**用规则基线做逐类验证，用 VLM 基线做最终收益确认。**

---

## 6. 基线与 5 类根因的对应关系

| 根因 | 规则基线失败项 | VLM 基线失败项 |
|---|---|---|
| RC1 地点/发布方正则贪婪或漏词 | 12 | 13（location 5 + issuer 8） |
| RC2 时间语义 | 12 | 9（event_time） |
| RC3 课程名抽取 | 4 | 2 |
| RC4 分类误判 | 3 | 0（VLM 下分类已满分） |
| RC5 event/deadline 双填 | 4 | 部分并入 event_time |
| **合计** | **37** | **24** |

---

## 7. S6 收尾（S4 + S6 已完成 ✅）

| 阶段   | 状态 | 产物 |
|--------|------|------|
| S4 锁定 before 基线 | ✅ | `BASELINE_BEFORE.md`、`BASELINE_MOCK.md`（92.5% / micro-F1 0.80）|
| S6 修复 RC1–RC5 | ✅ | `BASELINE_AFTER.md`（**100% / micro-F1 0.97，残余 5 条已知边界**）|

S6 实际收益详见 `BASELINE_AFTER.md`，要点：

- 分类 **92.5% → 100%**
- micro-F1 **0.80 → 0.97（+0.17）**
- 残余 5 条全为「跨年/缺词/晨 7 点歧义/issuer 覆盖度」等非 RC 主线根因，可由 VLM 兜底

---
_Last refreshed: 2026-09-08 · S6 收尾_
