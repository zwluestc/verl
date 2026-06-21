# Reward Scoring Logic — `qwen3_0p6b_deepseek_reward.py`

## 一、主流程 `compute_score()`

```
预测文本 (solution_str)
  │
  ├─ Step 0：提取预测答案
  │    ├─ 有 <final>…</final> 或 \boxed{} → pred_candidate，max_reward = 1.0
  │    └─ 都没有 → 取末尾 500 字符作为 pred_candidate，max_reward = 0.8
  │          └─ 末尾也为空 → 直接返回 0.0
  │
  ├─ Step 0b：提取 ground_truth 最终答案（_extract_official_answer）
  │    └─ 为空 → 直接返回 0.0
  │
  ├─ Step 1：本地规则打分 _local_rule_score()
  │    └─ score > 0 → 返回 score × max_reward（结束）
  │
  ├─ Step 2：LLM Judge（DeepSeek API）
  │    ├─ 返回 1.0/0.0 → 返回对应 max_reward 或 0（结束）
  │    └─ 返回 -1.0（API 不可用/报错）→ 继续
  │
  └─ Step 3：兜底——对已提取答案做 normalized 精确匹配
       └─ 返回 max_reward 或 0.0
```

> **重要**：当前代码**没有**按数据集屏蔽 API 的逻辑（旧版的 `LOCAL_RULE_ONLY_SOURCES` 已被移除）。
> 所有数据集在本地规则失败后，都会尝试 LLM Judge；只有 API 不可用时才退化到兜底匹配。

---

## 二、本地规则栈 `_local_rule_score()`

### WildSci（独立分支）

```python
if source == "wildsci":
    pred_choice = _extract_choice(pred_candidate)  # 提取 A/B/C/D
    gt_choice   = _extract_choice(gt_final)
    if pred_choice and gt_choice:
        return 1.0 if match else 0.0
    return 0.0, "local_choice_missing"   # ← 提取失败直接返回 0，不继续走下面的规则
```

WildSci 的本地规则**只走选项提取**，失败即为 0，不再尝试数值/符号等比较。

### 其他数学/科学数据集（按顺序短路）

| 步骤 | 函数 | 说明 |
|------|------|------|
| ① | `_normalize(pred) == _normalize(gt)` | 字符串精确匹配（去 LaTeX 符号、空白等） |
| ② | `_numeric_equal(pred, gt)` | 浮点数值比较，相对容差 1e-6 |
| ③ | `_prime_math_equal(pred, gt)` | 调用 `verl.utils.reward_score.prime_math.grade_answer` |
| ④ | `_math_verify_equal(pred, gt)` | 调用 `verl.utils.reward_score.math_verify.compute_score`；pred 若无 `\boxed` 则自动包裹 |
| ⑤ | 多元组匹配 | 逗号/分号拆分后，对每个 part 依次尝试 ①②③④⑤ |
| ⑥ | `_sympy_equal(pred, gt)` | SymPy 符号化简后差为 0 |

以上任一命中即返回 `score=1.0`，全部失败才进入 LLM Judge。

---

## 三、不用 API 时的覆盖情况

### 3.1 本地规则能覆盖的情况

| 类型 | 示例 | 覆盖路径 |
|------|------|---------|
| 纯整数/小数 | `42`、`3.14` | ① normalized 或 ② numeric |
| 百分数 | `50%` vs `0.5` | ② numeric（`%` → `/100`）|
| 分数（相同写法） | `\frac{1}{2}` vs `\frac{1}{2}` | ① normalized |
| 分数（不同写法） | `\frac{1}{2}` vs `1/2` | ③ prime_math 或 ④ math_verify 或 ⑥ sympy |
| 根式等价 | `\sqrt{2}/2` vs `\frac{\sqrt{2}}{2}` | ③④⑥ |
| 多元组 | `1, 2` vs `2, 1` | ⑤ 无序匹配 |
| MCQ（WildSci） | `(B)` vs `B` | WildSci 分支 `_extract_choice` |

### 3.2 本地规则覆盖不到、依赖 API 的情况

| 场景 | 原因 | 风险数据集 |
|------|------|-----------|
| WildSci 选项提取失败 | 模型输出推理链过长，无法干净提取单字母；提取失败后直接返回 0，不走其他规则，无 API 则落到兜底匹配（几乎必然失败） | WildSci |
| 含 `=`/`<`/`>` 的答案 | `_parse_sympy_expr` 遇到 `[<>=]` 直接返回 `None`，方程解（`x=3`）、不等式区间跳过 sympy | DeepMath、SciInstruct |
| `\frac` 写法差异且 prime_math/math_verify/sympy 均不可用 | `_normalize()` 把 `\frac{1}{2}` 处理为 `\frac12`（删大括号），与 `1/2` 字符串不等；依赖后续三个验证器补救 | 全部 |
| 度数与弧度不一致 | `_normalize()` 把 `^\\circ` 直接删除（`30°`→`30`），但弧度 `\pi/6` 保留原样，sympy 不做度弧转换 | WildSci、SciInstruct |
| 复杂符号式 | 超出 `_latex_to_sympy_text` 转换能力（嵌套命令、不支持的函数）导致 sympy parse 失败 | DeepMath、Metamath、SciInstruct |
| gt 无标准格式 | `_extract_official_answer` 在无 `<final>`/`\boxed{}`/`####`/`answer is` 时返回原始全文，后续所有比较几乎失败 | 任意 |

---

## 四、各数据集综合评估

### WildSci

- **本地规则**：专门走选项提取分支，格式规范时可靠。
- **无 API 风险**：模型输出冗长推理时选项提取失败，且失败后**不继续走数学规则**，直接返回 0 进 LLM Judge；无 API 则兜底匹配 MCQ 长文本，几乎必然失败。
- **建议**：通过 prompt 要求模型最后单独输出 `<final>A</final>` 或 `\boxed{A}`，确保 `_extract_choice` 能命中。

### DeepMath / OpenR1Math

- **本地规则**：数值答案覆盖好；符号式依赖 prime_math + math_verify + sympy 三层保险。
- **无 API 风险**：方程解形式（`x = 3`）被 `[<>=]` 过滤跳过 sympy；三个验证器均不可用时分数写法不一致会误判；兜底 normalized 匹配对提取后的答案字符串还算合理，但不等价写法仍会漏判。
- **建议**：确认 `prime_math` 和 `math_verify` 模块可用；竞赛题答案以数值/根式为主时无 API 基本可行。

### Metamath

- **本地规则**：形式化证明的结论若为标准数学表达式，同 DeepMath 路径。
- **无 API 风险**：Metamath 的 ground_truth 若是定理名或证明步骤（而非数值/解析式），`_extract_official_answer` 返回原文，normalized 匹配和 sympy 均不适用，本地规则必然失败；无 API 则判 0。
- **建议**：检查数据集 gt 格式，若为数值/解析式结论则可接受；若为形式化证明文本则必须有 API。

### SciInstruct

- **本地规则**：走标准数学规则路径，无特殊处理。
- **无 API 风险**：物理题常见问题：① 带单位答案（`9.8 m/s²` vs `9.8`）normalized 后仍不等；② 度/弧度混用；③ 向量、矩阵等结构化答案超出 sympy 解析能力。
- **建议**：物理/科学题强烈建议保留 API；或对 gt 做预处理统一单位和格式。

---

## 五、关键依赖项

| 依赖 | 检测方式 | 缺失时影响 |
|------|---------|-----------|
| `sympy` | `try/import` 包裹，失败则 `_sympy = None` | `_sympy_equal` 始终返回 `False` |
| `math_verify` | `importlib.util.find_spec("math_verify")` | `_math_verify_equal` 始终返回 `False` |
| `verl.utils.reward_score.prime_math` | 运行时 `try/import` | `_prime_math_equal` 始终返回 `False` |
| DeepSeek API Key | `DEEPSEEK_API_KEY` / `LLM_API_KEY` / `API_KEY` 任一非空 | 跳过 LLM Judge，落到兜底匹配 |

---

## 六、环境变量

| 变量 | 作用 | 默认值 |
|------|------|-------|
| `LLM_API_KEY` / `API_KEY` / `DEEPSEEK_API_KEY` | API 鉴权 Key（优先级从左到右） | 无 |
| `LLM_API_URL` / `DEEPSEEK_API_URL` | 完整 Chat Completions URL | 自动构建 |
| `LLM_API_BASE_URL` / `BASE_URL` / `DEEPSEEK_API_BASE_URL` | Base URL（自动补 `/v1/chat/completions`） | `https://api.deepseek.com` |
| `LLM_JUDGE_MODEL` / `DEEPSEEK_MODEL` | 判分模型名 | `deepseek-v4-flash` |
| `LLM_JUDGE_MAX_TOKENS` | Judge 响应最大 token 数 | `128` |
| `LOCAL_MATH_VERIFY_TIMEOUT` | `math_verify` 单次超时（秒） | `5` |
| `REWARD_DEBUG_LOG` | 调试日志路径（JSONL） | 不写日志 |
| `REWARD_DEBUG_LIMIT` | 调试日志最大条数（上限 1000） | `1000` |
