"""
deepseek_client.py — DeepSeek V3 integration (text-only).

Two-call pipeline:
  1. Board writing: natural language prompt (user's preferred style) → free-form text
  2. Reformat board writing to clean Markdown (content unchanged)
  3. Question analysis: knowledge points, approach, topic groups → JSON
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from openai import OpenAI

logger = logging.getLogger(__name__)

# ── System prompt for question analysis (JSON output) ────────────────────────

_ANALYSIS_SYSTEM_HOMEWORK = """你是一位高中物理教师助手，只输出 JSON，不输出任何其他内容。

对每道题给出：
1. knowledge_points：具体考察的知识点（要细，不要泛化，每道题的知识点必须与题目内容严格对应）
2. approach：解题思路，1-3句话，说清楚切入点和步骤逻辑
3. primary_topic：该题最核心的考点，用一个具体短语表示（5-12字），
   这是归类的唯一依据，要能直接看出这道题在考什么。
   示例："电阻率与温度的关系""U-I图线斜率与电阻""串联电路电压分配"
   禁止用："计算题""综合题""电学基础""电阻问题"等笼统表达

topic_groups 规则：
- 把 primary_topic 相同或高度相关的题目归为一组
- 组名直接用该类题目共同的核心考点（具体，不笼统）
- 一道题只归入一个最主要的组

严格按以下 JSON 格式输出，不要输出任何其他内容：
{
  "questions": [
    {
      "number": 题目编号(整数),
      "knowledge_points": ["知识点1", "知识点2"],
      "approach": "解题思路（1-3句）",
      "primary_topic": "该题最核心考点（5-12字）"
    }
  ],
  "topic_groups": [
    {
      "topic": "该组核心考点（具体短语）",
      "numbers": [题号列表]
    }
  ]
}"""

_ANALYSIS_SYSTEM_EXAM = """你是一位高中物理教师助手，只输出 JSON，不输出任何其他内容。

对每道题给出：
1. knowledge_points：具体考察的知识点（要细，不要泛化，必须与题目内容严格对应）
2. approach：解题思路，1-3句话，说清楚切入点和步骤逻辑
3. primary_topic：按以下规则确定（5-12字），这是归类的唯一依据：
   - 选择题（单选/多选）：用该题考察的题型大类，例如：
     "匀变速直线运动""牛顿第二定律综合""电场力做功与电势能""LC振荡电路"
   - 实验题：用实验名称，例如：
     "用单摆测重力加速度""描绘小灯泡的伏安特性曲线""验证机械能守恒定律"
   - 计算题（解答题）：用物理模型名称，例如：
     "带电粒子在匀强电场中的运动""楞次定律与安培力做功""弹簧振子与动量守恒"
   禁止用："计算题""选择题""综合题""电学题"等笼统表达

topic_groups 规则：
- 把 primary_topic 相同或高度相关的题目归为一组
- 组名直接用该类题目共同的 primary_topic（具体，不笼统）
- 一道题只归入一个最主要的组
- 同一道大题的各小问归入同一组

严格按以下 JSON 格式输出，不要输出任何其他内容：
{
  "questions": [
    {
      "number": 题目编号(整数),
      "knowledge_points": ["知识点1", "知识点2"],
      "approach": "解题思路（1-3句）",
      "primary_topic": "该题考点/题型/模型（5-12字）"
    }
  ],
  "topic_groups": [
    {
      "topic": "该组题型/模型名称（具体短语）",
      "numbers": [题号列表]
    }
  ]
}"""

# ── System prompt for Markdown reformatting ───────────────────────────────────
_REFORMAT_SYSTEM = """你是一个 Markdown 格式化工具，目标渲染器为 StackEdit（支持 KaTeX 数学公式、GFM 表格）。
将用户提供的板书内容转换为规范 Markdown，严格遵守以下规则：

【格式规则】
1. 内容、知识点、公式一字不改，只调整格式
2. 标题只用 ### 和 ####，标题前空一行，标题后不空行
3. 数学公式：
   - 行内变量/公式用 $...$（如 $R = \\rho L/S$）
   - 推导式、定义式独占一行用 $$...$$，上下各空一行
   - 变量用斜体（默认），单位用 \\text{} 正体（如 $\\text{Ω}$）
4. 列表只用 - 或 1.，不用 * 或 +，列表项之间不空行
5. 对比/归纳/分类内容**必须**使用 GFM 标准表格（包括公式对比、情况分类、易错总结等），表格行之间不空行：
   | 项目 | A | B |
   |------|---|---|
   | ...  |...|...|
6. 重点内容用 **加粗**，易错点用 ⚠️ **加粗提示**
7. 在语义合适的位置加 Unicode 图标（不要强行加，宁缺毋滥）：
   - 🔑 核心概念/关键公式   ⚡ 电学内容
   - ⚠️ 易错点/注意事项    ✅ 正确做法   ❌ 常见错误
   - 🔄 变化关系/循环      🆚 对比比较
   - 🌟 重要结论/口诀      🔋 电路/能量
   - ⬆ ⬇ 增大/减小关系   📐 几何/图形
8. 不加开场白，不加结尾客套话
9. 不用代码块包裹整个回答

直接输出格式化后的 Markdown，不要任何解释。"""


def analyze_questions(
    api_key: str,
    question_texts: list[str],
    question_numbers: list[int],
    homework_name: str = "",
    reference_names: list[str] | None = None,
    reference_texts: list[str] | None = None,
    api_base: str = "https://api.deepseek.com",
    generate_board: bool = True,
    analyze_json: bool = True,
    mode: str = "homework",
) -> dict[str, Any]:
    """
    generate_board : whether to call the board-writing prompts (2 API calls)
    analyze_json   : whether to call the question-analysis prompt (1 API call)
    At least one of them should be True when this function is called.
    """
    client = OpenAI(api_key=api_key, base_url=api_base)
    result: dict[str, Any] = {"questions": [], "topic_groups": [], "board_writing": ""}

    if generate_board:
        # ── Call 1: Board writing (natural prompt, free-form output) ─────────
        board_raw = _generate_board_writing(
            client, question_texts, question_numbers,
            homework_name, reference_names, reference_texts,
        )
        logger.info("Board writing generated, length=%d chars", len(board_raw))

        # ── Call 2: Reformat board writing to clean Markdown ─────────────────
        board_md = _reformat_to_markdown(client, board_raw)
        logger.info("Board writing reformatted, length=%d chars", len(board_md))
        result["board_writing"] = board_md

    if analyze_json:
        # ── Call 3: Question analysis → JSON ─────────────────────────────────
        analysis = _analyze_questions_json(
            client, question_texts, question_numbers, reference_texts, mode=mode,
        )
        result["questions"]    = analysis.get("questions", [])
        result["topic_groups"] = analysis.get("topic_groups", [])

    return result


# ── Board writing generation ──────────────────────────────────────────────────

def _generate_board_writing(
    client: OpenAI,
    question_texts: list[str],
    question_numbers: list[int],
    homework_name: str,
    reference_names: list[str] | None,
    reference_texts: list[str] | None,
) -> str:
    hw_label = f"《{homework_name}》" if homework_name else "《作业》"

    ref_labels = ""
    if reference_names:
        ref_labels = "".join(f"《{n}》" for n in reference_names)
    elif reference_texts:
        ref_labels = "《课件》"

    prompt_parts = [
        f"请根据作业{hw_label}"
        + (f"和我的课件{ref_labels}" if ref_labels else "")
        + "，帮我设计出明天上课的板书，越精简越好，但是要能做题。"
        + "适合用表格对比或归纳的地方（如公式对比、情况分类、易错点总结等）一律用表格。"
    ]

    if reference_texts:
        combined_ref = "\n\n".join(reference_texts[:10])[:8000]
        prompt_parts.append(f"\n【课件内容】\n{combined_ref}")

    prompt_parts.append(f"\n【作业题目（共{len(question_texts)}题）】")
    for q_num, q_text in zip(question_numbers, question_texts):
        prompt_parts.append(f"\n第{q_num}题：\n{q_text.strip()}")

    user_content = "\n".join(prompt_parts)

    try:
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": user_content}],
            max_tokens=4096,
            temperature=0.5,
        )
    except Exception as e:
        logger.error("Board writing API call failed: %s", e)
        raise RuntimeError(f"板书生成失败: {e}") from e

    return resp.choices[0].message.content or ""


# ── Markdown reformatting ─────────────────────────────────────────────────────

def _reformat_to_markdown(client: OpenAI, raw_text: str) -> str:
    try:
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": _REFORMAT_SYSTEM},
                {"role": "user", "content": raw_text},
            ],
            max_tokens=4096,
            temperature=0.1,
        )
    except Exception as e:
        logger.warning("Markdown reformat failed, using raw text: %s", e)
        return raw_text

    result = resp.choices[0].message.content or raw_text
    # Strip accidental code fence wrapping
    fence = re.search(r"```(?:markdown)?\s*([\s\S]*?)```", result)
    if fence:
        result = fence.group(1).strip()
    return _remove_extra_blank_lines(result)


def _remove_extra_blank_lines(md: str) -> str:
    """
    Remove all blank lines except one blank line immediately before a heading line.
    Consecutive blank lines before a heading are collapsed to one.
    """
    lines = md.split('\n')
    result: list[str] = []
    i = 0
    while i < len(lines):
        if lines[i].strip() == '':
            # Collect run of blank lines, then peek at next non-blank line
            j = i + 1
            while j < len(lines) and lines[j].strip() == '':
                j += 1
            if j < len(lines) and lines[j].lstrip().startswith('#'):
                result.append('')   # keep exactly one blank line before heading
            # else: discard all blank lines in this run
            i = j
        else:
            result.append(lines[i])
            i += 1
    return '\n'.join(result)


# ── Question analysis → JSON ──────────────────────────────────────────────────

def _analyze_questions_json(
    client: OpenAI,
    question_texts: list[str],
    question_numbers: list[int],
    reference_texts: list[str] | None,
    mode: str = "homework",
) -> dict[str, Any]:
    parts: list[str] = []

    if reference_texts:
        combined_ref = "\n\n".join(reference_texts[:10])[:4000]
        parts.append(f"【参考课件内容】\n{combined_ref}")

    # Deduplicate page texts — same page may appear for multiple questions;
    # send the full homework text once rather than repeating per question.
    seen, unique_texts = set(), []
    for t in question_texts:
        key = t.strip()[:200]   # use first 200 chars as dedup key
        if key not in seen:
            seen.add(key)
            unique_texts.append(t.strip())

    parts.append(f"【作业全文】\n" + "\n\n---\n\n".join(unique_texts))
    parts.append(
        f"\n作业共 {len(question_numbers)} 道题，题号为：{question_numbers}。\n"
        "请逐题分析，每道题的知识点和解题思路必须针对该题具体内容，不能千篇一律。\n"
        "严格按要求的JSON格式输出。"
    )
    user_content = "\n\n".join(parts)

    logger.info("Sending question analysis request, %d questions, %d chars",
                len(question_texts), len(user_content))

    system_prompt = _ANALYSIS_SYSTEM_EXAM if mode == "exam" else _ANALYSIS_SYSTEM_HOMEWORK

    try:
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            max_tokens=4096,
            temperature=0.2,
        )
    except Exception as e:
        logger.error("Question analysis API call failed: %s", e)
        raise RuntimeError(f"题目分析失败: {e}") from e

    raw = resp.choices[0].message.content or ""
    logger.debug("Analysis raw response (first 500): %s", raw[:500])
    return _parse_analysis(raw, question_numbers)


# ── JSON parsing ──────────────────────────────────────────────────────────────

def _parse_analysis(raw_text: str, question_numbers: list[int]) -> dict[str, Any]:
    text = raw_text.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        text = fence.group(1).strip()

    try:
        result = json.loads(text)
        _validate_and_fill(result, question_numbers)
        return result
    except json.JSONDecodeError:
        pass

    obj = re.search(r"\{[\s\S]*\}", text)
    if obj:
        try:
            result = json.loads(obj.group(0))
            _validate_and_fill(result, question_numbers)
            return result
        except json.JSONDecodeError:
            pass

    logger.error("Could not parse JSON from analysis response: %s", raw_text[:1000])
    return _fallback_result(question_numbers, raw_text)


def _validate_and_fill(result: dict, question_numbers: list[int]) -> None:
    if "questions" not in result:
        result["questions"] = []
    if "topic_groups" not in result:
        result["topic_groups"] = []

    existing_nums = {q.get("number") for q in result["questions"]}
    for qnum in question_numbers:
        if qnum not in existing_nums:
            result["questions"].append({
                "number": qnum,
                "knowledge_points": [],
                "approach": "",
            })

    result["questions"].sort(key=lambda q: q.get("number", 0))


def _fallback_result(question_numbers: list[int], raw_text: str) -> dict[str, Any]:
    return {
        "questions": [
            {"number": n, "knowledge_points": ["（解析失败，请手动填写）"], "approach": ""}
            for n in question_numbers
        ],
        "topic_groups": [],
        "board_writing": f"### 板书内容\n\n（AI返回内容解析失败）\n\n原始返回：\n\n{raw_text[:500]}",
    }
