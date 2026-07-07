"""
agent.py — 体检报告解读 Agent（ReAct 多步推理）

架构：4 步推理链
  Step 1: 报告结构分析 — 提取所有指标项（名称/数值/单位/参考范围/方向/是否异常）
  Step 2: 逐项解读 + 交叉关联 — 深度解读每个异常指标，发现指标间关联风险
  Step 3: 自检纠错 — 对比 Step1 指标列表，检查遗漏与严重程度一致性
  Step 4: 整合输出 — 合并修正，生成最终结构化报告

多轮追问：基于报告上下文进行多轮对话，支持用户追问
"""

import requests
import json
import re
from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL


# ============================================================
# LLM 调用基础设施
# ============================================================

def _call_llm(messages: list, temperature: float = 0.3, timeout: int = 60) -> str:
    """调用 DeepSeek Chat API（支持多轮 messages 格式）"""
    response = requests.post(
        DEEPSEEK_BASE_URL,
        headers={
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": DEEPSEEK_MODEL,
            "messages": messages,
            "temperature": temperature,
        },
        timeout=timeout,
    )
    if response.status_code != 200:
        return f"[API_ERROR:{response.status_code}] {response.text[:200]}"
    return response.json()["choices"][0]["message"]["content"]


def _call_llm_json(prompt: str, temperature: float = 0.3) -> dict:
    """调用 LLM 并解析为 JSON，失败返回空 dict"""
    raw = _call_llm([{"role": "user", "content": prompt}], temperature=temperature)
    return _safe_json_parse(raw)


def _safe_json_parse(raw: str) -> dict:
    """从 LLM 返回文本中提取 JSON"""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass
    return {}


# ============================================================
# Step 1: 报告结构分析
# ============================================================

STEP1_PROMPT = """你是一个体检报告解析器。请分析以下体检报告文本，提取所有检测指标。

任务：
1. 识别报告类型（常规体检/血液化验/肝功能/血脂/血糖/尿常规/影像报告等）
2. 提取每一个检测指标，包括：指标名称、检测结果（含单位）、参考范围、偏离方向（↑偏高/↓偏低/正常）
3. 标记哪些指标是异常的（有↑↓箭头，或数值超出参考范围）

输出纯 JSON（不要 markdown 代码块）：
{{
  "report_type": "报告类型描述",
  "indicators": [
    {{
      "name": "指标中文名",
      "value": "检测数值（含单位）",
      "reference": "参考范围",
      "direction": "偏高/偏低/正常",
      "is_abnormal": true
    }}
  ]
}}

以下是体检报告文本：
---
{report_text}
---

请直接输出 JSON："""


def step1_analyze_structure(report_text: str) -> dict:
    """Step 1: 分析报告结构，提取所有指标"""
    prompt = STEP1_PROMPT.format(report_text=report_text[:3000])
    data = _call_llm_json(prompt, temperature=0.1)
    return {
        "report_type": data.get("report_type", "体检报告"),
        "indicators": data.get("indicators", []),
        "total_count": len(data.get("indicators", [])),
        "abnormal_count": sum(1 for i in data.get("indicators", []) if i.get("is_abnormal")),
    }


# ============================================================
# Step 2: 逐项解读 + 交叉关联分析
# ============================================================

STEP2_PROMPT = """你是一位经验丰富的家庭医生，正在帮一位65岁的老年人解读体检报告。

我已经帮你提取了报告中的所有指标（结构化数据），请基于这些数据进行深度解读。

【核心原则（必须严格遵守）】
1. 严格基于当前数据，绝对禁止编造既往病史、用药史、家族史
2. 只描述偏离，不做确诊——用"可能存在""或可伴随"等不确定表述
3. 严禁"说明有""证明""可诊断"等绝对诊断词语

【任务】
1. 对每个【异常指标】进行通俗解读
2. 进行交叉关联分析：发现多个异常指标之间的关联风险
   例如：血糖偏高 + 甘油三酯偏高 + 低密度脂蛋白偏高 → 代谢综合征风险
3. 给出分级的生活建议和就医提醒

【严重程度分级】
- 注意：轻度偏离（<20%），居家调理
- 关注：明显偏离（20%-50%），生活干预+择期复查
- 警惕：严重偏离（>50%）或涉及关键器官，尽快就医

【输出纯 JSON】
{{
  "整体术语翻译": {{"原词": "大白话解释"}},
  "异常指标": [
    {{
      "指标": "名称",
      "结果": "检测值",
      "正常范围": "参考区间",
      "状态": "偏高/偏低",
      "严重程度": "注意/关注/警惕",
      "通俗解释": "大白话解释",
      "相关术语": [{{"原词": "词汇", "解释": "解释"}}],
      "复查建议": "建议X个月复查"
    }}
  ],
  "交叉关联分析": "多个异常指标之间的关联风险解读（如果没有关联则写'本次报告各项异常指标相对独立，未见明显关联风险'）",
  "生活建议": ["具体可操作的建议"],
  "就医提醒": ["需要看医生的提醒，注明科室"]
}}

以下是结构化指标数据：
---
{indicators_json}
---

以下是原始报告文本（供参考）：
---
{report_text}
---

请直接输出 JSON："""


def step2_interpret(structure: dict, report_text: str) -> dict:
    """Step 2: 逐项解读 + 交叉关联"""
    indicators_json = json.dumps(structure.get("indicators", []), ensure_ascii=False, indent=2)
    prompt = STEP2_PROMPT.format(
        indicators_json=indicators_json,
        report_text=report_text[:2000],
    )
    return _call_llm_json(prompt, temperature=0.3)


# ============================================================
# Step 3: 自检纠错
# ============================================================

STEP3_PROMPT = """你是一个质检审核员。请检查体检报告解读结果是否准确完整。

【检查项】
1. 完整性：对比原始指标列表，是否有异常指标被遗漏？
2. 一致性：严重程度判定是否合理？（轻度偏离不该判为"警惕"，严重偏离不该判为"注意"）
3. 合规性：是否存在绝对诊断用语？（如"说明有""确诊""证明"等违规表述）

【输出纯 JSON】
{{
  "passed": true/false,
  "missing_indicators": ["被遗漏的异常指标名称"],
  "severity_issues": [
    {{
      "指标": "名称",
      "当前判定": "注意/关注/警惕",
      "建议判定": "注意/关注/警惕",
      "原因": "为什么需要修正"
    }}
  ],
  "compliance_issues": ["不合规的表述列表"],
  "summary": "总体质量评价"
}}

【原始提取的异常指标列表】
{original_abnormals}

【解读结果中的异常指标列表】
{interpreted_items}

请直接输出 JSON："""


def step3_self_check(structure: dict, interpretation: dict) -> dict:
    """Step 3: 自检纠错"""
    original_abnormals = [i.get("name", "") for i in structure.get("indicators", []) if i.get("is_abnormal")]
    interpreted_items = [i.get("指标", "") for i in interpretation.get("异常指标", [])]

    prompt = STEP3_PROMPT.format(
        original_abnormals=json.dumps(original_abnormals, ensure_ascii=False),
        interpreted_items=json.dumps(interpreted_items, ensure_ascii=False),
    )
    return _call_llm_json(prompt, temperature=0.1)


# ============================================================
# Step 4: 整合输出
# ============================================================

def step4_integrate(interpretation: dict, check_result: dict) -> dict:
    """Step 4: 合并自检修正，输出最终结果"""

    items = interpretation.get("异常指标", [])

    # 应用严重程度修正
    severity_fixes = {f["指标"]: f["建议判定"] for f in check_result.get("severity_issues", [])}
    for item in items:
        name = item.get("指标", "")
        if name in severity_fixes:
            old = item.get("严重程度", "")
            new = severity_fixes[name]
            item["严重程度"] = new
            # 同步更新复查建议
            review_map = {"警惕": "建议1个月内就医复查", "关注": "建议3个月内复查", "注意": "建议半年内常规体检复查"}
            item["复查建议"] = review_map.get(new, item.get("复查建议", ""))

    # 构建自检报告
    check_report = {
        "passed": check_result.get("passed", True),
        "missing": check_result.get("missing_indicators", []),
        "severity_fixes": len(severity_fixes),
        "compliance_issues": check_result.get("compliance_issues", []),
        "summary": check_result.get("summary", ""),
    }

    return {
        "global_terms": interpretation.get("整体术语翻译", {}),
        "items": items,
        "advice": interpretation.get("生活建议", []),
        "alerts": interpretation.get("就医提醒", []),
        "cross_analysis": interpretation.get("交叉关联分析", ""),
        "check_report": check_report,
    }


# ============================================================
# Agent 主入口
# ============================================================

def interpret_report_agent(report_text: str, on_step=None) -> dict:
    """
    Agent 多步推理解读体检报告。

    Args:
        report_text: 体检报告文本
        on_step: 回调函数 on_step(step_num, step_name, result)，用于展示进度

    Returns:
        与 core.interpret_report 兼容的 dict，额外包含:
        - agent_steps: 各步骤的执行记录
        - cross_analysis: 交叉关联分析
        - check_report: 自检报告
    """
    steps_log = []

    def _log(step, name, result):
        steps_log.append({"step": step, "name": name, "result": result})
        if on_step:
            on_step(step, name, result)

    # ---- Step 1: 报告结构分析 ----
    _log(1, "报告结构分析", "⏳ 正在提取所有检测指标...")
    try:
        structure = step1_analyze_structure(report_text)
        if not structure.get("indicators"):
            return {"error": "未能从报告中提取到有效指标，请检查文本内容"}
        _log(1, "报告结构分析", f"✅ 提取到 {structure['total_count']} 项指标，其中 {structure['abnormal_count']} 项异常")
    except Exception as e:
        _log(1, "报告结构分析", f"❌ 失败：{e}")
        return {"error": f"报告结构分析失败：{e}"}

    # ---- Step 2: 逐项解读 + 交叉关联 ----
    _log(2, "逐项解读 + 交叉关联", "⏳ 正在深度解读异常指标...")
    try:
        interpretation = step2_interpret(structure, report_text)
        if not interpretation.get("异常指标"):
            return {"error": "解读未生成有效结果，请重试"}
        item_count = len(interpretation.get("异常指标", []))
        cross = interpretation.get("交叉关联分析", "无")
        _log(2, "逐项解读 + 交叉关联", f"✅ 解读 {item_count} 项异常指标，关联分析完成")
    except Exception as e:
        _log(2, "逐项解读 + 交叉关联", f"❌ 失败：{e}")
        return {"error": f"解读失败：{e}"}

    # ---- Step 3: 自检纠错 ----
    _log(3, "自检纠错", "⏳ 正在核查完整性与合规性...")
    try:
        check_result = step3_self_check(structure, interpretation)
        passed = check_result.get("passed", True)
        missing = len(check_result.get("missing_indicators", []))
        fixes = len(check_result.get("severity_issues", []))
        status = "✅ 通过" if passed else f"⚠️ 发现 {missing} 项遗漏、{fixes} 项修正"
        _log(3, "自检纠错", status)
    except Exception as e:
        _log(3, "自检纠错", f"⚠️ 自检跳过：{e}")
        check_result = {"passed": True, "missing_indicators": [], "severity_issues": [], "compliance_issues": [], "summary": "自检未执行"}

    # ---- Step 4: 整合输出 ----
    _log(4, "整合输出", "⏳ 正在生成最终报告...")
    final = step4_integrate(interpretation, check_result)
    _log(4, "整合输出", "✅ 报告生成完成")

    final["agent_steps"] = steps_log
    final["report_type"] = structure.get("report_type", "体检报告")
    return final


# ============================================================
# 多轮追问对话
# ============================================================

QA_SYSTEM_PROMPT = """你是一位耐心细致的家庭医生助手。用户已经上传了体检报告并获得了 AI 解读结果。
现在用户可以就报告内容向你追问。请基于以下信息回答：

【体检报告原文】
{report_text}

【AI 解读结果摘要】
{interpretation_summary}

【回答规则】
1. 只基于报告数据和解读结果回答，不要编造信息
2. 如果用户问的信息不在报告中，明确告知"本次报告未包含此项检查"
3. 用通俗语言回答，避免过多专业术语
4. 不做确诊，用"可能""建议进一步检查"等表述
5. 如果用户描述了报告之外的症状，可以给一般性建议但注明"这需要医生面诊判断"
6. 回答简洁，一般不超过 200 字"""


def ask_followup(question: str, report_text: str, interpretation: dict, chat_history: list = None) -> str:
    """
    多轮追问：基于报告上下文回答用户问题。

    Args:
        question: 用户问题
        report_text: 原始报告文本
        interpretation: Agent 解读结果
        chat_history: 之前的对话记录 [{role: "user"/"assistant", content: "..."}]

    Returns:
        回答文本
    """
    # 构建解读摘要
    items = interpretation.get("items", [])
    summary_parts = []
    for item in items:
        summary_parts.append(
            f"- {item.get('指标', '?')}：{item.get('结果', '?')}（{item.get('状态', '?')}，{item.get('严重程度', '?')}）"
        )
    cross = interpretation.get("cross_analysis", "")
    if cross:
        summary_parts.append(f"\n关联分析：{cross}")

    interpretation_summary = "\n".join(summary_parts) if summary_parts else "无明显异常指标"

    system_content = QA_SYSTEM_PROMPT.format(
        report_text=report_text[:2000],
        interpretation_summary=interpretation_summary,
    )

    messages = [{"role": "system", "content": system_content}]
    if chat_history:
        messages.extend(chat_history[-6:])  # 保留最近 3 轮对话
    messages.append({"role": "user", "content": question})

    return _call_llm(messages, temperature=0.4)
