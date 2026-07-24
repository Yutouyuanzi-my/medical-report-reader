"""
agent.py — 体检报告解读 Agent（ReAct + Function Calling）

架构升级：从写死的 4 步 pipeline → LLM 自主决策的 ReAct 循环

核心变化：
  1. LLM 自主决定是否调用工具（不再写死步骤顺序）
  2. 4 个医疗工具：eGFR 计算、单位换算、参考范围查询、偏离度评估
  3. ReAct 循环：观察→推理→行动→观察→...→输出
  4. 动态路由：根据报告内容决定分析路径

工具列表：
  - calculate_egfr:       根据血肌酐计算肾小球滤过率（MDRD 公式）
  - convert_unit:         医学检验单位换算（血糖/血脂/肌酐/尿酸等）
  - lookup_reference:     查询检验项目标准参考范围（内置知识库）
  - assess_severity:      量化评估指标偏离程度（百分比 + 严重等级）
"""

import requests
import json
import re
import math
from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL, _proxies


# ============================================================
# LLM 调用基础设施
# ============================================================

def _call_llm(messages: list, temperature: float = 0.3, timeout: int = 60) -> str:
    """调用 DeepSeek Chat API（纯文本，不带工具）"""
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
        proxies=_proxies,
    )
    if response.status_code != 200:
        return f"[API_ERROR:{response.status_code}] {response.text[:200]}"
    return response.json()["choices"][0]["message"]["content"]


def _call_llm_with_tools(messages: list, tools: list, temperature: float = 0.3, timeout: int = 90) -> dict:
    """
    调用 DeepSeek API（支持 Function Calling）。
    返回完整的 message 对象，包含 content 和/或 tool_calls。
    """
    response = requests.post(
        DEEPSEEK_BASE_URL,
        headers={
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": DEEPSEEK_MODEL,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "temperature": temperature,
        },
        timeout=timeout,
        proxies=_proxies,
    )
    if response.status_code != 200:
        return {"content": f"[API_ERROR:{response.status_code}] {response.text[:200]}", "tool_calls": None}
    choice = response.json()["choices"][0]
    return {
        "content": choice["message"].get("content"),
        "tool_calls": choice["message"].get("tool_calls"),
        "finish_reason": choice.get("finish_reason"),
    }


def _call_llm_with_tools_stream(messages: list, tools: list, on_token=None, temperature: float = 0.3, timeout: int = 120) -> dict:
    """
    流式调用 DeepSeek API（支持 Function Calling）。

    在流式接收的同时：
      - 通过 on_token(text) 实时回传 content 片段（用于 UI 流式展示）
      - 累积完整响应，返回包含 content / tool_calls / finish_reason 的 dict

    ReAct 循环中：工具调用轮次通常只有 tool_calls 增量（无 content），
    此时 on_token 不会被触发；只有最终输出轮（含 content）才会流式展示。
    """
    response = requests.post(
        DEEPSEEK_BASE_URL,
        headers={
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": DEEPSEEK_MODEL,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "temperature": temperature,
            "stream": True,
        },
        timeout=timeout,
        stream=True,
        proxies=_proxies,
    )
    if response.status_code != 200:
        return {"content": f"[API_ERROR:{response.status_code}] {response.text[:200]}", "tool_calls": None}

    full_content = ""
    tool_calls = []
    finish_reason = None

    for raw_line in response.iter_lines():
        if not raw_line:
            continue
        line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
        if not line.startswith("data:"):
            continue
        data_str = line[5:].strip()
        if data_str == "[DONE]":
            break
        try:
            chunk = json.loads(data_str)
        except json.JSONDecodeError:
            continue
        if not chunk.get("choices"):
            continue
        delta = chunk["choices"][0].get("delta", {})
        # 累积 content 并流式回传
        if delta.get("content"):
            piece = delta["content"]
            full_content += piece
            if on_token:
                on_token(piece)
        # 累积 tool_calls 增量
        if delta.get("tool_calls"):
            for tc_delta in delta["tool_calls"]:
                idx = tc_delta.get("index", 0)
                while len(tool_calls) <= idx:
                    tool_calls.append({"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                tc = tool_calls[idx]
                if tc_delta.get("id"):
                    tc["id"] = tc_delta["id"]
                if tc_delta.get("function"):
                    fn = tc_delta["function"]
                    if fn.get("name"):
                        tc["function"]["name"] += fn["name"]
                    if fn.get("arguments"):
                        tc["function"]["arguments"] += fn["arguments"]
        if chunk["choices"][0].get("finish_reason"):
            finish_reason = chunk["choices"][0]["finish_reason"]

    return {
        "content": full_content,
        "tool_calls": tool_calls if tool_calls else None,
        "finish_reason": finish_reason,
    }


def _safe_json_parse(raw: str) -> dict:
    """从 LLM 返回文本中提取 JSON（处理 markdown 包裹、think 块、混合文本、非法字符等）"""
    if not raw:
        return {}

    # 0. 基础清理：去掉 BOM、首尾空白、常见不可见控制字符（保留 \t\n\r）
    text = raw.strip()
    text = text.replace("\ufeff", "")
    text = "".join(ch for ch in text if ch == "\t" or ch == "\n" or ch == "\r" or ord(ch) >= 32)

    # 1. 去掉各种 think / reasoning 块
    text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"\tthinking.*?\t/think", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"```thinking\n.*?```", "", text, flags=re.DOTALL).strip()

    # 2. 去掉 markdown 代码块包裹
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    # 3. 尝试直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 4. 提取最外层匹配的 {...}（使用栈，避免只取到第一个子对象）
    start = -1
    depth = 0
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start != -1:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        pass
    if start == -1:
        start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass

    # 5. 清理后再次尝试：替换中文引号、去掉注释、去掉 trailing comma
    cleaned = text
    cleaned = cleaned.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    cleaned = re.sub(r"//[^\n]*", "", cleaned)
    cleaned = re.sub(r"/\*.*?\*/", "", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r",(\s*[}\]])", r"\1", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # 6. 提取 markdown 代码块中的 JSON（取最长）
    candidates = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", raw, flags=re.DOTALL)
    if candidates:
        for cand in sorted(candidates, key=len, reverse=True):
            for variant in [cand, cand.replace("'", '"'), re.sub(r",(\s*[}\]])", r"\1", cand)]:
                try:
                    return json.loads(variant)
                except json.JSONDecodeError:
                    pass

    return {}
# ============================================================
# 工具 1: calculate_egfr — 肾功能计算
# ============================================================

def _tool_calculate_egfr(creatinine_umol: float, age: float, gender: str) -> dict:
    """
    根据血肌酐计算估算肾小球滤过率（eGFR）。
    使用 MDRD 简化公式：eGFR = 175 × (Scr_mg/dL)^-1.154 × Age^-0.203 × (0.742 if female)
    """
    scr_mg = creatinine_umol / 88.4  # μmol/L → mg/dL
    base = 175 * (scr_mg ** -1.154) * (age ** -0.203)
    if gender == "女":
        base *= 0.742
    egfr = round(base, 1)

    # CKD 分期
    if egfr >= 90:
        stage, advice = "G1 正常", "肾功能正常"
    elif egfr >= 60:
        stage, advice = "G2 轻度下降", "建议定期监测，控制血压血糖"
    elif egfr >= 45:
        stage, advice = "G3a 中度下降", "建议肾内科就诊评估"
    elif egfr >= 30:
        stage, advice = "G3b 中重度下降", "建议尽快肾内科就诊"
    elif egfr >= 15:
        stage, advice = "G4 重度下降", "需肾内科紧急就诊"
    else:
        stage, advice = "G5 肾衰竭", "需立即就医"

    return {
        "egfr": egfr,
        "unit": "mL/min/1.73m²",
        "stage": stage,
        "advice": advice,
        "formula": "MDRD 简化公式",
        "input": f"肌酐 {creatinine_umol} μmol/L, {age}岁, {gender}",
    }


# ============================================================
# 工具 2: convert_unit — 医学单位换算
# ============================================================

# 换算因子表：(from_unit → to_unit) 对应的乘数，按检测项目区分
_UNIT_FACTORS = {
    "血糖":     {"mmol/L→mg/dL": 18.018, "mg/dL→mmol/L": 0.0555},
    "总胆固醇": {"mmol/L→mg/dL": 38.67,  "mg/dL→mmol/L": 0.0259},
    "甘油三酯": {"mmol/L→mg/dL": 88.5,   "mg/dL→mmol/L": 0.0113},
    "低密度脂蛋白": {"mmol/L→mg/dL": 38.67, "mg/dL→mmol/L": 0.0259},
    "高密度脂蛋白": {"mmol/L→mg/dL": 38.67, "mg/dL→mmol/L": 0.0259},
    "肌酐":     {"μmol/L→mg/dL": 0.01131, "mg/dL→μmol/L": 88.4, "umol/L→mg/dL": 0.01131},
    "尿素":     {"mmol/L→mg/dL": 6.01,   "mg/dL→mmol/L": 0.1665},
    "尿酸":     {"μmol/L→mg/dL": 0.01681, "mg/dL→μmol/L": 59.48, "umol/L→mg/dL": 0.01681},
    "总胆红素": {"μmol/L→mg/dL": 0.05847, "mg/dL→μmol/L": 17.1, "umol/L→mg/dL": 0.05847},
    "钙":       {"mmol/L→mg/dL": 4.0,    "mg/dL→mmol/L": 0.25},
}


def _tool_convert_unit(value: float, from_unit: str, to_unit: str, test_type: str = "") -> dict:
    """医学检验单位换算"""
    key = f"{from_unit}→{to_unit}"
    # 尝试精确匹配
    if test_type and test_type in _UNIT_FACTORS and key in _UNIT_FACTORS[test_type]:
        factor = _UNIT_FACTORS[test_type][key]
        converted = round(value * factor, 4)
        return {
            "original": f"{value} {from_unit}",
            "converted": f"{converted} {to_unit}",
            "factor": factor,
            "test_type": test_type,
        }
    # 模糊匹配 test_type
    for tname, factors in _UNIT_FACTORS.items():
        if test_type and test_type in tname and key in factors:
            factor = factors[key]
            converted = round(value * factor, 4)
            return {
                "original": f"{value} {from_unit}",
                "converted": f"{converted} {to_unit}",
                "factor": factor,
                "test_type": tname,
            }
    return {"error": f"不支持 {test_type} 的 {key} 换算，请检查单位或项目名称"}


# ============================================================
# 工具 3: lookup_reference — 参考范围查询
# ============================================================

_REFERENCE_DB = {
    "白细胞":       {"range": "3.5-9.5", "unit": "×10⁹/L", "note": "感染/炎症时升高，血液病时异常"},
    "红细胞":       {"range": "男 4.3-5.8 / 女 3.8-5.1", "unit": "×10¹²/L", "note": "贫血时偏低"},
    "血红蛋白":     {"range": "男 130-175 / 女 115-150", "unit": "g/L", "note": "贫血/脱水评估核心指标"},
    "血小板":       {"range": "125-350", "unit": "×10⁹/L", "note": "止血功能，偏低有出血风险"},
    "中性粒细胞":   {"range": "1.8-6.3", "unit": "×10⁹/L", "note": "细菌感染时升高"},
    "淋巴细胞":     {"range": "1.1-3.2", "unit": "×10⁹/L", "note": "病毒感染时升高"},
    "空腹血糖":     {"range": "3.9-6.1", "unit": "mmol/L", "note": "≥7.0 需警惕糖尿病"},
    "糖化血红蛋白": {"range": "4.0-6.0", "unit": "%", "note": "反映近3月平均血糖，≥6.5 提示糖尿病"},
    "总胆固醇":     {"range": "<5.2", "unit": "mmol/L", "note": "心血管风险指标"},
    "甘油三酯":     {"range": "<1.7", "unit": "mmol/L", "note": "≥2.3 为高甘油三酯血症"},
    "低密度脂蛋白": {"range": "<3.4", "unit": "mmol/L", "note": "'坏胆固醇'，越低越好"},
    "高密度脂蛋白": {"range": ">1.0", "unit": "mmol/L", "note": "'好胆固醇'，越高越好"},
    "谷丙转氨酶":   {"range": "7-40", "unit": "U/L", "note": "肝细胞损伤敏感指标"},
    "谷草转氨酶":   {"range": "13-35", "unit": "U/L", "note": "肝/心肌损伤指标"},
    "总胆红素":     {"range": "3.4-17.1", "unit": "μmol/L", "note": "黄疸评估指标"},
    "直接胆红素":   {"range": "0-6.8", "unit": "μmol/L", "note": "胆道梗阻时升高"},
    "肌酐":         {"range": "男 57-111 / 女 41-81", "unit": "μmol/L", "note": "肾功能核心指标"},
    "尿素":         {"range": "2.6-7.5", "unit": "mmol/L", "note": "肾功能+蛋白质代谢指标"},
    "尿酸":         {"range": "男 208-428 / 女 155-357", "unit": "μmol/L", "note": "痛风风险指标"},
    "总蛋白":       {"range": "65-85", "unit": "g/L", "note": "营养+肝功能指标"},
    "白蛋白":       {"range": "40-55", "unit": "g/L", "note": "营养状态+肝功能"},
    "钙":           {"range": "2.1-2.6", "unit": "mmol/L", "note": "骨代谢/神经肌肉功能"},
    "钾":           {"range": "3.5-5.3", "unit": "mmol/L", "note": "心律相关，危急值需紧急处理"},
    "钠":           {"range": "137-147", "unit": "mmol/L", "note": "水电解质平衡"},
    "尿蛋白":       {"range": "阴性", "unit": "", "note": "阳性提示肾损伤"},
    "尿糖":         {"range": "阴性", "unit": "", "note": "阳性提示血糖过高"},
    "血沉":         {"range": "男 <15 / 女 <20", "unit": "mm/h", "note": "炎症/肿瘤筛查"},
    "C反应蛋白":    {"range": "<10", "unit": "mg/L", "note": "急性炎症指标"},
}


def _tool_lookup_reference(test_name: str, gender: str = "", age: int = 0) -> dict:
    """查询检验项目标准参考范围"""
    # 精确匹配
    if test_name in _REFERENCE_DB:
        ref = _REFERENCE_DB[test_name]
        result = dict(ref)
        result["test_name"] = test_name
        return result
    # 模糊匹配
    for key, val in _REFERENCE_DB.items():
        if test_name in key or key in test_name:
            result = dict(val)
            result["test_name"] = key
            return result
    return {"error": f"未找到 '{test_name}' 的参考范围，数据库暂未收录此项"}


# ============================================================
# 工具 4: assess_severity — 偏离度量化评估
# ============================================================

def _tool_assess_severity(value: float, ref_low: float, ref_high: float, direction: str = "") -> dict:
    """
    量化评估指标偏离参考范围的程度。
    direction: "偏高" / "偏低" / ""（自动判断）
    """
    # 自动判断方向
    if not direction:
        if value > ref_high:
            direction = "偏高"
        elif value < ref_low:
            direction = "偏低"
        else:
            direction = "正常"

    if direction == "偏高" and ref_high > 0:
        deviation_pct = ((value - ref_high) / ref_high) * 100
    elif direction == "偏低" and ref_low > 0:
        deviation_pct = ((ref_low - value) / ref_low) * 100
    else:
        deviation_pct = 0.0

    deviation_pct = round(deviation_pct, 1)

    # 分级
    if direction == "正常":
        severity = "正常"
    elif deviation_pct < 20:
        severity = "注意"
    elif deviation_pct < 50:
        severity = "关注"
    else:
        severity = "警惕"

    return {
        "value": value,
        "reference": f"{ref_low}-{ref_high}",
        "direction": direction,
        "deviation_pct": deviation_pct,
        "severity": severity,
        "criteria": "注意(<20%) / 关注(20-50%) / 警惕(>50%)",
    }


# ============================================================
# 工具 Schema 定义（OpenAI Function Calling 格式）
# ============================================================

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "calculate_egfr",
            "description": "根据血肌酐值计算估算肾小球滤过率(eGFR)，用于评估肾功能。当报告中包含肌酐异常时建议调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "creatinine_umol": {"type": "number", "description": "血肌酐值（μmol/L）"},
                    "age": {"type": "number", "description": "患者年龄"},
                    "gender": {"type": "string", "enum": ["男", "女"], "description": "患者性别"},
                },
                "required": ["creatinine_umol", "age", "gender"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "convert_unit",
            "description": "医学检验单位换算。支持血糖、胆固醇、甘油三酯、肌酐、尿酸、胆红素等常见项目的单位转换。",
            "parameters": {
                "type": "object",
                "properties": {
                    "value": {"type": "number", "description": "原始数值"},
                    "from_unit": {"type": "string", "description": "原始单位，如 mmol/L、mg/dL、μmol/L"},
                    "to_unit": {"type": "string", "description": "目标单位"},
                    "test_type": {"type": "string", "description": "检测项目名称（如 血糖、肌酐），用于确定换算因子"},
                },
                "required": ["value", "from_unit", "to_unit", "test_type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_reference",
            "description": "查询医学检验项目的标准参考范围。当报告中缺少参考范围或需要验证时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "test_name": {"type": "string", "description": "检验项目名称（如 白细胞、血糖、肌酐）"},
                    "gender": {"type": "string", "enum": ["男", "女"], "description": "患者性别（部分指标有性别差异）"},
                    "age": {"type": "integer", "description": "患者年龄"},
                },
                "required": ["test_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "assess_severity",
            "description": "量化评估指标偏离参考范围的程度，返回偏离百分比和严重程度等级。当需要精确判断严重程度时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "value": {"type": "number", "description": "实际检测值"},
                    "ref_low": {"type": "number", "description": "参考范围下限"},
                    "ref_high": {"type": "number", "description": "参考范围上限"},
                    "direction": {"type": "string", "enum": ["偏高", "偏低", ""], "description": "偏离方向，留空则自动判断"},
                },
                "required": ["value", "ref_low", "ref_high"],
            },
        },
    },
]

# 工具名 → 执行函数 的映射
TOOL_EXECUTORS = {
    "calculate_egfr": _tool_calculate_egfr,
    "convert_unit": _tool_convert_unit,
    "lookup_reference": _tool_lookup_reference,
    "assess_severity": _tool_assess_severity,
}


def _execute_tool(tool_name: str, arguments: dict) -> dict:
    """执行工具调用，返回结果 dict"""
    func = TOOL_EXECUTORS.get(tool_name)
    if not func:
        return {"error": f"未知工具: {tool_name}"}
    try:
        return func(**arguments)
    except Exception as e:
        return {"error": f"工具执行失败: {e}"}


# ============================================================
# Agent System Prompt
# ============================================================

AGENT_SYSTEM_PROMPT = """你是一个医疗体检报告解读 Agent，具备自主推理和工具调用能力。

## 你的工作流程（ReAct 模式）
1. **观察**：阅读体检报告文本，识别所有检测指标
2. **推理**：判断哪些指标异常，哪些需要工具辅助分析
3. **行动**：主动调用可用工具进行精确计算/查询
4. **观察**：接收工具返回结果，整合到分析中
5. **重复**：继续推理和调用工具，直到分析完整
6. **输出**：生成最终结构化 JSON 报告

## 可用工具
- calculate_egfr: 根据血肌酐计算 eGFR（肾功能评估）。当报告含肌酐异常时**必须调用**。
- convert_unit: 医学单位换算。当需要标准单位对比时调用。
- lookup_reference: 查询参考范围。当报告缺少参考范围或需验证时调用。
- assess_severity: 量化偏离程度。对每个异常指标**建议调用**以精确分级。

## 关键规则（医疗合规）
1. 严格基于报告数据，**绝对禁止**编造既往病史、用药史、家族史
2. 不做确诊——使用"可能存在""或可伴随"等不确定表述
3. 严禁"说明有""确诊""证明"等绝对诊断词语
4. 有工具可用时**主动使用**，不要靠猜测
5. 如果报告中有肌酐值，**必须**调用 calculate_egfr 评估肾功能

## 最终输出格式（严格遵守）

**重要：你的最终回复必须是一个完整的、可被 JSON.parse() 直接解析的 JSON 对象。**
- 绝对不要输出 markdown 代码块（不要用 ```json 包裹）
- 绝对不要输出 `` 推理过程
- 绝对不要在 JSON 前后加任何解释文字
- 第一个字符必须是 `{`，最后一个字符必须是 `}`

输出 schema：
{
  "report_type": "报告类型描述",
  "整体术语翻译": {"原词": "大白话解释"},
  "异常指标": [
    {
      "指标": "名称",
      "结果": "检测值（含单位）",
      "正常范围": "参考区间",
      "状态": "偏高/偏低",
      "严重程度": "注意/关注/警惕",
      "通俗解释": "大白话解释",
      "相关术语": [{"原词": "词汇", "解释": "解释"}],
      "复查建议": "建议X个月复查",
      "就医提醒": "与该指标直接相关的就医提醒（只写1条最相关的，没有则写空字符串\"\"）",
      "tool_enhanced": "如有工具辅助分析，注明用了什么工具和结论"
    }
  ],
  "交叉关联分析": "多个异常指标的关联风险（无关联则写'各项异常相对独立'）",
  "工具分析结果": [{"tool": "工具名", "input": "输入", "output": "关键结论"}],
  "生活建议": ["具体可操作的建议"],
  "就医提醒": ["所有需就医提醒的汇总，每条注明科室"],
  "自检报告": {
    "passed": true/false,
    "missing": ["遗漏的异常指标"],
    "compliance_issues": ["不合规表述"],
    "summary": "质量评价"
  }
}

## 重要输出规则
1. 每个异常指标的"就医提醒"字段必须**只包含与该指标直接相关的提醒**（如维生素D低只写维生素D/骨科/内分泌相关，不要写尿酸或肾内科）。
2. 若该指标无需就医，写空字符串 ""，不要写无关内容。
3. "相关术语"字段只放与该指标相关的术语，不要放所有术语。
4. 全局"就医提醒"汇总所有需要就医的提醒，用于结果页底部统一展示。

## 严重程度分级
- 注意：轻度偏离（<20%），居家调理
- 关注：明显偏离（20%-50%），生活干预+择期复查
- 警惕：严重偏离（>50%）或涉及关键器官，尽快就医
"""


# ============================================================
# ReAct Agent 主循环
# ============================================================

def interpret_report_agent(report_text: str, on_step=None, on_token=None) -> dict:
    """
    ReAct Agent：LLM 自主推理 + 工具调用循环。

    Args:
        report_text: 体检报告文本
        on_step: 回调函数 on_step(phase, description, detail)
        on_token: 流式回调 on_token(text)，实时回传 LLM 生成的文本片段

    Returns:
        与 app.py 兼容的结构化 dict，额外包含:
        - agent_steps: 推理过程记录
        - tool_calls: 工具调用详情列表
    """
    steps_log = []
    tool_calls_log = []
    phase = 0

    def _log(desc, detail):
        nonlocal phase
        phase += 1
        entry = {"phase": phase, "desc": desc, "detail": detail}
        steps_log.append(entry)
        if on_step:
            on_step(phase, desc, detail)

    # 初始化消息
    messages = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": f"请解读以下体检报告：\n\n---\n{report_text[:4000]}\n---"},
    ]

    _log("🧠 Agent 启动", "开始 ReAct 推理循环（最多 8 轮）")

    max_iterations = 8
    final_content = None

    for iteration in range(max_iterations):
        _log(f"🔄 第 {iteration + 1} 轮推理", "LLM 正在分析并决定下一步行动...")

        # 调用 LLM（带工具）
        resp = _call_llm_with_tools_stream(messages, TOOL_SCHEMAS, on_token=on_token, temperature=0.3)

        # 检查 API 错误
        if resp.get("finish_reason") and "ERROR" in str(resp.get("content", "")):
            _log("❌ API 错误", resp["content"][:200])
            return {"error": f"API 调用失败: {resp['content'][:100]}"}

        tool_calls = resp.get("tool_calls")
        content = resp.get("content")

        # 情况 1：LLM 决定调用工具
        if tool_calls:
            # 把 assistant 的工具调用消息加入历史
            messages.append({
                "role": "assistant",
                "content": content,
                "tool_calls": [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": tc["function"],
                    }
                    for tc in tool_calls
                ],
            })

            # 逐个执行工具
            for tc in tool_calls:
                tool_name = tc["function"]["name"]
                try:
                    tool_args = json.loads(tc["function"]["arguments"])
                except json.JSONDecodeError:
                    tool_args = {}

                _log(f"🔧 调用工具: {tool_name}", f"参数: {json.dumps(tool_args, ensure_ascii=False)}")

                result = _execute_tool(tool_name, tool_args)

                # 记录工具调用
                tool_calls_log.append({
                    "tool": tool_name,
                    "args": tool_args,
                    "result": result,
                })

                result_str = json.dumps(result, ensure_ascii=False)
                _log(f"📋 工具返回: {tool_name}", result_str[:200])

                # 把工具结果喂回给 LLM
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result_str,
                })

            # 继续下一轮推理
            continue

        # 情况 2：LLM 输出最终结果（无工具调用）
        if content:
            final_content = content
            _log("✅ Agent 输出最终结果", "正在解析结构化 JSON...")
            break

    else:
        # 达到最大迭代次数
        _log("⚠️ 达到最大推理轮数", f"已执行 {max_iterations} 轮，使用最后一轮结果")
        if not final_content and messages[-1].get("role") == "tool":
            # 强制再调一次让 LLM 总结
            messages.append({"role": "user", "content": "请基于以上工具分析结果，输出最终的 JSON 报告。"})
            resp = _call_llm_with_tools_stream(messages, [], on_token=on_token, temperature=0.3)
            final_content = resp.get("content", "")

    if not final_content:
        return {"error": "Agent 未能生成结果，请重试"}

    # 解析最终 JSON（多轮兜底）
    data = _safe_json_parse(final_content)
    if not data:
        _log("⚠️ JSON 解析失败", "尝试让 LLM 重新输出纯 JSON...")
        messages.append({
            "role": "user",
            "content": "你刚才的输出无法被 JSON.parse 解析。请**只输出一个完整的 JSON 对象**，"
                       "不要 markdown 代码块，不要 thinking 标签，不要任何解释文字。"
                       "第一个字符必须是 {，最后一个必须是 }。直接开始输出。",
        })
        for attempt in range(2):
            resp2 = _call_llm_with_tools_stream(messages, [], on_token=on_token, temperature=0.2)
            retry_content = resp2.get("content", "")
            data = _safe_json_parse(retry_content)
            if data:
                final_content = retry_content
                _log("✅ 重新解析成功", f"第 {attempt + 1} 次重试成功")
                break
            else:
                _log(f"❌ 第 {attempt + 1} 次重试仍失败", "继续尝试...")
                messages.append({
                    "role": "user",
                    "content": "仍然解析失败。请确保只输出纯 JSON 对象，不要任何其他文字。",
                })

    if not data:
        # 最终兜底：返回一个可读的错误结果，并保留完整 raw 供调试
        return {
            "error": "Agent 输出解析失败，请重试或简化报告内容",
            "raw": final_content[:3000],
            "global_terms": {},
            "items": [],
            "advice": [],
            "alerts": ["AI 返回格式异常，请稍后重试或检查网络/API 状态"],
            "cross_analysis": "",
            "check_report": {"passed": False, "summary": "输出解析失败"},
            "report_type": "体检报告",
            "tool_analysis": [],
            "agent_steps": steps_log,
            "tool_calls": tool_calls_log,
        }

    # 转换为 app.py 兼容格式
    items = data.get("异常指标", [])
    for item in items:
        if "相关术语" not in item:
            item["相关术语"] = []
        if "就医提醒" not in item:
            item["就医提醒"] = ""

    result = {
        "global_terms": data.get("整体术语翻译", {}),
        "items": items,
        "advice": data.get("生活建议", []),
        "alerts": data.get("就医提醒", []),
        "cross_analysis": data.get("交叉关联分析", ""),
        "check_report": data.get("自检报告", {"passed": True, "summary": "Agent 自检完成"}),
        "report_type": data.get("report_type", "体检报告"),
        "tool_analysis": data.get("工具分析结果", []),
        "agent_steps": steps_log,
        "tool_calls": tool_calls_log,
    }

    tool_count = len(tool_calls_log)
    _log("🎉 完成", f"共 {len(steps_log)} 步推理，调用 {tool_count} 次工具")

    return result


# ============================================================
# 多轮追问对话（保持不变）
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
    """
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
        messages.extend(chat_history[-6:])
    messages.append({"role": "user", "content": question})

    return _call_llm(messages, temperature=0.4)
