"""
core.py — 体检报告解读核心逻辑
复用自医疗AI报告评估项目的 ask_ai() 模式
"""

import requests
import json
import re
from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL


def ask_ai(prompt: str, temperature: float = 0.3) -> str:
    """
    向 DeepSeek 提问，返回答案文本。
    temperature=0.3：体检解读需要严谨，输出更稳定。
    """
    response = requests.post(
        DEEPSEEK_BASE_URL,
        headers={
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": DEEPSEEK_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        },
        timeout=60,
    )

    if response.status_code != 200:
        return f"[API 调用失败：{response.status_code}] {response.text[:200]}"

    data = response.json()
    return data["choices"][0]["message"]["content"]


def _validate_input(text: str) -> str | None:
    """
    验证输入文本是否包含有效的体检指标信息。
    返回 None 表示验证通过，返回字符串表示错误信息。
    """
    if not text or len(text.strip()) < 10:
        return "文本太短，请粘贴完整的体检报告内容"

    # 检查是否包含常见体检指标关键词
    medical_keywords = [
        "白细胞", "红细胞", "血红蛋白", "血小板", "WBC", "RBC", "Hb", "PLT",
        "胆固醇", "甘油三酯", "低密度", "高密度", "HDL", "LDL",
        "血糖", "糖化", "空腹", "GLU", "HbA1c",
        "转氨酶", "ALT", "AST", "胆红素", "TBIL",
        "肌酐", "尿素", "尿酸", "Cr", "BUN", "UA",
        "尿蛋白", "尿糖", "PRO", "GLU",
        "↑", "↓", "mmol/L", "μmol/L", "×10",
        "正常", "参考", "范围",
    ]
    found = [kw for kw in medical_keywords if kw.lower() in text.lower()]
    if not found:
        return "未识别到有效体检指标，请粘贴完整的体检报告文字"

    return None


def interpret_report(report_text: str) -> dict:
    """
    解读体检报告：

    返回 dict:
      - items: 异常指标列表 [{指标, 结果, 正常范围, 状态, 严重程度, 通俗解释, 相关术语, 复查建议}]
      - global_terms: 全局术语翻译（不限于特定指标的通用术语）
      - advice: 生活建议列表
      - alerts: 就医提醒列表
      - error: 错误信息（仅在失败时存在）
    """
    # 输入验证
    validation_error = _validate_input(report_text)
    if validation_error:
        return {"error": validation_error}

    prompt = f"""你是一位经验丰富的家庭医生。你的任务是帮一位65岁的老年人分析体检报告。

【核心原则（必须严格遵守，违反将导致严重医疗合规问题）】
1. 严格基于当前提供的报告数值进行分析，绝对禁止推断或编造任何既往病史、用药史、手术史、家族史等报告中没有出现的信息
2. 只描述偏离，不做确诊——使用"可能存在""或可伴随""有一定几率"等不确定表述，严禁使用"说明有""证明""可诊断""肯定是"等绝对诊断词语
3. 正确表述示例："血红蛋白略低于参考区间，存在轻度贫血的可能性，日常可伴随头晕、乏力等表现"
4. 错误表述示例（严禁出现）："说明有轻度贫血""有痛风病史""长期高血压导致"

【输出格式 — 纯 JSON（不加任何前缀说明或 markdown 代码块）】
{{
  "整体术语翻译": {{"原词1": "大白话解释1", "原词2": "大白话解释2"}},
  "异常指标": [
    {{
      "指标": "指标中文名",
      "结果": "检测数值（含单位）",
      "正常范围": "参考区间",
      "状态": "偏高/偏低",
      "严重程度": "注意/关注/警惕",
      "通俗解释": "用大白话解释偏离意味着什么",
      "相关术语": [
        {{"原词": "与该指标相关的专业词汇", "解释": "大白话解释"}}
      ],
      "复查建议": "建议X个月内复查/就医"
    }}
  ],
  "生活建议": ["具体可操作的建议"],
  "就医提醒": ["需要看医生的提醒，注明挂哪个科室"]
}}

【严重程度分级标准】
- 注意：指标轻度偏离参考范围（偏离幅度<20%），居家调理即可
- 关注：指标明显偏离（偏离幅度20%-50%），建议生活干预+择期复查
- 警惕：指标严重偏离（偏离幅度>50%）或涉及关键器官，建议尽快就医

【相关术语填写规则（解决术语串项问题）】
- 每条指标的"相关术语"数组里，只放与该指标直接相关的专业词汇
- 例如：高尿酸指标的卡片里放{{"原词":"嘌呤","解释":"食物里的一种成分"}}，不要把肌酐相关的术语放进去
- 如果该指标没有需要解释的专业术语，写空数组 []

【复查建议】
- 警惕 → "建议1个月内就医复查"
- 关注 → "建议3个月内复查"
- 注意 → "建议半年内常规体检复查"

【术语翻译要求】
- "整体术语翻译"放通用的、不限于某个具体指标的专业词汇
- 用老年人能听懂的话，如"血小板就像身体的创可贴，出血时帮忙止血"

【生活建议要求】
- 接地气可操作，如"少吃肥肉""晚饭后快走20分钟""每天一个鸡蛋"
- 不要"适当运动""注意饮食"这种废话

【强制要求】
- 异常指标必须覆盖报告中所有带 ↑ 或 ↓ 的指标，不能遗漏
- 输出必须是纯 JSON，不要有 ```json ``` 等代码块标记
- 不要输出任何解释性文字在 JSON 之外

以下是体检报告内容：
---
{report_text[:3000]}
---

请直接输出 JSON："""

    result = ask_ai(prompt, temperature=0.3)
    return _parse_result(result)


def _parse_result(raw: str) -> dict:
    """从 AI 返回的文本中提取 JSON 并解析"""
    text = raw.strip()

    # 去掉 markdown 代码块标记
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            try:
                data = json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                return {"error": "解读失败，请检查文本格式后重试", "raw": raw[:500]}
        else:
            return {"error": "解读失败，请检查文本格式后重试", "raw": raw[:500]}

    return _normalize(data)


def _normalize(data: dict) -> dict:
    """统一输出格式，兼容 AI 可能返回的不同字段名"""
    if not isinstance(data, dict):
        return {"global_terms": {}, "items": [], "advice": [], "alerts": []}

    # 整体术语翻译
    global_terms = (
        data.get("整体术语翻译")
        or data.get("术语翻译")
        or data.get("terms")
        or _find_key_by_keywords(data, ["整体术语", "术语", "term"])
        or {}
    )
    if isinstance(global_terms, list):
        global_terms = {item.get("原词", ""): item.get("解释", item.get("大白话解释", ""))
                        for item in global_terms}
    if not isinstance(global_terms, dict):
        global_terms = {}

    # 异常指标
    items = (
        data.get("异常指标")
        or data.get("异常指标分析")
        or data.get("items")
        or data.get("abnormal_items")
        or _find_key_by_keywords(data, ["异常", "指标", "abnormal"])
        or []
    )
    if not isinstance(items, list):
        items = []

    # 确保每个 item 的 相关术语 是列表
    for item in items:
        if "相关术语" not in item:
            item["相关术语"] = []
        if not isinstance(item["相关术语"], list):
            item["相关术语"] = []

    # 生活建议
    advice = (
        data.get("生活建议")
        or data.get("advice")
        or data.get("suggestions")
        or data.get("recommendations")
        or _find_key_by_keywords(data, ["建议", "生活", "advice", "suggestion"])
        or []
    )
    if not isinstance(advice, list):
        advice = []

    # 就医提醒
    alerts = (
        data.get("就医提醒")
        or data.get("alerts")
        or data.get("就医建议")
        or data.get("medical_alerts")
        or _find_key_by_keywords(data, ["就医", "提醒", "alert", "doctor"])
        or []
    )
    if not isinstance(alerts, list):
        alerts = []

    return {
        "global_terms": global_terms,
        "items": items,
        "advice": advice,
        "alerts": alerts,
    }


def _find_key_by_keywords(data: dict, keywords: list) -> any:
    """根据关键词模糊匹配字典里的键"""
    for key in data.keys():
        key_lower = key.lower()
        for kw in keywords:
            if kw.lower() in key_lower:
                return data[key]
    return None
