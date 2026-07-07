"""
app.py — 体检报告解读助手（Streamlit Web 版）V2

支持：PDF 文件上传 / 图片OCR识别 / 文字粘贴
新增：隐私脱敏、合规话术、术语-指标绑定、风险分级卡片

启动方式：
  cd /Users/my/Desktop/体检报告解读助手
  bash run.sh
"""

import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
import pandas as pd
import json
import tempfile
import os
import re
import html as html_mod
from datetime import datetime
from core import interpret_report
from agent import interpret_report_agent, ask_followup


# ============================================================
# 工具函数
# ============================================================

SEVERITY_STYLES = {
    "警惕": {"emoji": "⚠️", "tag": "⚠️ 需警惕", "color": "#c62828", "bg": "#fff5f5", "border": "#e53935",
             "bg_card": "#fff0f0", "label": "建议尽快就医", "review": "建议1个月内就医复查", "icon": "⚠️"},
    "关注": {"emoji": "◆", "tag": "◆ 需关注", "color": "#e65100", "bg": "#fff8f0", "border": "#ef6c00",
             "bg_card": "#fff8f0", "label": "生活干预+复查", "review": "建议3个月内复查", "icon": "◆"},
    "注意": {"emoji": "●", "tag": "● 需注意", "color": "#0d47a1", "bg": "#f5f9ff", "border": "#1e88e5",
             "bg_card": "#f5f9ff", "label": "居家调理", "review": "建议半年内常规体检复查", "icon": "●"},
}

SEVERITY_ORDER = {"警惕": 0, "关注": 1, "注意": 2}


def _severity_style(level: str) -> dict:
    return SEVERITY_STYLES.get(level, SEVERITY_STYLES["注意"])


def mask_sensitive_info(text: str) -> str:
    """
    脱敏处理：遮蔽身份证号、手机号等隐私信息。
    保留姓名前缀供识别，但用 *** 替换后半部分。
    """
    if not text:
        return text

    # 中国居民身份证号 (18位: 6位地区码 + 4位年份 + 4位月日 + 3位顺序 + 1位校验)
    text = re.sub(
        r'\b\d{6}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx]\b',
        '身份证号[已隐藏]',
        text,
    )

    # 手机号 (11位，以1开头)
    text = re.sub(
        r'\b1[3-9]\d{9}\b',
        '手机号[已隐藏]',
        text,
    )

    return text


def _generate_export_text(result: dict, include_header: bool = True) -> str:
    """生成可导出的纯文本解读报告"""
    lines = []
    if include_header:
        lines = [
            "=" * 50,
            "体检报告解读结果",
            f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "=" * 50, "",
        ]

    for item in result.get("items", []):
        s = _severity_style(item.get("严重程度", "注意"))
        lines.append(
            f"{s['tag']} {item.get('指标', '?')}: {item.get('结果', '?')} "
            f"(正常：{item.get('正常范围', '?')}) {item.get('状态', '')}"
        )
        lines.append(f"  → {item.get('通俗解释', '')}")

        # 相关术语
        related_terms = item.get("相关术语", [])
        for t in related_terms:
            lines.append(f"    术语：{t.get('原词', '?')} → {t.get('解释', '?')}")

        review = item.get("复查建议", s.get("review", ""))
        if review:
            lines.append(f"  → 复查建议：{review}")
        lines.append("")

    global_terms = result.get("global_terms", {})
    if global_terms:
        lines.append("【通用术语解释】")
        for k, v in global_terms.items():
            lines.append(f"  {k}：{v}")
        lines.append("")

    for i, tip in enumerate(result.get("advice", []), 1):
        lines.append(f"  {i}. {tip}")

    if result.get("alerts"):
        lines.append("")
        lines.append("【就医提醒】")
        for alert in result["alerts"]:
            lines.append(f"  🏥 {alert}")

    if include_header:
        lines += ["", "=" * 50,
                  "本解读由 AI 生成，仅供参考，不构成医疗建议。",
                  "如有疑问，请咨询专业医生。"]
    return "\n".join(lines)


def _esc(text: str) -> str:
    """HTML 转义，防止 AI 输出内容破坏 HTML 结构"""
    return html_mod.escape(str(text)) if text else ""


def _find_in_original_text(indicator_name: str, report_text: str, context: int = 60) -> str | None:
    """
    在原文中查找指标对应位置，返回上下文片段。
    尝试多种匹配方式：全名 → 中文部分 → 英文缩写。
    """
    if not indicator_name or not report_text:
        return None

    search_terms = [indicator_name]
    # 提取括号内英文缩写
    m = re.search(r'[\(（]([A-Za-z][A-Za-z0-9\-/]+)[\)）]', indicator_name)
    if m:
        search_terms.append(m.group(1))
    # 提取中文部分
    cn = re.sub(r'[\(（].*?[\)）]', '', indicator_name).strip()
    if cn and cn != indicator_name:
        search_terms.append(cn)

    for term in search_terms:
        idx = report_text.find(term)
        if idx != -1:
            start = max(0, idx - context)
            end = min(len(report_text), idx + len(term) + context)
            snippet = report_text[start:end].replace('\n', ' ')
            prefix = "…" if start > 0 else ""
            suffix = "…" if end < len(report_text) else ""
            return f"{prefix}{snippet}{suffix}"
    return None


def _generate_share_card_html(result: dict, items: list, advice: list, alerts: list) -> str:
    """生成分享图卡片 HTML（供 html2canvas 截图下载）"""
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    alert_count = sum(1 for i in items if i.get("严重程度") == "警惕")
    warn_count = sum(1 for i in items if i.get("严重程度") == "关注")
    note_count = sum(1 for i in items if i.get("严重程度") == "注意")

    color_map = {"警惕": "#e53935", "关注": "#ef6c00", "注意": "#1e88e5"}
    bg_map = {"警惕": "#fff5f5", "关注": "#fff8f0", "注意": "#f5f9ff"}
    tag_map = {"警惕": "⚠️ 需警惕", "关注": "◆ 需关注", "注意": "● 需注意"}

    # 异常指标卡片（最多 5 条）
    findings_html = ""
    for item in items[:5]:
        sev = item.get("严重程度", "注意")
        color = color_map.get(sev, "#1e88e5")
        bg = bg_map.get(sev, "#f5f9ff")
        tag = tag_map.get(sev, "● 需注意")
        findings_html += f"""
        <div style="background:{bg};border-left:5px solid {color};border-radius:8px;padding:14px 16px;margin-bottom:10px;">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
                <span style="font-size:18px;font-weight:700;color:#333;">{_esc(item.get('指标', '?'))}</span>
                <span style="font-size:13px;font-weight:700;color:{color};background:#ffffff;padding:2px 10px;border-radius:4px;">{tag}</span>
            </div>
            <div style="font-size:15px;color:#555;margin-bottom:4px;">
                检测值：<b style="color:{color};">{_esc(item.get('结果', '?'))}</b>
                <span style="color:#999;margin-left:10px;">正常：{_esc(item.get('正常范围', '?'))}</span>
                <span style="color:{color};font-weight:600;margin-left:10px;">{_esc(item.get('状态', ''))}</span>
            </div>
            <div style="font-size:15px;color:#444;line-height:1.6;">{_esc(item.get('通俗解释', ''))}</div>
        </div>"""

    # 生活建议（最多 3 条）
    advice_html = ""
    if advice:
        advice_html = '<div style="margin-top:16px;"><div style="font-size:16px;font-weight:700;color:#1a73e8;margin-bottom:8px;">💡 生活建议</div>'
        for i, tip in enumerate(advice[:3], 1):
            advice_html += f'<div style="font-size:15px;color:#444;line-height:1.8;padding:4px 0;">{i}. {_esc(tip)}</div>'
        advice_html += '</div>'

    # 就医提醒（最多 2 条）
    alerts_html = ""
    if alerts:
        alerts_html = '<div style="margin-top:12px;padding:12px 16px;background:#fff3e0;border-radius:8px;border:1px solid #ffb74d;">'
        alerts_html += '<div style="font-size:16px;font-weight:700;color:#e65100;margin-bottom:6px;">🏥 就医提醒</div>'
        for a in alerts[:2]:
            alerts_html += f'<div style="font-size:15px;color:#555;line-height:1.6;">· {_esc(a)}</div>'
        alerts_html += '</div>'

    return f"""
    <div id="share-card" style="width:560px;padding:32px;background:#ffffff;font-family:-apple-system,BlinkMacSystemFont,'PingFang SC','Microsoft YaHei',sans-serif;">
        <div style="background:linear-gradient(135deg,#1a73e8,#4fc3f7);border-radius:12px;padding:20px 24px;margin-bottom:20px;color:#ffffff;">
            <div style="font-size:24px;font-weight:700;">🩺 体检报告解读摘要</div>
            <div style="font-size:14px;opacity:0.9;margin-top:4px;">生成时间：{now_str} · 共 {len(items)} 项异常</div>
        </div>
        <div style="display:flex;gap:12px;margin-bottom:20px;">
            <div style="flex:1;background:#fff5f5;border:2px solid #e53935;border-radius:10px;padding:12px;text-align:center;">
                <div style="font-size:28px;font-weight:700;color:#c62828;">{alert_count}</div>
                <div style="font-size:13px;color:#c62828;">⚠️ 需警惕</div>
            </div>
            <div style="flex:1;background:#fff8f0;border:2px solid #ef6c00;border-radius:10px;padding:12px;text-align:center;">
                <div style="font-size:28px;font-weight:700;color:#e65100;">{warn_count}</div>
                <div style="font-size:13px;color:#e65100;">◆ 需关注</div>
            </div>
            <div style="flex:1;background:#f5f9ff;border:2px solid #1e88e5;border-radius:10px;padding:12px;text-align:center;">
                <div style="font-size:28px;font-weight:700;color:#0d47a1;">{note_count}</div>
                <div style="font-size:13px;color:#0d47a1;">● 需注意</div>
            </div>
        </div>
        <div style="font-size:16px;font-weight:700;color:#333;margin-bottom:10px;">📋 异常指标解读</div>
        {findings_html}
        {advice_html}
        {alerts_html}
        <div style="margin-top:20px;padding-top:14px;border-top:1px solid #eeeeee;text-align:center;">
            <div style="font-size:12px;color:#bbbbbb;">🩺 体检报告解读助手 · AI 生成内容仅供参考 · 不构成医疗建议</div>
        </div>
    </div>
    """


# ============================================================
# Session State 初始化
# ============================================================

if "report_text" not in st.session_state:
    st.session_state.report_text = ""
if "result" not in st.session_state:
    st.session_state.result = None
if "analyzing" not in st.session_state:
    st.session_state.analyzing = False
if "font_size" not in st.session_state:
    st.session_state.font_size = 1
if "input_mode" not in st.session_state:
    st.session_state.input_mode = "file"
if "extracted_text" not in st.session_state:
    st.session_state.extracted_text = ""
if "show_raw" not in st.session_state:
    st.session_state.show_raw = False
if "privacy_on" not in st.session_state:
    st.session_state.privacy_on = True  # 默认启用脱敏
if "ocr_char_count" not in st.session_state:
    st.session_state.ocr_char_count = 0
if "risk_filter" not in st.session_state:
    st.session_state.risk_filter = "全部"
if "card_expanded" not in st.session_state:
    st.session_state.card_expanded = {}
if "editing_raw" not in st.session_state:
    st.session_state.editing_raw = False
if "raw_edit_area" not in st.session_state:
    st.session_state.raw_edit_area = ""
if "share_image_data" not in st.session_state:
    st.session_state.share_image_data = None
if "use_agent" not in st.session_state:
    st.session_state.use_agent = True
if "agent_steps_display" not in st.session_state:
    st.session_state.agent_steps_display = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "show_chat" not in st.session_state:
    st.session_state.show_chat = False

FONT_SIZES = {
    0: {"base": "13px", "h": "1.5rem", "card": "13px", "label": "小号"},
    1: {"base": "16px", "h": "2rem", "card": "15px", "label": "标准"},
    2: {"base": "20px", "h": "2.5rem", "card": "18px", "label": "大号"},
}
fs = FONT_SIZES[st.session_state.font_size]

EXAMPLE_REPORT = """体检报告
姓名：张大爷  性别：男  年龄：65岁

血常规：
白细胞计数(WBC)：7.2 ×10^9/L  （正常：3.5-9.5）
红细胞计数(RBC)：4.1 ×10^12/L （正常：4.3-5.8）↓
血红蛋白(Hb)：128 g/L  （正常：130-175）↓
血小板计数(PLT)：215 ×10^9/L  （正常：125-350）

血脂：
总胆固醇(TC)：6.2 mmol/L  （正常：<5.2）↑
甘油三酯(TG)：2.4 mmol/L  （正常：<1.7）↑
低密度脂蛋白(LDL-C)：4.1 mmol/L  （正常：<3.4）↑
高密度脂蛋白(HDL-C)：1.1 mmol/L  （正常：>1.0）

血糖：
空腹血糖(GLU)：7.8 mmol/L  （正常：3.9-6.1）↑
糖化血红蛋白(HbA1c)：7.2%  （正常：<6.0）↑

肝功能：
谷丙转氨酶(ALT)：45 U/L  （正常：9-50）
谷草转氨酶(AST)：38 U/L  （正常：15-40）
总胆红素(TBIL)：18 μmol/L  （正常：3.4-17.1）↑

肾功能：
肌酐(Cr)：95 μmol/L  （正常：44-133）
尿素氮(BUN)：6.5 mmol/L  （正常：3.2-7.1）

尿常规：
尿蛋白(PRO)：+  （正常：阴性）↑
尿糖(GLU)：++  （正常：阴性）↑"""


# ============================================================
# 页面设置
# ============================================================

st.set_page_config(
    page_title="体检报告解读助手",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 自定义样式
# ============================================================

st.markdown(f"""
<style>
    :root {{
        --font-base: {fs['base']};
        --font-h: {fs['h']};
        --font-card: {fs['card']};
    }}
    .stApp {{ font-size: var(--font-base); }}

    /* ----- 固定免责条 ----- */
    .disclaimer-bar {{
        position: sticky; top: 0; z-index: 999;
        background: #fafafa; border-bottom: 1px solid #eee;
        padding: 6px 20px; text-align: center;
        font-size: 12px; color: #999; margin: -1rem -1rem 1rem -1rem;
    }}

    /* ----- 标题区 ----- */
    .main-title {{ font-size: var(--font-h); font-weight: 700; color: #1a73e8; text-align: center; margin-bottom: 0.2rem; }}
    .subtitle {{ text-align: center; color: #888; font-size: var(--font-base); margin-bottom: 1.5rem; }}

    /* ----- 隐私提示 ----- */
    .privacy-notice {{
        background: #e8f5e9; border-radius: 8px; padding: 10px 14px;
        font-size: 13px; color: #2e7d32; display: flex; align-items: center; gap: 6px;
        margin-top: 10px;
    }}

    /* ----- 统计卡片（四列） ----- */
    .stat-card {{
        padding: 14px 10px; border-radius: 10px; text-align: center;
        cursor: pointer; transition: transform 0.15s, box-shadow 0.15s;
    }}
    .stat-card:hover {{ transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,0,0,0.08); }}
    .stat-card.danger {{ background: #fff5f5; border: 2px solid #e53935; }}
    .stat-card.warning {{ background: #fff8f0; border: 2px solid #ef6c00; }}
    .stat-card.info {{ background: #f5f9ff; border: 2px solid #1e88e5; }}
    .stat-card.total {{ background: #f5f5f5; border: 2px solid #9e9e9e; }}
    .stat-card.active {{ box-shadow: 0 0 0 3px rgba(26,115,232,0.3); }}
    .stat-number {{ font-size: 2rem; font-weight: 700; }}
    .stat-danger {{ color: #c62828; }} .stat-warning {{ color: #e65100; }} .stat-info {{ color: #0d47a1; }}

    /* ----- 风险图例 ----- */
    .legend-row {{
        display: flex; gap: 24px; flex-wrap: wrap; margin-top: 10px;
        padding: 8px 12px; background: #fafafa; border-radius: 8px;
    }}
    .legend-item {{ display: flex; align-items: center; gap: 6px; font-size: 12px; color: #555; }}
    .legend-dot {{ width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }}

    /* ----- 指标卡片 ----- */
    .report-card {{
        padding: 16px; margin: 10px 0; border-radius: 10px;
        border-left: 5px solid #ccc; font-size: var(--font-card); line-height: 1.7;
        position: relative;
    }}
    .report-card.danger {{ background: #fff5f5; border: 1px solid #ffcdd2; border-left-width: 5px; border-left-color: #e53935; }}
    .report-card.warning {{ background: #fff8f0; border: 1px solid #ffe0b2; border-left-width: 5px; border-left-color: #ef6c00; }}
    .report-card.info {{ background: #f5f9ff; border: 1px solid #bbdefb; border-left-width: 5px; border-left-color: #1e88e5; }}
    .card-header {{
        display: flex; align-items: center; gap: 8px; margin-bottom: 10px;
        font-weight: 600; font-size: calc(var(--font-card) + 2px);
    }}
    .card-tag {{
        font-size: 11px; font-weight: 700; padding: 3px 10px; border-radius: 4px;
        text-transform: uppercase; letter-spacing: 0.5px;
    }}
    .card-tag.danger {{ background: #ffcdd2; color: #c62828; }}
    .card-tag.warning {{ background: #ffe0b2; color: #e65100; }}
    .card-tag.info {{ background: #bbdefb; color: #0d47a1; }}
    .highlight-section {{
        background: rgba(255,255,255,0.7); padding: 14px; border-radius: 8px;
        margin: 8px 0; font-size: calc(var(--font-card) + 1px);
        border: 1px dashed #ddd; line-height: 1.8;
    }}
    .detail-section {{ padding: 8px 0; border-top: 1px solid #eee; margin-top: 8px; }}
    .term-badge {{
        display: inline-block; background: #f3e5f5; color: #7b1fa2;
        padding: 2px 8px; border-radius: 4px; font-size: 12px; margin: 2px 4px 2px 0;
    }}
    .review-box {{
        background: #e3f2fd; border-left: 3px solid #1a73e8;
        padding: 8px 14px; margin: 8px 0; border-radius: 4px;
        font-size: 13px; color: #1565c0;
    }}
    .card-disclaimer {{ font-size: 11px; color: #bbb; margin-top: 10px; text-align: right; }}

    /* ----- 生活建议 -- */
    .advice-item {{
        padding: 0.6rem 1rem; margin: 0.3rem 0; background: #f5f7fa;
        border-radius: 8px; border-left: 3px solid #1a73e8;
    }}

    /* ----- 原文区 ----- */
    .raw-block {{
        background: #f9f9f9; border: 1px solid #e0e0e0; border-radius: 6px;
        padding: 10px; max-height: 260px; overflow-y: auto;
        font-size: 12px; color: #666; font-family: monospace;
        white-space: pre-wrap; word-break: break-all; line-height: 1.5;
    }}

    /* ----- 上传空状态 ----- */
    .upload-target {{
        border: 2px dashed #ccc; border-radius: 12px; padding: 30px;
        text-align: center; color: #999; transition: border-color 0.2s;
    }}
    .upload-target:hover {{ border-color: #1a73e8; }}

    /* ----- 页脚 ----- */
    .footer {{ text-align: center; color: #ccc; font-size: 0.75rem; margin-top: 3rem; padding-bottom: 1rem; }}

    /* ----- 移动端适配 ----- */
    @media (max-width: 768px) {{
        .stat-card {{ padding: 10px 6px; }} .stat-number {{ font-size: 1.5rem; }}
        .report-card {{ padding: 10px 12px; }}
        .legend-row {{ gap: 8px; flex-direction: column; }}
        button[kind="secondary"] {{ min-height: 48px; }}
        .main-title {{ font-size: 1.6rem; }}
        .subtitle {{ font-size: 13px; }}
        .highlight-section {{ padding: 10px; font-size: 14px; }}
        .card-header {{ flex-wrap: wrap; gap: 4px; }}
        .disclaimer-bar {{ font-size: 11px; padding: 4px 12px; }}
        .raw-block {{ max-height: 180px; font-size: 11px; }}
    }}

    /* ----- 小屏手机适配 ----- */
    @media (max-width: 480px) {{
        .stat-card {{ padding: 8px 4px; }}
        .stat-number {{ font-size: 1.3rem; }}
        .report-card {{ padding: 8px 10px; margin: 6px 0; }}
        .card-tag {{ font-size: 10px; padding: 2px 6px; }}
        .term-badge {{ font-size: 11px; padding: 1px 6px; }}
        .review-box {{ padding: 6px 10px; font-size: 12px; }}
    }}

    /* ----- 滚动条美化 ----- */
    ::-webkit-scrollbar {{ width: 6px; height: 6px; }}
    ::-webkit-scrollbar-track {{ background: transparent; }}
    ::-webkit-scrollbar-thumb {{ background: #ccc; border-radius: 3px; }}
    ::-webkit-scrollbar-thumb:hover {{ background: #aaa; }}

    /* ----- 平滑滚动 ----- */
    html {{ scroll-behavior: smooth; }}

    /* ----- Expander 样式优化 ----- */
    .streamlit-expanderHeader {{ font-size: var(--font-card); font-weight: 600; }}

    /* ----- 按钮焦点态（无障碍） ----- */
    button:focus-visible {{ outline: 2px solid #1a73e8; outline-offset: 2px; }}

    /* ----- 卡片入场动画 ----- */
    .report-card {{ animation: fadeInUp 0.3s ease-out; }}
    @keyframes fadeInUp {{
        from {{ opacity: 0; transform: translateY(8px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
</style>
""", unsafe_allow_html=True)


# ============================================================
# 页面标题区
# ============================================================

st.markdown("""
<div class="disclaimer-bar">
    ⚕ 本工具解读仅为健康科普参考，不能替代执业医师诊断，指标异常请及时线下就医
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🩺 体检报告解读助手</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">上传PDF / 拍照上传 · 帮爸妈看懂体检报告 · 术语翻译 · 异常提醒 · 生活建议</div>',
    unsafe_allow_html=True,
)

# 字体调节
_, font_col = st.columns([6, 1])
with font_col:
    fa, fb, fc = st.columns(3)
    with fa:
        if st.button("A⁻", key="font_small", use_container_width=True,
                     disabled=st.session_state.font_size == 0):
            st.session_state.font_size = max(0, st.session_state.font_size - 1)
            st.rerun()
    with fb:
        st.caption(fs["label"])
    with fc:
        if st.button("A⁺", key="font_large", use_container_width=True,
                     disabled=st.session_state.font_size == 2):
            st.session_state.font_size = min(2, st.session_state.font_size + 1)
            st.rerun()


# ============================================================
# 侧边栏
# ============================================================

with st.sidebar:
    st.header("📋 使用说明")

    st.markdown(f"""
    <div style="font-size:var(--font-base);line-height:1.8;">
        <b>①</b> 粘贴体检报告文字，或上传PDF/照片<br>
        <b>②</b> 点击「开始解读」，AI自动翻译成大白话<br>
        <b>③</b> 查看异常分级、生活建议与就医提醒
    </div>
    """, unsafe_allow_html=True)

    if st.button("📝 加载示例报告", use_container_width=True, key="load_example_sidebar"):
        st.session_state.report_text = EXAMPLE_REPORT
        st.session_state.report_input = EXAMPLE_REPORT
        st.session_state.extracted_text = ""
        st.session_state.last_uploaded_file_id = None
        st.session_state.ocr_char_count = len(EXAMPLE_REPORT)
        st.session_state.show_raw = False
        st.session_state.privacy_on = True
        st.session_state.result = None
        st.session_state.editing_raw = False
        st.session_state.show_share_image = False
        st.session_state.chat_history = []
        st.session_state.agent_steps_display = []
        st.session_state.input_mode = "paste"  # 切换到粘贴模式，让用户能看到示例文本
        st.rerun()

    st.caption("* 仅支持血常规、血脂、血糖、肝肾功能等文本类报告，影像/手写单据识别效果有限")

    # Agent 模式开关
    st.divider()
    st.subheader("🤖 Agent 模式")
    st.session_state.use_agent = st.toggle(
        "启用多步推理",
        value=st.session_state.use_agent,
        help="Agent 模式：4步推理（结构分析→逐项解读→自检纠错→整合输出），更准确但稍慢\n关闭后使用快速模式（单次调用）",
    )
    if st.session_state.use_agent:
        st.caption("✅ 已启用 · 结构分析 → 逐项解读 → 自检纠错 → 整合输出")
    else:
        st.caption("⚡ 快速模式 · 单次调用")

    # 历史解读
    st.divider()
    st.markdown("#### 📜 历史解读")
    st.caption("（存储在本地浏览器，不上传服务器）")

    if "history" not in st.session_state:
        st.session_state.history = []

    if st.session_state.history:
        for i, h in enumerate(st.session_state.history[:3]):
            with st.expander(f"{h.get('time', '未知')}", expanded=False):
                st.markdown(h.get("summary", "暂无摘要"))
                if st.button("加载这条记录", key=f"load_hist_{i}"):
                    st.session_state.report_text = h.get("text", "")
                    st.session_state.report_input = h.get("text", "")
                    st.session_state.extracted_text = ""
                    st.session_state.last_uploaded_file_id = None
                    st.session_state.ocr_char_count = len(h.get("text", ""))
                    st.session_state.show_raw = False
                    st.session_state.result = None
                    st.session_state.editing_raw = False
                    st.session_state.show_share_image = False
                    st.session_state.chat_history = []
                    st.session_state.agent_steps_display = []
                    st.session_state.input_mode = "paste"
                    st.rerun()
        if st.button("清空历史", key="clear_history"):
            st.session_state.history = []
            st.rerun()
    else:
        st.caption("暂无历史解读记录")

    st.divider()
    st.caption("💡 数据仅用于AI解读，不会存储")


# ============================================================
# 双栏布局
# ============================================================

col1, col2 = st.columns([9, 11])


# ============================================================
# 左栏：原文区（P1-1：上传固定顶部 + 原文默认折叠）
# ============================================================

with col1:
    st.markdown("### 📄 体检报告输入")

    # ---- 模式切换 ----
    mode_c1, mode_c2 = st.columns(2)
    with mode_c1:
        file_active = st.session_state.input_mode == "file"
        if st.button("📎 上传文件", use_container_width=True,
                     type="primary" if file_active else "secondary", key="mode_file"):
            st.session_state.input_mode = "file"
            st.rerun()
    with mode_c2:
        paste_active = st.session_state.input_mode == "paste"
        if st.button("✏️ 粘贴文字", use_container_width=True,
                     type="primary" if paste_active else "secondary", key="mode_paste"):
            st.session_state.input_mode = "paste"
            st.rerun()

    st.caption("支持 PDF 文件、手机截图、纸质报告拍照上传，AI 自动提取文字")

    # ================================================================
    # 文件上传模式
    # ================================================================
    if st.session_state.input_mode == "file":
        uploaded_file = st.file_uploader(
            "选择文件",
            type=["pdf", "png", "jpg", "jpeg"],
            label_visibility="collapsed",
            key="file_uploader",
        )

        if uploaded_file is not None:
            file_type = uploaded_file.type
            file_ext = os.path.splitext(uploaded_file.name)[1].lower()

            if "pdf" in file_type or file_ext == ".pdf":
                proc_type = "pdf"
                type_label = "PDF"
            else:
                proc_type = "image"
                type_label = "图片"

            # 文件ID：用于判断是否是同一个文件，避免每次rerun都重复OCR
            file_id = f"{uploaded_file.name}_{uploaded_file.size}"
            is_new_file = st.session_state.get("last_uploaded_file_id") != file_id

            if is_new_file:
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name

                with st.spinner(f"正在识别{type_label}中的文字..."):
                    from ocr_helper import extract_text
                    ocr_result = extract_text(tmp_path, proc_type)

                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

                if "error" in ocr_result:
                    st.error(f"❌ {ocr_result['error']}")
                    extracted = ""
                else:
                    extracted = ocr_result.get("text", "")

                if extracted.strip():
                    st.session_state.extracted_text = extracted
                    st.session_state.report_text = extracted
                    st.session_state.report_input = extracted
                    st.session_state.ocr_char_count = len(extracted)
                    st.session_state.show_raw = False
                    st.session_state.result = None
                    st.session_state.editing_raw = False
                    st.session_state.show_share_image = False
                    st.session_state.chat_history = []
                    st.session_state.agent_steps_display = []
                    st.session_state.last_uploaded_file_id = file_id
            else:
                # 同一文件：复用缓存，不重置result
                extracted = st.session_state.get("extracted_text", "")

            if extracted.strip():
                # 识别状态提示
                st.success(f"✅ 已从{type_label}中识别到 {len(extracted)} 个字符")

                # OCR 准确率提示（加分项）
                target_kw = ["白细胞", "红细胞", "血红蛋白", "胆固醇", "血糖",
                             "转氨酶", "肌酐", "尿酸", "mmol/L", "×10"]
                found_kw = [kw for kw in target_kw if kw in extracted]
                accuracy = min(99, max(70, int(len(found_kw) / max(len(target_kw), 1) * 100)))
                st.caption(f"📊 识别准确率预估：{accuracy}%+ · 如个别数值有误，可切换到「粘贴文字」模式手动修改")

                # 原文查看与编辑（默认折叠，P1-1 + P1-3：支持编辑纠错）
                with st.expander("📝 查看 / 编辑报告原文", expanded=False):
                    if st.session_state.editing_raw:
                        # ---- 编辑模式：显示真实文本供修改 ----
                        st.caption("✏️ 编辑模式 · 可修正 OCR 识别错误，修改后点击保存即可重新解读")
                        if st.session_state.privacy_on:
                            st.info("💡 编辑模式下需显示完整文本，隐私信息暂未脱敏")
                        edited_text = st.text_area(
                            "报告原文（可编辑）",
                            value=st.session_state.report_text,
                            height=220,
                            label_visibility="collapsed",
                            key="raw_edit_area",
                        )
                        ce1, ce2 = st.columns(2)
                        with ce1:
                            if st.button("💾 保存并更新", key="save_edit", use_container_width=True, type="primary"):
                                st.session_state.report_text = edited_text
                                st.session_state.report_input = edited_text
                                st.session_state.ocr_char_count = len(edited_text)
                                st.session_state.editing_raw = False
                                st.session_state.result = None
                                st.session_state.last_uploaded_file_id = None  # 编辑后视为新内容
                                st.toast("✅ 已保存修改，可重新解读", icon="✅")
                                st.rerun()
                        with ce2:
                            if st.button("↩️ 取消", key="cancel_edit", use_container_width=True):
                                st.session_state.editing_raw = False
                                st.rerun()
                    else:
                        # ---- 查看模式 ----
                        display = extracted
                        if st.session_state.privacy_on:
                            display = mask_sensitive_info(display)
                        st.text(display[:3000] + ("..." if len(display) > 3000 else ""))

                        c_e1, c_e2, c_e3 = st.columns(3)
                        with c_e1:
                            privacy_label = "🔒 已脱敏" if st.session_state.privacy_on else "🔓 完整显示"
                            if st.button(privacy_label, key="toggle_privacy_left", use_container_width=True):
                                st.session_state.privacy_on = not st.session_state.privacy_on
                                st.rerun()
                        with c_e2:
                            if st.button("✏️ 编辑修正", key="start_edit", use_container_width=True,
                                         help="OCR 识别可能有误，点击此处手动修正"):
                                st.session_state.editing_raw = True
                                st.session_state.raw_edit_area = st.session_state.report_text
                                st.rerun()
                        with c_e3:
                            if st.button("📋 切到粘贴模式", key="switch_to_paste", use_container_width=True):
                                st.session_state.input_mode = "paste"
                                st.rerun()
            elif not is_new_file and not extracted.strip():
                # 同一文件但之前OCR失败/无文字
                st.warning(f"⚠️ {type_label}中未识别到文字内容，请尝试粘贴模式")
            elif is_new_file and not extracted.strip():
                st.warning(f"⚠️ {type_label}中未识别到文字内容，请尝试粘贴模式")

        else:
            # 空状态
            st.markdown("""
            <div class="upload-target">
                <div style="font-size:40px;margin-bottom:10px;">📎</div>
                <div style="font-size:16px;margin-bottom:6px;">点击上方选择文件上传</div>
                <div style="font-size:13px;color:#bbb;">支持 PDF / PNG / JPG，最大 200MB</div>
                <div style="font-size:13px;color:#bbb;">手机截图、纸质报告拍照均可自动识别</div>
            </div>
            """, unsafe_allow_html=True)

    # ================================================================
    # 粘贴模式
    # ================================================================
    else:
        # 一键粘贴 JS
        components.html("""
        <script>
        (function() {
            const observer = new MutationObserver(() => {
                const ta = document.querySelector('textarea[aria-label="请在此粘贴体检报告文字内容..."]');
                if (ta && !ta.dataset.hooked) {
                    ta.dataset.hooked = '1';
                    window.readClipboard = async function() {
                        try {
                            const text = await navigator.clipboard.readText();
                            const nativeInputValueSetter = Object.getOwnPropertyDescriptor(
                                window.HTMLTextAreaElement.prototype, 'value'
                            ).set;
                            nativeInputValueSetter.call(ta, text);
                            ta.dispatchEvent(new Event('input', {bubbles: true}));
                            const btn = document.querySelector('[data-paste-btn]');
                            if (btn) { btn.textContent = '✅ 已粘贴'; setTimeout(() => { btn.textContent = '📋 一键粘贴'; }, 1500); }
                        } catch(e) {
                            alert('无法读取剪贴板，请手动粘贴（Ctrl+V / Cmd+V）');
                        }
                    };
                }
            });
            observer.observe(document.body, {childList: true, subtree: true});
        })();
        </script>
        <div style="display:flex;justify-content:flex-end;margin-bottom:-36px;position:relative;z-index:10;">
            <button data-paste-btn onclick="readClipboard()"
                style="border:1px solid #ddd;background:#fff;border-radius:6px;padding:4px 12px;
                       cursor:pointer;font-size:13px;color:#666;">
                📋 一键粘贴
            </button>
        </div>
        """, height=0)

        display_text = st.session_state.report_text
        report_text = st.text_area(
            "请在此粘贴体检报告文字内容...",
            value=display_text,
            height=260,
            placeholder="请在此粘贴体检报告文字内容...\n\n例如：\n血常规：白细胞 7.2，红细胞 4.1↓\n血脂：总胆固醇 6.2↑，甘油三酯 2.4↑\n...",
            label_visibility="collapsed",
            key="report_input",
        )
        if report_text != st.session_state.report_text:
            st.session_state.report_text = report_text
            st.session_state.ocr_char_count = len(report_text)

    # ---- 隐私声明 ----
    st.markdown("""
    <div class="privacy-notice">
        ✅ 所有报告仅在本地浏览器处理，不上传服务器，不会存储您的隐私数据
    </div>
    """, unsafe_allow_html=True)

    # ---- 全局隐私开关（加分项） ----
    if st.session_state.report_text.strip():
        priv_col1, priv_col2 = st.columns([4, 6])
        with priv_col1:
            privacy_on = st.toggle(
                "🔒 隐藏个人信息" if st.session_state.privacy_on else "🔓 显示完整信息",
                value=st.session_state.privacy_on,
                key="privacy_toggle",
                help="开启后自动遮蔽身份证号、手机号等敏感信息，截图分享更安心",
            )
            if privacy_on != st.session_state.privacy_on:
                st.session_state.privacy_on = privacy_on
                st.rerun()
        with priv_col2:
            if not st.session_state.privacy_on:
                st.caption("敏感信息已暴露，分享时请注意")

    # ---- 开始解读 ----
    has_text = bool(st.session_state.report_text.strip())
    analyze_clicked = st.button(
        "🔍 开始解读" if not st.session_state.analyzing else "⏳ 解读中...",
        type="primary",
        use_container_width=True,
        disabled=not has_text or st.session_state.analyzing,
    )


# ============================================================
# 右栏：解读结果
# ============================================================

with col2:
    st.markdown("### 📊 解读结果")

    # ---- 执行解读 ----
    if analyze_clicked and has_text and not st.session_state.analyzing:
        st.session_state.analyzing = True
        st.rerun()

    if st.session_state.analyzing:
        # ---- Agent 多步推理展示 ----
        progress = st.progress(0, text="🤖 Agent 启动中...")
        step_container = st.container()

        agent_steps_log = []

        def on_step_callback(step_num, step_name, status):
            agent_steps_log.append({"step": step_num, "name": step_name, "status": status})
            pct = int(step_num / 4 * 100)
            progress.progress(pct, text=f"Step {step_num}/4: {step_name} — {status}")
            with step_container:
                st.markdown(f"**Step {step_num} · {step_name}** → {status}")

        try:
            if st.session_state.use_agent:
                result = interpret_report_agent(
                    st.session_state.report_text.strip(),
                    on_step=on_step_callback,
                )
            else:
                progress.progress(50, text="AI 正在分析报告...")
                result = interpret_report(st.session_state.report_text.strip())

            st.session_state.result = result
            st.session_state.agent_steps_display = agent_steps_log

            if "error" not in result:
                items = result.get("items", [])
                summary_parts = []
                for it in items[:3]:
                    summary_parts.append(f"{it.get('指标','?')}: {it.get('状态','?')}")
                preview = "；".join(summary_parts) if summary_parts else "未发现异常"
                st.session_state.history.insert(0, {
                    "time": datetime.now().strftime("%m-%d %H:%M"),
                    "preview": f"{len(items)}项异常",
                    "summary": f"异常指标：{preview}",
                    "text": st.session_state.report_text,
                })
                st.session_state.history = st.session_state.history[:3]
                st.session_state.risk_filter = "全部"
                st.session_state.chat_history = []  # 重置追问历史
        except Exception as e:
            st.session_state.result = {"error": f"解读失败：{str(e)}"}
        st.session_state.analyzing = False
        st.rerun()

    result = st.session_state.result

    if result:
        if "error" in result:
            st.error(f"❌ {result['error']}")
            if result.get("raw"):
                with st.expander("调试信息"):
                    st.code(result["raw"][:1000])
        else:
            items = result.get("items", [])
            global_terms = result.get("global_terms", {})
            advice = result.get("advice", [])
            alerts = result.get("alerts", [])

            # ---- P1-2: 异常统计卡片 + 点击筛选 ----
            alert_count = sum(1 for i in items if i.get("严重程度") == "警惕")
            warn_count = sum(1 for i in items if i.get("严重程度") == "关注")
            note_count = sum(1 for i in items if i.get("严重程度") == "注意")

            alert_names = [i.get("指标", "?") for i in items if i.get("严重程度") == "警惕"]
            warn_names = [i.get("指标", "?") for i in items if i.get("严重程度") == "关注"]
            note_names = [i.get("指标", "?") for i in items if i.get("严重程度") == "注意"]

            c1, c2, c3, c4 = st.columns(4)

            # 统计卡片（改为 HTML 渲染，避免 st.button 把标签当纯文本显示）
            with c1:
                st.markdown(
                    f'<div class="stat-card total"><div style="font-size:12px;color:#999;">异常总计</div>'
                    f'<div class="stat-number" style="color:#555;">{len(items)}</div>'
                    f'<div style="font-size:11px;color:#aaa;">项</div></div>',
                    unsafe_allow_html=True,
                )

            with c2:
                active_cls = " active" if st.session_state.risk_filter == "警惕" else ""
                st.markdown(
                    f'<div class="stat-card danger{active_cls}" title="{'、'.join(alert_names) if alert_names else '无'}">'
                    f'<div style="font-size:12px;color:#c62828;">⚠️ 需警惕</div>'
                    f'<div class="stat-number stat-danger">{alert_count}</div>'
                    f'<div style="font-size:11px;color:#c62828;">建议尽快就医</div></div>',
                    unsafe_allow_html=True,
                )

            with c3:
                active_cls = " active" if st.session_state.risk_filter == "关注" else ""
                st.markdown(
                    f'<div class="stat-card warning{active_cls}" title="{'、'.join(warn_names) if warn_names else '无'}">'
                    f'<div style="font-size:12px;color:#e65100;">◆ 需关注</div>'
                    f'<div class="stat-number stat-warning">{warn_count}</div>'
                    f'<div style="font-size:11px;color:#e65100;">生活干预+复查</div></div>',
                    unsafe_allow_html=True,
                )

            with c4:
                active_cls = " active" if st.session_state.risk_filter == "注意" else ""
                st.markdown(
                    f'<div class="stat-card info{active_cls}" title="{'、'.join(note_names) if note_names else '无'}">'
                    f'<div style="font-size:12px;color:#0d47a1;">● 需注意</div>'
                    f'<div class="stat-number stat-info">{note_count}</div>'
                    f'<div style="font-size:11px;color:#0d47a1;">居家调理</div></div>',
                    unsafe_allow_html=True,
                )

            # ---- P1-3: 风险图例（整洁并排） ----
            st.markdown("""
            <div class="legend-row">
                <div class="legend-item"><span class="legend-dot" style="background:#e53935;"></span> ⚠️ 需警惕：高危异常，建议尽快就医</div>
                <div class="legend-item"><span class="legend-dot" style="background:#ef6c00;"></span> ◆ 需关注：中度异常，建议生活干预+复查</div>
                <div class="legend-item"><span class="legend-dot" style="background:#1e88e5;"></span> ● 需注意：轻度偏离，建议居家调理</div>
            </div>
            """, unsafe_allow_html=True)

            st.divider()

            # ---- P1-4 + P2-2: 工具栏（搜索 + 风险筛选 + 复制 + 导出 + 分享图） ----
            if items:
                s_col, f_col, b1_col, b2_col, b3_col = st.columns([2.5, 1.2, 1, 1, 1])

                with s_col:
                    search_query = st.text_input(
                        "🔍 搜索指标名称",
                        placeholder="输入指标名称快速筛选，如：胆固醇、血糖...",
                        label_visibility="collapsed",
                        key="search_indicators",
                    )
                    # 清空按钮（P2-2）
                    if search_query:
                        if st.button("✕ 清空", key="clear_search"):
                            st.session_state.search_indicators = ""
                            st.rerun()

                with f_col:
                    # 风险等级筛选下拉（P2-2）
                    risk_options = ["全部", "警惕", "关注", "注意"]
                    current_idx = risk_options.index(st.session_state.risk_filter) if st.session_state.risk_filter in risk_options else 0
                    risk_filter = st.selectbox(
                        "风险等级",
                        risk_options,
                        index=current_idx,
                        label_visibility="collapsed",
                        key="risk_filter",
                    )

                with b1_col:
                    if st.button("📋 复制", use_container_width=True, key="copy_all",
                                 help="一键复制全部解读文本到剪贴板"):
                        export_text = _generate_export_text(result)
                        components.html(f"""
                        <script>
                        navigator.clipboard.writeText({json.dumps(export_text)})
                            .then(() => {{}}).catch(() => {{}});
                        </script>""", height=0)
                        st.toast("✅ 已复制到剪贴板！", icon="✅")

                with b2_col:
                    st.download_button(
                        label="📥 导出",
                        data=_generate_export_text(result),
                        file_name=f"体检报告解读_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                        mime="text/plain",
                        use_container_width=True,
                        help="下载为 TXT 文本文件，方便保存和转发",
                    )

                with b3_col:
                    if st.button("📷 分享图", use_container_width=True, key="share_image",
                                 help="生成一张可转发给家人的解读摘要图片"):
                        st.session_state.show_share_image = not st.session_state.show_share_image
                        st.rerun()

                # ---- 分享图渲染 ----
                if st.session_state.show_share_image:
                    share_html = _generate_share_card_html(result, items, advice, alerts)
                    ts = datetime.now().strftime("%Y%m%d_%H%M")
                    components.html(f"""
                    <script src="https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js"></script>
                    <div id="share-status" style="text-align:center;padding:20px;color:#666;font-size:15px;">
                        ⏳ 正在生成分享图...
                    </div>
                    <div id="capture-zone" style="position:absolute;left:-9999px;top:0;">
                    {share_html}
                    </div>
                    <script>
                    (function() {{
                        function tryCapture() {{
                            if (typeof html2canvas === 'undefined') {{
                                setTimeout(tryCapture, 150);
                                return;
                            }}
                            var card = document.getElementById('share-card');
                            if (!card) {{ setTimeout(tryCapture, 150); return; }}
                            html2canvas(card, {{
                                scale: 2,
                                useCORS: true,
                                backgroundColor: '#ffffff',
                                logging: false
                            }}).then(function(canvas) {{
                                var link = document.createElement('a');
                                link.download = '体检报告解读_{ts}.png';
                                link.href = canvas.toDataURL('image/png');
                                link.click();
                                var s = document.getElementById('share-status');
                                s.innerHTML = '✅ 分享图已生成并下载，请查看浏览器下载文件夹';
                                s.style.color = '#4caf50';
                                s.style.fontWeight = '700';
                            }}).catch(function(err) {{
                                var s = document.getElementById('share-status');
                                s.innerHTML = '❌ 生成失败：' + err.message + '<br>请检查网络后重试';
                                s.style.color = '#f44336';
                            }});
                        }}
                        setTimeout(tryCapture, 300);
                    }})();
                    </script>
                    """, height=100)

                st.divider()

                # ---- 指标卡片排序与筛选 ----
                sorted_items = sorted(items, key=lambda x: SEVERITY_ORDER.get(x.get("严重程度", "注意"), 9))

                # 按风险等级筛选（P2-2）
                if st.session_state.risk_filter != "全部":
                    sorted_items = [i for i in sorted_items if i.get("严重程度") == st.session_state.risk_filter]

                # 按搜索词筛选
                if search_query:
                    ql = search_query.lower()
                    filtered = []
                    for i in sorted_items:
                        match_name = ql in i.get("指标", "").lower()
                        match_explain = ql in i.get("通俗解释", "").lower()
                        match_terms = any(ql in t.get("原词", "").lower() or ql in t.get("解释", "").lower()
                                          for t in i.get("相关术语", []))
                        if match_name or match_explain or match_terms:
                            filtered.append(i)
                    sorted_items = filtered
                    if not sorted_items:
                        st.info(f"未找到包含「{search_query}」的指标")

                # ---- 渲染每条指标卡片（P2-1 + P2-3 + P2-4） ----
                for item in sorted_items:
                    severity = item.get("严重程度", "注意")
                    s = _severity_style(severity)
                    indicator_name = item.get("指标", "?")

                    # 生成唯一 key
                    card_key = f"card_{indicator_name}"

                    # 卡片容器（P2-3：按风险等级差异化配色）
                    card_class = {"警惕": "danger", "关注": "warning", "注意": "info"}.get(severity, "info")

                    with st.container():
                        # 卡片头部
                        st.markdown(f"""
                        <div class="report-card {card_class}">
                            <div class="card-header">
                                <span class="card-tag {card_class}">{s['tag']}</span>
                                <span style="font-weight:700;">{indicator_name}</span>
                                <span style="color:#888;">—</span>
                                <span style="color:{s['color']};font-weight:600;">{item.get('状态', '?')}</span>
                            </div>
                            <div style="margin-left:4px;">
                                <b>检测结果：</b>
                                <span style="color:{s['color']};font-weight:600;">{item.get('结果', '?')}</span>
                                &nbsp;&nbsp;（正常参考范围：{item.get('正常范围', '?')}）
                            </div>
                        """, unsafe_allow_html=True)

                        # 通俗解读（P2-1：突出显示，放大字号）
                        st.markdown(f"""
                        <div class="highlight-section" style="background:{s['bg']};border-left:4px solid {s['border']};">
                            <b>📖 通俗解读：</b>{item.get('通俗解释', '暂无解释')}
                        </div>
                        """, unsafe_allow_html=True)

                        # 详细内容（默认折叠，P2-1）
                        with st.expander("🔍 展开详情（相关术语、建议、提醒）", expanded=False):
                            # 相关术语
                            related_terms = item.get("相关术语", [])
                            if related_terms:
                                st.markdown("<b>🔤 相关术语：</b>", unsafe_allow_html=True)
                                for t in related_terms:
                                    st.markdown(
                                        f'<span class="term-badge">{t.get("原词", "?")}</span>'
                                        f'<span style="color:#888;font-size:13px;"> → {t.get("解释", "?")}</span>',
                                        unsafe_allow_html=True,
                                    )
                                st.markdown("<br>", unsafe_allow_html=True)

                            # 复查建议（P2-4：行动指引）
                            review = item.get("复查建议", s.get("review", ""))
                            if review:
                                st.markdown(f"""
                                <div class="review-box">
                                    <b>📅 复查建议：</b>{review}
                                </div>""", unsafe_allow_html=True)

                            # 该指标相关的就医提醒
                            indicator_alerts = [
                                a for a in alerts
                                if indicator_name[:2] in a or a[:2] in indicator_name
                            ]
                            if indicator_alerts:
                                st.markdown("<b>🏥 就医提醒：</b>", unsafe_allow_html=True)
                                for a in indicator_alerts:
                                    st.warning(f"🏥 {a}")
                            elif alerts:
                                st.markdown("<b>🏥 就医提醒：</b>", unsafe_allow_html=True)
                                for a in alerts[:2]:
                                    st.warning(f"🏥 {a}")

                            # 全局术语（如果有相关的话）
                            if global_terms:
                                st.markdown("<b>🔤 报告通用术语：</b>", unsafe_allow_html=True)
                                term_items = list(global_terms.items())
                                for k, v in term_items[:5]:
                                    st.markdown(
                                        f'<span class="term-badge">{k}</span>'
                                        f'<span style="color:#888;font-size:13px;"> → {v}</span>',
                                        unsafe_allow_html=True,
                                    )

                            # 定位原文（解读-原文联动）
                            st.markdown("<br>", unsafe_allow_html=True)
                            locate_key = f"locate_{card_key}"
                            if st.button("📍 定位原文", key=locate_key, use_container_width=True,
                                         help="在原始报告中查找该指标的对应位置"):
                                st.session_state[f"show_locate_{card_key}"] = True

                            if st.session_state.get(f"show_locate_{card_key}"):
                                snippet = _find_in_original_text(indicator_name, st.session_state.report_text)
                                if snippet:
                                    st.markdown(f"""
                                    <div style="background:#fffde7;border:1px solid #fff9c4;border-left:3px solid #fbc02d;
                                                padding:10px 14px;border-radius:6px;margin-top:6px;font-size:13px;
                                                color:#5d4037;line-height:1.6;">
                                        <b>📝 原文片段：</b><br>{_esc(snippet)}
                                    </div>""", unsafe_allow_html=True)
                                else:
                                    st.caption("未在原文中直接找到该指标名称，可能使用了不同的表述方式")

                        # 免责小字
                        st.caption("解读仅供参考，具体请以临床诊断为准")

                        # 闭合 report-card div
                        st.markdown("</div>", unsafe_allow_html=True)

                # ---- 柱状图 ----
                if len(sorted_items) > 1 and not search_query:
                    st.divider()
                    st.caption("📊 异常指标偏离程度对比")
                    chart_data = []
                    for item in sorted_items:
                        chart_data.append({
                            "指标": item.get("指标", "?")[:10],
                            "严重程度": item.get("严重程度", ""),
                        })
                    if chart_data:
                        df = pd.DataFrame(chart_data)
                        df["分数"] = df["严重程度"].map({"警惕": 3, "关注": 2, "注意": 1})
                        fig = go.Figure(data=[go.Bar(
                            x=df["指标"], y=df["分数"],
                            marker_color=df["严重程度"].map(
                                {"警惕": "#e53935", "关注": "#ef6c00", "注意": "#1e88e5"}
                            ),
                            text=df["严重程度"], textposition="outside",
                        )])
                        fig.update_layout(
                            yaxis=dict(tickvals=[1, 2, 3], ticktext=["注意", "关注", "警惕"], range=[0, 3.8]),
                            margin=dict(t=10, b=10), height=240,
                        )
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("🎉 未发现明显异常指标，报告总体良好！")

            # ---- 生活建议（无搜索词时显示） ----
            if advice and not search_query:
                st.divider()
                st.markdown("### 💡 生活建议")
                for i, tip in enumerate(advice, 1):
                    st.markdown(f"""
                    <div class="advice-item" style="font-size:var(--font-card);">
                        <span style="color:#1a73e8;font-weight:700;">{i}.</span> {tip}
                    </div>""", unsafe_allow_html=True)

            # ---- 就医提醒（无搜索词时显示） ----
            if alerts and not search_query:
                st.divider()
                st.markdown("### 🏥 就医提醒")
                for alert in alerts:
                    st.warning(f"🏥 {alert}")

            # ---- 交叉关联分析（Agent 模式专属）----
            cross = result.get("cross_analysis", "")
            if cross and not search_query:
                st.divider()
                st.markdown("### 🔗 指标关联分析")
                st.info(f"🔍 {cross}")

            # ---- Agent 推理过程（可折叠）----
            agent_steps = result.get("agent_steps", []) or st.session_state.agent_steps_display
            if agent_steps and not search_query:
                with st.expander("🤖 Agent 推理过程", expanded=False):
                    for s in agent_steps:
                        step_num = s.get("step", "?")
                        name = s.get("name", "")
                        status = s.get("status", "")
                        st.markdown(f"**Step {step_num} · {name}**")
                        st.caption(status)

                    check = result.get("check_report", {})
                    if check:
                        st.divider()
                        passed = check.get("passed", True)
                        st.markdown(f"**自检结果：** {'✅ 通过' if passed else '⚠️ 有修正'}")
                        if check.get("missing"):
                            st.warning(f"遗漏指标：{', '.join(check['missing'])}")
                        if check.get("severity_fixes"):
                            st.info(f"严重程度修正：{check['severity_fixes']} 处")
                        if check.get("compliance_issues"):
                            st.warning(f"合规修正：{', '.join(check['compliance_issues'][:3])}")
                        if check.get("summary"):
                            st.caption(check["summary"])

            # ---- 多轮追问对话 ----
            if not search_query:
                st.divider()
                st.markdown("### 💬 追问报告内容")
                st.caption("🤖 基于你的报告上下文回答，可追问指标含义、严重程度、注意事项等")

                # 显示历史对话
                for msg in st.session_state.chat_history:
                    if msg["role"] == "user":
                        st.chat_message("user").write(msg["content"])
                    else:
                        st.chat_message("assistant").write(msg["content"])

                # 输入框
                user_question = st.chat_input("输入你的问题，如：我的血糖严重吗？需要吃药吗？")
                if user_question:
                    st.chat_message("user").write(user_question)
                    st.session_state.chat_history.append({"role": "user", "content": user_question})

                    with st.spinner("🤖 思考中..."):
                        try:
                            answer = ask_followup(
                                user_question,
                                st.session_state.report_text,
                                result,
                                st.session_state.chat_history,
                            )
                        except Exception as e:
                            answer = f"抱歉，回答失败：{e}"

                    st.chat_message("assistant").write(answer)
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                    st.rerun()

    elif not st.session_state.analyzing:
        # 空状态
        st.markdown("""
        <div style="text-align:center;padding:60px 20px;color:#bbb;">
            <div style="font-size:48px;margin-bottom:16px;">📊</div>
            <div style="font-size:18px;margin-bottom:8px;">暂无解读结果</div>
            <div style="font-size:14px;">👈 在左侧上传报告或粘贴文字，点击「开始解读」即可看到分析结果</div>
        </div>
        """, unsafe_allow_html=True)


# ============================================================
# 页脚
# ============================================================

st.markdown(
    '<div class="footer">🩺 体检报告解读助手 · AI 生成内容仅供参考 · 不构成医疗建议</div>',
    unsafe_allow_html=True,
)
