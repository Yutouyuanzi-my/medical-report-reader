# config.py -- DeepSeek API 配置
#
# API Key 优先级：
#   1. Streamlit secrets  (.streamlit/secrets.toml 或 Cloud Secrets)
#   2. 环境变量 DEEPSEEK_API_KEY
#
# 本地开发：在 .streamlit/secrets.toml 中写入：
#   DEEPSEEK_API_KEY = "sk-xxxx"

import os
import sys


def _load_api_key() -> str:
    """按优先级读取 API Key"""
    # 1. Streamlit secrets（本地 secrets.toml / Cloud Secrets）
    try:
        import streamlit as st
        key = st.secrets.get("DEEPSEEK_API_KEY", "")
        if key:
            return key
    except Exception:
        pass

    # 2. 环境变量
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    if key:
        return key

    # 3. 未找到
    print("[config] 警告：未找到 DEEPSEEK_API_KEY，请在 .streamlit/secrets.toml 或环境变量中配置", file=sys.stderr)
    return ""


DEEPSEEK_API_KEY = _load_api_key()
DEEPSEEK_BASE_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"
