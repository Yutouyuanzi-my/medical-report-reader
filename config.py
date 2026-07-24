# config.py -- DeepSeek API 配置
#
# API Key 优先级：
#   1. Streamlit secrets  (.streamlit/secrets.toml 或 Cloud Secrets)
#   2. 环境变量 DEEPSEEK_API_KEY
#
# 本地开发：在 .streamlit/secrets.toml 中写入：
#   DEEPSEEK_API_KEY = "sk-xxxx"

import os
import re
import socket
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


def _check_and_disable_broken_local_proxy():
    """
    检测环境变量中的本地代理（127.0.0.1 / localhost）是否可用。
    如果不可用，清空相关环境变量，避免 requests 因连接失败代理而报错。
    """
    proxy_keys = ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
                  "http_proxy", "https_proxy", "all_proxy"]
    local_hosts = ("127.0.0.1", "localhost", "::1")

    for key in proxy_keys:
        value = os.environ.get(key, "")
        if not value:
            continue
        # 提取 host:port，例如 http://127.0.0.1:7890
        match = re.search(r"(?:https?|socks5?)://([^/:@]+):(\d+)", value)
        if not match:
            continue
        host, port = match.group(1), int(match.group(2))
        if host not in local_hosts:
            continue
        # 测试本地代理端口是否开放
        try:
            with socket.create_connection((host, port), timeout=1.5):
                # 端口开放，保留代理
                return value
        except (OSError, socket.timeout):
            # 端口未开放，清空该代理环境变量
            os.environ.pop(key, None)
            print(f"[config] 检测到 {key}={value} 不可用，已自动禁用", file=sys.stderr)
    return None


# 初始化时自动检测代理；同时导出 _proxies 供 requests 使用
_CHECKED_PROXY = _check_and_disable_broken_local_proxy()
if _CHECKED_PROXY:
    _proxies = None  # 让 requests 使用环境变量中的代理
else:
    _proxies = {"http": None, "https": None}  # 显式禁用代理


DEEPSEEK_API_KEY = _load_api_key()
DEEPSEEK_BASE_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"
