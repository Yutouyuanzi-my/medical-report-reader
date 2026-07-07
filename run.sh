#!/bin/bash
# 一键启动体检报告解读助手 Streamlit 服务
# 使用 nohup 让服务在关闭终端后仍然运行

cd "$(dirname "$0")"

# 如果已有服务在运行，先停止
if lsof -i :8501 >/dev/null 2>&1; then
    echo "检测到 8501 端口被占用，正在停止旧服务..."
    pkill -f "streamlit run app.py" 2>/dev/null
    sleep 2
fi

# 用 nohup 后台启动，使用完整路径
VENV_PYTHON="$(dirname "$0")/.venv/bin/python3"
echo "使用 Python: $VENV_PYTHON"
nohup "$VENV_PYTHON" -m streamlit run app.py --server.port 8501 --server.headless true --browser.gatherUsageStats false > streamlit.log 2>&1 &

sleep 3

if lsof -i :8501 >/dev/null 2>&1; then
    echo "✅ 服务已启动：http://localhost:8501"
else
    echo "❌ 服务启动失败，请查看 streamlit.log"
    cat streamlit.log
fi
