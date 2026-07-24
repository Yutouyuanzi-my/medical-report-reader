#!/bin/bash
# 一键启动体检报告解读助手 Streamlit 服务
# 使用 nohup 让服务在关闭终端后仍然运行

cd "$(dirname "$0")"

# 如果已有服务在运行，先停止
if lsof -i :8502 >/dev/null 2>&1; then
    echo "检测到 8502 端口被占用，正在停止旧服务..."
    pkill -f "streamlit run app.py" 2>/dev/null
    sleep 2
fi

# 用 nohup 后台启动
# 优先使用本地 .venv（现在已用系统 Python 重建），回退到 managed Python venv
if [ -x "$(dirname "$0")/.venv/bin/python3" ]; then
    VENV_PYTHON="$(dirname "$0")/.venv/bin/python3"
else
    VENV_PYTHON="/Users/my/.workbuddy/binaries/python/envs/default/bin/python3"
fi
echo "使用 Python: $VENV_PYTHON"
nohup "$VENV_PYTHON" -m streamlit run app.py --server.port 8502 --server.headless true --browser.gatherUsageStats false --server.enableCORS false --server.enableXsrfProtection false > streamlit.log 2>&1 &

sleep 12

# 等 streamlit 启动完成（首次 import 约需 20-30s）
for i in 1 2 3 4 5 6 7 8 9 10; do
    if lsof -i :8502 >/dev/null 2>&1; then
        break
    fi
    sleep 3
done

if lsof -i :8502 >/dev/null 2>&1; then
    echo "✅ 服务已启动：http://localhost:8502"
else
    echo "❌ 服务启动失败，请查看 streamlit.log"
    tail -20 streamlit.log
fi
