# 部署指南：Streamlit Cloud

> 代码已推送到 GitHub 仓库：`https://github.com/Yutouyuanzi-my/medical-report-reader`
> 本指南带你把项目部署成一个公开可访问的 Web 应用。

## 前置条件
- GitHub 账号已登录，且仓库 `medical-report-reader` 已包含最新代码（main 分支）
- 一个 DeepSeek API Key（`sk-` 开头）

## 步骤

### 1. 打开 Streamlit Cloud
访问 **https://streamlit.io/cloud** → 点击右上角 **Sign in with GitHub** → 授权登录。

### 2. 新建应用
登录后点击 **New app**（或 "Create app"）。

在弹窗中填写：
- **Repository**：`Yutouyuanzi-my/medical-report-reader`
- **Branch**：`main`
- **Main file path**：`app.py`

### 3. 配置密钥（必须）
展开 **Advanced settings** → **Secrets**，粘贴以下内容（把 `sk-xxxx` 换成你的真实 Key）：

```toml
DEEPSEEK_API_KEY = "sk-xxxx"
```

> ⚠️ 不填这个，网页能打开但解读会报错（API 无密钥）。
> 本地 `.streamlit/secrets.toml` 不会被上传（已在 .gitignore），所以云端必须单独填。

### 4. 部署
点击 **Deploy**。首次构建会安装 `requirements.txt` 里的依赖（含 pdfplumber、pytesseract 等），约 1–3 分钟。

构建完成后会分配一个公开 URL，形如：
`https://medical-report-reader-xxxx.streamlit.app`

### 5. 验证
打开分配到的 URL，点「加载示例报告」或上传一份体检报告 PDF，确认能正常输出解读结果。

---

## 常见问题

**Q：部署后页面报错 `API_ERROR` 或超时？**
A：DeepSeek API 从美国服务器直连偶尔不稳定。先确认 Secrets 里的 Key 正确无误；若持续超时，可考虑在 `config.py` 中切换 `DEEPSEEK_BASE_URL` 为国际版 endpoint。

**Q：PDF 上传后识别为空？**
A：文本型 PDF 用 pdfplumber 直接抽取；扫描件图片 PDF 依赖 OCR（已配 `tesseract-ocr-chi-sim` 中文包）。纯图片扫描件若识别不佳，可先转成文本型 PDF 再上传。

**Q：改了代码怎么更新线上？**
A：本地改完 → `git add .` → `git commit` → `git push`。Streamlit Cloud 会自动检测并重新部署（也可在后台手动 Reboot）。

---

## 部署架构
```
用户浏览器
    │  HTTPS
    ▼
Streamlit Cloud (美国)
    ├── app.py          # Web UI + 流式展示
    ├── agent.py        # ReAct Agent + 4 医疗工具
    └── DeepSeek API    # 大模型推理 (直连)
```
