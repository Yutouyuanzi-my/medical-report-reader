"""
ocr_helper.py -- PDF / 图片文字提取（跨平台）

PDF  -> pdfplumber 提取
图片 -> macOS: Vision 框架 OCR（内置中文识别，零依赖）
        Linux: pytesseract（需安装 tesseract-ocr + tesseract-ocr-chi-sim）
"""

import subprocess
import json
import sys

# 顶层导入，避免首次上传 PDF 时内部 import 触发 Streamlit 脚本超时
# pdfplumber 首次编译 import 可能耗时 30s+，顶层导入在启动时完成一次即可缓存
try:
    import pdfplumber
except ImportError:
    pdfplumber = None  # type: ignore


# ============================================================
# macOS Vision OCR（JXA 脚本）
# ============================================================

_OCR_JXA = r'''
ObjC.import("Vision");
ObjC.import("AppKit");

function run(argv) {
    let imgPath = argv[0];
    let imgData = $.NSData.dataWithContentsOfFile($(imgPath));
    if (!imgData || imgData.length === 0) {
        return JSON.stringify({ error: "无法读取图片文件: " + imgPath });
    }

    let handler = $.VNImageRequestHandler.alloc.initWithDataOptions(
        imgData, $.NSDictionary.alloc.init
    );
    let request = $.VNRecognizeTextRequest.alloc.init;
    request.recognitionLevel = $.VNRequestTextRecognitionLevelAccurate;

    // 中文优先识别
    let langs = $.NSMutableArray.alloc.init;
    langs.addObject($("zh-Hans"));
    langs.addObject($("en"));
    request.recognitionLanguages = langs;

    let error = $();
    let success = handler.performRequestsError($([request]), error);
    if (!success) {
        return JSON.stringify({ error: "OCR 处理失败" });
    }

    let results = request.results;
    if (!results || results.count === 0) {
        return JSON.stringify({ error: "图片中未识别到文字" });
    }

    let lines = [];
    for (let i = 0; i < results.count; i++) {
        let topCandidate = results.objectAtIndex(i)
            .topCandidates(1)
            .objectAtIndex(0);
        if (topCandidate) {
            lines.push(topCandidate.string.js);
        }
    }
    return JSON.stringify({ text: lines.join("\n") });
}
'''


def _ocr_macos_vision(file_path: str) -> dict:
    """macOS Vision 框架 OCR"""
    try:
        result = subprocess.run(
            ["osascript", "-l", "JavaScript", "-e", _OCR_JXA, file_path],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            err = result.stderr.strip() or "未知错误"
            return {"error": f"OCR 调用失败：{err}"}
        return json.loads(result.stdout.strip())
    except subprocess.TimeoutExpired:
        return {"error": "OCR 处理超时，请尝试更小的图片"}
    except json.JSONDecodeError:
        return {"error": "OCR 返回格式异常，请重试"}
    except Exception as e:
        return {"error": f"OCR 异常：{str(e)}"}


def _ocr_tesseract(file_path: str) -> dict:
    """Linux/云端 pytesseract OCR"""
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        return {"error": "pytesseract 未安装，图片 OCR 不可用。请使用 PDF 或粘贴文字模式。"}

    try:
        img = Image.open(file_path)
        # chi_sim+eng 中文+英文识别
        text = pytesseract.image_to_string(img, lang="chi_sim+eng")
        if not text.strip():
            return {"error": "图片中未识别到文字，请尝试更清晰的图片或使用粘贴模式"}
        return {"text": text.strip()}
    except Exception as e:
        err_msg = str(e)
        if "tesseract is not installed" in err_msg or "not installed" in err_msg:
            return {"error": "tesseract 未安装，图片 OCR 不可用。请使用 PDF 或粘贴文字模式。"}
        return {"error": f"OCR 异常：{err_msg}"}


# ============================================================
# 公共接口
# ============================================================

def extract_from_pdf(file_path: str) -> dict:
    """从 PDF 中提取文字"""
    if pdfplumber is None:
        return {"error": "pdfplumber 未安装，请运行：pip install pdfplumber"}

    try:
        with pdfplumber.open(file_path) as pdf:
            all_text = []
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    all_text.append(text)
            if not all_text:
                return {"error": "PDF 中未提取到文字，可能是扫描件（图片型PDF）。请截图后上传图片格式。"}
            return {"text": "\n\n".join(all_text)}
    except Exception as e:
        return {"error": f"PDF 解析失败：{str(e)}"}


def extract_from_image(file_path: str) -> dict:
    """从图片中提取文字（自动选择 macOS Vision / pytesseract）"""
    if sys.platform == "darwin":
        return _ocr_macos_vision(file_path)
    else:
        return _ocr_tesseract(file_path)


def extract_text(file_path: str, file_type: str) -> dict:
    """
    统一的文字提取入口

    file_type: "pdf" | "image"
    返回: {"text": "..."} 或 {"error": "..."}
    """
    if file_type == "pdf":
        return extract_from_pdf(file_path)
    elif file_type in ("image", "png", "jpg", "jpeg"):
        return extract_from_image(file_path)
    else:
        return {"error": f"不支持的文件类型：{file_type}"}
