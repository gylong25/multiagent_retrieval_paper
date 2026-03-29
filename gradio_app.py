"""
Gradio 前端：上传 PDF + 对话聊天。

启动方式：
  python gradio_app.py
"""

from __future__ import annotations

import os
from typing import Dict, List

import gradio as gr
import requests
from dotenv import load_dotenv

load_dotenv()

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:5000").rstrip("/")


def _format_error(message: str) -> str:
    return f"❌ {message}"


def _format_ok(message: str) -> str:
    return f"✅ {message}"


def upload_files(files: List[gr.File]) -> str:
    if not files:
        return _format_error("未选择文件。")

    multipart = []
    opened = []
    for file in files:
        if not file or not getattr(file, "name", ""):
            continue
        file_obj = open(file.name, "rb")
        opened.append(file_obj)
        multipart.append(
            ("files", (os.path.basename(file.name), file_obj, "application/pdf"))
        )

    if not multipart:
        return _format_error("没有可用的 PDF 文件。")

    try:
        resp = requests.post(f"{BACKEND_URL}/upload", files=multipart, timeout=300)
        if resp.status_code != 200:
            return _format_error(resp.json().get("error", resp.text))
        data = resp.json()
        stats = data.get("index_stats", {})
        return _format_ok(
            f"已上传 {len(data.get('saved', []))} 个文件，"
            f"索引：{stats.get('pdf_count', 0)} 篇 PDF，"
            f"{stats.get('chunk_count', 0)} 个文本块。"
        )
    except Exception as exc:  # noqa: BLE001
        return _format_error(f"上传失败：{exc}")
    finally:
        for file_obj in opened:
            try:
                file_obj.close()
            except Exception:
                pass


def chat(message: str, history: List[Dict[str, str]]) -> tuple[List[Dict[str, str]], str]:
    question = (message or "").strip()
    if not question:
        return history, ""
    try:
        resp = requests.post(
            f"{BACKEND_URL}/chat", json={"question": question}, timeout=300
        )
        if resp.status_code != 200:
            error = resp.json().get("error", resp.text)
            history.extend(
                [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": _format_error(error)},
                ]
            )
            return history, ""
        answer = resp.json().get("answer", "")
        history.extend(
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ]
        )
        return history, ""
    except Exception as exc:  # noqa: BLE001
        history.extend(
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": _format_error(f"请求失败：{exc}")},
            ]
        )
        return history, ""


with gr.Blocks(title="多智能体 PDF 文献问答") as demo:
    gr.Markdown(
        """
# 📚 多智能体 PDF 文献检索与总结
上传论文 PDF 后即可发起对话，系统会进行检索与总结，输出结构化的中文报告。

**提示：**
- 支持多文件上传（建议先上传后再开始聊天）。
- 首次构建索引可能需要一点时间。
- 可在 `.env` 中为不同智能体配置不同模型。
        """
    )

    with gr.Row():
        uploader = gr.File(
            label="上传 PDF",
            file_types=[".pdf"],
            file_count="multiple",
        )
        upload_btn = gr.Button("上传并构建索引", variant="primary")

    upload_status = gr.Markdown("")
    upload_btn.click(upload_files, inputs=[uploader], outputs=[upload_status])

    gr.Markdown("---")
    chatbot = gr.Chatbot(label="文献对话", height=420)
    user_input = gr.Textbox(
        label="问题",
        placeholder="例如：总结一下目前关于大模型在 RAG 系统中的应用方法。",
    )
    send_btn = gr.Button("发送")

    send_btn.click(chat, inputs=[user_input, chatbot], outputs=[chatbot, user_input])
    user_input.submit(chat, inputs=[user_input, chatbot], outputs=[chatbot, user_input])

    gr.Markdown(
        f"后端地址：`{BACKEND_URL}`  |  先启动 Flask，再启动本页面。"
    )


if __name__ == "__main__":
    server_name = os.getenv("GRADIO_SERVER_NAME", "127.0.0.1")
    server_port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    share = os.getenv("GRADIO_SHARE", "false").strip().lower() in {"1", "true", "yes"}
    demo.queue().launch(server_name=server_name, server_port=server_port, share=share)
