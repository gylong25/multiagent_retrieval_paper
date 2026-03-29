"""
Flask 后端服务：提供 PDF 上传与聊天接口。

启动方式：
  python flask_server.py
"""

from __future__ import annotations

import os
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List

from dotenv import load_dotenv
from flask import Flask, jsonify, request

from main import AppConfig, MultiAgentLiteratureRAG, PDFRAGService, save_report

load_dotenv()

config = AppConfig.from_env()
config.ensure_directories()
config.validate_runtime()

rag_service = PDFRAGService(config)
agent_system = MultiAgentLiteratureRAG(config, rag_service)
runtime_lock = Lock()

app = Flask(__name__)
max_upload_mb = int(os.getenv("MAX_UPLOAD_MB", "50"))
app.config["MAX_CONTENT_LENGTH"] = max_upload_mb * 1024 * 1024


def _safe_filename(filename: str) -> str:
    name = Path(filename).name
    return name.strip()


def _is_pdf(filename: str) -> bool:
    return filename.lower().endswith(".pdf")


@app.route("/health", methods=["GET"])
def health() -> Any:
    return jsonify(
        {
            "status": "ok",
            "collection_count": rag_service.collection.count(),
            "papers_dir": str(config.papers_dir.resolve()),
        }
    )


@app.route("/upload", methods=["POST"])
def upload() -> Any:
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "未收到文件"}), 400

    saved: List[str] = []
    skipped: List[str] = []

    for file in files:
        if not file or not file.filename:
            continue
        filename = _safe_filename(file.filename)
        if not filename or not _is_pdf(filename):
            skipped.append(filename or "<empty>")
            continue
        destination = config.papers_dir / filename
        file.save(destination)
        saved.append(filename)

    if not saved:
        return jsonify({"error": "没有可用的 PDF 文件", "skipped": skipped}), 400

    with runtime_lock:
        index_stats = rag_service.build_index()

    return jsonify(
        {
            "saved": saved,
            "skipped": skipped,
            "index_stats": index_stats,
        }
    )


@app.route("/chat", methods=["POST"])
def chat() -> Any:
    payload: Dict[str, Any] = request.get_json(silent=True) or {}
    question = str(payload.get("question", "")).strip()
    if not question:
        return jsonify({"error": "问题不能为空"}), 400

    if rag_service.collection.count() == 0:
        return jsonify({"error": "当前索引为空，请先上传 PDF 构建索引。"}), 400

    with runtime_lock:
        answer = agent_system.run(question)
        report_path = save_report(answer, config.output_dir)

    return jsonify({"answer": answer, "report_path": str(report_path.resolve())})


if __name__ == "__main__":
    host = os.getenv("FLASK_HOST", "127.0.0.1")
    port = int(os.getenv("FLASK_PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "false").strip().lower() in {"1", "true", "yes"}
    app.run(host=host, port=port, debug=debug)
