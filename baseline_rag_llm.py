"""
基线脚本：RAG 检索 + 拼接上下文 + 单 LLM 回答。

运行：
  python baseline_rag_llm.py --question "你的问题"
输出文件保存到 outputs 目录。
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

from main import AppConfig, PDFRAGService


def _build_client(config: AppConfig) -> OpenAI:
    return OpenAI(
        api_key=config.llm_api_key,
        base_url=config.llm_base_url,
        timeout=60,
        max_retries=1,
    )


def _write_output(output_dir: Path, name: str, content: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / name
    path.write_text(content, encoding="utf-8")
    return path


def _build_context(results: List[Dict[str, Any]]) -> str:
    sections: List[str] = []
    for idx, item in enumerate(results, start=1):
        title = item.get("title") or item.get("file_name") or "unknown"
        content = item.get("content", "")
        snippets = item.get("child_snippets") or []
        snippet_text = "\n".join(f"- {s}" for s in snippets[:3] if s)
        section = [
            f"[{idx}] 标题：{title}",
            "内容：",
            content,
        ]
        if snippet_text:
            section.extend(["子块片段：", snippet_text])
        sections.append("\n".join(section))
    return "\n\n".join(sections)


def run(question: str, output_name: Optional[str] = None, top_k: Optional[int] = None) -> Path:
    load_dotenv()
    config = AppConfig.from_env()
    config.ensure_directories()
    config.validate_runtime()

    rag_service = PDFRAGService(config)
    if rag_service.collection.count() == 0:
        rag_service.build_index()

    top_k_value = top_k or config.top_k
    raw = rag_service.search(question, top_k=top_k_value)
    payload = json.loads(raw)
    results = payload.get("results", [])

    context = _build_context(results)
    prompt = (
        "以下是从 PDF 文献中检索到的内容，请仅依据这些内容回答问题，"
        "若信息不足请明确说明：\n\n"
        f"{context}\n\n问题：{question}"
    )

    client = _build_client(config)
    response = client.chat.completions.create(
        model=config.llm_model,
        messages=[
            {"role": "system", "content": "你是学术助手，请用中文回答问题。"},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    answer = response.choices[0].message.content or ""

    if not output_name:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_name = f"baseline_rag_llm_{stamp}.md"

    content = (
        f"# 问题\n{question}\n\n"
        f"# 回答\n{answer}\n\n"
        f"# 检索片段\n{context}\n"
    )
    return _write_output(config.output_dir, output_name, content)


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG+单 LLM 基线")
    parser.add_argument(
        "--question",
        default=None,
        help="问题内容（默认读取 .env 的 DEMO_QUESTION）",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="检索返回条数（默认使用 TOP_K）",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="输出文件名（默认自动生成）",
    )
    args = parser.parse_args()

    load_dotenv()
    config = AppConfig.from_env()
    question = (args.question or config.demo_question or "").strip()
    if not question:
        raise SystemExit("问题不能为空，请通过 --question 传入。")

    output_path = run(question, args.output, args.top_k)
    print(f"[OK] 结果已保存到：{output_path.resolve()}")


if __name__ == "__main__":
    main()
