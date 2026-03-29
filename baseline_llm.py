"""
基线脚本：单个 LLM 直接回答问题（无检索）。

运行：
  python baseline_llm.py --question "你的问题"
输出文件保存到 outputs 目录。
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

from main import AppConfig


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


def run(question: str, output_name: Optional[str] = None) -> Path:
    load_dotenv()
    config = AppConfig.from_env()
    config.ensure_directories()
    config.validate_runtime()

    client = _build_client(config)
    messages = [
        {
            "role": "system",
            "content": "你是学术助手，请用中文回答问题，结构清晰、简洁客观。",
        },
        {"role": "user", "content": question},
    ]
    response = client.chat.completions.create(
        model=config.llm_model, messages=messages, temperature=0.2
    )
    answer = response.choices[0].message.content or ""

    if not output_name:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_name = f"baseline_llm_{stamp}.md"

    content = f"# 问题\n{question}\n\n# 回答\n{answer}\n"
    return _write_output(config.output_dir, output_name, content)


def main() -> None:
    parser = argparse.ArgumentParser(description="单 LLM 直接回答基线")
    parser.add_argument(
        "--question",
        default=None,
        help="问题内容（默认读取 .env 的 DEMO_QUESTION）",
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

    output_path = run(question, args.output)
    print(f"[OK] 结果已保存到：{output_path.resolve()}")


if __name__ == "__main__":
    main()
