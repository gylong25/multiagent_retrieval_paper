"""
用于快速测试 Embedding 模型/接口是否可用的脚本。

用法（PowerShell）：
  1) 先确保已配置 .env（或直接传参）：
     python test_embedding.py

  2) 显式指定参数（会覆盖 .env）：
     python test_embedding.py --base-url https://dashscope.aliyuncs.com/compatible-mode/v1 --model text-embedding-v4 --api-key xxx

说明：
  - 该脚本只做一次最小 embedding 请求（输入 1 条短文本）。
  - 成功会打印向量维度、耗时等信息；失败会打印可定位的错误原因与建议。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Optional

from dotenv import load_dotenv
from openai import APIConnectionError, APIError, APITimeoutError, OpenAI, RateLimitError


def _mask_key(key: str) -> str:
    if not key:
        return ""
    key = key.strip()
    if len(key) <= 8:
        return "*" * len(key)
    return key[:4] + "*" * (len(key) - 8) + key[-4:]


def test_embedding(
    *,
    api_key: str,
    base_url: str,
    model: str,
    timeout: float,
    text: str,
) -> int:
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=0)

    print("== Embedding 连通性测试 ==")
    print(f"- base_url: {base_url}")
    print(f"- model:    {model}")
    print(f"- api_key:  {_mask_key(api_key)}")
    print(f"- timeout:  {timeout}s")
    print(f"- input:    {text!r}")

    started = time.perf_counter()
    try:
        resp = client.embeddings.create(model=model, input=[text])
    except (APIConnectionError, APITimeoutError) as exc:
        elapsed = time.perf_counter() - started
        print("\n[FAIL] 连接/超时错误：", repr(exc))
        print(f"耗时：{elapsed:.2f}s")
        print("建议：")
        print("- 检查网络能否访问 EMBEDDING_BASE_URL（必要时使用代理/VPN/DNS）。")
        print("- 检查 base_url 是否写对（需兼容 OpenAI embeddings 接口路径）。")
        return 2
    except RateLimitError as exc:
        elapsed = time.perf_counter() - started
        print("\n[FAIL] 触发限流：", repr(exc))
        print(f"耗时：{elapsed:.2f}s")
        print("建议：稍后重试或降低并发。")
        return 3
    except APIError as exc:
        elapsed = time.perf_counter() - started
        print("\n[FAIL] API 返回错误：", repr(exc))
        print(f"耗时：{elapsed:.2f}s")
        print("建议：")
        print("- 检查 EMBEDDING_API_KEY 是否有效/未过期。")
        print("- 检查 model 名称是否正确、当前账号是否有权限。")
        return 4
    except Exception as exc:  # noqa: BLE001
        elapsed = time.perf_counter() - started
        print("\n[FAIL] 未知错误：", repr(exc))
        print(f"耗时：{elapsed:.2f}s")
        return 5

    elapsed = time.perf_counter() - started
    if not resp.data or not getattr(resp.data[0], "embedding", None):
        print("\n[FAIL] 返回结果不包含 embedding 向量，响应结构异常。")
        return 6

    vec = resp.data[0].embedding
    print("\n[OK] Embedding 请求成功")
    print(f"- 向量维度: {len(vec)}")
    print(f"- 耗时:     {elapsed:.2f}s")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(description="测试 Embedding 模型/接口是否可用")
    parser.add_argument(
        "--api-key",
        default=os.getenv("EMBEDDING_API_KEY") or os.getenv("LLM_API_KEY") or "",
        help="Embedding API Key（默认读取 EMBEDDING_API_KEY，其次 LLM_API_KEY）",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("EMBEDDING_BASE_URL", "").strip()
        or os.getenv("LLM_BASE_URL", "").strip()
        or "https://dashscope.aliyuncs.com/compatible-mode/v1",
        help="OpenAI 兼容 Base URL（默认读取 EMBEDDING_BASE_URL，其次 LLM_BASE_URL）",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("EMBEDDING_MODEL", "text-embedding-v4").strip(),
        help="Embedding 模型名（默认读取 EMBEDDING_MODEL）",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(os.getenv("EMBEDDING_TIMEOUT", "30")),
        help="请求超时秒数（默认读取 EMBEDDING_TIMEOUT=30）",
    )
    parser.add_argument(
        "--text",
        default="embedding connectivity test",
        help="测试文本（默认是一条短句）",
    )

    args = parser.parse_args(argv)
    if not args.api_key.strip():
        print("[FAIL] 未配置 API Key：请设置 EMBEDDING_API_KEY（或通过 --api-key 传入）。")
        return 1

    return test_embedding(
        api_key=args.api_key.strip(),
        base_url=args.base_url.strip(),
        model=args.model.strip(),
        timeout=float(args.timeout),
        text=str(args.text),
    )


if __name__ == "__main__":
    raise SystemExit(main())

