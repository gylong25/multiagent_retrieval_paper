"""
一个可直接运行的多智能体 RAG（检索增强生成）示例项目。

requirements.txt 依赖如下：
- pyautogen
- chromadb
- langchain-text-splitters
- openai
- python-dotenv
- requests
- beautifulsoup4
- lxml
- pypdf
- reportlab

运行步骤：
1. 先执行：pip install -r requirements.txt
2. 复制：.env.example -> .env，并填入可用的 API Key / Base URL
3. 如果本机已启动 Grobid，本脚本会优先使用 Grobid 解析 PDF
4. 执行：python main.py
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import autogen
import chromadb
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import APIConnectionError, APIError, APITimeoutError, OpenAI, RateLimitError
from pypdf import PdfReader
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


load_dotenv()


def _configure_stdio() -> None:
    """避免 Windows 控制台默认 GBK 编码导致的 UnicodeEncodeError。"""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                # 部分运行环境不允许重配；忽略即可（至少 errors=replace 会避免崩溃）
                pass


_configure_stdio()

@dataclass
class AppConfig:
    """统一管理项目配置，便于后续替换模型或迁移部署环境。"""

    papers_dir: Path
    chroma_dir: Path
    output_dir: Path
    collection_name: str
    top_k: int
    chunk_size: int
    chunk_overlap: int
    child_chunk_size: int
    child_chunk_overlap: int
    llm_model: str
    coordinator_model: str
    retrieval_model: str
    writer_model: str
    manager_model: str
    llm_api_key: str
    llm_base_url: str
    embedding_model: str
    embedding_api_key: str
    embedding_base_url: str
    embedding_timeout: float
    embedding_max_retries: int
    embedding_retry_backoff: float
    embedding_batch_size: int
    grobid_url: str
    demo_question: str

    @classmethod
    def from_env(cls) -> "AppConfig":
        """从 .env 中读取配置；如果未配置则使用更适合演示的默认值。"""
        def _as_bool(value: str, default: bool = False) -> bool:
            if value is None:
                return default
            return value.strip().lower() in {"1", "true", "yes", "y", "on"}

        llm_api_key = os.getenv("LLM_API_KEY", "").strip()
        embedding_api_key = os.getenv("EMBEDDING_API_KEY", llm_api_key).strip()
        llm_model = os.getenv("LLM_MODEL", "qwen-plus").strip()

        def _pick_model(env_name: str) -> str:
            value = os.getenv(env_name, "").strip()
            return value or llm_model

        chunk_size = int(os.getenv("CHUNK_SIZE", "900"))
        chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "150"))
        child_chunk_size = int(
            os.getenv("CHILD_CHUNK_SIZE", str(max(200, chunk_size // 3)))
        )
        child_chunk_overlap = int(
            os.getenv("CHILD_CHUNK_OVERLAP", str(max(50, child_chunk_size // 4)))
        )

        return cls(
            papers_dir=Path(os.getenv("PAPERS_DIR", "./papers")),
            chroma_dir=Path(os.getenv("CHROMA_DIR", "./chroma_db")),
            output_dir=Path(os.getenv("OUTPUT_DIR", "./outputs")),
            collection_name=os.getenv("CHROMA_COLLECTION", "pdf_rag_demo").strip(),
            top_k=int(os.getenv("TOP_K", "2")),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            child_chunk_size=child_chunk_size,
            child_chunk_overlap=child_chunk_overlap,
            llm_model=llm_model,
            coordinator_model=_pick_model("COORDINATOR_MODEL"),
            retrieval_model=_pick_model("RETRIEVAL_MODEL"),
            writer_model=_pick_model("WRITER_MODEL"),
            manager_model=_pick_model("MANAGER_MODEL"),
            llm_api_key=llm_api_key,
            llm_base_url=os.getenv(
                "LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
            ).strip(),
            embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-v4").strip(),
            embedding_api_key=embedding_api_key,
            embedding_base_url=os.getenv(
                "EMBEDDING_BASE_URL",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
            ).strip(),
            embedding_timeout=float(os.getenv("EMBEDDING_TIMEOUT", "30")),
            embedding_max_retries=int(os.getenv("EMBEDDING_MAX_RETRIES", "2")),
            embedding_retry_backoff=float(os.getenv("EMBEDDING_RETRY_BACKOFF", "1.6")),
            embedding_batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "10")),
            grobid_url=os.getenv("GROBID_URL", "http://localhost:8070").strip(),
            demo_question=os.getenv(
                "DEMO_QUESTION",
                "总结一下目前关于大模型在 RAG 系统中的应用方法。",
            ).strip(),
        )

    def ensure_directories(self) -> None:
        """首次运行时自动创建需要的目录。"""
        self.papers_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def validate_runtime(self) -> None:
        """在真正调用模型前检查关键配置，避免运行到中途才失败。"""
        missing = []
        if not self.llm_api_key:
            missing.append("LLM_API_KEY")
        if not self.embedding_api_key :
            missing.append("EMBEDDING_API_KEY")

        if missing:
            raise RuntimeError(
                "缺少必要环境变量："
                + ", ".join(missing)
                + "。请先复制 .env.example 为 .env，并填入可用配置。"
            )

        if self.chunk_size <= 0 or self.child_chunk_size <= 0:
            raise RuntimeError("CHUNK_SIZE/CHILD_CHUNK_SIZE 必须是正整数。")
        if self.chunk_overlap < 0 or self.child_chunk_overlap < 0:
            raise RuntimeError("CHUNK_OVERLAP/CHILD_CHUNK_OVERLAP 不能为负数。")
        if self.chunk_overlap >= self.chunk_size:
            raise RuntimeError("CHUNK_OVERLAP 必须小于 CHUNK_SIZE。")
        if self.child_chunk_overlap >= self.child_chunk_size:
            raise RuntimeError("CHILD_CHUNK_OVERLAP 必须小于 CHILD_CHUNK_SIZE。")

        if self.embedding_batch_size <= 0:
            raise RuntimeError(
                "EMBEDDING_BATCH_SIZE 必须是正整数。"
                f"当前值：{self.embedding_batch_size!r}"
            )


def extract_pdf_text(pdf_path: Path, grobid_url: str) -> str:
    """通过 Grobid 解析论文，获得更接近学术场景的结构化文本（深度绑定章节标题版）。"""
    endpoint = grobid_url.rstrip("/") + "/api/processFulltextDocument"

    with pdf_path.open("rb") as pdf_file:
        response = requests.post(
            endpoint,
            files={"input": (pdf_path.name, pdf_file, "application/pdf")},
            data={"consolidateHeader": "0", "segmentSentences": "1"},
            timeout=120,
        )

    response.raise_for_status()

    soup = BeautifulSoup(response.text, "xml")
    blocks: List[str] = []

    # 【核心优化】：状态变量，用于记忆当前所处的章节标题
    current_section = ""

    # 按文档顺序遍历所有的标题和段落标签
    for tag in soup.find_all(["head", "p"]):
        text = tag.get_text(" ", strip=True)
        if not text:
            continue

        if tag.name == "head":
            # 遇到新的标题，更新状态变量。
            # 此时不再把它单独作为一个 block 加入列表，避免被切分器孤立
            current_section = text
        elif tag.name == "p":
            # 遇到正文段落，检查是否有记忆的章节标题
            if current_section:
                # 强绑定：用明确的格式将章节标题与段落内容缝合
                blocks.append(f"【所属章节：{current_section}】\n{text}")
            else:
                # 应对文章开头没有小标题的摘要（Abstract）等段落
                blocks.append(text)

    # 用双换行符拼接所有拼装好的段落
    # LangChain 的 RecursiveCharacterTextSplitter 会优先按 "\n\n" 切分
    # 这样就能完美保证 "标题 + 对应段落" 被完整切入同一个父块中
    full_text = "\n\n".join(blocks).strip()

    if not full_text:
        raise ValueError(f"Grobid 成功返回，但未从 {pdf_path.name} 中提取到文本。")

    return full_text

class OpenAICompatibleEmbedder:
    """对 OpenAI 兼容 Embedding 接口做一层封装，便于切换提供商。"""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        timeout: float,
        max_retries: int,
        retry_backoff: float,
    ) -> None:
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=0,
        )
        self.model = model
        self.max_retries = max_retries
        self.retry_backoff = max(1.0, retry_backoff)



    def embed_texts(self, texts: List[str], batch_size: int = 10) -> List[List[float]]:
        """分批生成向量，避免一次请求过长文本列表。"""
        vectors: List[List[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]

            last_error: Optional[BaseException] = None
            for attempt in range(self.max_retries + 1):
                try:
                    response = self.client.embeddings.create(
                        model=self.model, input=batch
                    )
                    vectors.extend(item.embedding for item in response.data)
                    last_error = None
                    break
                except (APIConnectionError, APITimeoutError) as exc:
                    last_error = exc
                    if attempt < self.max_retries:
                        time.sleep(self.retry_backoff**attempt)
                        continue
                    break
                except (RateLimitError, APIError) as exc:
                    last_error = exc
                    message = str(exc)
                    if "batch size is invalid" in message.lower() and len(batch) > 10:
                        raise RuntimeError(
                            "Embedding 接口报错：当前提供商限制单次 embeddings 请求的 batch size "
                            "不得超过 10。请将 EMBEDDING_BATCH_SIZE 设为 10（或更小）。"
                        ) from exc
                    if attempt < self.max_retries:
                        time.sleep(self.retry_backoff**attempt)
                        continue
                    break

            if last_error is not None:
                raise RuntimeError(
                    "Embedding 接口调用失败，请检查网络、EMBEDDING_BASE_URL 或密钥配置。"
                ) from last_error
        return vectors


class PDFRAGService:
    """负责 PDF 文献入库、向量化、检索，是整个 RAG 系统的知识底座。"""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.chroma_client = chromadb.PersistentClient(path=str(config.chroma_dir))
        self.collection = self.chroma_client.get_or_create_collection(
            name=config.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        existing_dim = self._infer_existing_dim()
        self.embedder = OpenAICompatibleEmbedder(
            api_key=config.embedding_api_key,
            base_url=config.embedding_base_url,
            model=config.embedding_model,
            timeout=config.embedding_timeout,
            max_retries=config.embedding_max_retries,
            retry_backoff=config.embedding_retry_backoff,
        )
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=["\n\n", "\n", "。", ".", " ", ""],
        )
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.child_chunk_size,
            chunk_overlap=config.child_chunk_overlap,
            separators=["\n\n", "\n", "。", ".", " ", ""],
        )

    def _infer_existing_dim(self) -> Optional[int]:
        """如果 Collection 已有数据，尝试推断向量维度，避免兜底维度不一致。"""
        try:
            if self.collection.count() == 0:
                return None
            sample = self.collection.get(limit=1, include=["embeddings"])
            embeddings = sample.get("embeddings", [])
            if embeddings and embeddings[0]:
                return len(embeddings[0])
        except Exception:
            return None
        return None

    def _fingerprint_pdf(self, pdf_path: Path) -> str:
        """基于文件元信息生成指纹，用于判断是否需要重建索引。"""
        stat = pdf_path.stat()
        return f"{stat.st_mtime_ns}-{stat.st_size}"

    def _is_already_indexed(self, source: str, fingerprint: str) -> bool:
        """判断指定 PDF 是否已在向量库中且与当前文件一致。"""
        try:
            existing = self.collection.get(
                where={"source": source},
                include=["metadatas"],
                limit=1,
            )
        except Exception:
            return False

        metadatas = existing.get("metadatas") or []
        if not metadatas:
            return False
        metadata = metadatas[0] or {}
        return metadata.get("source_fingerprint") == fingerprint

    def build_index(self) -> Dict[str, int]:
        """扫描 PDF 目录、切分文本、生成向量，并写入 Chroma Collection。"""
        pdf_files = sorted(self.config.papers_dir.glob("*.pdf"))
        if not pdf_files:
            raise RuntimeError(
                f"未在 {self.config.papers_dir.resolve()} 中找到 PDF 文件。"
            )

        pdf_count = 0
        chunk_count = 0
        skipped_count = 0

        for pdf_path in pdf_files:
            source = str(pdf_path.resolve())
            fingerprint = self._fingerprint_pdf(pdf_path)
            if self._is_already_indexed(source, fingerprint):
                print(f"[INFO] {pdf_path.name} 已存在且未变更，跳过索引。")
                skipped_count += 1
                continue

            raw_text = extract_pdf_text(pdf_path, self.config.grobid_url)
            parent_chunks = [
                chunk.strip()
                for chunk in self.parent_splitter.split_text(raw_text)
                if chunk.strip()
            ]
            if not parent_chunks:
                print(f"[WARN] {pdf_path.name} 未生成有效文本块，已跳过。")
                continue

            embedding_inputs: List[str] = []
            documents: List[str] = []
            ids: List[str] = []
            metadatas: List[Dict[str, Any]] = []

            for parent_idx, parent_chunk in enumerate(parent_chunks):
                parent_id = hashlib.md5(
                    f"{source}::parent::{parent_idx}::{parent_chunk[:80]}".encode(
                        "utf-8"
                    )
                ).hexdigest()[:16]
                parent_title = pdf_path.stem
                child_chunks = [
                    child.strip()
                    for child in self.child_splitter.split_text(parent_chunk)
                    if child.strip()
                ]
                if not child_chunks:
                    child_chunks = [parent_chunk]

                for child_idx, child_chunk in enumerate(child_chunks):
                    stable_hash = hashlib.md5(
                        f"{source}::{parent_id}::{child_idx}::{child_chunk[:80]}".encode(
                            "utf-8"
                        )
                    ).hexdigest()[:16]
                    ids.append(
                        f"{pdf_path.stem}-p{parent_idx}-c{child_idx}-{stable_hash}"
                    )
                    documents.append(child_chunk)
                    embedding_inputs.append(f"{parent_title}\n{child_chunk}".strip())
                    metadatas.append(
                        {
                            "source": source,
                            "file_name": pdf_path.name,
                            "parent_id": parent_id,
                            "parent_index": parent_idx,
                            "parent_title": parent_title,
                            "parent_content": parent_chunk,
                            "child_index": child_idx,
                            "source_fingerprint": fingerprint,
                        }
                    )

            if not documents:
                print(f"[WARN] {pdf_path.name} 未生成子块，已跳过。")
                continue

            embeddings = self.embedder.embed_texts(
                embedding_inputs, batch_size=self.config.embedding_batch_size
            )

            try:
                self.collection.delete(where={"source": source})
            except Exception:
                # 某些情况下文件第一次入库不存在旧数据，这里静默跳过即可。
                pass

            self.collection.upsert(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
            )

            pdf_count += 1
            chunk_count += len(documents)

        return {
            "pdf_count": pdf_count,
            "chunk_count": chunk_count,
            "skipped_count": skipped_count,
        }

    def search(self, question: str, top_k: int = 4) -> str:
        """供 Retrieval Agent 调用的工具函数，返回结构化 JSON 字符串。"""
        if self.collection.count() == 0:
            raise RuntimeError("当前 Chroma Collection 为空，请先构建索引。")

        query_embedding = self.embedder.embed_texts([question])[0]
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=max(top_k * 3, top_k),
            include=["documents", "metadatas", "distances"],
        )

        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        payload: Dict[str, Any] = {"question": question, "results": []}
        seen_parents: Dict[str, Dict[str, Any]] = {}
        for index, document in enumerate(documents):
            metadata = metadatas[index] if index < len(metadatas) else {}
            distance = distances[index] if index < len(distances) else None
            parent_id = metadata.get("parent_id") or (
                f"{metadata.get('source', '')}::{metadata.get('parent_index', -1)}"
            )
            entry = seen_parents.get(parent_id)
            if entry is None:
                entry = {
                    "rank": len(payload["results"]) + 1,
                    "title": metadata.get("parent_title")
                    or metadata.get("file_name", "unknown.pdf"),
                    "file_name": metadata.get("file_name", "unknown.pdf"),
                    "parent_index": metadata.get("parent_index", -1),
                    "source": metadata.get("source", ""),
                    "distance": distance,
                    "content": metadata.get("parent_content") or document,
                    "child_snippets": [],
                }
                seen_parents[parent_id] = entry
                payload["results"].append(entry)

            if document:
                snippets = entry.setdefault("child_snippets", [])
                if document not in snippets and len(snippets) < 3:
                    snippets.append(document)
            if distance is not None and (
                entry.get("distance") is None or distance < entry["distance"]
            ):
                entry["distance"] = distance

        payload["results"] = payload["results"][:top_k]

        return json.dumps(payload, ensure_ascii=False, indent=2)


class MultiAgentLiteratureRAG:
    """封装 AutoGen 的多智能体编排逻辑。"""

    def __init__(self, config: AppConfig, rag_service: PDFRAGService) -> None:
        self.config = config
        self.rag_service = rag_service

        self.tool_executor = autogen.UserProxyAgent(
            name="ToolExecutor",
            human_input_mode="NEVER",
            code_execution_config=False,
            llm_config=False,
            system_message=(
                "你只负责执行工具函数，不直接产出最终结论。"
                "不要输出 emoji 或特殊符号，避免终端编码问题。"
            ),
        )

        self.coordinator = autogen.AssistantAgent(
            name="Coordinator",
            system_message=(
                "你是 Coordinator Agent，负责接收用户问题、管理流程，并在最后"
                "输出一份面向用户的结构化结论。你不能编造文献证据，必须严格基于"
                "RetrievalAnalyst 和 Writer 的内容做最终整合。"
                "请只输出纯文本（不要 emoji / 特殊符号）。"
            ),
            llm_config=self._build_llm_config(self.config.coordinator_model),
        )

        self.retrieval_agent = autogen.AssistantAgent(
            name="RetrievalAnalyst",
            system_message=(
                "你是 Retrieval & Analysis Agent。你的第一步必须调用工具 "
                "`search_pdf_knowledge_base` 检索相关 PDF 文本块。然后根据"
                "检索结果提炼：1. 核心观点 2. 关键证据 3. 可以支撑总结的结论。"
                "不要凭空补充未检索到的事实。"
                "你只输出“要点提炼/证据摘录/结论”，不要写最终长报告（由 Writer 完成）。"
                "请只输出纯文本（不要 emoji / 特殊符号）。"
            ),
            llm_config=self._build_llm_config(self.config.retrieval_model),
        )

        self.writer_agent = autogen.AssistantAgent(
            name="Writer",
            system_message=(
                "你是 Writer Agent。请把 RetrievalAnalyst 提炼出的信息组织成"
                "专业、客观、结构化的中文文献总结报告。报告必须包含：引言、"
                "核心方法/观点概述、总结、参考来源。若证据不足，要明确指出。"
                "请只输出纯文本（不要 emoji / 特殊符号）。"
            ),
            llm_config=self._build_llm_config(self.config.writer_model),
        )

        def search_pdf_knowledge_base(
            question: str, top_k: Optional[int] = None
        ) -> str:
            """为 AutoGen 工具调用包装检索函数，避免绑定方法无法设置属性。"""
            try:
                top_k_value = int(top_k) if top_k is not None else self.config.top_k
            except (TypeError, ValueError):
                top_k_value = self.config.top_k

            if top_k_value <= 0:
                top_k_value = self.config.top_k

            return self.rag_service.search(question, top_k=top_k_value)

        self.search_tool = search_pdf_knowledge_base

        autogen.register_function(
            self.search_tool,
            caller=self.retrieval_agent,
            executor=self.tool_executor,
            name="search_pdf_knowledge_base",
            description="从本地 Chroma 向量数据库中检索与用户问题最相关的 PDF 文本块。",
        )

        self.groupchat = autogen.GroupChat(
            agents=[
                self.coordinator,
                self.retrieval_agent,
                self.writer_agent,
                self.tool_executor,
            ],
            messages=[],
            max_round=8,
            speaker_selection_method=self._speaker_selection,
        )

        self.manager = autogen.GroupChatManager(
            groupchat=self.groupchat,
            llm_config=self._build_llm_config(self.config.manager_model),
        )

    def _build_llm_config(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """构建 AutoGen 所需的 OpenAI 兼容配置。"""
        model = (model_name or "").strip() or self.config.llm_model
        return {
            "temperature": 0.2,
            "config_list": [
                {
                    "model": model,
                    "api_key": self.config.llm_api_key,
                    "base_url": self.config.llm_base_url,
                }
            ],
        }

    def _speaker_selection(self, last_speaker: Any, groupchat: Any) -> Any:
        """用一个固定工作流确保三位核心 Agent 按预期顺序协作。"""
        last_name = getattr(last_speaker, "name", "")
        last_message = groupchat.messages[-1] if groupchat.messages else {}

        # 如果上一轮产生了工具调用请求，优先让 ToolExecutor 执行。
        pending_tool_call = bool(
            last_message.get("tool_calls")
            or last_message.get("function_call")
        )
        if last_name == self.retrieval_agent.name and pending_tool_call:
            return self.tool_executor

        # 工具执行完后，让 RetrievalAnalyst 读取工具输出并完成分析。
        if last_name == self.tool_executor.name:
            return self.retrieval_agent

        if last_name == self.coordinator.name:
            return self.retrieval_agent
        if last_name == self.retrieval_agent.name:
            return self.writer_agent
        if last_name == self.writer_agent.name:
            return None
        return self.coordinator

    def _extract_final_report(self) -> str:
        """从对话记录中挑选最后一条有效报告内容。"""
        preferred = {self.writer_agent.name, self.coordinator.name}
        for message in reversed(self.groupchat.messages):
            content = message.get("content")
            if content and message.get("name") in preferred:
                return content
        for message in reversed(self.groupchat.messages):
            content = message.get("content")
            if content:
                return content
        raise RuntimeError("最终消息为空，请检查模型配置或 Agent 提示词。")

    def run(self, question: str) -> str:
        """触发一次完整的多智能体协作，并返回最终报告。"""
        self.groupchat.messages.clear()

        self.coordinator.initiate_chat(
            self.manager,
            message=(
                "请围绕下面这个问题启动一次 PDF 文献检索与总结流程，并给出最终报告：\n"
                f"{question}"
            ),
            clear_history=True,
            summary_method="last_msg",
        )

        if not self.groupchat.messages:
            raise RuntimeError("GroupChat 未生成任何消息。")
        return self._extract_final_report()


def save_report(report: str, output_dir: Path) -> Path:
    """将最终报告保存到本地文件，方便二次编辑或分享。"""
    report_path = output_dir / "latest_report.md"
    report_path.write_text(report, encoding="utf-8")
    return report_path


def main() -> None:
    """演示完整流程：准备 PDF -> 建索引 -> 启动多 Agent -> 输出报告。"""
    config = AppConfig.from_env()
    config.ensure_directories()


    # 模型与向量接口都依赖有效的 API Key，因此在这里做统一校验。
    config.validate_runtime()

    rag_service = PDFRAGService(config)
    index_stats = rag_service.build_index()

    print(
        f"[INFO] 已完成索引构建：{index_stats['pdf_count']} 篇 PDF，"
        f"{index_stats['chunk_count']} 个文本块。"
    )
    skipped = index_stats.get("skipped_count", 0)
    if skipped:
        print(f"[INFO] 跳过 {skipped} 篇已索引且未变更的 PDF。")

    agent_system = MultiAgentLiteratureRAG(config, rag_service)
    final_report = agent_system.run(config.demo_question)
    report_path = save_report(final_report, config.output_dir)

    print("\n================ 最终报告 ================\n")
    print(final_report)
    print("\n==========================================\n")
    print(f"[INFO] 报告已保存到：{report_path.resolve()}")


if __name__ == "__main__":
    main()
