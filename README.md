nlpagent：多智能体 PDF 文献检索与总结（父子索引版）
====================================================

这是一个可直接运行的多智能体 RAG（Retrieval-Augmented Generation）示例项目：
从本地 PDF 文献库中构建向量索引，在对话中检索相关证据，并由多个智能体协作生成结构化的中文文献阅读报告。

核心特点
--------
1) 父子索引（Parent/Child Index）
   - 父块（parent chunk）：更长的上下文，用于最终返回与拼接摘要，避免只命中“几个词”就结束。
   - 子块（child chunk）：更短、更细粒度，用于向量检索，提高召回与精度。
   - 检索时先命中子块，再按 parent_id 聚合成父块返回（并附带少量 child snippets）。

2) 标题/章节绑定
   - PDF 通过 Grobid 解析后，将“章节标题（head）+ 段落（p）”强绑定在一起（形如“【所属章节：…】\n正文…”），
     尽量保证切分后仍能保留章节语义。

3) 去重跳过（增量索引）
   - 已入库且 PDF 未变更（mtime + size 指纹一致）会直接跳过，不重复生成 embedding。

4) 多智能体协作（AutoGen）
   - Coordinator：流程协调与最终整合
   - RetrievalAnalyst：强制先检索，再做证据提炼
   - Writer：将提炼结果写成结构化报告
   - ToolExecutor：只负责执行检索工具，不产出结论
   - GroupChatManager：负责编排对话轮次（可配置单独模型）

目录结构
--------
- main.py                多智能体 RAG 主程序（建索引 + 对话 + 保存报告）
- test_embedding.py      Embedding 接口连通性测试脚本
- baseline_llm.py        基线：单 LLM 直接回答（无检索）
- baseline_rag_llm.py    基线：仅 RAG 检索 + 拼接上下文 + 单 LLM 回答
- flask_server.py        Flask 后端（上传 PDF / 构建索引 / 聊天）
- gradio_app.py          Gradio 前端（上传 + 对话框）
- papers/                放置待入库的 PDF 文献
- chroma_db/             Chroma 持久化向量库目录（索引数据）
- outputs/               运行结果输出目录（报告/基线输出）
- .env                   运行配置（不要提交你的密钥）
- .env.example           配置示例（复制后填写）
- requirements.txt       依赖列表

技术栈
------
- Python 3.10+（Windows/PowerShell 友好）
- 智能体编排：pyautogen（Microsoft AutoGen）
- 向量数据库：chromadb（本地持久化）
- 文本切分：langchain-text-splitters（RecursiveCharacterTextSplitter）
- 模型调用：openai Python SDK（适配 OpenAI 兼容 API）
- PDF 解析：Grobid（外部服务，输出 TEI XML）+ BeautifulSoup/lxml 解析
- 服务化/前端（可选）：Flask + Gradio

运行前准备
----------
1) 安装依赖（建议使用当前项目的虚拟环境解释器）
   - PowerShell 示例：
     - .\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt

2) 配置环境变量
   - 将 .env.example 复制为 .env，然后填入你的 Key/Base URL：
     - LLM：LLM_API_KEY / LLM_BASE_URL / LLM_MODEL
     - Embedding：EMBEDDING_API_KEY / EMBEDDING_BASE_URL / EMBEDDING_MODEL
   - 注意：通义兼容 embeddings 常见限制单次 batch <= 10，建议配置：EMBEDDING_BATCH_SIZE=10

3) 启动 Grobid（必须）
   - 确保 GROBID_URL 可访问（默认：http://localhost:8070）
   - 若 Grobid 未启动，PDF 解析会直接失败（当前版本不再自动回退到 pypdf）。

推荐执行顺序（命令行）
----------------------
按下面顺序运行，最容易定位问题：

步骤 1：验证 Embedding 可用性
  - .\\.venv\\Scripts\\python.exe test_embedding.py
  - 结果：控制台打印 base_url/model/向量维度/耗时；成功说明 embedding 配置可用。

步骤 2：准备 PDF
  - 将论文 PDF 放入 papers/（或使用后端上传接口）。

步骤 3：一键跑通多智能体 RAG
  - .\\.venv\\Scripts\\python.exe main.py
  - 执行内容：
    1) 扫描 papers/ -> Grobid 解析
    2) 父子切分（CHUNK_SIZE/CHILD_CHUNK_SIZE）
    3) 子块向量化写入 Chroma（chroma_db/）
    4) 多智能体对话（Coordinator -> RetrievalAnalyst(工具检索) -> Writer）
  - 结果：
    - 终端打印最终报告
    - 输出文件：outputs/latest_report.md（每次运行会覆盖）

基线脚本（用于对比）
--------------------
1) 基线 A：单 LLM 直接回答（无检索）
  - .\\.venv\\Scripts\\python.exe baseline_llm.py --question \"你的问题\"
  - 结果：outputs/baseline_llm_YYYYMMDD_HHMMSS.md

2) 基线 B：RAG 检索 + 拼接上下文 + 单 LLM 回答
  - .\\.venv\\Scripts\\python.exe baseline_rag_llm.py --question \"你的问题\"
  - 可选：--top-k 4（覆盖 TOP_K）
  - 结果：outputs/baseline_rag_llm_YYYYMMDD_HHMMSS.md（包含：问题/回答/检索片段）

服务化（可选：后端 + 前端）
--------------------------
1) 启动 Flask 后端
  - .\\.venv\\Scripts\\python.exe flask_server.py
  - 接口：
    - GET  /health     查看服务状态与索引数量
    - POST /upload     multipart/form-data，字段名 files（支持多文件）；保存到 papers/ 并构建/增量更新索引
    - POST /chat       JSON：{\"question\": \"...\"}；返回 answer，并写入 outputs/latest_report.md

2) 启动 Gradio 前端
  - .\\.venv\\Scripts\\python.exe gradio_app.py
  - 功能：上传 PDF + 聊天对话框；通过 BACKEND_URL 调用 Flask。

关键配置项（.env）速查
----------------------
【LLM】
- LLM_MODEL / LLM_API_KEY / LLM_BASE_URL
- COORDINATOR_MODEL / RETRIEVAL_MODEL / WRITER_MODEL / MANAGER_MODEL
  - 不配置则自动回退到 LLM_MODEL

【Embedding】
- EMBEDDING_MODEL / EMBEDDING_API_KEY / EMBEDDING_BASE_URL
- EMBEDDING_TIMEOUT / EMBEDDING_MAX_RETRIES / EMBEDDING_RETRY_BACKOFF
- EMBEDDING_BATCH_SIZE（建议 <= 10）

【解析与索引】
- GROBID_URL
- CHROMA_DIR / CHROMA_COLLECTION（想强制重建可改 collection 名或删除 chroma_db/）
- CHUNK_SIZE / CHUNK_OVERLAP（父块）
- CHILD_CHUNK_SIZE / CHILD_CHUNK_OVERLAP（子块）
- TOP_K（检索返回父块数量）

【输出与目录】
- PAPERS_DIR / OUTPUT_DIR（默认输出目录对应 D:\\ALLM\\nlpagent\\outputs）

常见问题
--------
1) embeddings 报 400：batch size invalid
   - 将 EMBEDDING_BATCH_SIZE 设为 10 或更小。

2) 修改了切分参数/索引结构后检索效果异常
   - 建议删除 chroma_db/ 或更换 CHROMA_COLLECTION，重新建索引。

3) Grobid 连接失败
   - 检查 GROBID_URL、端口与网络；确保 Grobid 已启动并可访问。
