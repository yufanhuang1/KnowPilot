
# 🤖 KnowPilot 智能对话助手

一个融合本地知识库与网络检索的智能 Agent 对话系统，支持上下文记忆、多工具调用、本地文档上传检索等功能。
适用于企业内部问答、知识检索型助手、教育答疑等场景。

## 🚀 项目亮点

- ✅ **支持多轮上下文记忆**：对话历史自动存储与追踪
- ✅ **知识增强生成（RAG）**：支持本地文档上传，基于向量库检索增强回答
- ✅ **多工具调用 Agent**：内置搜索、计算器、知识库等工具，支持 LangChain Agent 执行
- ✅ **融合 BM25 与向量检索**：兼顾关键词与语义相关性，提升召回质量


## 🧱 技术栈

| 分类 | 技术 |
|------|------|
| 编程语言 | Python 3.10+ |
| 大模型接口 | OpenAI / DashScope API |
| 框架 | LangChain, LlamaIndex, FastAPI, Streamlit |
| 知识库 | ChromaDB + BM25 融合检索 |
| 嵌入模型 | DashScope Embedding / OpenAI Embedding |
| 工具调用 | LangChain Agent Tool 调度系统 |


## 🧪 使用方法

### 1️⃣ 安装依赖

\`\`\`bash
pip install -r requirements.txt
\`\`\`

### 2️⃣ 启动后端接口

\`\`\`bash
uvicorn app:app --host 0.0.0.0 --port 8000
\`\`\`

### 3️⃣ 启动前端（Streamlit）

\`\`\`bash
streamlit run app.py
\`\`\`

## 🧠 示例问法

\`\`\`
- 什么是 Transformer？
- 帮我用 Python 写一个折线图代码
- 计算 3.14 × (1 + 2) 的结果
- 上传后的文档提到的项目名称是什么？
\`\`\`

## 📌 TODO

- [ ] 支持多模态知识库上传与构建
- [ ] 添加rerank模块优化数据检索
- [ ] 增加 Web UI 上传多个文档的批处理支持

