# VibeMatch 项目进度追踪

## 项目概述
基于开源项目 Movies++ 进行改造，构建符合 AIE6002 课程要求的 RAG 语义电影推荐系统。保留 Movies++ 的前端交互优势，重构后端为本地可控的 LangChain + ChromaDB 架构，并补充完整的学术评估体系。

---

## 当前状态
- [x] 已下载并导入 Movies++ 项目
- [x] 已完成项目结构分析
- [x] 已完成技术栈确定
- [x] 已完成与课程要求的差距分析
- [x] 已初始化 Git 仓库并推送到 GitHub
- [x] **Phase 1: 数据预处理与向量化** ✅ 完成
- [ ] **Phase 2: RAG 核心后端开发** 🔄 代码已写，待测试
- [ ] **Phase 3: Baseline 实现**
- [ ] **Phase 4: 评估体系搭建**
- [ ] **Phase 5: 前端适配**
- [ ] **Phase 6: 系统集成与测试**
- [ ] **Phase 7: 论文与 Presentation**

---

## Phase 1: 数据预处理与向量化 ✅ 完成 (2026-05-01)

### 已完成的工作

#### 1.1 后端环境搭建
- 创建 `backend/` 目录
- 编写 `requirements.txt`（ChromaDB 1.5.x 预编译版本，无需 C++ 编译器）
- 安装所有 Python 依赖

#### 1.2 数据预处理 (`data_processor.py`)
- 加载 TMDB 5000 Movie Dataset（`tmdb_5000_movies.csv`）
- 解析 JSON 列（genres, keywords）
- 移除缺失 overview 的电影（4 部）
- 拼接富文本：Title + Year + Genres + Keywords + Overview
- 输出 `backend/data/movies_processed.json`（4799 部电影）

#### 1.3 模型下载 (`download_model.py`)
- 使用 `huggingface_hub.snapshot_download` 下载到项目目录
- 支持国内镜像源（hf-mirror.com）
- 模型保存到 `backend/models/all-MiniLM-L6-v2/`
- 下载大小约 80MB，Embedding 维度 384

#### 1.4 向量化 (`vectorstore.py`)
- 使用 `langchain-huggingface` 的 `HuggingFaceEmbeddings`
- 自动检测本地模型，不存在时回退到 OpenAI Embedding
- 构建 ChromaDB 向量数据库，持久化到 `backend/chroma_db/`
- 支持 Cosine Similarity 和 MMR 两种检索模式
- 测试通过：查询 "sci-fi movie about space travel" 返回 Gravity、Interstellar、Space Cowboys

### 解决的环境问题

| 问题 | 解决方案 |
|:---|:---|
| `chroma-hnswlib` 编译失败 | 升级到 ChromaDB 1.5.8（预编译 wheel） |
| NumPy 2.x 与 ChromaDB 0.4.x 不兼容 | 升级 ChromaDB 到 1.5.x |
| HuggingFace 模型下载超时 | 使用 hf-mirror.com 国内镜像 |
| `HuggingFaceEmbeddings` 弃用警告 | 改用 `langchain-huggingface` 包 |
| 模型下载路径不匹配 | 使用 `snapshot_download(local_dir=...)` 直接下载到目标目录 |

---

## Phase 2: RAG 核心后端开发 🔄 代码已写，待测试

### 已编写的代码

#### 2.1 Prompt 工程 (`prompts.py`)
- RAG 推荐 Prompt：严格规则（只推荐 Context 中的电影，禁止编造）
- Pure-LLM Baseline Prompt
- 幻觉检测 Prompt

#### 2.2 RAG Chain (`rag_chain.py`)
- `RAGPipeline` 类：封装完整 RAG 流程
- 支持 Similarity 和 MMR 两种检索模式
- `format_docs()` 格式化检索结果
- `create_pure_llm_chain()`：Pure-LLM Baseline
- `create_retrieval_chain()`：Retrieval-Only Baseline

#### 2.3 FastAPI 服务 (`main.py`)
- `POST /chat`：主推荐接口
- `POST /baseline/pure-llm`：Pure-LLM Baseline
- `POST /baseline/retrieval-only`：Retrieval-Only Baseline
- `GET /health`：健康检查
- CORS 配置：允许 `localhost:3000` 访问

### 待完成
- [ ] 配置 OpenAI API Key 并测试端到端
- [ ] 验证 LLM 生成回复无幻觉
- [ ] 验证 MMR 模式比 Similarity 模式更多样
- [ ] 验证 API 响应时间 < 3 秒

---

## Phase 3: Baseline 实现 (待开始)

### 3.1 Baseline 1: Pure-LLM
- 代码已在 `rag_chain.py` 中实现
- 需要端到端测试

### 3.2 Baseline 2: Tag-Based Filtering
- 需要新建 `backend/baselines.py`
- 基于 TMDB genre 标签的精确匹配

### 3.3 Baseline 3: Retrieval-Only
- 代码已在 `main.py` 中实现
- 需要端到端测试

---

## Phase 4: 评估体系搭建 (待开始)

### 4.1 测试查询集
- 构造 20-30 个覆盖不同复杂度的查询
- 输出：`backend/evaluation/test_queries.json`

### 4.2 评估指标
- 幻觉率 (Hallucination Rate)
- 检索多样性 (Intra-List Diversity)
- 响应延迟 (Latency)
- 用户满意度 (Likert 1-5)

### 4.3 消融实验
- Full System vs w/o MMR vs w/o Chunking vs w/o LLM

---

## Phase 5: 前端适配 (待开始)

### 5.1 后端 API 对接
- 修改 `app/Ai.tsx`，替换 `LangflowClient` 为调用本地 FastAPI

### 5.2 新增溯源面板
- 新建 `app/SourcePanel.tsx`

### 5.3 新增配置面板
- 切换检索模式、调节 Top-K、调节 MMR Lambda

---

## Phase 6: 系统集成与测试 (待开始)

### 6.1 端到端测试
- 后端 + 前端联调

### 6.2 性能优化
- 向量检索缓存
- LLM 响应流式传输

---

## Phase 7: 论文与 Presentation (待开始)

### 7.1 Final Paper
- Abstract, Introduction, Related Work, Methodology, Experiments, Results, Discussion, Conclusion

### 7.2 Presentation (10 分钟)
- Hook → Problem → Solution → Demo → Results → Conclusion

---

## 依赖关系图

```
Phase 1 (环境+数据) ✅
    │
    ▼
Phase 2 (RAG核心) 🔄 ──▶ Phase 3 (Baseline)
    │                       │
    ▼                       ▼
Phase 5 (前端适配) ◀── Phase 4 (评估体系)
    │
    ▼
Phase 6 (集成测试)
    │
    ▼
Phase 7 (论文+Presentation)
```

---

## 风险与应对

| 风险 | 概率 | 影响 | 应对策略 |
|:---|:---|:---|:---|
| OpenAI API 额度不足 | 中 | 高 | 使用 GPT-4o-mini，成本低；准备备用 Key |
| 前端对接复杂 | 中 | 中 | 简化前端，优先保证后端实验 |
| 评估人工标注耗时 | 高 | 中 | 提前招募同学，设计清晰标注指南 |
| 论文时间不够 | 中 | 高 | Phase 7 预留 5 天，每天固定写作时间 |

---

*最后更新：2026-05-01*
*当前进度：Phase 1 完成，Phase 2 代码已写待测试*
