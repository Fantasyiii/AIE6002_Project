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
- [x] **Phase 2: RAG 核心后端开发** ✅ 完成
- [x] **Phase 3: Baseline 实现** ✅ 完成
- [x] **Phase 4: 评估体系搭建** ✅ 完成
- [x] **Phase 5: 前端适配** ✅ 完成
- [ ] **Phase 6: 系统集成与测试** 🔄 进行中
- [ ] **Phase 7: 论文与 Presentation** 待开始

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

## Phase 2: RAG 核心后端开发 ✅ 完成 (2026-05-01)

### 已完成的工作

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
- 修复 `RunnableLambda` 包装 retriever 函数，解决 LCEL `|` 运算符 TypeError

#### 2.3 FastAPI 服务 (`main.py`)
- `POST /chat`：主推荐接口
- `POST /baseline/pure-llm`：Pure-LLM Baseline
- `POST /baseline/retrieval-only`：Retrieval-Only Baseline
- `GET /health`：健康检查
- CORS 配置：允许 `localhost:3000` 访问
- 修复 `global pipeline` SyntaxError，改用函数参数传递

### API 配置
- 使用 NVIDIA API (`qwen/qwen3.5-122b-a10b`)
- API Key 配置在 `backend/.env` 中
- 支持 OpenAI 回退

---

## Phase 3: Baseline 实现 ✅ 完成 (2026-05-01)

### 3.1 Baseline 1: Pure-LLM (`baselines.py`)
- 直接调用 LLM，无检索上下文
- 用于对比 RAG 的幻觉率降低效果

### 3.2 Baseline 2: Tag-Based Filtering (`baselines.py`)
- 基于 TMDB genre 标签的关键词匹配
- 提取查询中的 genre 关键词，精确匹配电影类型
- 按 vote_average 排序返回 Top 5

### 3.3 Baseline 3: Retrieval-Only (`baselines.py`)
- 仅向量检索，无 LLM 生成
- 用于评估 LLM 生成对推荐质量的影响

---

## Phase 4: 评估体系搭建 ✅ 完成 (2026-05-02)

### 4.1 测试查询集 (`evaluation/test_queries.json`)
- 构造 15 个覆盖不同复杂度的查询：
  - 简单查询（3个）：明确类型/主题
  - Vibe 查询（4个）：情感/氛围描述
  - 多条件查询（4个）：类型 + 年代 + 风格
  - 极端/边缘查询（4个）：小众需求

### 4.2 评估指标 (`evaluation/metrics.py`)
- **幻觉率 (Hallucination Rate)**：`extract_movie_titles()` 提取回答中的电影名，与检索来源 fuzzy match
- **多样性 (Diversity)**：基于 genre Jaccard distance 计算 intra-list diversity
- **相关性 (Relevance)**：查询关键词与电影 genre/overview 的重叠度
- **延迟 (Latency)**：端到端响应时间

### 4.3 自动化评估脚本 (`evaluation/run_eval.py`)
- 对 5 个系统各运行 15 个查询
- 自动聚合指标，生成 JSON 和 Markdown 报告
- 输出：`evaluation/results/evaluation_report.md`

### 4.4 评估结果

| System | Queries | Hallucination Rate | Avg Latency | Avg Recommendations |
|--------|---------|-------------------|-------------|-------------------|
| VibeMatch (RAG) | 15 | 61.00% | 17232ms | 5.0 |
| VibeMatch (MMR) | 15 | 54.00% | 7012ms | 5.0 |
| Pure-LLM | 15 | 100.00% | 27633ms | 0.0 |
| Tag-Based | 15 | 100.00% | 91ms | 0.0 |
| Retrieval-Only | 15 | 11.00% | 292ms | 5.0 |

**分析**：
- RAG 相比 Pure-LLM 降低了幻觉率（100% → 61%）
- MMR 模式比 Similarity 模式更快（7s vs 17s），幻觉率更低
- Retrieval-Only 幻觉率最低（11%），但缺乏 LLM 的解释能力
- Tag-Based 无法返回推荐（查询多为 vibe 描述，不含明确 genre 关键词）

---

## Phase 5: 前端适配 ✅ 完成 (2026-05-02)

### 5.1 移除 AI SDK RSC 依赖
- 删除 `app/Ai.tsx`（原 Langflow + OpenAI RSC 架构）
- 删除 `app/useMovieSearch.ts` 中的 `ai/rsc` 依赖
- 改为标准 React Hooks + `fetch` 调用 FastAPI

### 5.2 重写核心组件

#### `app/useMovieSearch.ts`
- 定义 `Message`、`MovieSource`、`ChatResponse` 类型
- `useMovieSearch()` Hook：管理消息状态、加载状态
- `search()` 函数：POST 到 `http://localhost:8000/chat`
- 支持错误处理和加载指示

#### `app/SearchForm.tsx`
- 添加加载状态指示器（旋转动画）
- 更新 placeholder 为 "Describe the vibe you're looking for..."
- 添加示例提示文本

#### `app/page.tsx`
- 重写为聊天界面布局
- 用户消息：右侧气泡
- 助手消息：左侧，包含 Markdown 渲染的回答 + Sources 卡片
- SourceCard 组件：显示电影标题、年份、类型、简介
- 显示响应时间

#### `app/layout.tsx`
- 更新 title: "VibeMatch - AI Movie Recommendations"
- 更新 description: "RAG-powered semantic movie recommendation system"
- 移除 `Ai` Provider 包装

### 5.3 构建验证
- `npm install` 成功
- `npm run build` 成功（Next.js 15.3.2，Static prerendering）
- 输出 `.next/` 目录，包含静态资源

---

## Phase 6: 系统集成与测试 🔄 进行中

### 6.1 启动流程
```bash
# Terminal 1: 启动后端
cd backend
python main.py

# Terminal 2: 启动前端
cd ..
npm run dev
```

### 6.2 联调检查清单
- [ ] 前端 `localhost:3000` 能正常访问
- [ ] 输入查询后，前端显示 "Thinking..."
- [ ] 后端 `/chat` 接收请求并返回结果
- [ ] 前端正确渲染 Markdown 回答
- [ ] Sources 卡片正确显示检索来源
- [ ] 响应时间显示正常
- [ ] 网络错误时显示友好提示

### 6.3 待优化项
- [ ] 添加流式响应 (SSE)
- [ ] 添加检索模式切换（Similarity / MMR）
- [ ] 添加 Top-K 调节滑块
- [ ] 向量检索缓存

---

## Phase 7: 论文与 Presentation 待开始

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
Phase 2 (RAG核心) ✅ ──▶ Phase 3 (Baseline) ✅
    │                       │
    ▼                       ▼
Phase 5 (前端适配) ✅ ◀── Phase 4 (评估体系) ✅
    │
    ▼
Phase 6 (集成测试) 🔄
    │
    ▼
Phase 7 (论文+Presentation)
```

---

## 风险与应对

| 风险 | 概率 | 影响 | 应对策略 |
|:---|:---|:---|:---|
| NVIDIA API 额度不足 | 中 | 高 | 已配置 qwen 模型，成本较低；准备备用 Key |
| 前端对接复杂 | 低 | 中 | 已完成基础对接，Phase 6 进行联调 |
| 评估人工标注耗时 | 高 | 中 | 已设计自动化评估指标，减少人工依赖 |
| 论文时间不够 | 中 | 高 | Phase 7 预留充足时间，每天固定写作时间 |

---

*最后更新：2026-05-02*
*当前进度：Phase 5 完成，Phase 6 进行中*
