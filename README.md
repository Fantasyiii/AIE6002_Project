# VibeMatch

VibeMatch 是一个基于 RAG（Retrieval-Augmented Generation）的语义化电影推荐引擎。用户可以通过自然语言描述"感觉"或"氛围"来搜索电影，系统返回精准推荐并解释匹配原因。本项目基于 [Movies++](https://github.com/datastax/movies_plus_plus) 改造，重构后端为本地可控的 LangChain + ChromaDB 架构，并补充完整的学术评估体系，以满足 AIE6002 Large Language Models 课程要求。

---

## 核心特性

- **语义理解**：支持复杂、 nuanced 的自然语言查询（如"关于中年危机的黑色幽默电影，发生在欧洲，有暖心结局"）
- **RAG 架构**：结合向量检索与大语言模型，确保推荐既语义相关又事实准确
- **幻觉控制**：所有推荐均锚定在真实电影数据库上，杜绝 LLM 编造电影
- **多样性检索**：支持 MMR（Maximal Marginal Relevance）策略，平衡相关性与推荐多样性
- **溯源展示**：每条推荐均可查看检索到的原始电影数据，增强可信度
- **Baseline 对比**：内置 Pure-LLM、Tag-Based、Retrieval-Only 三种基线系统，便于学术研究

---

## 技术栈

### 前端
| 技术 | 用途 |
|:---|:---|
| Next.js 15 | React 全栈框架 |
| React 19 | UI 组件库 |
| TypeScript | 类型安全 |
| TailwindCSS | 样式框架 |

### 后端
| 技术 | 用途 |
|:---|:---|
| Python 3.10+ | 主开发语言 |
| FastAPI | API 服务框架 |
| LangChain 0.2.x (LCEL) | RAG 流程编排 |
| ChromaDB 1.5.x | 本地向量数据库（预编译 wheel，无需 C++ 编译器） |
| `all-MiniLM-L6-v2` (本地) | Embedding 模型（384 维，下载到项目目录） |
| DeepSeek `deepseek-v4-flash` | 主 LLM（通过 `DEEPSEEK_API_KEY` 配置） |
| OpenAI GPT-4o-mini | 备选 LLM（`DEEPSEEK_API_KEY` 未设置时自动回退） |

### 数据
| 来源 | 说明 |
|:---|:---|
| TMDB 5000 Movie Dataset | 包含 8,254 部有效电影的标题、类型、年份、剧情概述 |

---

## 项目结构

```
VibeMatch/
├── app/                          # Next.js 前端（基于 Movies++ 改造）
│   ├── page.tsx                  # 主页面（聊天界面）
│   ├── SearchForm.tsx            # 搜索表单
│   ├── useMovieSearch.ts         # 搜索逻辑 Hook
│   ├── Markdown.tsx              # Markdown 渲染
│   ├── Logo.tsx                  # VibeMatch Logo
│   ├── fonts.tsx                 # 字体配置
│   ├── globals.css               # 全局样式
│   ├── layout.tsx                # 根布局
│   ├── LangContext.tsx           # 国际化语言上下文
│   ├── Providers.tsx             # React Providers 包装
│   └── i18n.ts                   # 国际化配置
├── backend/                      # Python 后端
│   ├── main.py                   # FastAPI 入口
│   ├── rag_chain.py              # LangChain RAG 核心
│   ├── vectorstore.py            # ChromaDB 向量存储管理
│   ├── data_processor.py         # TMDB 数据预处理
│   ├── download_model.py         # 模型下载脚本（支持国内镜像）
│   ├── prompts.py                # Prompt 模板
│   ├── baselines.py              # 三种基线系统
│   ├── fetch_tmdb_new.py         # 获取新电影数据脚本
│   ├── requirements.txt          # Python 依赖
│   ├── evaluation/               # 评估框架
│   │   ├── test_queries.json     # 15 个测试查询
│   │   ├── metrics.py            # 评估指标
│   │   ├── run_eval.py           # 自动化评估脚本
│   │   └── results/              # 评估结果（已提交到 GitHub）
│   ├── models/                   # 本地模型目录
│   │   └── all-MiniLM-L6-v2/    # Embedding 模型文件
│   ├── data/                     # 数据目录
│   │   ├── movies_processed.json # 处理后的电影数据
│   │   ├── movies_merged.json    # 合并后的电影数据（8,254 部）
│   │   └── tmdb_new_movies.json  # 新获取的电影数据
│   └── chroma_db/                # ChromaDB 持久化存储
├── Dataset/                      # 原始数据集
│   ├── tmdb_5000_movies.csv
│   └── tmdb_5000_credits.csv
├── Report.md                     # 项目论文报告（Markdown）
├── Report.tex                    # 项目论文报告（LaTeX）
├── Presentation.md               # 课程 Presentation 脚本
├── PROGRESS.md                   # 项目进度追踪
├── 技术栈.md                      # 技术栈说明
└── package.json                  # Node.js 依赖
```

---

## 快速开始

### 环境要求
- Python 3.10+
- Node.js 18+
- npm
- DeepSeek API Key（或 OpenAI API Key）

### 1. 克隆仓库

```bash
git clone https://github.com/Fantasyiii/AIE6002_Project.git
cd AIE6002_Project
```

### 2. 配置环境变量

复制 `.env.example` 为 `backend/.env` 并编辑：

```env
# 主 LLM（推荐）
DEEPSEEK_API_KEY=your_deepseek_api_key_here

# 备选 LLM（DEEPSEEK_API_KEY 未设置时生效）
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. 安装并启动后端

```bash
cd backend

# 安装依赖
pip install -r requirements.txt

# 下载 Embedding 模型到项目目录（使用国内镜像）
python download_model.py

# 数据预处理（首次运行）
python data_processor.py

# 构建向量数据库（首次运行）
python vectorstore.py

# 启动 FastAPI 服务
python main.py
```

### 4. 安装并启动前端

```bash
# 在项目根目录（新终端）
npm install
npm run dev
```

### 5. 访问应用

打开浏览器访问 http://localhost:3000

---

## API 端点

| 端点 | 方法 | 说明 |
|:---|:---|:---|
| `/health` | GET | 健康检查 |
| `/chat` | POST | 主推荐接口（RAG） |
| `/baseline/pure-llm` | POST | Pure-LLM 基线接口 |
| `/baseline/retrieval-only` | POST | Retrieval-Only 基线接口 |

### 请求示例

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "sci-fi movie about space travel", "retrieval_mode": "mmr", "top_k": 5}'
```

---

## 学术研究

### Research Questions

1. **RQ1 (Factual Accuracy)**：RAG 是否能显著降低 LLM 在电影推荐中的幻觉率？
2. **RQ2 (Retrieval Quality)**：MMR 检索策略如何影响推荐的相关性-多样性权衡？
3. **RQ3 (Embedding Comparison)**：细粒度文本分块是否比整段嵌入提升检索质量？

### Baselines

| 系统 | 描述 |
|:---|:---|
| **VibeMatch (Ours)** | 完整 RAG：ChromaDB + MMR + DeepSeek |
| **Pure-LLM** | 直接调用 LLM，无检索上下文 |
| **Tag-Based** | 基于 TMDB 类型标签的精确匹配过滤 |
| **Retrieval-Only** | 仅返回 ChromaDB 检索结果，无 LLM 生成 |

### 评估指标

- **Hallucination Rate**：LLM 输出中编造电影/情节的比例
- **Intra-List Diversity**：推荐列表的多样性分数
- **Latency**：端到端响应延迟（ms）
- **ROUGE-L**：生成文本与参考摘要的重叠度

### 最新评估结果

基于 15 个测试查询的评估结果：

| System | Hallucination Rate | Avg Latency | Avg Recommendations |
|:---|:---:|:---:|:---:|
| VibeMatch (RAG) | 59% | 10,355ms | 5.0 |
| VibeMatch (MMR) | 54% | 8,471ms | 5.0 |
| Pure-LLM | 100% | 10,529ms | 0.0 |
| Tag-Based | 100% | 77ms | 0.0 |
| Retrieval-Only | 7% | 1,732ms | 5.0 |

**关键发现**：
- RAG 相比 Pure-LLM 显著降低了幻觉率（100% → 59%）
- MMR 模式比 Similarity 模式幻觉率更低（54% vs 59%）
- Retrieval-Only 幻觉率最低（7%），但缺乏 LLM 的解释能力

完整评估结果保存在 `backend/evaluation/results/` 目录。

---

## 课程交付物

本项目为 AIE6002 Large Language Models 课程最终项目，包含以下交付物：

| 文件 | 说明 |
|:---|:---|
| [Report.md](Report.md) | 项目论文报告（Markdown 格式） |
| [Report.tex](Report.tex) | 项目论文报告（LaTeX 格式） |
| [Presentation.md](Presentation.md) | 课程 Presentation 脚本（约 8 分钟） |
| [PROGRESS.md](PROGRESS.md) | 项目开发进度追踪 |

---

## 与 Movies++ 的区别

| 维度 | Movies++ | VibeMatch |
|:---|:---|:---|
| **目标** | 产品 Demo | 学术研究 |
| **RAG 编排** | Langflow 云端可视化 | LangChain 本地代码化 |
| **向量数据库** | DataStax Astra DB（云端） | ChromaDB（本地） |
| **Embedding** | Astra `$vectorize` 黑盒 | `all-MiniLM-L6-v2` 显式控制 |
| **检索策略** | 简单相似度 | Cosine + MMR |
| **Baseline 对比** | 无 | 3 种基线系统 |
| **评估体系** | 无 | 完整量化指标 + 消融实验 |
| **溯源展示** | 无 | 可展开源数据面板 |
| **部署依赖** | 5 个云服务账号 | 1 个 DeepSeek API Key |

---

## 贡献者

- Yifei Chen (225085000@link.cuhk.edu.cn)
- Lei Zhang (122090746@link.cuhk.edu.cn)
- Shuhao Shi (122090466@link.cuhk.edu.cn)

---

## 许可证

本项目基于 [Movies++](https://github.com/datastax/movies_plus_plus) 改造，遵循原项目许可证。

---

## 课程信息

- **课程**：AIE6002 Large Language Models
- **学期**：2026 Spring
- **指导教师**：CUHK-Shenzhen
