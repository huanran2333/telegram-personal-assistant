# 部署与本地验证

## 已验证的范围

本仓库已在 **Windows、Python 3.12、PostgreSQL 18.6（未安装 vector 扩展）** 上，使用全新隔离数据库验证：Schema/Checkpoint/Store 初始化、FastAPI 本地启动、Web 工作台 HTTPS 登录、Todo 创建/重复提交/读取/删除。测试环境没有 Bot Token、Google OAuth，也没有调用真实模型。另用固定响应的 Fake Model 验证了 Agent→Todo→Checkpoint。以下步骤用于复制这个**基础 Web Todo 开发环境**，不代表 Telegram 完整部署。

## 1. 安装依赖

安装 [PostgreSQL 18 的 Windows 版本](https://www.postgresql.org/download/windows/)。当前默认文字检索、LangGraph Checkpointer 和未配置向量索引的 Store 不需要 pgvector。确认 `initdb.exe`、`pg_ctl.exe` 位于同一个 PG18 的 `bin` 目录。只需安装 PostgreSQL 二进制；不要把开发库建在现有生产 PostgreSQL 实例里。

还需要 Python 3.12、Node.js/npm。在**这份独立副本**根目录运行：

```powershell
py -3.12 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
Copy-Item .env.example .env
Push-Location web
npm ci
npm test
npm run build
Pop-Location
```

`.env` 被 Git 忽略。模板中的模型地址和 Key 是**不可用占位值**：它能让设置校验与图装配运行，不会让 Agent 真正回答。要运行 Agent，换成自己的 `LLM_PROVIDER`、`LLM_MODEL`、对应 Key 和 Base URL；模型 ID 应以服务提供方的模型列表为准。本轮未调用真实模型。

## 2. 初始化独立 PostgreSQL

Windows 开发辅助脚本只使用仓库内被忽略的 `.local-tools/`（数据）与 `.release-local/`（随机凭据、日志），只监听 `127.0.0.1`，拒绝已存在的数据目录和正式端口。它创建固定的 staging 库 `personal_assistant_stage` 和角色 `pa_stage_app`，核对服务端数据目录、版本和账号，然后调用项目已有的 `offline_initialize --mode staging` 完成基础业务表及 LangGraph Checkpoint/Store 初始化。凭据不会进入命令行、Git 或报告。

```powershell
& .\.venv\Scripts\python.exe ops/local_staging.py init --port 56520 --pg-bin 'C:\Program Files\PostgreSQL\18\bin'
```

若 PostgreSQL 安装在别处，修改 `--pg-bin`。端口必须是未使用的高位端口；不要使用 5432、55432 或 55632。`init` **只用于全新目录**，重复执行会拒绝覆盖。已有本地数据目录要重新启动时使用 `start`；完成后可使用 `stop`，两者都会核对是同一个数据目录：

```powershell
& .\.venv\Scripts\python.exe ops/local_staging.py start --port 56520
& .\.venv\Scripts\python.exe ops/local_staging.py stop --port 56520
```

不要删除或移动正在运行的数据目录。当前基本 Todo/Web 验证不需要会议草稿表；该表的 G6 迁移是独立显式操作，不能把基础初始化描述成已执行所有历史/可选迁移。已有旧数据库的迁移也不属于此全新 staging 流程。

## 3. 本地启动与 Web Todo 验证

仅验证 API 时，确保隔离 PostgreSQL 已启动，然后运行：

```powershell
& .\.venv\Scripts\python.exe ops/local_staging.py serve --port 56520
```

它使用 Windows Selector 事件循环，在 `http://127.0.0.1:8000/` 提供本地服务。**不要在 Windows 直接替换为普通 `uvicorn` 命令**；后者的默认 Proactor 事件循环会让异步 PostgreSQL Store 初始化失败。`/` 返回 200 可证明 HTTP 服务启动；`/health` 在没有 Bot 时按完整产品就绪标准返回 503，应检查其中数据库、Store、Agent 图和 Checkpoint 的组件状态，不能把 `/` 的 200 称为 Bot 已就绪。

验证 Web 页面和 Todo 时，先为**合成测试用户**绑定本地密码并生成仅用于回环地址的自签名证书：

```powershell
& .\.venv\Scripts\python.exe ops/local_staging.py web-user --port 56520
& .\.venv\Scripts\python.exe ops/local_staging.py serve --port 56520 --web
```

另开一个终端访问 `https://127.0.0.1:18443/workbench/`，或运行：

```powershell
& .\.venv\Scripts\python.exe ops/smoke_local_web.py --port 56520
```

这个烟雾测试使用本地证书校验 HTTPS，从被忽略的凭据目录读取合成用户密码，验证登录、Todo 创建、同一提交标识去重、列表读取，并删除它创建的 Todo。**不需要信任或上传自签名证书**；浏览器可能提示本地测试证书不受信任。`/health` 仍会因关闭 Telegram 入站而返回 503，但 `web_auth` 应为就绪。按 Ctrl+C 停止后端，再用前述 `stop` 停止测试数据库。密码、私钥和数据目录均不得提交或分享。

## 4. 四种验收层级

| 层级 | 当前可核实内容 | 不可由此推断 |
| --- | --- | --- |
| 离线 Mock 测试 | 精选 Python 测试、前端测试与构建 | 真实数据库、模型或 Bot 已工作 |
| 本地开发 | 独立 PG18、基础 API、HTTPS Web Todo 闭环 | Agent 已调用真实模型；Telegram 已收发消息 |
| Telegram 测试部署 | **本轮未执行**；需要专用 Bot、独立 HTTPS Webhook 和受核验的测试数据库 | 不能用生产 Token 或正式 Outbox 代替测试 |
| 完整生产部署 | **本副本未验证**；需正式身份、备份、迁移、操作员预检和真实服务验收 | Docker 构建通过也不等于可上线 |

正式 Telegram 启动保护仍要求特定 Windows PG18 数据库身份、`APP_ENV=production`、HTTPS Web 来源和显式预检确认，绑定了原部署约束。上面的 staging 辅助脚本**强制关闭** Bot、Webhook、Scheduler、Google 和主动投递，绝不绕过该保护。新开发者要测试 Telegram，需要另行设计、审查独立测试 Bot 的安全启动路径；本轮不能声称 Telegram 功能已可移植部署。Calendar/Gmail 可不授权而使用基础 Web Todo；单账号 OAuth 的限制见[Google 集成](google-integration.md)。

当前 `docker-compose.yml` 只有镜像构建检查配置，不提供可运行的 PG/Agent/Bot 栈。Docker 引擎未参与本轮验证；`docker compose config --quiet` 仅通过静态检查。不要用 `docker compose up` 代替上述 staging 步骤。

## 5. 测试入口与失败边界

```powershell
& .\ops\check_public_copy.ps1
```

该脚本在进程内覆盖外部服务变量，运行已验证的当前业务离线测试、`npm test` 和 `npm run build`；先完成 `npm ci`。完整默认 pytest 仍包含需要未公开历史轨迹的测试，具体分类见[公开测试说明](testing.md)。不要把精选测试的通过数量当成完整测试套件的结果。
