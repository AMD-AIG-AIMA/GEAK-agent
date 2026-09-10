# 快速上手:在你自己的机器上跑 GEAK 可切换后端(codex / cursor)

> 这个分支的 runtime **自包含、零 npm 依赖**(只用 Node 内置模块)。`git pull` 后,只需装好 CLI、设几个环境变量,即可跑。设计/架构见 `DESIGN.md`。

本层目录:`interface/runtime/`。以下命令假设在**仓库根**执行。

---

> **原 SaFE 网关(`global.primus-safe.amd.com`)已停用**,相关配置全部移除,由 AMD 网关
> (`llm-api.amd.com/Unified`,见下方 `AMDKEY`)接替。原 A 节那套"经 SaFE 跑 claude"的
> 步骤连同 `codex-opus48` / `codex-gpt54` / `qwen-opus48` / `kimi-opus48` 一起删掉了。
>
> AMD 网关认的是 `Ocp-Apim-Subscription-Key` 头(**只**认这个,单给 Bearer 是 401),而
> **qwen / kimi 没有下发自定义头的通路**(只会把 key 放进 Bearer),所以这两个 CLI 目前
> 接不上 AMD 网关,暂不支持;要用就自己 `export OPENAI_BASE_URL=...` 指一个 Bearer 鉴权的端点。

## A. codex 自动按 key 选网关(OpenAI 官方 / AMD)

codex 的 provider **自动配置**,无需手写 config.toml、无需 `setup.sh`、无需选 provider;
配好 key 同时也**选中了 codex 这个后端**,连 `GEAK_AGENT_BACKEND=codex` 都不用设。
runtime 在启动 codex 时按下面顺序解析(第一个命中),生成 `-c model_providers.geak_auto.*` 覆盖:

1. **显式 `OPENAI_BASE_URL`**(或所选 model 的 base_url)→ 直接用它(任意 OpenAI 兼容网关)。
2. 否则**按"哪个 key 非空"自动选**:`AMDKEY`→AMD、`OPENAI_API_KEY`→OpenAI 官方。

自动选中的那条 provider 还带自己的 **`default_model`**(`GEAK_CODEX_MODEL` 没设时用它),
目前两条都是 `gpt-5.6-sol`。端点和 model id 写在同一条里是有意的:id 只在自己的端点上有效。
同一份 `provider_autoselect` 也是**后端选择**的依据:没显式指定
`--agent` / `--profile` / `GEAK_AGENT_BACKEND` 时,key 自己就能把这次运行落到 codex 上。

判定看的是**整个凭据环境的形状**,不是"某个 key 在不在"——一个 key 只有在**没有其他后端的
凭据也配着**的时候才选中自己那个后端:

| 环境 | 跑哪个 |
| --- | --- |
| 只有 `AMDKEY` 或 `OPENAI_API_KEY` | **codex** |
| 只有 anthropic 侧(`ANTHROPIC_API_KEY` / `ANTHROPIC_BASE_URL` / `ANTHROPIC_AUTH_TOKEN` / `CLAUDE_CODE_OAUTH_TOKEN` 任一) | claude |
| **两边都配** | claude(`default_profile`) |
| 什么都没配 | claude |

两边都配时不猜、退回默认,是为了不劫持一套已经在跑的 claude 部署:只要有人 export 了
`AMDKEY`,原本的 claude 流程就被静默换掉,这种事很难排查。什么都没配时也退回 claude,
因为那种环境下 claude CLI 很可能是用别的方式认证的(已登录的 CLI、Bedrock)。
这套规则和 hyperloom `common/llm_config.py` 的 `is_openai_only()` / `is_anthropic_only()`
一致。要在两边都配的环境里强制走 codex,显式写 `--agent codex` 或
`GEAK_AGENT_BACKEND=codex` 即可;`GEAK_AGENT_AUTO=0` 则完全关掉自动选。

### 前置:安装 codex CLI(不要假设已装好)
```bash
# 1) Node.js v20+(codex 依赖)
node -v        # 无 node 或 <20:先装 Node 20+(nvm / 系统包管理器 / nodejs.org)

# 2) 安装 codex CLI —— 务必 pin 0.146.1(0.147 与网关不兼容)
npm i -g @openai/codex@0.146.1
#   若没有 /usr/local 写权限,用用户级 prefix:
#   npm config set prefix "$HOME/.npm-global"
#   export PATH="$HOME/.npm-global/bin:$PATH"      # 建议写进 ~/.bashrc
#   npm i -g @openai/codex@0.146.1

# 3) 验证
codex --version        # 期望 0.146.1
```

### 第 1 步:选 provider —— 设对应的 key(二选一)
```bash
# AMD 网关(自动补 Ocp-Apim-Subscription-Key 头 + llm-api.amd.com/Unified)
export AMDKEY="<32位hex 订阅 key>"
# 证书是公信的,不需要 SSL_CERT_FILE

# 或 —— 官方 OpenAI(公网 CA,无需 shim / SSL_CERT_FILE / config.toml)
# export OPENAI_API_KEY="sk-....."
```
> `GEAK_CODEX_MODEL` 是**可选**的:不设就用那条 provider 的 `default_model`(两条都是 `gpt-5.6-sol`)。
> 要覆盖时注意 id 只在自己的端点上有效:AMD 有 `gpt-5.6-sol` / `-terra` / `-luna`,但**没有**
> 不带后缀的 `gpt-5.6`;官方账号能用哪些取决于你的 entitlement。
>
> 想直接用**官方的 `gpt-5.6`**(不带后缀,只有官方端点有)就 pin 死那条 profile:
> `--profile codex-gpt56`。pin 住的 model 自带 `base_url` 和 `OPENAI_API_KEY`,优先级
> 高于按 key 自动选,所以即使环境里还留着 `AMDKEY` 也不会被切到网关上去。
> 这条**未实测**(手边没有官方 key),你的账号若 404/400 就退回
> `--profile codex-openai` + `GEAK_CODEX_MODEL=<你能用的 id>`。
>
> AMD 网关上 **gpt 系两个协议都正常**(`/v1/responses` 含流式、`/v1/chat/completions` 实测均 200);
> 但 **claude 系基本不答**(Opus 全 500、Sonnet-5 504,只有 `Claude-Sonnet-4.5` 通),所以
> registry 里没有 pin 任何 claude 模型。

### 第 2 步:运行(第 1 步的 key 已选中 codex,直接跑)
```bash
# 想覆盖自动选择才需要设,例如回到 claude:export GEAK_AGENT_BACKEND=claude

# e2e(整模型吞吐):用 handoff.json 描述任务(字段/示例见 interface/run_e2e.md)
python3 interface/run_e2e.py <handoff.json> <result.json>

# 单核:
node interface/runtime/run_workflow.mjs kernel_workflow/kernel_workflow.js --agent codex \
  --args '{"kernel_path":"/abs/kernel","workflow_dir":"'"$PWD"'/kernel_workflow","budget":6}'
```

### 覆盖 / 关闭 / 排错
- **thinking level(reasoning effort)默认拉满**。注意 codex 自己的档位只有
  `none`/`low`/`medium`/`high`/`xhigh`,**没有 `max`** —— `xhigh` 就是它的最高档。所以
  runtime 下发的是 `-c model_reasoning_effort=xhigh`;`GEAK_CODEX_EFFORT=max` 仍可写,会被
  翻译成 `xhigh`(和 hyperloom `resolve_codex_reasoning_effort` 同一套映射),写别的非法值
  会直接报错而不是丢给 codex。也可以用
  `GEAK_CODEX_EXTRA_ARGS="-c model_reasoning_effort=high"` 显式钉(优先)。
- 任意网关:`export OPENAI_BASE_URL=https://你的网关/v1`(+ 对应 key)——优先于 key 自动选。
- 关闭自动配置:`export GEAK_CODEX_AUTOCONFIG=0`(回落到 `codex-home/config.toml`)。
- 关闭"按 key 自动选后端":`export GEAK_AGENT_AUTO=0`(回到 registry 的 `default_profile`,即 claude)。
- 手动指定 provider:`export GEAK_CODEX_EXTRA_ARGS="-c model_provider=openai"`(优先于自动)。
- base_url 指向 `127.0.0.1`/`localhost`(即本地 shim)时**不会**自动覆盖,保留 config.toml 的 `safe_shim` 路径。
- 401 → key 空/无效;404 model → `GEAK_CODEX_MODEL` 不可用或不支持 Responses API;TLS 错 → 内网网关需 `SSL_CERT_FILE`(官方 OpenAI 不需要)。

---

## B. cursor(注意:走 Cursor 私有云,**不经**任何网关)

cursor 与 shim/网关无关,不需要 `setup.sh`。

### 前提
1. `cursor-agent` CLI 装好。
2. **你自己的 Cursor Team 账号**:`cursor-agent login`(登录态存 `~/.config/cursor/auth.json`)。别人的账号带不过去;需要你有对应 Team 的访问权。或设 `export CURSOR_API_KEY=...`。

### 步骤
```bash
cursor-agent login          # 首次
# 可选:选模型(Cursor 侧 id,如 composer-2.5 / sonnet-4-thinking)
export GEAK_CURSOR_MODEL="composer-2.5"
node interface/runtime/run_workflow.mjs <workflow.js> --profile cursor
```

> 提醒:cursor 的请求和代码会**外发到 Cursor 云**,且模型是 Cursor 侧模型 —— 因此它**不能**和 codex/qwen 做"同网关同模型"的严格对照。

---

## 验证 runtime 本身没坏(可选,不需网络/GPU)
```bash
node interface/runtime/selftest.mjs      # 期望 50/50
```

## 各文件是什么
- `run_workflow.mjs` runtime 核心 · `config.mjs`+`registry.json` 后端/模型配置
- `backends/` 后端契约与 generic 实现 · `schema.mjs` 结构化输出
- `responses_shim.mjs` codex+claude 的 de-stream 代理 · `setup.sh` 一键起环境
- `codex-home/config.toml` 仓库内 CODEX_HOME(providers:`safe_shim` 默认 / `openai` 官方)
