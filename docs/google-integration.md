# Google Calendar 与 Gmail 集成

## 两条不同的接入路径

Calendar 使用本地启动的第三方 `@cocal/google-calendar-mcp@2.7.0` Server，通过 MCP 将底层日历能力提供给项目适配层。MCP 解决程序与外部工具之间的发现和调用协议；模型 Tool Calling 解决模型选择哪个**项目工具**及填写什么参数。项目工具仍需核对可信用户和业务条件，MCP Server 本身不能代替这层授权。此 Server 不是 Google 官方提供的 MCP Server。

Gmail 使用独立 OAuth 授权和项目自己的读取工具，当前能力是搜索、未读邮件和线程读取，不支持发送邮件。Calendar 的 OAuth token 与 Gmail 只读 token 不能混用。

## 配置与账号范围

两个集成默认关闭。启用 Calendar 需要为本地 MCP 准备 Google OAuth 客户端凭据、Node.js/npm、可安装的第三方 MCP 包，以及 `GOOGLE_CALENDAR_OWNER_TELEGRAM_ID`。启用 Gmail 需要另一份只读 token，按 `python -m app.gmail_oauth` 的本地授权流程生成，并设置 `GMAIL_OWNER_TELEGRAM_ID`。真实凭据和 token 只保存在本机受限位置，不得提交或放进镜像。

当前每种集成分别绑定一个配置账号，不是每位 Telegram 用户各自完成 OAuth。其他用户可以使用自己的本地 Todo，但不能因为服务器已持有某个 Google token 就读取该账号的数据。无授权时相应工具应不可见；执行时仍须复核身份。每日简报也遵守同一账号边界。

## Calendar 写入的两类业务

独立 Calendar 事件的创建、修改和删除是直接操作远端事件；在 Telegram Agent 中受审批，在 Web 明确表单中受身份、版本和重复提交约束。Todo 排期后自动同步的 Google 时间块由另一个后台路径处理。Todo 数据库事务与 Google 写入不是同一个提交，日历查询成功也不等于事件创建成功。

远端写入遇到超时可能已成功。项目会保存操作身份并查询远端证据，以区分确定成功、确定失败和仍未知；不承诺 Google 端只有一次请求或绝对不重复。Google 读取失败时，不应让模型编造日历或邮件事实。