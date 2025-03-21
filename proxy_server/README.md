# OpenAI API 代理服务器

这是一个用于代理 OpenAI API 请求的本地服务器，支持将请求转发到硅基流动的API服务。

## 功能特点

- 支持 HTTP 和 HTTPS 协议
- 支持 SSE（Server-Sent Events）流式响应
- 自动添加 appid 和 apikey 认证信息
- 监听本地 5566 端口
- 支持所有 OpenAI API 格式的标准请求

## 环境要求

- Python 3.11 或更高版本
- 所需的 Python 包已在 requirements.txt 中列出

## 安装

1. 克隆或下载此代码库
2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 配置

在运行服务器之前，需要设置以下环境变量：

1. 创建 `.env` 文件：
```bash
APP_ID=your_app_id
API_KEY=your_api_key
```

或者直接设置环境变量：
```bash
export APP_ID=your_app_id
export API_KEY=your_api_key
```

## 运行

直接运行 Python 脚本：
```bash
python proxy_server.py
```

服务器将在本地 5566 端口启动，同时支持 HTTP 和 HTTPS 协议。

## 使用方法

将你的 AI 开发工具（如 Cursor、cline 等）的 API 基础 URL 设置为：
```
http://localhost:5566
```
或
```
https://localhost:5566
```

所有发往此地址的请求将被自动添加认证信息并转发到硅基流动的 API 服务。

## 注意事项

- 确保已正确设置 APP_ID 和 API_KEY
- 服务器默认超时时间设置为 10 分钟
- 如遇到 SSL 证书问题，请确保系统已安装必要的证书