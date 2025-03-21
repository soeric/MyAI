#!/usr/bin/env python3
import os
import json
import ssl
import logging
import traceback
from logging.handlers import RotatingFileHandler
from aiohttp import web, ClientSession, ClientTimeout
from aiohttp.web import middleware
from aiohttp_sse import sse_response
from dotenv import load_dotenv

# 配置日志
def setup_logger():
    """配置日志记录器"""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)  # 设置为DEBUG级别以获取更多信息

    # 创建日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 添加文件处理器（限制文件大小为5MB，保留3个备份）
    file_handler = RotatingFileHandler(
        'server.log',
        maxBytes=5*1024*1024,  # 5MB
        backupCount=3,
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    # 添加控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    # 将处理器添加到日志记录器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

# 初始化日志记录器
logger = setup_logger()

try:
    # 加载环境变量
    load_dotenv()

    # 配置
    HOST = "0.0.0.0"
    PORT = 5566
    OPENAI_API_BASE = "https://api.siliconflow.com"
    APP_ID = os.getenv("APP_ID", "")
    API_KEY = os.getenv("API_KEY", "")

    if not APP_ID or not API_KEY:
        raise ValueError("APP_ID and API_KEY must be set in .env file")

    logger.info(f"Starting server with APP_ID: {APP_ID}")
    logger.debug("Environment variables loaded successfully")

except Exception as e:
    logger.error(f"Error during initialization: {str(e)}")
    logger.debug(f"Detailed error: {traceback.format_exc()}")
    raise

async def forward_stream(response, ws):
    """转发流式响应"""
    try:
        async for chunk in response.content:
            if chunk:
                logger.debug(f"Receiving chunk: {chunk[:100]}...")
                await ws.write(chunk)
                await ws.flush()
                logger.debug("Chunk forwarded successfully")
    except Exception as e:
        logger.error(f"Error in forwarding stream: {str(e)}")
        logger.debug(f"Detailed error: {traceback.format_exc()}")
        raise
    finally:
        await ws.write(b"")
        logger.debug("Stream forwarding completed")

@middleware
async def error_middleware(request, handler):
    """错误处理中间件"""
    try:
        logger.debug(f"Processing request: {request.method} {request.path}")
        response = await handler(request)
        logger.debug(f"Request processed successfully: {response.status}")
        return response
    except Exception as e:
        logger.error(f"Error handling request: {str(e)}")
        logger.debug(f"Detailed error: {traceback.format_exc()}")
        return web.Response(
            status=500,
            text=json.dumps({
                "error": str(e),
                "type": type(e).__name__
            }),
            content_type="application/json"
        )

async def proxy_handler(request):
    """处理代理请求"""
    try:
        # 获取原始请求数据
        body = await request.read()
        body_str = body.decode('utf-8') if body else ""
        logger.debug(f"Received request body: {body_str[:200]}...")

        # 设置请求头
        headers = {
            "Content-Type": "application/json",
            "X-AppId": APP_ID,  # 修改header名称
            "X-ApiKey": API_KEY  # 修改header名称
        }

        # 获取目标路径
        path = request.path.lstrip('/')  # 移除开头的斜杠
        target_url = f"{OPENAI_API_BASE}/{path}"

        logger.info(f"Proxying request to: {target_url}")
        logger.debug(f"Request headers: {headers}")

        # 创建新的请求
        timeout = ClientTimeout(total=600)  # 10分钟超时
        async with ClientSession(timeout=timeout) as session:
            logger.debug("Created client session")
            async with session.request(
                method=request.method,
                url=target_url,
                headers=headers,
                data=body,
                ssl=False
            ) as resp:
                logger.info(f"Response status: {resp.status}")
                logger.debug(f"Response headers: {resp.headers}")

                # 检查是否为流式响应
                if "text/event-stream" in resp.headers.get("content-type", ""):
                    logger.debug("Handling streaming response")
                    response = web.StreamResponse(
                        status=resp.status,
                        headers={
                            "Content-Type": "text/event-stream",
                            "Cache-Control": "no-cache",
                            "Connection": "keep-alive",
                        }
                    )
                    await response.prepare(request)
                    await forward_stream(resp, response)
                    return response
                else:
                    # 非流式响应
                    content = await resp.read()
                    content_str = content.decode('utf-8') if content else ""
                    logger.debug(f"Response content: {content_str[:200]}...")
                    return web.Response(
                        body=content,
                        status=resp.status,
                        headers={"Content-Type": resp.headers.get("Content-Type", "application/json")}
                    )
    except Exception as e:
        logger.error(f"Proxy error: {str(e)}")
        logger.debug(f"Detailed error: {traceback.format_exc()}")
        return web.Response(
            status=500,
            text=json.dumps({
                "error": str(e),
                "type": type(e).__name__
            }),
            content_type="application/json"
        )

def create_ssl_context():
    """创建SSL上下文"""
    ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    ssl_context.load_default_certs()
    return ssl_context

def main():
    """主函数"""
    try:
        app = web.Application(middlewares=[error_middleware])
        app.router.add_route("*", "/{tail:.*}", proxy_handler)

        logger.info(f"Starting server on {HOST}:{PORT}")

        # 启动服务器
        web.run_app(
            app,
            host=HOST,
            port=PORT,
            access_log=logger
        )
    except Exception as e:
        logger.error(f"Server error: {str(e)}")
        logger.debug(f"Detailed error: {traceback.format_exc()}")
        raise

if __name__ == "__main__":
    main()