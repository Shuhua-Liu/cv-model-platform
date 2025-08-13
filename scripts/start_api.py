#!/usr/bin/env python3
"""
API服务启动脚本

启动CV Model Platform的REST API服务
"""

import sys
import argparse
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def main():
    parser = argparse.ArgumentParser(description='启动CV Model Platform API服务')
    
    parser.add_argument('--host', 
                      type=str, 
                      default='0.0.0.0',
                      help='服务器主机地址')
    
    parser.add_argument('--port', 
                      type=int, 
                      default=8000,
                      help='服务器端口')
    
    parser.add_argument('--workers', 
                      type=int, 
                      default=1,
                      help='工作进程数')
    
    parser.add_argument('--reload', 
                      action='store_true',
                      help='启用代码热重载（开发模式）')
    
    parser.add_argument('--log-level', 
                      type=str, 
                      default='info',
                      choices=['debug', 'info', 'warning', 'error'],
                      help='日志级别')
    
    args = parser.parse_args()
    
    try:
        import uvicorn
        from cv_platform.api.models1.requests import app
        from src.cv_platform.utils.logger import setup_logger
        
        # 设置日志
        setup_logger(args.log_level.upper())
        
        print(f"🚀 启动CV Model Platform API服务器...")
        print(f"📡 服务地址: http://{args.host}:{args.port}")
        print(f"📚 API文档: http://{args.host}:{args.port}/docs")
        print(f"🔧 ReDoc文档: http://{args.host}:{args.port}/redoc")
        print(f"👥 工作进程: {args.workers}")
        print(f"🔄 热重载: {'启用' if args.reload else '禁用'}")
        print("=" * 50)
        
        # 启动服务器
        uvicorn.run(
            "src.cv_platform.api.rest_api:app",
            host=args.host,
            port=args.port,
            workers=1 if args.reload else args.workers,  # 热重载模式下只能用1个worker
            reload=args.reload,
            log_level=args.log_level
        )
        
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        print("请确保已安装必要依赖: pip install fastapi uvicorn")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 服务器已停止")
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()