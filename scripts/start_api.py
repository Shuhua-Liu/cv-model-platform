#!/usr/bin/env python3
"""
API service startup script

Start the REST API service of CV Model Platform
"""

import sys
import argparse
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def main():
    parser = argparse.ArgumentParser(description='Start CV Model Platform API service')
    
    parser.add_argument('--host', 
                      type=str, 
                      default='0.0.0.0',
                      help='Server host address')
    
    parser.add_argument('--port', 
                      type=int, 
                      default=8000,
                      help='Server port')
    
    parser.add_argument('--workers', 
                      type=int, 
                      default=1,
                      help='Number of work processes')
    
    parser.add_argument('--reload', 
                      action='store_true',
                      help='Enable code hot reloading (development mode)')
    
    parser.add_argument('--log-level', 
                      type=str, 
                      default='info',
                      choices=['debug', 'info', 'warning', 'error'],
                      help='Log level')
    
    args = parser.parse_args()
    
    try:
        import uvicorn
        from src.cv_platform.api.main import app
        from src.cv_platform.utils.logger import setup_logger
        
        # setup logger
        setup_logger(args.log_level.upper())
        
        print(f"🚀 CV Model Platform API Server...")
        print(f"📡 Address: http://{args.host}:{args.port}")
        print(f"📚 API docs: http://{args.host}:{args.port}/docs")
        print(f"🔧 ReDoc docs: http://{args.host}:{args.port}/redoc")
        print(f"👥 Workers: {args.workers}")
        print(f"🔄 Hot reload: {'enabled' if args.reload else 'disabled'}")
        print("=" * 50)
        
        # start server
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            workers=1 if args.reload else args.workers,  # Only one worker can be used in hot heavy load mode.
            reload=args.reload,
            log_level=args.log_level
        )
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please ensure necessary dependencies are installed: pip install fastapi uvicorn")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 Server has stopped")
    except Exception as e:
        print(f"❌ Startup failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
