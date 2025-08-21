#!/usr/bin/env python3
import os 
import sys
import argparse
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

def main():
    parser = argparse.ArgumentParser(description='CV Model Platform API Server', prog='python -m cv_platform.api')
    
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
    
    print("CV Platform API Server")
    print("=" * 40)
    print(f"Address: http://{args.host}:{args.post}")    
    print(f"API Docs: http://{args.host}:{args.port}/docs")
    print(f"Workers: {args.workers}")
    print("=" * 40)
    
    try:
        import uvicorn
        from .main import app
      
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            workers=1 if args.reload else args.workers,  
            reload=args.reload,
            log_level=args.log_level
        )
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        print("Please install: pip install uvicorn[standard]")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 Server has stopped")
    except Exception as e:
        print(f"❌ Startup failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
