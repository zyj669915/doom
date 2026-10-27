"""支持 -I 的多路线入口，不依赖调用者工作目录。"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from flyrl_next.cli import main

if __name__ == '__main__':
    main()
