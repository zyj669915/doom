"""从任意工作目录启动本地受信任实验包，也支持 python -I。"""
from pathlib import Path
import sys

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from flyrl.cli import main
    main()
