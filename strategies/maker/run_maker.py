import asyncio
import sys
from pathlib import Path

# Ensure project root on sys.path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from maker_strategy import MakerStrategy


def main():
    strategy = MakerStrategy()
    asyncio.run(strategy.run())


if __name__ == "__main__":
    main()

