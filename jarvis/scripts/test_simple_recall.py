"""Simple test to isolate recall issue."""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.memory import Memory
from config.settings import SETTINGS


def main():
    print("Creating Memory instance...")
    memory = Memory(SETTINGS)
    
    print("\nChecking embedder status...")
    embedder = memory._get_embedder()
    print(f"Embedder: {embedder is not None}")
    
    print("\nChecking collection status...")
    collection = memory._get_collection()
    print(f"Collection: {collection is not None}")
    
    print("\nTesting simple recall...")
    start = time.time()
    
    # Try a simple query
    try:
        results = memory.recall("Krish", limit=3)
        elapsed = time.time() - start
        
        print(f"\nRecall completed in {elapsed:.3f}s")
        print(f"Results: {len(results)}")
        for i, r in enumerate(results, 1):
            print(f"  {i}. {r[:80]}")
    except KeyboardInterrupt:
        print("\n\nInterrupted!")
        elapsed = time.time() - start
        print(f"Was running for {elapsed:.1f}s")
    except Exception as e:
        elapsed = time.time() - start
        print(f"\nError after {elapsed:.3f}s: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
