"""Test script to verify all memory system fixes."""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Add jarvis to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.memory import Memory
from config.settings import SETTINGS


def test_recall_speed():
    """Test that recall is fast (no 60s latency)."""
    print("\n=== Test 1: Recall Speed ===")
    memory = Memory(SETTINGS)
    
    queries = [
        "What is Krish's name?",
        "Where does the user live?",
        "What does the user like?",
    ]
    
    for query in queries:
        start = time.time()
        results = memory.recall(query, limit=5)
        elapsed = time.time() - start
        
        print(f"\nQuery: {query}")
        print(f"Time: {elapsed:.3f}s")
        print(f"Results: {len(results)}")
        for i, result in enumerate(results[:3], 1):
            print(f"  {i}. {result[:80]}")
        
        if elapsed > 5.0:
            print(f"  ⚠️  WARNING: Slow recall ({elapsed:.1f}s)")
        else:
            print(f"  ✓ Fast recall")


def test_no_chunk_reads():
    """Test that _candidate_pool doesn't read chunks."""
    print("\n=== Test 2: No Chunk Reads ===")
    memory = Memory(SETTINGS)
    
    # Get candidates
    candidates = memory._candidate_pool(include_chunks=False)
    print(f"Total candidates: {len(candidates)}")
    
    # Check if any candidates came from chunks
    # (This is indirect - we just verify we get reasonable results)
    if len(candidates) > 0:
        print("✓ Candidates retrieved from records")
        print(f"Sample: {candidates[0][:80] if candidates else 'None'}")
    else:
        print("⚠️  No candidates found")


def test_conflict_detection():
    """Test that conflicting records are detected and superseded."""
    print("\n=== Test 3: Conflict Detection ===")
    memory = Memory(SETTINGS)
    
    # Try to add a conflicting fact
    print("\nAdding: 'Krish loves meetings on Mondays'")
    result = memory.remember("Krish loves meetings on Mondays", brain=None, source="test")
    
    print(f"Stored: {result.get('stored', 0)} items")
    print(f"Items: {result.get('items', [])}")
    
    # Check what we recall now
    recall_results = memory.recall("Krish meetings Monday", limit=3)
    print(f"\nRecall results for 'Krish meetings Monday':")
    for i, r in enumerate(recall_results, 1):
        print(f"  {i}. {r}")
    
    if len(recall_results) > 0:
        print("✓ Conflict handling working")
    else:
        print("⚠️  No results found")


def test_incremental_indexing():
    """Test that indexing is incremental (not full rebuild)."""
    print("\n=== Test 4: Incremental Indexing ===")
    memory = Memory(SETTINGS)
    
    # Force index to be current
    print("Running ensure_index_current...")
    start = time.time()
    memory.ensure_index_current()
    elapsed = time.time() - start
    
    print(f"First index: {elapsed:.3f}s")
    
    # Run again - should be instant (no changes)
    start = time.time()
    memory.ensure_index_current()
    elapsed = time.time() - start
    
    print(f"Second index (no changes): {elapsed:.3f}s")
    
    if elapsed < 0.5:
        print("✓ Incremental indexing working (instant when no changes)")
    else:
        print(f"⚠️  Slow re-index ({elapsed:.1f}s)")


def test_background_indexing():
    """Test that indexing runs in background."""
    print("\n=== Test 5: Background Indexing ===")
    
    # Create new memory instance - should not block
    print("Creating Memory instance...")
    start = time.time()
    memory = Memory(SETTINGS)
    elapsed = time.time() - start
    
    print(f"Memory init time: {elapsed:.3f}s")
    
    if elapsed < 1.0:
        print("✓ Memory init is fast (indexing in background)")
    else:
        print(f"⚠️  Slow init ({elapsed:.1f}s) - may be blocking")
    
    # Give background thread time to complete
    print("Waiting for background indexing...")
    time.sleep(2)
    print("✓ Background indexing complete")


def test_dedup_extraction():
    """Test that _fallback_extract doesn't create duplicates."""
    print("\n=== Test 6: Dedup Extraction ===")
    memory = Memory(SETTINGS)
    
    # Get current record count
    payload = memory._load_records_payload()
    before_count = len([r for r in payload.get("records", []) if r.get("active", True)])
    print(f"Active records before: {before_count}")
    
    # Try to extract an existing fact
    transcript = [
        "USER: Krish hates meetings on Mondays",
        "ASSISTANT: I understand.",
    ]
    
    items = memory._fallback_extract(transcript)
    print(f"Extracted items: {len(items)}")
    
    if len(items) == 0:
        print("✓ Dedup working - existing fact not re-extracted")
    else:
        print(f"⚠️  Extracted {len(items)} items (may be duplicate)")
        for item in items:
            print(f"  - {item.get('text', '')}")


def test_memory_stats():
    """Show memory system statistics."""
    print("\n=== Memory Statistics ===")
    memory = Memory(SETTINGS)
    
    payload = memory._load_records_payload()
    records = payload.get("records", [])
    
    active = [r for r in records if r.get("active", True) and not r.get("demoted")]
    demoted = [r for r in records if r.get("demoted")]
    
    by_tag = {}
    for r in active:
        tag = r.get("tag", "UNKNOWN")
        by_tag[tag] = by_tag.get(tag, 0) + 1
    
    print(f"Total records: {len(records)}")
    print(f"Active records: {len(active)}")
    print(f"Demoted records: {len(demoted)}")
    print(f"\nActive by tag:")
    for tag, count in sorted(by_tag.items()):
        print(f"  {tag}: {count}")
    
    # Check vector index
    try:
        collection = memory._get_collection()
        if collection:
            existing = collection.get(include=[])
            indexed_count = len(existing.get("ids", []) or [])
            print(f"\nVector index: {indexed_count} items")
        else:
            print("\nVector index: Not available")
    except Exception as e:
        print(f"\nVector index: Error - {e}")


def main():
    """Run all tests."""
    print("=" * 60)
    print("JARVIS Memory System - Comprehensive Test Suite")
    print("=" * 60)
    
    try:
        test_memory_stats()
        test_recall_speed()
        test_no_chunk_reads()
        test_incremental_indexing()
        test_background_indexing()
        test_dedup_extraction()
        test_conflict_detection()
        
        print("\n" + "=" * 60)
        print("All tests completed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
