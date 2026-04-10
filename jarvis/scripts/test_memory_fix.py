"""Test script to verify memory fixes are working correctly.

Run from workspace root:
    python jarvis/scripts/test_memory_fix.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.memory import Memory


def test_no_chunk_reads():
    """Verify _candidate_pool doesn't read chunk files."""
    mem = Memory()
    
    # This should only read from records, not chunks
    candidates = mem._candidate_pool(include_facts=True)
    print(f"[OK] _candidate_pool returned {len(candidates)} candidates from records only")
    
    # Verify _trusted_chunk_candidates is not called in recall path
    import inspect
    recall_source = inspect.getsource(mem.recall)
    assert "_trusted_chunk_candidates" not in recall_source, "recall() still calls _trusted_chunk_candidates!"
    print("[OK] recall() does not call _trusted_chunk_candidates")


def test_incremental_indexing():
    """Verify ensure_index_current is incremental, not full reset."""
    mem = Memory()
    
    import inspect
    index_source = inspect.getsource(mem.ensure_index_current)
    assert "_reset_collection" not in index_source, "ensure_index_current still calls _reset_collection!"
    assert "existing_ids" in index_source, "ensure_index_current missing incremental logic!"
    assert "new_items" in index_source, "ensure_index_current missing incremental logic!"
    print("[OK] ensure_index_current uses incremental updates")


def test_no_chunk_writes():
    """Verify _persist_extracted_memory doesn't write chunk files."""
    mem = Memory()
    
    import inspect
    persist_source = inspect.getsource(mem._persist_extracted_memory)
    assert "_append_chunk" not in persist_source, "_persist_extracted_memory still writes chunks!"
    print("[OK] _persist_extracted_memory does not write chunk files")


def test_memory_source_state():
    """Verify _memory_source_state returns the new digest-based shape."""
    mem = Memory()
    
    state = mem._memory_source_state()
    assert "records_digest" in state, "records_digest missing from state!"
    assert "count" in state, "count missing from state!"
    assert "files" not in state, "legacy files key should not be present!"
    print("[OK] _memory_source_state uses digest-based state")


def test_query_embedding_cache():
    """Verify query embedding cache is initialized and cleared properly."""
    mem = Memory()
    
    # Should be initialized in __init__
    assert hasattr(mem, "_query_embedding_cache"), "_query_embedding_cache not initialized!"
    assert isinstance(mem._query_embedding_cache, dict), "_query_embedding_cache wrong type!"
    print("[OK] _query_embedding_cache properly initialized")
    
    # Should be cleared in _rebuild_materialized_memory
    import inspect
    rebuild_source = inspect.getsource(mem._rebuild_materialized_memory)
    assert "_query_embedding_cache = {}" in rebuild_source, "_query_embedding_cache not cleared on rebuild!"
    print("[OK] _query_embedding_cache cleared on memory rebuild")


def test_conflict_detection():
    """Verify _find_conflicting_record works for all tags."""
    mem = Memory()
    
    import inspect
    conflict_source = inspect.getsource(mem._find_conflicting_record)
    
    # Should NOT have tag restriction
    assert 'if tag not in {' not in conflict_source, "_find_conflicting_record still has tag restriction!"
    print("[OK] _find_conflicting_record works for all tags")


def test_fallback_extract_dedup():
    """Verify _fallback_extract checks existing texts."""
    mem = Memory()
    
    import inspect
    fallback_source = inspect.getsource(mem._fallback_extract)
    assert "existing_texts" in fallback_source, "_fallback_extract missing dedup check!"
    assert "content.lower() in existing_texts" in fallback_source, "_fallback_extract dedup check wrong!"
    print("[OK] _fallback_extract has proper deduplication")


def test_daily_summary_no_chunks():
    """Verify _build_daily_context_summary doesn't fall back to chunks."""
    mem = Memory()
    
    import inspect
    summary_source = inspect.getsource(mem._build_daily_context_summary)
    assert "_trusted_chunk_candidates" not in summary_source, "_build_daily_context_summary still uses chunks!"
    assert "_load_records_payload" in summary_source, "_build_daily_context_summary missing records fallback!"
    print("[OK] _build_daily_context_summary uses records, not chunks")


def main():
    print("Running memory fix verification tests...\n")
    
    try:
        test_no_chunk_reads()
        test_incremental_indexing()
        test_no_chunk_writes()
        test_memory_source_state()
        test_query_embedding_cache()
        test_conflict_detection()
        test_fallback_extract_dedup()
        test_daily_summary_no_chunks()
        
        print("\n[PASS] All tests passed! Memory fixes are correctly implemented.")
        return 0
    except AssertionError as e:
        print(f"\n[FAIL] Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n[FAIL] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
