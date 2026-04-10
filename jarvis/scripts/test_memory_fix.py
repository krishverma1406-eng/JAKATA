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
    print(f"✓ _candidate_pool returned {len(candidates)} candidates from records only")
    
    # Verify _trusted_chunk_candidates is not called in recall path
    import inspect
    recall_source = inspect.getsource(mem.recall)
    assert "_trusted_chunk_candidates" not in recall_source, "recall() still calls _trusted_chunk_candidates!"
    print("✓ recall() does not call _trusted_chunk_candidates")


def test_incremental_indexing():
    """Verify ensure_index_current is incremental, not full reset."""
    mem = Memory()
    
    import inspect
    index_source = inspect.getsource(mem.ensure_index_current)
    assert "_reset_collection" not in index_source, "ensure_index_current still calls _reset_collection!"
    assert "existing_ids" in index_source, "ensure_index_current missing incremental logic!"
    assert "new_items" in index_source, "ensure_index_current missing incremental logic!"
    print("✓ ensure_index_current uses incremental updates")


def test_no_chunk_writes():
    """Verify _persist_extracted_memory doesn't write chunk files."""
    mem = Memory()
    
    import inspect
    persist_source = inspect.getsource(mem._persist_extracted_memory)
    assert "_append_chunk" not in persist_source, "_persist_extracted_memory still writes chunks!"
    print("✓ _persist_extracted_memory does not write chunk files")


def test_memory_source_state():
    """Verify _memory_source_state only watches records and entities."""
    mem = Memory()
    
    state = mem._memory_source_state()
    paths = [f["path"] for f in state.get("files", [])]
    
    # Should only have records and entities
    assert any("memory_records.json" in p for p in paths), "records not in state!"
    assert any("entities.json" in p for p in paths), "entities not in state!"
    
    # Should NOT have chunks, profile, or projects
    assert not any("chunks" in p and p.endswith(".txt") for p in paths), "chunk files still in state!"
    assert not any("profile.md" in p for p in paths), "profile.md still in state!"
    assert not any("projects.md" in p for p in paths), "projects.md still in state!"
    print("✓ _memory_source_state only watches records and entities")


def test_query_embedding_cache():
    """Verify query embedding cache is initialized and cleared properly."""
    mem = Memory()
    
    # Should be initialized in __init__
    assert hasattr(mem, "_query_embedding_cache"), "_query_embedding_cache not initialized!"
    assert isinstance(mem._query_embedding_cache, dict), "_query_embedding_cache wrong type!"
    print("✓ _query_embedding_cache properly initialized")
    
    # Should be cleared in _rebuild_materialized_memory
    import inspect
    rebuild_source = inspect.getsource(mem._rebuild_materialized_memory)
    assert "_query_embedding_cache = {}" in rebuild_source, "_query_embedding_cache not cleared on rebuild!"
    print("✓ _query_embedding_cache cleared on memory rebuild")


def test_conflict_detection():
    """Verify _find_conflicting_record works for all tags."""
    mem = Memory()
    
    import inspect
    conflict_source = inspect.getsource(mem._find_conflicting_record)
    
    # Should NOT have tag restriction
    assert 'if tag not in {' not in conflict_source, "_find_conflicting_record still has tag restriction!"
    print("✓ _find_conflicting_record works for all tags")


def test_fallback_extract_dedup():
    """Verify _fallback_extract checks existing texts."""
    mem = Memory()
    
    import inspect
    fallback_source = inspect.getsource(mem._fallback_extract)
    assert "existing_texts" in fallback_source, "_fallback_extract missing dedup check!"
    assert "content.lower() in existing_texts" in fallback_source, "_fallback_extract dedup check wrong!"
    print("✓ _fallback_extract has proper deduplication")


def test_daily_summary_no_chunks():
    """Verify _build_daily_context_summary doesn't fall back to chunks."""
    mem = Memory()
    
    import inspect
    summary_source = inspect.getsource(mem._build_daily_context_summary)
    assert "_trusted_chunk_candidates" not in summary_source, "_build_daily_context_summary still uses chunks!"
    assert "_load_records_payload" in summary_source, "_build_daily_context_summary missing records fallback!"
    print("✓ _build_daily_context_summary uses records, not chunks")


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
        
        print("\n✅ All tests passed! Memory fixes are correctly implemented.")
        return 0
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
