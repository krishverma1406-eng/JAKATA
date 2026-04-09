User says something
      ↓
Planner decides: simple task or multi-step?
      ↓
If multi-step → break into subtasks
      ↓
For each step:
   LLM picks a tool (or responds directly)
   Tool executes, returns result
   Result fed back to LLM as observation
   LLM decides: done? or need another tool?
      ↓
Final answer assembled and returned to user
      ↓
Entire exchange saved to JSONL + vector DB
