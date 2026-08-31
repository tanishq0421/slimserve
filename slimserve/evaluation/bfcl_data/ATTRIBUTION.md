# BFCL test data (vendored)

These files are the official test set + possible-answer keys from the
**Berkeley Function-Calling Leaderboard (BFCL)** — gorilla-llm/gorilla, Apache-2.0
— for the AST categories we evaluate: `simple`, `multiple`, `parallel` (Python).
Vendored so the BFCL evaluator is self-contained (no `bfcl-eval` install needed).
The scorer is BFCL's own `ast_checker`, vendored in `slimserve/evaluation/_bfcl_ast.py`.
