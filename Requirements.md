1. Expression interning/hash consing
2. SMT-LIB2 based solver (problems: UNSAT CORE, SMT-LIB2)
3. Parsing SMT-LIB2
4. Z3-solver
5. Unsupported features -> Unsupported exeception
6. Expression naming
7. Z3 Test -> Test system
8. External process - mirror process - to kill
9. Portfolio solver
10. Benchmark solver request
11. Don't use Z3Java API
12. Bitwoozla as second priority solver
13. Bv, Arrays, Ints, FP, Q-
14. Recommendation - switch to other solver (if fails)
15. Easy deployment via JitPack
16. Incremental via lemma. Drop? Assumption incrementality (check-sat with params)
17. Pop/push
18. Options for each solver implementation
19. Timeout - suspend function?
20. Model - on kotlin part in terms of Expression (model completion!)
21. Generics?

### Naming
- org.ksmt
- KAndExpression
- KModel
- KSolver
- KArrayStoreExpression
