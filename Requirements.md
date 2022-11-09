
| Feature                                       | Status |
|-----------------------------------------------|--------|
| [Rich expression system](#rich-expression-system)    | -      |
| [Expression interning](#expression-interning) | -      |
| ...                                           | -      |


### Rich expression system
Instead of wrapper object for solver expression z3 expression 

### Expression interning
Aka [hash consing](https://en.wikipedia.org/wiki/Hash_consing) is needed for ...

### Serialize to SMT-LIB2

And provide interface to pass this SMT-LIB2 to corresponding solver. 
This way is also an easy to interact with new solver

### Parse SMT-LIB2 
For benchmarking - we can use rich database of benchmarks from SMT-COMP to verify correctness

### Solvers support
This is the most detailed sections. Please write here what features of what solvers 
we support now, which are going to support and which will not support

### Theories support
Bv, Arrays, Ints, FP, Q-

### Unsupported features
If some solver doesn't support some theory (like Bitvec) or some feature(unsat-core) we need to throw 
specialized exception. Exception should contain recommendation to switch to other solver.

### Expression naming
Add debug information into expression for future visualization

### External process

### Portfolio solver

### Performance tests
Describe how to write performance tests that measure overhead of KSMT for typical operations.
After done, add results here

### Solver Native API support
Don't use Z3Java API

### Ease of use
Easy deployment via JitPack, maven central

### Solver options
Options for each solver implementation

###  Pop/push

### Incremental lemma
Incremental via lemma. Drop? Assumption incrementality (check-sat with params)

### Solver timeout
What happens if timeout. Suspend function?

### Evaluate model
On kotlin part in terms of Expression (model completion)

