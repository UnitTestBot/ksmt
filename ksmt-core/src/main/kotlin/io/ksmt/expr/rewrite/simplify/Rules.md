## Simplification rules

Note: eval constants -- evaluate an expression if all arguments are constants

Note: todo -- additional simplification rules implemented in Z3

### Core

1. KDistinctExpr
   * ``(distinct a) ==> true``
   * ``(distinct a b) ==> (not (= a b))``
   * ``(distinct a a) ==> false``
   * todo: bool_rewriter.cpp:786
2. KAndExpr
   * flattening: ``(and a (and b c)) ==> (and a b c)``
   * ``(and a b true) ==> (and a b)``
   * ``(and a b false) ==> false``
   * ``(and a b a) ==> (and a b)``
   * ``(and a (not a)) ==> false``
3. KOrExpr
   * flattening: ``(or a (or a c)) ==> (or a b c)``
   * ``(or a b true) ==> true``
   * ``(or a b false) ==> (or a b)``
   * ``(or a b a) ==> (or a b)``
   * ``(or a (not a)) ==> true``
4. KNotExpr
   * eval constants
   * ``(not (not x)) ==> x``
5. KIteExpr
   * ``(ite true t e) ==> t``
   * ``(ite false t e) ==> e``
   * ``(ite c t t) ==> t``
   * ``(ite (not c) a b) ==> (ite c b a)``
   * ``(ite c (ite c t1 t2) t3)  ==> (ite c t1 t3)``
   * ``(ite c t1 (ite c t2 t3))  ==> (ite c t1 t3)``
   * ``(ite c t1 (ite c2 t1 t2)) ==> (ite (or c c2) t1 t2)``
   * ite with bool sort rules
      * ``(ite c true e) ==> (or c e)``
      * ``(ite c false e) ==> (and (not c) e)``
      * ``(ite c t true) ==> (or (not c) t)``
      * ``(ite c t false) ==> (and c t)``
      * ``(ite c t c) ==> (and c t)``
      * ``(ite c c e) ==> (or c e)``
   * todo: ite extra rules: bool_rewriter.cpp:846
6. KImpliesExpr
   * ``(=> p q) ==> (or (not p) q)``
7. KXorExpr
   * ``(xor a b) ==> (= (not a) b)``
8. KEqExpr(KBool, KBool)
   * eval constants
   * ``(= (not a) (not b)) ==> (= a b)``
   * ``(= a (not a)) ==> false``
   * todo: bool_rewriter.cpp:700

### Array

1. KEqExpr(KArray, KArray)
   * ```
       (= (store a i v) (store b x y)) ==>
         (and
             (= (select a i) (select b i))
             (= (select a x) (select b x))
             (= a b)
         )
      ```
   * ``(= (const a) (const b)) ==> (= a b)``
   * todo: array_rewriter.cpp:740
2. KArrayStore
   * ``(store (const v) i v) ==> (const v)``
   * ``(store a i (select a i)) ==> a``
   * ``(store (store a i x) i y) ==> (store a i y)``
   * ``(store (store a i x) j y), i != j ==> (store (store a j y) i x)``
3. KArraySelect
   * ``(select (store i v) i) ==> v``
   * ``(select (store a i v) j), i != j ==> (select a j)``
   * ``(select (const v) i) ==> v``
   * ``(select (lambda x body) i) ==> body[i/x]``
   * todo: array_rewriter.cpp:199

### Arithmetic

1. KLtArithExpr
    * eval constants
    * todo: lt, gt, le, ge: arith_rewriter.cpp:514
2. KLeArithExpr
    * eval constants
3. KGtArithExpr
    * eval constants
4. KGeArithExpr
    * eval constants
5. KAddArithExpr
    * eval constants
    * ``(+ a 0) ==> a``
6. KMulArithExpr
    * eval constants
    * ``(* a 0) ==> 0``
    * ``(* a 1) ==> a``
7. KSubArithExpr
    * ``(- a b) ==> (+ a -b)``
8. KUnaryMinusArithExpr
    * eval constants
9. KDivArithExpr
    * eval constants
    * ``(intdiv a a) ==> (ite (= a 0) (intdiv 0 0) 1)``
    * ``(div a 1) ==> a``
    * ``(div a -1) ==> -a``
    * todo: arith_rewriter.cpp:1046
10. KPowerArithExpr
    * todo: arith_rewriter.cpp:1319
11. KModIntExpr
    * eval constants
    * ``(mod a 1) ==> 0``
    * ``(mod a -1) ==> 0``
    * todo: arith_rewriter.cpp:1196
12. KRemIntExpr
    * eval constants
    * ``(rem a 1) ==> 0``
    * ``(rem a -1) ==> 0``
    * todo: arith_rewriter.cpp:1257
13. KToIntRealExpr
    * eval constants
    * ``(real2int (int2real x)) ==> x``
    * todo: arith_rewriter.cpp:1487
14. KIsIntRealExpr
    * eval constants
    * ``(isInt (int2real x)) ==> true``
15. KToRealIntExpr
    * eval constants
    * todo: arith_rewriter.cpp:1540
16. KEqExpr(KInt, KInt)
    * eval constants
    * todo: arith_rewriter.cpp:694
17. KEqExpr(KReal, KReal)
    * eval constants
    * todo: arith_rewriter.cpp:694

### Bv

1. KBvUnsignedLessOrEqualExpr
   * eval constants
   * ``a <= b, b == MIN_VALUE ==> a == b``
   * ``a <= b, b == MAX_VALUE ==> true``
   *  ``a <= b, a == MIN_VALUE ==> true``
   * ``a <= b, a == MAX_VALUE ==> a == b``
   * todo: bv_rewriter.cpp:433
2. KBvSignedLessOrEqualExpr
   * same as for KBvUnsignedLessOrEqualExpr
3. KBvUnsignedGreaterOrEqualExpr
   * ``(uge a b) ==> (ule b a)``
4. KBvUnsignedLessExpr
   * ``(ult a b) ==> (not (ule b a))``
5. KBvUnsignedGreaterExpr
   * ``(ugt a b) ==> (not (ule a b))``
6. KBvSignedGreaterOrEqualExpr
   * ``(sge a b) ==> (sle b a)``
7. KBvSignedLessExpr
   * ``(slt a b) ==> (not (sle b a))``
8. KBvSignedGreaterExpr
   * ``(sgt a b) ==> (not (sle a b))``
9. KBvAddExpr
   * eval constants
   * ``(+ const1 (+ const2 x)) ==> (+ (+ const1 const2) x)``
   * ``(+ x 0) ==> x``
10. KBvSubExpr
    * ``(- a b) ==> (+ a -b)``
11. KBvMulExpr
    * eval constants
    * ``(* const1 (* const2 x)) ==> (* (* const1 const2) x)``
    * ``(* 0 a) ==> 0``
    * ``(* 1 a) ==> a``
    * ``(* -1 a) ==> -a``
12. KBvNegationExpr
    * eval constants
13. KBvSignedDivExpr
    * eval constants
    * ``(sdiv a 1) ==> a``
14. KBvUnsignedDivExpr
    * eval constants
    * ``(udiv a 1) ==> a``
    * ``(udiv a x), x == 2^n ==> (lshr a n)``
15. KBvSignedRemExpr
    * eval constants
    * ``(srem a 1) ==> 0``
16. KBvUnsignedRemExpr
    * eval constants
    * ``(urem a 1) ==> 0``
    * ``(urem a x), x == 2^n ==> (concat 0 (extract [n-1:0] a))``
17. KBvSignedModExpr
    * eval constants
    * ``(smod a 1) ==> 0``
18. KBvNotExpr
    * eval constants
    * ``(bvnot (bvnot a)) ==> a``
    * ``(bvnot (concat a b)) ==> (concat (bvnot a) (bvnot b))``
    * ``(bvnot (ite c a b)) ==> (ite c (bvnot a) (bvnot b))``
    * todo: bv_rewriter.cpp:2007
19. KBvOrExpr
    * eval constants
    * ``(bvor const1 (bvor const2 x)) ==> (bvor (bvor const1 const2) x)``
    * ``(bvor a b a) ==> (bvor a b)``
    * ``(bvor a (bvnot a)) ==> 0xFFFF...``
    * ``(bvor 0xFFFF... a) ==> 0xFFFF...``
    * ``(bvor 0 a) ==> a``
    * ``(bvor a a) ==> a``
    * ```
       (bvor (concat a b) c) ==>
          (concat
            (bvor (extract (0, <a_size>) c))
            (bvor b (extract (<a_size>, <a_size> + <b_size>) c))
          )
      ```
    * todo: bv_rewriter.cpp:1638
20. KBvXorExpr
    * eval constants
    * ``(bvxor const1 (bvxor const2 x)) ==> (bvxor (bvxor const1 const2) x)``
    * ``(bvxor 0 a) ==> a``
    * ``(bvxor 0xFFFF... a) ==> (bvnot a)``
    * ``(bvxor a a) ==> 0``
    * ``(bvxor (bvnot a) a) ==> 0xFFFF...``
    * ```
      (bvxor (concat a b) c) ==>
          (concat
              (bvxor (extract (0, <a_size>) c))
              (bvxor b (extract (<a_size>, <a_size> + <b_size>) c))
          )
      ```
    * todo: bv_rewriter.cpp:1810
21. KBvAndExpr
    * ``(bvand a b) ==> (bvnot (bvor (bvnot a) (bvnot b)))``
22. KBvNAndExpr
    * ``(bvnand a b) ==> (bvor (bvnot a) (bvnot b))``
23. KBvNorExpr
    * ``(bvnor a b) ==> (bvnot (bvor a b))``
24. KBvXNorExpr
    * ``(bvxnor a b) ==> (bvnot (bvxor a b))``
25. KBvReductionAndExpr
    * eval constants
26. KBvReductionOrExpr
    * eval constants
27. KBvConcatExpr
    * eval constants
    * ``(concat (concat a const1) (concat const2 b)) ==> (concat a (concat (concat const1 const2) b))``
    * ``(concat (extract[h1, l1] a) (extract[h2, l2] a)), l1 == h2 + 1 ==> (extract[h1, l2] a)``
28. KBvExtractExpr
    * eval constants
    * ``(extract [size-1:0] x) ==> x``
    * ``(extract[high:low] (extract[_:nestedLow] x)) ==> (extract[high+nestedLow : low+nestedLow] x)``
    * ``(extract (concat a b)) ==> (concat (extract a) (extract b))``
    * ``(extract [h:l] (bvnot x)) ==> (bvnot (extract [h:l] x))``
    * ``(extract [h:l] (bvor a b)) ==> (bvor (extract [h:l] a) (extract [h:l] b))``
    * ``(extract [h:l] (bvxor a b)) ==> (bvxor (extract [h:l] a) (extract [h:l] b))``
    * ``(extract [h:0] (bvadd a b)) ==> (bvadd (extract [h:0] a) (extract [h:0] b))``
    * ``(extract [h:0] (bvmul a b)) ==> (bvmul (extract [h:0] a) (extract [h:0] b))``
    * todo: bv_rewriter.cpp:681
29. KBvShiftLeftExpr
    * eval constants
    * ``(x << 0) ==> x``
    * ``(x << shift), shift >= size ==> 0``
    * ``(bvshl x shift) ==> (concat (extract [size-1-shift:0] x) 0..[shift]..0)``
    * ```
      (bvshl (bvshl x nestedShift) shift) ==>
           (ite (bvule nestedShift (+ nestedShift shift)) (bvshl x (+ nestedShift shift)) 0)
      ```
30. KBvLogicalShiftRightExpr
    * eval constants
    * ``(x >>> 0) ==> x``
    * ``(x >>> shift), shift >= size ==> 0``
    * ``(bvlshr x shift) ==> (concat 0..[shift]..0 (extract [size-1:shift] x))``
    * ``(x >>> x) ==> 0``
31. KBvArithShiftRightExpr
    * eval constants
    * ``(x >> 0) ==> x``
    * todo: bv_rewriter.cpp:923
32. KBvRepeatExpr
    * ``(repeat a x) ==> (concat a a ..[x].. a)``
33. KBvZeroExtensionExpr
    * ``(zeroext a) ==> (concat 0 a)``
34. KBvSignExtensionExpr
    * eval constants
35. KBvRotateLeftIndexedExpr
    * ``(rotateLeft a x) ==> (concat (extract [size-1-x:0] a) (extract [size-1:size-x] a))``
36. KBvRotateRightIndexedExpr
    * ``(rotateRight a x) ==> (rotateLeft a (- size x))``
37. KBvRotateLeftExpr
    * eval constants
38. KBvRotateRightExpr
    * eval constants
39. KBvAddNoOverflowExpr
    * signed
      * ```
        (bvadd no ovf signed a b) ==>
            (=> (and (bvslt 0 a) (bvslt 0 b)) (bvslt 0 (bvadd a b)))
        ```
    * unsigned
      * ```
        (bvadd no ovf unsigned a b) ==>
            (= 0 (extract [highestBit] (bvadd (concat 0 a) (concat 0 b))))
        ```
40. KBvAddNoUnderflowExpr
    * ```
      (bvadd no udf a b) ==>
          (=> (and (bvslt a 0) (bvslt b 0)) (bvslt (bvadd a b) 0))
      ```
41. KBvSubNoOverflowExpr
    * ```
      (bvsub no ovf a b) ==>
          (ite (= b MIN_VALUE) (bvslt a 0) (bvadd no ovf signed a (bvneg b)))
      ```
42. KBvSubNoUnderflowExpr
    * signed
      * ```
        (bvsub no udf signed a b) ==>
            (=> (bvslt 0 b) (bvadd no udf (bvneg b)))
        ```
    * unsigned
      * ```
        (bvsub no udf unsigned a b) ==>
            (bvule b a)
        ```
43. KBvNegNoOverflowExpr
    * ```(bvneg no ovf a) ==> (not (= a MIN_VALUE))```
44. KBvDivNoOverflowExpr
    * ```(bvsdiv no ovf a b) ==> (not (and (= a MSB) (= b -1)))```
45. KBvMulNoOverflowExpr
    * eval constants
46. KBvMulNoUnderflowExpr
    * eval constants
47. KBv2IntExpr
    * eval constants
48. KEqExpr(KBv, KBv)
    * eval constants
     ```
     (= (concat a b) c) ==>
        (and
           (= a (extract (0, <a_size>) c))
           (= b (extract (<a_size>, <a_size> + <b_size>) c)
        )
      ```
    * todo: bv_rewriter.cpp:2681

### Fp

1. KFpAbsExpr
   * eval constants
2. KFpNegationExpr
   * eval constants
3. KFpAddExpr
   * eval constants
4. KFpSubExpr
   * ``(- a b) ==> (+ a -b)``
5. KFpMulExpr
   * eval constants
6. KFpDivExpr
   * eval constants
7. KFpFusedMulAddExpr
   * eval constants
8. KFpSqrtExpr
   * eval constants
9. KFpRoundToIntegralExpr
   * eval constants
10. KFpRemExpr
    * eval constants
11. KFpMinExpr
    * eval constants
    * ``(min a NaN) ==> a``
12. KFpMaxExpr
    * eval constants
    * ``(max a NaN) ==> a``
13. KFpLessOrEqualExpr
    * eval constants
    * ``(<= a NaN) ==> false``
    * ``(<= NaN a) ==> false``
14. KFpLessExpr
    * eval constants
    * ``(< a NaN) ==> false``
    * ``(< NaN a) ==> false``
    * ``(< +Inf a) ==> false``
    * ``(< a -Inf) ==> false``
15. KFpGreaterOrEqualExpr
    * ``(>= a b) ==> (<= b a)``
16. KFpGreaterExpr
    * ``(> a b) ==> (< b a)``
17. KFpEqualExpr
    * eval constants
18. KFpIsNormalExpr
    * eval constants
19. KFpIsSubnormalExpr
    * eval constants
20. KFpIsZeroExpr
    * eval constants
21. KFpIsInfiniteExpr
    * eval constants
22. KFpIsNaNExpr
    * eval constants
23. KFpIsNegativeExpr
    * eval constants
24. KFpIsPositiveExpr
    * eval constants
25. KFpFromBvExpr
    * eval constants
26. KFpToIEEEBvExpr
    * eval constants
27. KFpToFpExpr
    *  todo: fpa_rewriter.cpp:169
28. KRealToFpExpr
    * todo: fpa_rewriter.cpp:160
29. KFpToRealExpr
    * eval constants
30. KBvToFpExpr
    * todo: fpa_rewriter.cpp:179
31. KFpToBvExpr
    * eval constants
32. KEqExpr(KFp, KFp)
    * eval constants

### Strings

1. KStringConcat
    * eval constants
    * ``((concat a const1) const2) ==> (concat a (concat const1 const2))``
    * ``((concat const1 (concat const2 a)) => (concat (concat const1 const2) a)``
    * ``((concat (concat a const1) (concat const2 b)) ==> (concat a (concat (concat const1 const2) b))``
2. KStringLen
    * eval constants
3. KStringSuffixOf
    * eval constants
4. KStringPrefixOf
    * eval constants
5. KStringLt
    * eval constants
6. KStringLe
    * eval constants 
7. KStringGt
    * eval constants
8. KStringGe
    * eval constants
9. KStringContains
    * eval constants
10. KStringSingletonSub
    * eval constants
11. KStringSub
    * eval constants
12. KStringIndexOf
    * eval constants
13. KStringReplace
    * eval constants
14. KStringReplaceAll
    * eval constants 
15. KStringToLower
    * eval constants
16. KStringToUpper
    * eval constants
17. KStringReverse
    * eval constants
18. KStringIsDigit
    * eval constants
19. KStringToCode
    * eval constants 
20. KStringFromCode
    * eval constants 
21. KStringToInt
    * eval constants 
22. KStringFromInt
    * eval constants
