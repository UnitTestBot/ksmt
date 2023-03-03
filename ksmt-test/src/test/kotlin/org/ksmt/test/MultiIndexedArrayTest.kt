package org.ksmt.test

import com.microsoft.z3.Context
import org.ksmt.KContext
import org.ksmt.KContext.SimplificationMode.NO_SIMPLIFY
import org.ksmt.expr.KExpr
import org.ksmt.expr.rewrite.KExprUninterpretedDeclCollector
import org.ksmt.solver.KSolver
import org.ksmt.solver.KSolverStatus
import org.ksmt.solver.bitwuzla.KBitwuzlaContext
import org.ksmt.solver.bitwuzla.KBitwuzlaExprConverter
import org.ksmt.solver.bitwuzla.KBitwuzlaExprInternalizer
import org.ksmt.solver.bitwuzla.KBitwuzlaSolver
import org.ksmt.solver.z3.KZ3Context
import org.ksmt.solver.z3.KZ3ExprConverter
import org.ksmt.solver.z3.KZ3ExprInternalizer
import org.ksmt.solver.z3.KZ3Solver
import org.ksmt.sort.KArray2Sort
import org.ksmt.sort.KArray3Sort
import org.ksmt.sort.KArraySort
import org.ksmt.sort.KArraySortBase
import org.ksmt.sort.KBv32Sort
import org.ksmt.sort.KBvSort
import org.ksmt.sort.KSort
import org.ksmt.utils.uncheckedCast
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.time.Duration.Companion.seconds

class MultiIndexedArrayTest {

    @Test
    fun testMultiIndexedArraysZ3WithZ3Oracle(): Unit = with(KContext(simplificationMode = NO_SIMPLIFY)) {
        KZ3Solver(this).use { oracleSolver ->
            mkZ3Context(this).use { z3NativeCtx ->
                runMultiIndexedArraySamples(oracleSolver) { expr ->
                    internalizeAndConvertZ3(z3NativeCtx, expr)
                }
            }
        }
    }

    @Test
    fun testMultiIndexedArraysBitwuzlaWithZ3Oracle(): Unit = with(KContext(simplificationMode = NO_SIMPLIFY)) {
        KZ3Solver(this).use { oracleSolver ->
            KBitwuzlaContext(this).use { z3NativeCtx ->
                runMultiIndexedArraySamples(oracleSolver) { expr ->
                    internalizeAndConvertBitwuzla(z3NativeCtx, expr)
                }
            }
        }
    }

    @Test
    fun testMultiIndexedArraysZ3WithBitwuzlaOracle(): Unit = with(KContext(simplificationMode = NO_SIMPLIFY)) {
        KBitwuzlaSolver(this).use { oracleSolver ->
            mkZ3Context(this).use { z3NativeCtx ->
                runMultiIndexedArraySamples(oracleSolver) { expr ->
                    internalizeAndConvertZ3(z3NativeCtx, expr)
                }
            }
        }
    }

    @Test
    fun testMultiIndexedArraysBitwuzlaWithBitwuzlaOracle(): Unit = with(KContext(simplificationMode = NO_SIMPLIFY)) {
        KBitwuzlaSolver(this).use { oracleSolver ->
            KBitwuzlaContext(this).use { z3NativeCtx ->
                runMultiIndexedArraySamples(oracleSolver) { expr ->
                    internalizeAndConvertBitwuzla(z3NativeCtx, expr)
                }
            }
        }
    }

    private inline fun KContext.runMultiIndexedArraySamples(
        oracle: KSolver<*>,
        process: (KExpr<KSort>) -> KExpr<KSort>
    ) {
        val sorts = listOf(
            mkArraySort(bv32Sort, bv32Sort),
//            mkArraySort(bv32Sort, bv16Sort, bv32Sort),
//            mkArraySort(bv32Sort, bv16Sort, bv8Sort, bv32Sort),
//            mkArrayNSort(listOf(bv32Sort, bv16Sort, bv8Sort, bv32Sort, bv8Sort), bv32Sort)
        )

        for (sort in sorts) {
            val expressions = mkArrayExpressions(sort)
            for (expr in expressions) {
                val processed = process(expr)
                assertEquals(oracle, expr, processed)
            }
        }
    }

    private fun <A : KArraySortBase<KBv32Sort>> KContext.mkArrayExpressions(sort: A): List<KExpr<KSort>> {
        var arrayExpressions = listOf(
            mkConst(sort),
            mkAsArray(sort), // disable as-array because it is too hard to check equality
            mkArrayConst(sort) { mkConst("cv", bv32Sort) },
            mkLambda(sort) { mkConst("lv", bv32Sort) }
        )

        arrayExpressions = arrayExpressions + arrayExpressions.map {
            mkStore(it) { mkConst("v", bv32Sort) }
        }

        arrayExpressions = arrayExpressions + arrayExpressions.flatMap { first ->
            arrayExpressions.map { second ->
                mkIte(mkConst("cond", boolSort), first, second)
            }
        }

        val arrayEq = arrayExpressions.zipWithNext().map { (first, second) -> first eq second }

        var arraySelects = arrayExpressions.map { mkSelect { it } }

        val arrayValues = arraySelects + listOf(mkConst("x", bv32Sort))
        arrayExpressions = arrayExpressions + arrayValues.flatMap { value ->
            listOf(
                mkArrayConst(sort) { value },
                mkLambda(sort) { value },
            )
        }

        arrayExpressions = arrayExpressions + arrayExpressions.flatMap { array ->
            arrayValues.map { value ->
                mkStore(array) { value }
            }
        }

        arraySelects = arraySelects + arrayExpressions.map { mkSelect { it } }

        return listOf(
            arrayExpressions,
            arraySelects,
            arrayEq
        ).flatten().uncheckedCast()
    }

    private fun <A : KArraySortBase<KBv32Sort>> KContext.mkConst(sort: A): KExpr<A> =
        mkFreshConst("c", sort)

    private fun <A : KArraySortBase<KBv32Sort>> KContext.mkArrayConst(
        sort: A,
        value: () -> KExpr<KBv32Sort>
    ): KExpr<A> = mkArrayConst(sort, value())

    private fun <A : KArraySortBase<KBv32Sort>> KContext.mkAsArray(sort: A): KExpr<A> {
        val function = mkFreshFuncDecl("f", sort.range, sort.domainSorts)
        return mkFunctionAsArray(sort, function)
    }

    private fun <A : KArraySortBase<KBv32Sort>> KContext.mkLambda(
        sort: A,
        mkBody: (List<KExpr<KBvSort>>) -> KExpr<KBv32Sort>
    ): KExpr<A> {
        val indices = sort.domainSorts.map { mkFreshConst("i", it) }
        val body = mkBody(indices.uncheckedCast())
        return when (indices.size) {
            KArraySort.DOMAIN_SIZE -> mkArrayLambda(indices.single().decl, body)
            KArray2Sort.DOMAIN_SIZE -> mkArrayLambda(indices.first().decl, indices.last().decl, body)
            KArray3Sort.DOMAIN_SIZE -> {
                val (i0, i1, i2) = indices
                mkArrayLambda(i0.decl, i1.decl, i2.decl, body)
            }

            else -> mkArrayNLambda(indices.map { it.decl }, body)
        }.uncheckedCast()
    }

    private fun <A : KArraySortBase<KBv32Sort>> KContext.mkStore(
        array: KExpr<A>,
        mkValue: () -> KExpr<KBv32Sort>
    ): KExpr<A> {
        val indices = array.sort.domainSorts.map { mkFreshConst("i", it) }
        val value = mkValue()
        return when (indices.size) {
            KArraySort.DOMAIN_SIZE -> mkArrayStore(array.uncheckedCast(), indices.single(), value)
            KArray2Sort.DOMAIN_SIZE -> mkArrayStore(array.uncheckedCast(), indices.first(), indices.last(), value)
            KArray3Sort.DOMAIN_SIZE -> {
                val (i0, i1, i2) = indices
                mkArrayStore(array.uncheckedCast(), i0, i1, i2, value)
            }

            else -> mkArrayNStore(array.uncheckedCast(), indices, value)
        }.uncheckedCast()
    }

    private fun <A : KArraySortBase<KBv32Sort>> KContext.mkSelect(
        mkArray: () -> KExpr<A>
    ): KExpr<KBv32Sort> {
        val array = mkArray()
        val indices = array.sort.domainSorts.map { mkFreshConst("i", it) }
        return when (indices.size) {
            KArraySort.DOMAIN_SIZE -> mkArraySelect(array.uncheckedCast(), indices.single())
            KArray2Sort.DOMAIN_SIZE -> mkArraySelect(array.uncheckedCast(), indices.first(), indices.last())
            KArray3Sort.DOMAIN_SIZE -> {
                val (i0, i1, i2) = indices
                mkArraySelect(array.uncheckedCast(), i0, i1, i2)
            }

            else -> mkArrayNSelect(array.uncheckedCast(), indices)
        }
    }

    private fun <T : KSort> KContext.internalizeAndConvertBitwuzla(
        nativeCtx: KBitwuzlaContext, expr: KExpr<T>
    ): KExpr<T> {
        val internalized = with(KBitwuzlaExprInternalizer(nativeCtx)) {
            expr.internalizeExpr()
        }

        val converted = with(KBitwuzlaExprConverter(this, nativeCtx)) {
            internalized.convertExpr(expr.sort)
        }

        return converted
    }

    private fun <T : KSort> KContext.internalizeAndConvertZ3(nativeCtx: Context, expr: KExpr<T>): KExpr<T> {
        val z3InternCtx = KZ3Context(nativeCtx)
        val z3ConvertCtx = KZ3Context(nativeCtx)

        val internalized = with(KZ3ExprInternalizer(this, z3InternCtx)) {
            expr.internalizeExpr()
        }

        // Copy declarations since we have fresh decls
        val declarations = KExprUninterpretedDeclCollector.collectUninterpretedDeclarations(expr)
        declarations.forEach {
            val nativeDecl = z3InternCtx.findInternalizedDecl(it)
            z3ConvertCtx.saveConvertedDecl(nativeDecl, it)
        }

        val converted = with(KZ3ExprConverter(this, z3ConvertCtx)) {
            internalized.convertExpr<T>()
        }

        return converted
    }

    private fun <T : KSort> KContext.assertEquals(oracle: KSolver<*>, expected: KExpr<T>, actual: KExpr<T>) {
        if (expected == actual) {
            return
        }

//        println("#".repeat(20))
//        println(expected)
//        println("-".repeat(20))
//        println(actual)

        oracle.push()

        // Check expressions are possible to be SAT
        oracle.assert(expected eq actual)
        val exprPossibleStatus = oracle.check(timeout = 1.seconds)
        if (exprPossibleStatus == KSolverStatus.UNKNOWN) {
            System.err.println("IGNORED: ${oracle.reasonOfUnknown()}")
            return
        }
        assertEquals(KSolverStatus.SAT, exprPossibleStatus)

        oracle.pop()
        oracle.push()

        // Check expressions are equal
        oracle.assert(expected neq actual)
        val exprEqualStatus = oracle.check()
        if (exprEqualStatus != KSolverStatus.UNSAT) {
            assertEquals(expected, actual, "Expressions are not equal")
        }

        oracle.pop()
    }

    private fun mkZ3Context(ctx: KContext): Context {
        KZ3Solver(ctx).close()
        return Context()
    }
}
