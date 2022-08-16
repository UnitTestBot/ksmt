package org.ksmt.solver.bitwuzla

import org.ksmt.KContext
import org.ksmt.expr.KApp
import org.ksmt.expr.KExpr
import org.ksmt.expr.transformer.KTransformer
import org.ksmt.solver.bitwuzla.bindings.BitwuzlaKind
import org.ksmt.solver.bitwuzla.bindings.BitwuzlaOption
import org.ksmt.solver.bitwuzla.bindings.BitwuzlaResult
import org.ksmt.solver.bitwuzla.bindings.BitwuzlaTerm
import org.ksmt.solver.bitwuzla.bindings.FilePtrUtils
import org.ksmt.solver.bitwuzla.bindings.Native
import org.ksmt.sort.KBv1Sort
import org.ksmt.sort.KSort
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class ConverterTest {
    private val ctx = KContext()
    private val bitwuzlaCtx = KBitwuzlaContext()
    private val internalizer = KBitwuzlaExprInternalizer(ctx, bitwuzlaCtx)
    private val converter = KBitwuzlaExprConverter(ctx, bitwuzlaCtx)
    private val sortChecker = SortChecker(ctx)

    init {
        Native.bitwuzlaSetOption(bitwuzlaCtx.bitwuzla, BitwuzlaOption.BITWUZLA_OPT_INCREMENTAL, 1)
        Native.bitwuzlaSetOption(bitwuzlaCtx.bitwuzla, BitwuzlaOption.BITWUZLA_OPT_PRODUCE_MODELS, 1)
    }

    @Test
    fun testSimpleBoolExpr(): Unit = with(ctx) {
        val a by boolSort
        val b by boolSort
        val c by boolSort
        val d by boolSort
        val expr = mkIte(a, mkAnd(a, b, d) or c, a and c or b)
        val term = with(internalizer) { expr.internalize() }
        val converted = with(converter) { term.convertExpr(boolSort) }
        converted.accept(sortChecker)
        // compare with term instead of original expr due to internal Bitwuzla rewritings
        val convertedTerm = with(internalizer) { converted.internalize() }
        assertEquals(term, convertedTerm)
    }

    @Test
    fun testSimpleBV1Expr(): Unit = with(ctx) {
        val a by bv1Sort
        val b by bv1Sort
        val c by bv1Sort
        val d by bv1Sort
        val expr = mkIte(
            a.toBool(),
            mkAnd(a.toBool(), b.toBool(), d.toBool()) or c.toBool(),
            a.toBool() and c.toBool() or b.toBool()
        )
        val term = with(internalizer) { expr.internalize() }
        val convertedBool = with(converter) { term.convertExpr(boolSort) }
        convertedBool.accept(sortChecker)
        val convertedBv = with(converter) { term.convertExpr(bv1Sort) }
        convertedBv.accept(sortChecker)
        val convertedBoolTerm = with(internalizer) { convertedBool.internalize() }
        val convertedBvTerm = with(internalizer) { convertedBv.internalize() }
        assertEquals(term, convertedBoolTerm)
        assertEquals(term, convertedBvTerm)
    }

    @Test
    fun testBoolArrayEquality(): Unit = with(ctx) {
        val a by mkArraySort(bv1Sort, bv1Sort)
        val b by mkArraySort(boolSort, boolSort)
        val aTerm = with(internalizer) { a.internalize() }
        val bTerm = with(internalizer) { b.internalize() }
        val term = Native.bitwuzlaMkTerm2(bitwuzlaCtx.bitwuzla, BitwuzlaKind.BITWUZLA_KIND_EQUAL, aTerm, bTerm)
        val converted = with(converter) { term.convertExpr(boolSort) }
        converted.accept(sortChecker)
        val convertedTerm = with(internalizer) { converted.internalize() }
        assertTrue(checkEquivalent(term, convertedTerm))
    }


    @Test
    fun testBvToBoolArrayExpr(): Unit = with(ctx) {
        val x by bv1Sort
        val onlyRange by mkArraySort(mkBv16Sort(), bv1Sort)
        val onlyDomain by mkArraySort(bv1Sort, mkBv16Sort())
        val rangeAndDomain by mkArraySort(bv1Sort, bv1Sort)
        val expr = rangeAndDomain.select(onlyRange.select(onlyDomain.select(x)))
        val term = with(internalizer) { expr.internalize() }
        val convertedBool = with(converter) { term.convertExpr(boolSort) }
        convertedBool.accept(sortChecker)
        val convertedBv1 = with(converter) { term.convertExpr(bv1Sort) }
        convertedBv1.accept(sortChecker)
        val convertedBoolTerm = with(internalizer) { convertedBool.internalize() }
        val convertedBv1Term = with(internalizer) { convertedBv1.internalize() }
        assertTrue(checkEquivalent(term, convertedBoolTerm))
        assertTrue(checkEquivalent(term, convertedBv1Term))
    }

    @Test
    fun testArrayToBVExpr(): Unit = with(ctx) {
        val a by mkArraySort(bv1Sort, bv1Sort)
        val term = with(internalizer) { a.store(mkBv(true), mkBv(true)).internalize() }
        val converted = with(converter) { term.convertExpr(a.sort) }
        converted.accept(sortChecker)
        val convertedTerm = with(internalizer) { converted.internalize() }
        assertTrue(checkEquivalent(term, convertedTerm))
    }

    @Test
    fun testBoolToBvExpr(): Unit = with(ctx) {
        val f = mkFuncDecl("f", bv1Sort, listOf(boolSort, bv1Sort))
        val bool by boolSort
        val bv1 by bv1Sort

        val (fAppTerm, eqTerm) = with(internalizer) {
            val fAppTerm = f.apply(listOf(bool, bv1)).internalize()
            val fTerm = Native.bitwuzlaTermGetChildren(fAppTerm)[0]
            val boolTerm = bool.internalize()
            val bv1Term = bv1.internalize()
            val fAppTermInversed = Native.bitwuzlaMkTerm3(
                bitwuzlaCtx.bitwuzla, BitwuzlaKind.BITWUZLA_KIND_APPLY, fTerm, bv1Term, boolTerm
            )
            val term = Native.bitwuzlaMkTerm2(
                bitwuzlaCtx.bitwuzla,
                BitwuzlaKind.BITWUZLA_KIND_EQUAL,
                fAppTerm,
                fAppTermInversed
            )
            fAppTermInversed to term
        }
        val converted = with(converter) { eqTerm.convertExpr(boolSort) }
        converted.accept(sortChecker)
        val convertedFBool = with(converter) { fAppTerm.convertExpr(boolSort) }
        convertedFBool.accept(sortChecker)
        val convertedFBv = with(converter) { fAppTerm.convertExpr(bv1Sort) }
        convertedFBv.accept(sortChecker)
        val convertedTerm = with(internalizer) { converted.internalize() }
        assertTrue(checkEquivalent(eqTerm, convertedTerm))
    }

    private fun KExpr<KBv1Sort>.toBool() = with(ctx) {
        mkIte(this@toBool eq mkBv(true), trueExpr, falseExpr)
    }

    private class SortChecker(override val ctx: KContext) : KTransformer {
        override fun <T : KSort> transformApp(expr: KApp<T, *>): KExpr<T> = with(ctx) {
            // apply internally check arguments sorts
            expr.decl.apply(expr.args)
            return super.transformApp(expr)
        }
    }

    private fun checkEquivalent(lhs: BitwuzlaTerm, rhs: BitwuzlaTerm): Boolean = with(bitwuzlaCtx) {
        val checkExpr = Native.bitwuzlaMkTerm1(
            bitwuzla,
            BitwuzlaKind.BITWUZLA_KIND_NOT,
            Native.bitwuzlaMkTerm2(bitwuzla, BitwuzlaKind.BITWUZLA_KIND_EQUAL, lhs, rhs)
        )
        Native.bitwuzlaPush(bitwuzla, 1)
        Native.bitwuzlaAssert(bitwuzla, checkExpr)
        val status = Native.bitwuzlaCheckSat(bitwuzla)
        if (status == BitwuzlaResult.BITWUZLA_SAT) {
            Native.bitwuzlaPrintModel(bitwuzla, "smt2", FilePtrUtils.stdout())
        }
        Native.bitwuzlaPop(bitwuzla, 1)
        status == BitwuzlaResult.BITWUZLA_UNSAT
    }
}
