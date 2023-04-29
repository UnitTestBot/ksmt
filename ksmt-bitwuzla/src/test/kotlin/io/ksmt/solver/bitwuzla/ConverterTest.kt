package io.ksmt.solver.bitwuzla

import io.ksmt.KContext
import io.ksmt.expr.KApp
import io.ksmt.expr.KBitVecValue
import io.ksmt.expr.KExpr
import io.ksmt.expr.rewrite.simplify.KExprSimplifier
import io.ksmt.expr.transformer.KNonRecursiveTransformer
import org.ksmt.solver.bitwuzla.bindings.BitwuzlaBVBase
import org.ksmt.solver.bitwuzla.bindings.BitwuzlaKind
import org.ksmt.solver.bitwuzla.bindings.BitwuzlaOption
import org.ksmt.solver.bitwuzla.bindings.BitwuzlaResult
import org.ksmt.solver.bitwuzla.bindings.BitwuzlaTerm
import org.ksmt.solver.bitwuzla.bindings.Native
import io.ksmt.sort.KBv1Sort
import io.ksmt.sort.KBvSort
import io.ksmt.sort.KFpSort
import io.ksmt.sort.KSort
import io.ksmt.utils.getValue
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class ConverterTest {
    private val ctx = KContext()
    private val bitwuzlaCtx = KBitwuzlaContext(ctx)
    private val internalizer = KBitwuzlaExprInternalizer(bitwuzlaCtx)
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

        assertEquals(convertedBoolTerm, convertedBvTerm)
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

    private class SortChecker(ctx: KContext) : KNonRecursiveTransformer(ctx) {
        override fun <T : KSort, A : KSort> transformApp(expr: KApp<T, A>): KExpr<T> = with(ctx) {
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
        val status = Native.bitwuzlaCheckSatResult(bitwuzla)
        Native.bitwuzlaPop(bitwuzla, 1)
        status == BitwuzlaResult.BITWUZLA_UNSAT
    }

    @Test
    fun testBvValueConversion() = with(bitwuzlaCtx) {
        val ctx = KContext()
        val converter = KBitwuzlaExprConverter(ctx, this)

        val ones52 = Native.bitwuzlaMkBvOnes(bitwuzla, Native.bitwuzlaMkBvSort(bitwuzla, 52))
        val ksmtOnes52 = with(converter) { ones52.convertExpr(ctx.mkBvSort(52u)) }
        assertEquals(52, (ksmtOnes52 as KBitVecValue<*>).stringValue.count { it == '1' })

        val ones32 = Native.bitwuzlaMkBvOnes(bitwuzla, Native.bitwuzlaMkBvSort(bitwuzla, 32))
        val ksmtOnes32 = with(converter) { ones32.convertExpr(ctx.bv32Sort) }
        assertEquals(32, (ksmtOnes32 as KBitVecValue<*>).stringValue.count { it == '1' })

        val someBvSorts = listOf(
            ctx.bv32Sort,
            ctx.bv64Sort,
            ctx.mkBvSort(17u),
            ctx.mkBvSort(517u),
            ctx.mkBvSort(1024u),
            ctx.mkBvSort(2048u),
        )

        for (sort in someBvSorts) {
            val randomBits = randomBits(sort.sizeBits)

            val nativeSort = with(internalizer) { sort.internalizeSort() }
            val randomBitsTerm = Native.bitwuzlaMkBvValue(
                bitwuzla, nativeSort, randomBits, BitwuzlaBVBase.BITWUZLA_BV_BASE_BIN
            )
            val randomBitsConverted = with(converter) { randomBitsTerm.convertExpr(sort) }
            val randomBitsKsmt = ctx.mkBv(randomBits, sort.sizeBits)
            assertEquals(randomBitsKsmt, randomBitsConverted)
        }
    }

    @Test
    fun testBvValueInternalization() = with(bitwuzlaCtx) {
        val ctx = KContext()
        val internalizer = KBitwuzlaExprInternalizer(this)
        val converter = KBitwuzlaExprConverter(ctx, this)

        val ones64 = ctx.mkBvConcatExpr(ctx.mkBv(-1), ctx.mkBv(-1))
        val ones128 = ctx.mkBvConcatExpr(ctx.mkBv(-1L), ctx.mkBv(-1L))
        val ones125 = ctx.mkBv("1".repeat(125), 125u)

        val bzlaOnes64 = with(internalizer) { ones64.internalize() }
        val bzlaOnes128 = with(internalizer) { ones128.internalize() }
        val bzlaOnes125 = with(internalizer) { ones125.internalize() }

        val ksmtOnes64 = with(converter) { bzlaOnes64.convertExpr(ones64.sort) }
        val ksmtOnes128 = with(converter) { bzlaOnes128.convertExpr(ones128.sort) }
        val ksmtOnes125 = with(converter) { bzlaOnes125.convertExpr(ones125.sort) }

        assertEquals(64, (ksmtOnes64 as KBitVecValue<*>).stringValue.count { it == '1' })
        assertEquals(128, (ksmtOnes128 as KBitVecValue<*>).stringValue.count { it == '1' })
        assertEquals(125, (ksmtOnes125 as KBitVecValue<*>).stringValue.count { it == '1' })

        val someBvSorts = listOf(
            ctx.bv32Sort,
            ctx.bv64Sort,
            ctx.mkBvSort(17u),
            ctx.mkBvSort(517u),
            ctx.mkBvSort(1024u),
            ctx.mkBvSort(2048u),
        )

        for (sort in someBvSorts) {
            val randomBits = randomBits(sort.sizeBits)

            val randomBitsKsmt = ctx.mkBv(randomBits, sort.sizeBits)
            val randomBitsInternalized = with(internalizer) { randomBitsKsmt.internalize() }

            Native.bitwuzlaCheckSat(bitwuzla) // Get value is not available before check-sat
            val internalizedBits = Native.bitwuzlaGetBvValue(bitwuzla, randomBitsInternalized)

            assertEquals(randomBits, internalizedBits)
        }
    }

    @Test
    fun testFpValueConversion() = with(bitwuzlaCtx) {
        val ctx = KContext()
        val fpCustom = ctx.mkFpSort(9u, 11u)

        val internalizer = KBitwuzlaExprInternalizer(this)
        val fp32sort = with(internalizer) { ctx.fp32Sort.internalizeSort() }
        val fp64sort = with(internalizer) { ctx.fp64Sort.internalizeSort() }
        val fpCustomSort = with(internalizer) { fpCustom.internalizeSort() }

        val fp32NegInf = Native.bitwuzlaMkFpNegInf(bitwuzla, fp32sort)
        val fp64NegInf = Native.bitwuzlaMkFpNegInf(bitwuzla, fp64sort)
        val fpCustomNegInf = Native.bitwuzlaMkFpNegInf(bitwuzla, fpCustomSort)

        val converter = KBitwuzlaExprConverter(ctx, this)
        val ksmt32NegInf = with(converter) { fp32NegInf.convertExpr(ctx.fp32Sort) }
        val ksmt64NegInf = with(converter) { fp64NegInf.convertExpr(ctx.fp64Sort) }
        val ksmtCustomNegInf = with(converter) { fpCustomNegInf.convertExpr(fpCustom) }

        assertEquals(ctx.mkFpInf(sort = ctx.fp32Sort, signBit = true), ksmt32NegInf)
        assertEquals(ctx.mkFpInf(sort = ctx.fp64Sort, signBit = true), ksmt64NegInf)
        assertEquals(ctx.mkFpInf(sort = fpCustom, signBit = true), ksmtCustomNegInf)

        val someFpSorts = listOf(ctx.fp32Sort, ctx.fp64Sort, ctx.mkFpSort(17u, 29u))
        for (someFpSort in someFpSorts) {
            val someBvSort = ctx.mkBvSort(someFpSort.exponentBits + someFpSort.significandBits)

            val randomBits = randomBits(someBvSort.sizeBits)
            val nativeBvSort = with(internalizer) { someBvSort.internalizeSort() }
            val randomBitsTerm = Native.bitwuzlaMkBvValue(
                bitwuzla, nativeBvSort, randomBits, BitwuzlaBVBase.BITWUZLA_BV_BASE_BIN
            )
            val randomBitsFp = Native.bitwuzlaMkTerm1Indexed2(
                bitwuzla, BitwuzlaKind.BITWUZLA_KIND_FP_TO_FP_FROM_BV,
                randomBitsTerm, someFpSort.exponentBits.toInt(), someFpSort.significandBits.toInt()
            )
            val convertedRandomFp = with(converter) { randomBitsFp.convertExpr(someFpSort) }
            val convertedBvValue = KExprSimplifier(ctx).apply(ctx.mkFpToIEEEBvExpr(convertedRandomFp))

            val expectedBv = ctx.normalizeFpBits(someFpSort, randomBits)
            assertEquals(expectedBv, convertedBvValue)
        }
    }

    /**
     * Normalize fp bits to ensure equivalent bit representation for NaN values.
     * */
    private fun KContext.normalizeFpBits(sort: KFpSort, bits: String): KExpr<KBvSort> {
        val sign = mkBv(bits[0] == '1')
        val exponent = mkBv(
            bits.substring(1, sort.exponentBits.toInt() + 1),
            sort.exponentBits
        )
        val significand = mkBv(
            bits.substring(sort.exponentBits.toInt() + 1),
            sort.significandBits - 1u
        )
        val normalizedBits = mkFpToIEEEBvExpr(mkFpFromBvExpr(sign, exponent, significand))
        val normalizedBv = KExprSimplifier(this).apply(normalizedBits)
        assertTrue(normalizedBv is KBitVecValue<*>)
        return normalizedBv
    }

    private fun randomBits(size: UInt) =
        String(CharArray(size.toInt()) { if (Random.nextBoolean()) '0' else '1' })
}
