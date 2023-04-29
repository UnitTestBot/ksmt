package io.ksmt.solver.yices

import com.sri.yices.Constructor
import com.sri.yices.Terms
import com.sri.yices.Terms.Component
import com.sri.yices.Terms.NULL_TERM
import com.sri.yices.Types
import io.ksmt.KContext
import io.ksmt.decl.KConstDecl
import io.ksmt.decl.KDecl
import io.ksmt.expr.KExpr
import io.ksmt.expr.printer.ExpressionPrinter
import io.ksmt.expr.transformer.KNonRecursiveTransformer
import io.ksmt.expr.transformer.KTransformer
import io.ksmt.expr.transformer.KTransformerBase
import io.ksmt.solver.util.ExprConversionResult
import io.ksmt.solver.util.KExprConverterBase
import io.ksmt.sort.KArithSort
import io.ksmt.sort.KArray2Sort
import io.ksmt.sort.KArray3Sort
import io.ksmt.sort.KArrayNSort
import io.ksmt.sort.KArraySort
import io.ksmt.sort.KArraySortBase
import io.ksmt.sort.KBoolSort
import io.ksmt.sort.KBvSort
import io.ksmt.sort.KIntSort
import io.ksmt.sort.KRealSort
import io.ksmt.sort.KSort
import io.ksmt.sort.KUninterpretedSort
import io.ksmt.utils.BvUtils.bvMaxValueUnsigned
import io.ksmt.utils.BvUtils.bvZero
import io.ksmt.utils.uncheckedCast

open class KYicesExprConverter(
    private val ctx: KContext,
    private val yicesCtx: KYicesContext
) : KExprConverterBase<YicesTerm>() {
    fun <T : KSort> YicesTerm.convert(expectedSort: T): KExpr<T> =
        convertFromNative<KSort>()
            .ensureSort(expectedSort)

    override fun findConvertedNative(expr: YicesTerm): KExpr<*>? =
        yicesCtx.findConvertedExpr(expr)

    override fun saveConvertedNative(native: YicesTerm, converted: KExpr<*>) {
        yicesCtx.saveConvertedExpr(native, converted)
    }

    open fun convertDecl(decl: YicesTerm): KDecl<*> = yicesCtx.convertDecl(decl) {
        generateFreshDecl(decl)
    }

    open fun convertVar(variable: YicesTerm): KDecl<*> = yicesCtx.convertVar(variable) {
        generateFreshDecl(variable)
    }

    open fun findOrCreateFunctionDecl(term: YicesTerm): KDecl<*> =
        yicesCtx.findConvertedDecl(term)
            ?: yicesCtx.findConvertedVar(term)
            ?: convertDecl(term)

    open fun generateFreshDecl(term: YicesTerm): KDecl<*> {
        val name = Terms.getName(term) ?: "yices"
        val sort = Terms.typeOf(term)

        if (!Terms.isFunction(term)) {
            return ctx.mkFreshConstDecl(name, convertSort(sort))
        }

        val childrenTypes = Types.children(sort).map { convertSort(it) }
        return ctx.mkFreshFuncDecl(name, childrenTypes.last(), childrenTypes.dropLast(1))
    }

    open fun convertSort(sort: YicesSort): KSort = yicesCtx.convertSort(sort) {
        with(ctx) {
            when (sort) {
                yicesCtx.int -> intSort
                yicesCtx.real -> realSort
                yicesCtx.bool -> boolSort
                else -> when {
                    Types.isBitvector(sort) -> mkBvSort(Types.bvSize(sort).toUInt())
                    Types.isUninterpreted(sort) -> mkUninterpretedSort(
                        Types.getName(sort) ?: error("Unexpected null name")
                    )

                    Types.isFunction(sort) -> {
                        val sortChildren = Types.children(sort)
                        val domain = sortChildren.dropLast(1).map { convertSort(it) }
                        val range = convertSort(sortChildren.last())

                        mkAnyArraySort(domain, range)
                    }

                    else -> error("Not supported sort")
                }
            }
        }
    }

    override fun convertNativeExpr(expr: YicesTerm): ExprConversionResult {
        return when {
            Terms.isAtomic(expr) -> convertAtomic(expr)
            Terms.isProjection(expr) -> convertProjection(expr)
            Terms.isSum(expr) -> convertSum(expr)
            Terms.isBvSum(expr) -> convertBvSum(expr)
            Terms.isProduct(expr) -> convertProduct(expr)
            Terms.isComposite(expr) -> convertComposite(expr)
            else -> error("Invalid term")
        }
    }

    private fun convertAtomic(expr: YicesTerm) = with(ctx) {
        when (Terms.constructor(expr)) {
            Constructor.BOOL_CONSTANT -> convert { Terms.boolConstValue(expr).expr }
            Constructor.ARITH_CONSTANT -> convert {
                val value = Terms.arithConstValue(expr)

                mkArithNum(value)
            }

            Constructor.BV_CONSTANT -> convert {
                mkBv(Terms.bvConstValue(expr), Terms.bitSize(expr).toUInt())
            }

            Constructor.VARIABLE -> convert {
                val convertedDecl = convertVar(expr)
                check(convertedDecl is KConstDecl<*>) { "Unexpected variable $convertedDecl" }

                convertedDecl.apply()
            }

            Constructor.UNINTERPRETED_TERM -> convert {
                val convertedDecl = convertDecl(expr)
                check(convertedDecl is KConstDecl<*>) { "Unexpected declaration $convertedDecl" }

                convertedDecl.apply()
            }

            Constructor.SCALAR_CONSTANT -> convert {
                val idx = Terms.scalarConstantIndex(expr)
                val valueIdx = yicesCtx.convertUninterpretedSortValueIndex(idx)

                val sort = convertSort(Terms.typeOf(expr))

                mkUninterpretedSortValue(sort as KUninterpretedSort, valueIdx)
            }

            else -> error("Not supported term ${Terms.toString(expr)}")
        }
    }

    private fun convertProjection(expr: YicesTerm) = with(ctx) {
        val constructor: Constructor = Terms.constructor(expr)
        check(constructor == Constructor.BIT_TERM) { "Not supported term ${Terms.toString(expr)}" }

        expr.convert(arrayOf(Terms.projArg(expr))) { bv: KExpr<KBvSort> ->
            BvBitExtractRoot(bv, Terms.projIndex(expr))
        }
    }

    private inline fun <K, T: KSort, S : KSort?> YicesTerm.convertComponents(
        getComponent: (YicesTerm, Int) -> Component<K>,
        expectedTermSort: S,
        wrapConstant: KContext.(K, S) -> KExpr<*>,
        mkComponentTerm: KContext.(KExpr<T>, KExpr<T>) -> KExpr<T>,
        reduceComponentTerms: KContext.(KExpr<T>, KExpr<T>) -> KExpr<T>
    ): ExprConversionResult {
        val simpleConstants = mutableListOf<K>()
        val termConstants = mutableListOf<K>()
        val terms = mutableListOf<YicesTerm>()

        for (i in 0 until Terms.numChildren(this)) {
            val component = getComponent(this, i)
            if (component.term == NULL_TERM) {
                simpleConstants.add(component.constValue)
            } else {
                termConstants.add(component.constValue)
                terms.add(component.term)
            }
        }

        return convertList(terms.toTypedArray()) { convertedTerms: List<KExpr<KSort>> ->
            val wrappedSimpleConstants = simpleConstants.map { ctx.wrapConstant(it, expectedTermSort) }
            val wrappedTermConstants = termConstants.map { ctx.wrapConstant(it, expectedTermSort) }

            val expectedSort = (wrappedSimpleConstants + wrappedTermConstants + convertedTerms)
                .map { it.sort }
                .reduce { acc, exprSort -> mergeSorts(acc, exprSort) }

            val expressions = mutableListOf<KExpr<T>>()
            wrappedSimpleConstants.mapTo(expressions) { it.ensureSort(expectedSort).uncheckedCast() }

            convertedTerms.zip(wrappedTermConstants) { term, const ->
                val termWithCorrectSort: KExpr<T> = term.ensureSort(expectedSort).uncheckedCast()
                val constWithCorrectSort: KExpr<T> = const.ensureSort(expectedSort).uncheckedCast()
                expressions += ctx.mkComponentTerm(constWithCorrectSort, termWithCorrectSort)
            }

            expressions.reduce { acc, e -> ctx.reduceComponentTerms(acc, e) }
        }
    }

    private fun convertSum(expr: YicesTerm): ExprConversionResult = expr.convertComponents<_, KArithSort, _>(
        getComponent = { _, idx -> Terms.sumComponent(expr, idx) },
        expectedTermSort = null,
        wrapConstant = { value, _ -> mkArithNum(value) },
        mkComponentTerm = { const, term -> mkArithMul(const, term) },
        reduceComponentTerms = { acc, term -> mkArithAdd(acc, term) }
    )

    private fun convertBvSum(expr: YicesTerm): ExprConversionResult = expr.convertComponents<_, KBvSort, _>(
        getComponent = { _, idx -> Terms.sumbvComponent(expr, idx) },
        expectedTermSort = ctx.mkBvSort(Terms.bitSize(expr).toUInt()),
        wrapConstant = { value, sort -> mkBv(value.toBooleanArray(), sort) },
        mkComponentTerm = { const, term -> mkBvMulExpr(const, term) },
        reduceComponentTerms = { acc, term -> mkBvAddExpr(acc, term) }
    )

    private fun convertProduct(expr: YicesTerm): ExprConversionResult =
        if (Terms.isBitvector(expr)) {
            convertBvProduct(expr)
        } else {
            convertRealProduct(expr)
        }

    private inner class BvPowerWrapper<T : KBvSort>(val value: Int, override val sort: T) : KExpr<T>(ctx) {
        override fun print(printer: ExpressionPrinter) {
            printer.append("(wrap $value)")
        }

        override fun accept(transformer: KTransformerBase): KExpr<T> = error("Transformers are not used for wrapper")
        override fun internEquals(other: Any): Boolean = error("Interning is not used for wrapper")
        override fun internHashCode(): Int = error("Interning is not used for wrapper")
    }

    private fun convertBvProduct(expr: YicesTerm): ExprConversionResult = expr.convertComponents<_, KBvSort, _>(
        getComponent = { _, idx -> Terms.productComponent(expr, idx) },
        expectedTermSort = ctx.mkBvSort(Terms.bitSize(expr).toUInt()),
        wrapConstant = { value, sort -> BvPowerWrapper(value, sort) },
        mkComponentTerm = { const, term -> mkBvPow(term, (const as BvPowerWrapper<*>).value) },
        reduceComponentTerms = { acc, term -> mkBvMulExpr(acc, term) }
    )

    private fun convertRealProduct(expr: YicesTerm): ExprConversionResult = expr.convertComponents<_, KArithSort, _>(
        getComponent = { _, idx -> Terms.productComponent(expr, idx) },
        expectedTermSort = ctx.realSort,
        wrapConstant = { value, _ -> mkIntNum(value) },
        mkComponentTerm = { const, term -> mkArithPower(term, const) },
        reduceComponentTerms = { acc, term -> mkArithMul(acc, term) }
    )

    private fun <S : KBvSort> KContext.mkBv(value: BooleanArray, sort: S): KExpr<S> =
        mkBv(value, sort.sizeBits).uncheckedCast()

    private fun KContext.mkBvPow(base: KExpr<KBvSort>, exp: Int): KExpr<KBvSort> {
        check(exp > 0)

        if (exp == 1)
            return base

        val t = mkBvPow(base, exp / 2).let {
            mkBvMulExpr(it, it)
        }

        return if (exp % 2 == 0)
            t
        else
            mkBvMulExpr(base, t)
    }

    private fun mkFloor(arg: KExpr<KRealSort>): KExpr<KIntSort> = ctx.mkRealToInt(arg)

    private fun mkCeil(arg: KExpr<KRealSort>): KExpr<KIntSort> = with(ctx) {
        val floor = mkFloor(arg)
        val realFloor = mkIntToReal(floor)
        val condition = mkEq(arg, realFloor)

        mkIte(condition, floor, mkArithAdd(floor, mkIntNum(1)))
    }

    private fun mkIDiv(arg0: KExpr<KRealSort>, arg1: KExpr<KRealSort>): KExpr<KIntSort> = with(ctx) {
        val condition = mkArithGt(arg1, ctx.mkRealNum(0))
        val div = mkArithDiv(arg0, arg1)

        mkIte(condition, mkFloor(div), mkCeil(div))
    }

    private fun converterMkIte(
        condition: KExpr<KBoolSort>,
        trueBranch: KExpr<KSort>,
        falseBranch: KExpr<KSort>
    ): KExpr<KSort> = with(ctx) {
        if (trueBranch is BvBitExtractExpr || falseBranch is BvBitExtractExpr) {
            return BvBitExtractIte(condition, trueBranch.uncheckedCast(), falseBranch.uncheckedCast()).uncheckedCast()
        }

        val expectedSort = mergeSorts(trueBranch.sort, falseBranch.sort)
        return mkIte(
            condition.ensureSort(boolSort),
            trueBranch.ensureSort(expectedSort),
            falseBranch.ensureSort(expectedSort)
        )
    }

    private fun converterMkEq(
        lhs: KExpr<KSort>,
        rhs: KExpr<KSort>
    ): KExpr<KBoolSort> = with(ctx) {
        if (lhs is BvBitExtractExpr || rhs is BvBitExtractExpr) {
            return BvBitExtractEq(lhs.uncheckedCast(), rhs.uncheckedCast())
        }

        val expectedSort = mergeSorts(lhs.sort, rhs.sort)
        return mkEq(lhs.ensureSort(expectedSort), rhs.ensureSort(expectedSort))
    }

    private fun converterMkDistinct(args: List<KExpr<KSort>>): KExpr<KBoolSort> = with(ctx) {
        if (args.any { it is BvBitExtractExpr }) {
            return BvBitExtractDistinct(args.uncheckedCast())
        }

        val expectedSort = args.map { it.sort }.reduce(::mergeSorts)
        return mkDistinct(args.map { it.ensureSort(expectedSort) })
    }

    private fun converterMkNot(arg: KExpr<KBoolSort>): KExpr<KBoolSort> = with(ctx) {
        if (arg is BvBitExtractExpr) {
            return BvBitExtractNot(arg)
        }

        return mkNot(arg)
    }

    private fun converterMkOr(args: List<KExpr<KBoolSort>>): KExpr<KBoolSort> = with(ctx) {
        if (args.any { it is BvBitExtractExpr }) {
            return BvBitExtractOr(args)
        }

        return mkOr(args)
    }

    private fun converterMkXor(args: List<KExpr<KBoolSort>>): KExpr<KBoolSort> = with(ctx) {
        if (args.any { it is BvBitExtractExpr }) {
            return BvBitExtractXor(args)
        }

        return args.reduce(::mkXor)
    }

    private fun convertBvArray(args: List<KExpr<KBoolSort>>): KExpr<KBvSort> =
        processBvArrayChunked(args.asReversed())

    private fun converterMkFunctionApp(decl: KDecl<*>, args: List<KExpr<KSort>>): KExpr<*> = with(ctx) {
        check(decl.argSorts.size == args.size) { "Arguments size mismatch" }
        val argsWithCorrectSorts = args.zip(decl.argSorts) { arg, sort -> arg.ensureSort(sort) }
        decl.apply(argsWithCorrectSorts)
    }

    private fun YicesTerm.convertFunctionApp(args: Array<YicesTerm>) = convertFunctionExpression(
        args = args,
        functionExpression = { decl, convertedArgs -> converterMkFunctionApp(decl, convertedArgs) },
        arrayExpression = { array, convertedArgs -> ctx.mkAnyArraySelect(array, convertedArgs) }
    )

    private fun YicesTerm.convertFunctionStore(args: Array<YicesTerm>) = convertFunctionExpression(
        args = args,
        functionExpression = { decl, convertedArgs ->
            val sort = ctx.mkAnyArraySort(decl.argSorts, decl.sort)
            val array = ctx.mkFunctionAsArray(sort, decl.uncheckedCast())
            ctx.mkAnyArrayStore(
                array = array,
                indices = convertedArgs.dropLast(1),
                value = convertedArgs.last()
            )
        },
        arrayExpression = { array, convertedArgs ->
            ctx.mkAnyArrayStore(
                array = array,
                indices = convertedArgs.dropLast(1),
                value = convertedArgs.last()
            )
        }
    )

    private inline fun YicesTerm.convertFunctionExpression(
        args: Array<YicesTerm>,
        functionExpression: (KDecl<*>, List<KExpr<KSort>>) -> KExpr<*>,
        arrayExpression: (KExpr<KArraySortBase<*>>, List<KExpr<KSort>>) -> KExpr<*>
    ): ExprConversionResult {
        val functionTerm = args.first()
        val functionIsDecl = Terms.isAtomic(functionTerm)

        val appArgs = if (functionIsDecl) {
            // convert function decl separately
            args.copyOfRange(fromIndex = 1, toIndex = args.size)
        } else {
            // convert function expression as part of arguments
            args
        }

        return convertList(appArgs) { convertedArgs: List<KExpr<KSort>> ->
            if (functionIsDecl) {
                val funcDecl = findOrCreateFunctionDecl(functionTerm)
                if (convertedArgs.isNotEmpty() && funcDecl is KConstDecl<*> && funcDecl.sort is KArraySortBase<*>) {
                    val array: KExpr<KArraySortBase<*>> = ctx.mkConstApp(funcDecl).uncheckedCast()
                    arrayExpression(array, convertedArgs)
                } else {
                    functionExpression(funcDecl, convertedArgs)
                }
            } else {
                val array: KExpr<KArraySortBase<*>> = convertedArgs.first().uncheckedCast()
                val arrayArgs = convertedArgs.drop(1)
                arrayExpression(array, arrayArgs)
            }
        }
    }

    @Suppress("LongMethod", "ComplexMethod")
    private fun convertComposite(expr: YicesTerm) = with(ctx) {
        val yicesArgs = Terms.children(expr).toTypedArray()

        when (Terms.constructor(expr)) {
            Constructor.APP_TERM -> expr.convertFunctionApp(yicesArgs)
            Constructor.UPDATE_TERM -> expr.convertFunctionStore(yicesArgs)

            Constructor.EQ_TERM -> expr.convert(yicesArgs, ::converterMkEq)
            Constructor.DISTINCT_TERM -> expr.convertList(yicesArgs, ::converterMkDistinct)
            Constructor.ITE_TERM -> expr.convert(yicesArgs, ::converterMkIte)

            Constructor.FORALL_TERM -> {
                expr.convert(yicesArgs.takeLast(1).toTypedArray()) { body: KExpr<KBoolSort> ->
                    val bounds = yicesArgs.dropLast(1).map { convertVar(it) }
                    ctx.mkUniversalQuantifier(body.ensureSort(boolSort), bounds)
                }
            }

            Constructor.LAMBDA_TERM -> {
                expr.convert(yicesArgs.takeLast(1).toTypedArray()) { body: KExpr<KSort> ->
                    val bounds = yicesArgs.dropLast(1).map { convertVar(it) }
                    mkAnyArrayLambda(bounds, body.eliminateBitExtract())
                }
            }

            Constructor.NOT_TERM -> expr.convert(yicesArgs, ::converterMkNot)
            Constructor.OR_TERM -> expr.convertList(yicesArgs, ::converterMkOr)
            Constructor.XOR_TERM -> expr.convertList(yicesArgs, ::converterMkXor)

            Constructor.BV_ARRAY -> expr.convertList(yicesArgs, ::convertBvArray)

            Constructor.BV_DIV -> expr.convert(yicesArgs, ::mkBvUnsignedDivExpr)
            Constructor.BV_REM -> expr.convert(yicesArgs, ::mkBvUnsignedRemExpr)
            Constructor.BV_SDIV -> expr.convert(yicesArgs, ::mkBvSignedDivExpr)
            Constructor.BV_SREM -> expr.convert(yicesArgs, ::mkBvSignedRemExpr)
            Constructor.BV_SMOD -> expr.convert(yicesArgs, ::mkBvSignedModExpr)
            Constructor.BV_SHL -> expr.convert(yicesArgs, ::mkBvShiftLeftExpr)
            Constructor.BV_LSHR -> expr.convert(yicesArgs, ::mkBvLogicalShiftRightExpr)
            Constructor.BV_ASHR -> expr.convert(yicesArgs, ::mkBvArithShiftRightExpr)
            Constructor.BV_GE_ATOM -> expr.convert(yicesArgs, ::mkBvUnsignedGreaterOrEqualExpr)
            Constructor.BV_SGE_ATOM -> expr.convert(yicesArgs, ::mkBvSignedGreaterOrEqualExpr)

            Constructor.ARITH_GE_ATOM -> expr.convert(yicesArgs) { lhs: KExpr<KArithSort>, rhs: KExpr<KArithSort> ->
                val expectedSort: KArithSort = mergeSorts(lhs.sort, rhs.sort).uncheckedCast()
                mkArithGe(lhs.ensureSort(expectedSort), rhs.ensureSort(expectedSort))
            }

            Constructor.ABS -> expr.convert(yicesArgs) { x: KExpr<KArithSort> ->
                val condition = mkArithUnaryExpr(
                    expr = x,
                    intExpr = { mkArithGe(it, mkIntNum(0)) },
                    realExpr = { mkArithGe(it, mkRealNum(0)) }
                )
                mkIte(condition, x, mkArithUnaryMinus(x))
            }

            Constructor.CEIL -> expr.convert(yicesArgs) { x: KExpr<KArithSort> ->
                mkArithUnaryExpr(
                    expr = x,
                    intExpr = { it },
                    realExpr = { mkCeil(it) }
                )
            }

            Constructor.FLOOR -> expr.convert(yicesArgs) { x: KExpr<KArithSort> ->
                mkArithUnaryExpr(
                    expr = x,
                    intExpr = { it },
                    realExpr = { mkFloor(it) }
                )
            }

            Constructor.RDIV -> expr.convert(yicesArgs) { lhs: KExpr<KArithSort>, rhs: KExpr<KArithSort> ->
                mkArithDiv(lhs.ensureSort(realSort), rhs.ensureSort(realSort))
            }

            Constructor.IDIV -> expr.convert(yicesArgs) { lhs: KExpr<KArithSort>, rhs: KExpr<KArithSort> ->
                mkArithBinaryExpr(
                    lhs = lhs, rhs = rhs,
                    intExpr = { l, r -> mkArithDiv(l, r) },
                    realExpr = { l, r -> mkIDiv(l, r) }
                )
            }

            Constructor.IMOD -> expr.convert(yicesArgs) { lhs: KExpr<KArithSort>, rhs: KExpr<KArithSort> ->
                mkArithBinaryExpr(
                    lhs = lhs, rhs = rhs,
                    intExpr = { l, r -> mkIntMod(l, r) },
                    realExpr = { l, r ->
                        val integerQuotient = mkIDiv(l, r).ensureSort(realSort)
                        mkArithSub(l, mkArithMul(r, integerQuotient))
                    }
                )
            }

            Constructor.IS_INT_ATOM -> expr.convert(yicesArgs) { arg: KExpr<KArithSort> ->
                mkArithUnaryExpr(
                    expr = arg,
                    intExpr = { trueExpr },
                    realExpr = { mkRealIsInt(it) }
                )
            }

            Constructor.DIVIDES_ATOM -> expr.convert(yicesArgs) { lhs: KExpr<KArithSort>, rhs: KExpr<KArithSort> ->
                mkArithBinaryExpr(
                    lhs = lhs, rhs = rhs,
                    intExpr = { l, r -> mkIntRem(r, l) eq mkIntNum(0) },
                    realExpr = { l, r -> mkRealIsInt(mkArithDiv(r, l)) }
                )
            }

            Constructor.ARITH_ROOT_ATOM -> TODO("ARITH_ROOT conversion is not supported")
            Constructor.TUPLE_TERM -> TODO("Tuple conversion is not supported")
            Constructor.CONSTRUCTOR_ERROR -> error("Constructor error")
            else -> error("Unexpected constructor ${Terms.constructor(expr)}")
        }
    }

    private inline fun <T : KSort> KContext.mkArithUnaryExpr(
        expr: KExpr<KArithSort>,
        intExpr: (KExpr<KIntSort>) -> KExpr<T>,
        realExpr: (KExpr<KRealSort>) -> KExpr<T>
    ) = when (expr.sort) {
        intSort -> intExpr(expr.uncheckedCast())
        realSort -> realExpr(expr.uncheckedCast())
        else -> error("Unexpected arith expr ${expr}")
    }

    private inline fun KContext.mkArithBinaryExpr(
        lhs: KExpr<KArithSort>, rhs: KExpr<KArithSort>,
        intExpr: (KExpr<KIntSort>, KExpr<KIntSort>) -> KExpr<*>,
        realExpr: (KExpr<KRealSort>, KExpr<KRealSort>) -> KExpr<*>
    ): KExpr<*> = when (val expectedSort = mergeSorts(lhs.sort, rhs.sort)) {
        intSort -> intExpr(lhs.ensureSort(intSort), rhs.ensureSort(intSort))
        realSort -> realExpr(lhs.ensureSort(realSort), rhs.ensureSort(realSort))
        else -> error("Unexpected arith sort ${expectedSort}")
    }

    private fun mergeSorts(lhs: KSort, rhs: KSort): KSort {
        if (lhs == rhs) return lhs

        if (lhs is KArithSort && rhs is KArithSort) {
            return when {
                lhs is KRealSort && rhs is KIntSort -> lhs
                lhs is KIntSort && rhs is KRealSort -> rhs
                else -> error("Can't merge arith sorts $lhs and $rhs")
            }
        }

        if (lhs is KArraySortBase<*> && rhs is KArraySortBase<*>) {
            check(lhs.domainSorts.size == rhs.domainSorts.size) {
                "Can't merge arrays $lhs and $rhs"
            }

            val mergedDomain = lhs.domainSorts.zip(rhs.domainSorts) { l, r -> mergeSorts(l, r) }
            val mergedRange = mergeSorts(lhs.range, rhs.range)

            return ctx.mkAnyArraySort(mergedDomain, mergedRange)
        }

        error("Unexpected sorts merge: $lhs and $rhs")
    }

    private fun castToInt(expr: KExpr<KRealSort>): KExpr<KIntSort> = ctx.mkRealToInt(expr)
    private fun castToReal(expr: KExpr<KIntSort>): KExpr<KRealSort> = ctx.mkIntToReal(expr)

    private fun castArray(
        expr: KExpr<KArraySortBase<*>>,
        sort: KArraySortBase<*>
    ): KExpr<out KArraySortBase<*>> = with(ctx) {
        val expectedDomain = sort.domainSorts
        val actualDomain = expr.sort.domainSorts

        val expectedIndices = expectedDomain.map { mkFreshConst("i", it) }
        val actualIndices = expectedIndices.zip(actualDomain) { idx, actualSort ->
            idx.ensureSort(actualSort)
        }

        val actualBody = mkAnyArraySelectUnchecked(expr, actualIndices)
        val expectedBody = actualBody.ensureSort(sort.range)

        mkAnyArrayLambda(expectedIndices.map { it.decl }, expectedBody)
    }

    private fun <S : KSort> KExpr<*>.ensureSort(sort: S): KExpr<S> {
        val exprSort = this.sort
        return when {
            sort == ctx.boolSort && this is BvBitExtractExpr -> eliminateBitExtract().uncheckedCast()
            exprSort == sort -> this.uncheckedCast()
            exprSort is KIntSort && sort is KRealSort -> castToReal(this.uncheckedCast()).uncheckedCast()
            exprSort is KRealSort && sort is KIntSort -> castToInt(this.uncheckedCast()).uncheckedCast()
            exprSort is KArraySortBase<*> && sort is KArraySortBase<*> -> {
                castArray(this.uncheckedCast(), sort).uncheckedCast()
            }

            else -> error("Unexpected cast from ${this.sort} to $sort")
        }
    }

    private fun <A : KArraySortBase<*>> KContext.mkAnyArrayStore(
        array: KExpr<A>,
        indices: List<KExpr<KSort>>,
        value: KExpr<KSort>
    ): KExpr<A> {
        val expectedValueSort = mergeSorts(array.sort.range, value.sort)
        val valueWithCorrectSort = value.ensureSort(expectedValueSort)
        return mkAnyArrayOperation(
            array, expectedValueSort, indices,
            { a, d0 -> mkArrayStore(a, d0, valueWithCorrectSort) },
            { a, d0, d1 -> mkArrayStore(a, d0, d1, valueWithCorrectSort) },
            { a, d0, d1, d2 -> mkArrayStore(a, d0, d1, d2, valueWithCorrectSort) },
            { a, domain -> mkArrayNStore(a, domain, valueWithCorrectSort) }
        ).uncheckedCast()
    }

    private fun <A : KArraySortBase<*>> KContext.mkAnyArraySelect(
        array: KExpr<A>,
        indices: List<KExpr<KSort>>
    ): KExpr<KSort> = mkAnyArrayOperation(
        array, array.sort.range, indices,
        { a, d0 -> mkArraySelect(a, d0) },
        { a, d0, d1 -> mkArraySelect(a, d0, d1) },
        { a, d0, d1, d2 -> mkArraySelect(a, d0, d1, d2) },
        { a, domain -> mkArrayNSelect(a, domain) }
    )

    private fun <A : KArraySortBase<*>> KContext.mkAnyArraySelectUnchecked(
        array: KExpr<A>,
        indices: List<KExpr<KSort>>
    ): KExpr<KSort> = mkAnyArrayOperation(
        indices,
        { d0 -> mkArraySelect(array.uncheckedCast(), d0) },
        { d0, d1 -> mkArraySelect(array.uncheckedCast(), d0, d1) },
        { d0, d1, d2 -> mkArraySelect(array.uncheckedCast(), d0, d1, d2) },
        { domain -> mkArrayNSelect(array.uncheckedCast(), domain) }
    )

    @Suppress("LongParameterList")
    private inline fun <A : KArraySortBase<*>, R> KContext.mkAnyArrayOperation(
        array: KExpr<A>,
        expectedArrayRange: KSort,
        indices: List<KExpr<KSort>>,
        array1: (KExpr<KArraySort<KSort, KSort>>, KExpr<KSort>) -> R,
        array2: (KExpr<KArray2Sort<KSort, KSort, KSort>>, KExpr<KSort>, KExpr<KSort>) -> R,
        array3: (KExpr<KArray3Sort<KSort, KSort, KSort, KSort>>, KExpr<KSort>, KExpr<KSort>, KExpr<KSort>) -> R,
        arrayN: (KExpr<KArrayNSort<KSort>>, List<KExpr<KSort>>) -> R
    ): R {
        val expectedIndicesSorts = array.sort.domainSorts.zip(indices) { domainSort, index ->
            mergeSorts(domainSort, index.sort)
        }
        val expectedArraySort = mkAnyArraySort(expectedIndicesSorts, expectedArrayRange)

        val arrayWithCorrectSort = array.ensureSort(expectedArraySort)
        val indicesWithCorrectSorts = indices.zip(expectedIndicesSorts) { index, expectedSort ->
            index.ensureSort(expectedSort)
        }

        return mkAnyArrayOperation(
            indicesWithCorrectSorts,
            { d0 -> array1(arrayWithCorrectSort.uncheckedCast(), d0) },
            { d0, d1 -> array2(arrayWithCorrectSort.uncheckedCast(), d0, d1) },
            { d0, d1, d2 -> array3(arrayWithCorrectSort.uncheckedCast(), d0, d1, d2) },
            { arrayN(arrayWithCorrectSort.uncheckedCast(), it) }
        )
    }

    private fun KContext.mkAnyArrayLambda(domain: List<KDecl<*>>, body: KExpr<*>) =
        mkAnyArrayOperation(
            domain,
            { d0 -> mkArrayLambda(d0, body) },
            { d0, d1 -> mkArrayLambda(d0, d1, body) },
            { d0, d1, d2 -> mkArrayLambda(d0, d1, d2, body) },
            { mkArrayNLambda(it, body) }
        )

    private fun KContext.mkAnyArraySort(domain: List<KSort>, range: KSort): KArraySortBase<KSort> =
        mkAnyArrayOperation(
            domain,
            { d0 -> mkArraySort(d0, range) },
            { d0, d1 -> mkArraySort(d0, d1, range) },
            { d0, d1, d2 -> mkArraySort(d0, d1, d2, range) },
            { mkArrayNSort(it, range) }
        )

    private inline fun <T, R> mkAnyArrayOperation(
        domain: List<T>,
        array1: (T) -> R,
        array2: (T, T) -> R,
        array3: (T, T, T) -> R,
        arrayN: (List<T>) -> R
    ): R = when (domain.size) {
        KArraySort.DOMAIN_SIZE -> array1(domain.single())
        KArray2Sort.DOMAIN_SIZE -> array2(domain.first(), domain.last())
        KArray3Sort.DOMAIN_SIZE -> {
            val (d0, d1, d2) = domain
            array3(d0, d1, d2)
        }

        else -> arrayN(domain)
    }

    /**
     * Yices use bit-blasting for bv logical expressions resulting in an enormous
     * amount of expressions.
     *
     * Usually the following pattern occurred:
     * 1. Single bit extraction operation (bv1 -> bool)
     * 2. Boolean operations with the extracted bits
     * 3. BvArray operation that concatenates individual bits to a single Bv
     *
     * We delay bit extraction operations until we definitely know it usage context:
     * 1. Bit is used as a boolean expression. We eliminate all lazy bit-level operations
     * and replace them with normal boolean operations.
     * 2. Bit us used in a BvArray expression. We can try to merge bits into a single Bv
     * and apply normal Bv operations (e.g. BvAnd).
     * */
    private sealed class BvBitExtractExpr(ctx: KContext) : KExpr<KBoolSort>(ctx) {
        override val sort: KBoolSort = ctx.boolSort

        abstract fun canBeJoinedWith(other: BvBitExtractExpr): Boolean

        override fun print(printer: ExpressionPrinter) {
            printer.append("(bit extract)")
        }

        override fun accept(transformer: KTransformerBase): KExpr<KBoolSort> {
            transformer as? BvBitExtractTransformer ?: error("Leaked bit extract aux expr")
            return accept(transformer)
        }

        abstract fun accept(transformer: BvBitExtractTransformer): KExpr<KBoolSort>

        override fun internEquals(other: Any): Boolean = error("Interning is not used for bv bit extract")
        override fun internHashCode(): Int = error("Interning is not used for bv bit extract")
    }

    private interface BvBitExtractTransformer : KTransformerBase {
        fun transform(expr: BvBitExtractRoot): KExpr<KBoolSort>

        fun transform(expr: BvBitExtractEq): KExpr<KBoolSort>
        fun transform(expr: BvBitExtractIte): KExpr<KBoolSort>
        fun transform(expr: BvBitExtractDistinct): KExpr<KBoolSort>

        fun transform(expr: BvBitExtractNot): KExpr<KBoolSort>
        fun transform(expr: BvBitExtractOr): KExpr<KBoolSort>
        fun transform(expr: BvBitExtractXor): KExpr<KBoolSort>
    }

    private inner class BvBitExtractEliminator : KNonRecursiveTransformer(ctx), BvBitExtractTransformer {
        override fun transform(expr: BvBitExtractRoot): KExpr<KBoolSort> = with(ctx) {
            val bv1 = mkBvExtractExpr(expr.idx, expr.idx, expr.expr)
            val condition = mkEq(mkBv(true, 1u), bv1)
            mkIte(condition, mkTrue(), mkFalse())
        }

        override fun transform(expr: BvBitExtractEq): KExpr<KBoolSort> =
            transformExprAfterTransformed(expr, expr.lhs, expr.rhs) { lhs, rhs ->
                converterMkEq(lhs.uncheckedCast(), rhs.uncheckedCast())
            }

        override fun transform(expr: BvBitExtractIte): KExpr<KBoolSort> =
            transformExprAfterTransformed(expr, expr.trueBranch, expr.falseBranch) { tb, fb ->
                converterMkIte(expr.condition, tb.uncheckedCast(), fb.uncheckedCast()).uncheckedCast()
            }

        override fun transform(expr: BvBitExtractDistinct): KExpr<KBoolSort> =
            transformExprAfterTransformed(expr, expr.args) { args ->
                converterMkDistinct(args.uncheckedCast())
            }

        override fun transform(expr: BvBitExtractNot): KExpr<KBoolSort> =
            transformExprAfterTransformed(expr, expr.arg) { arg ->
                converterMkNot(arg)
            }

        override fun transform(expr: BvBitExtractOr): KExpr<KBoolSort> =
            transformExprAfterTransformed(expr, expr.args) { args ->
                converterMkOr(args.uncheckedCast())
            }

        override fun transform(expr: BvBitExtractXor): KExpr<KBoolSort> =
            transformExprAfterTransformed(expr, expr.args) { args ->
                converterMkXor(args.uncheckedCast())
            }
    }

    private fun KExpr<*>.eliminateBitExtract(): KExpr<*> {
        if (this !is BvBitExtractExpr) return this
        return BvBitExtractEliminator().apply(this)
    }

    private inner class BvBitExtractRoot(
        val expr: KExpr<KBvSort>,
        val idx: Int
    ) : BvBitExtractExpr(ctx) {
        override fun canBeJoinedWith(other: BvBitExtractExpr): Boolean =
            other is BvBitExtractRoot && other.expr == expr

        override fun accept(transformer: BvBitExtractTransformer): KExpr<KBoolSort> =
            transformer.transform(this)
    }

    private inner class BvBitExtractEq(
        val lhs: KExpr<KBoolSort>,
        val rhs: KExpr<KBoolSort>
    ) : BvBitExtractExpr(ctx) {
        override fun canBeJoinedWith(other: BvBitExtractExpr): Boolean =
            other is BvBitExtractEq

        override fun accept(transformer: BvBitExtractTransformer): KExpr<KBoolSort> =
            transformer.transform(this)
    }

    private inner class BvBitExtractIte(
        val condition: KExpr<KBoolSort>,
        val trueBranch: KExpr<KBoolSort>,
        val falseBranch: KExpr<KBoolSort>
    ) : BvBitExtractExpr(ctx) {
        override fun canBeJoinedWith(other: BvBitExtractExpr): Boolean =
            other is BvBitExtractIte && other.condition == condition

        override fun accept(transformer: BvBitExtractTransformer): KExpr<KBoolSort> =
            transformer.transform(this)
    }

    private inner class BvBitExtractDistinct(
        val args: List<KExpr<KBoolSort>>
    ) : BvBitExtractExpr(ctx) {
        override fun canBeJoinedWith(other: BvBitExtractExpr): Boolean =
            other is BvBitExtractDistinct && args.size == other.args.size

        override fun accept(transformer: BvBitExtractTransformer): KExpr<KBoolSort> =
            transformer.transform(this)
    }

    private inner class BvBitExtractNot(
        val arg: KExpr<KBoolSort>
    ) : BvBitExtractExpr(ctx) {
        override fun canBeJoinedWith(other: BvBitExtractExpr): Boolean =
            other is BvBitExtractNot

        override fun accept(transformer: BvBitExtractTransformer): KExpr<KBoolSort> =
            transformer.transform(this)
    }

    private inner class BvBitExtractOr(
        val args: List<KExpr<KBoolSort>>
    ) : BvBitExtractExpr(ctx) {
        override fun canBeJoinedWith(other: BvBitExtractExpr): Boolean =
            other is BvBitExtractOr && args.size == other.args.size

        override fun accept(transformer: BvBitExtractTransformer): KExpr<KBoolSort> =
            transformer.transform(this)
    }

    private inner class BvBitExtractXor(
        val args: List<KExpr<KBoolSort>>
    ) : BvBitExtractExpr(ctx) {
        override fun canBeJoinedWith(other: BvBitExtractExpr): Boolean =
            other is BvBitExtractXor && args.size == other.args.size

        override fun accept(transformer: BvBitExtractTransformer): KExpr<KBoolSort> =
            transformer.transform(this)
    }

    private sealed interface BvArrayChunk {
        fun isEmpty(): Boolean
        fun add(expr: KExpr<KBoolSort>): BvArrayChunk
        fun process(): KExpr<KBvSort>
    }

    private inner class BitExtractChunk(
        val chunk: MutableList<BvBitExtractExpr>
    ) : BvArrayChunk {
        override fun isEmpty(): Boolean = chunk.isEmpty()

        override fun add(expr: KExpr<KBoolSort>): BvArrayChunk = when {
            expr !is BvBitExtractExpr -> OtherChunk(mutableListOf(expr))
            !chunk.last().canBeJoinedWith(expr) -> BitExtractChunk(mutableListOf(expr))
            else -> this.also { chunk.add(expr.uncheckedCast()) }
        }

        override fun process(): KExpr<KBvSort> = applyBitExtractOperation(ctx, chunk.first(), chunk)
    }

    private inner class OtherChunk(val chunk: MutableList<KExpr<KBoolSort>>) : BvArrayChunk {
        override fun isEmpty(): Boolean = chunk.isEmpty()
        override fun add(expr: KExpr<KBoolSort>): BvArrayChunk = if (expr is BvBitExtractExpr) {
            BitExtractChunk(mutableListOf(expr))
        } else {
            this.also { chunk += expr }
        }

        override fun process(): KExpr<KBvSort> = with(ctx) {
            val bvArgs: List<KExpr<KBvSort>> = chunk.map { element: KExpr<KBoolSort> ->
                mkIte(element, mkBv(true), mkBv(false))
            }.uncheckedCast()

            bvArgs.reduceConcat()
        }
    }

    private fun List<KExpr<KBvSort>>.reduceConcat(): KExpr<KBvSort> = with(ctx){
        reduce(::mkBvConcatExpr)
    }

    private fun findBvArrayChunks(args: List<KExpr<KBoolSort>>): List<BvArrayChunk> {
        var chunk: BvArrayChunk = OtherChunk(mutableListOf())
        val chunks = mutableListOf(chunk)

        for (arg in args) {
            val newChunk = chunk.add(arg)
            if (newChunk !== chunk) {
                chunk = newChunk
                chunks += newChunk
            }
        }

        return chunks.filterNot { it.isEmpty() }
    }

    private fun processBvArrayChunked(args: List<KExpr<KBoolSort>>): KExpr<KBvSort> {
        val unprocessedChunks = findBvArrayChunks(args)
        val chunksExpressions = unprocessedChunks.map { it.process() }
        return chunksExpressions.reduceConcat()
    }

    private inner class BitExtractApplyOperation(
        override val ctx: KContext,
        val chunk: List<BvBitExtractExpr>
    ) : BvBitExtractTransformer, KTransformer {
        lateinit var result: KExpr<KBvSort>

        override fun transform(expr: BvBitExtractRoot): KExpr<KBoolSort> {
            result = processRoot(chunk.uncheckedCast())
            return expr
        }

        override fun transform(expr: BvBitExtractEq): KExpr<KBoolSort> {
            result = processEq(chunk.uncheckedCast())
            return expr
        }

        override fun transform(expr: BvBitExtractIte): KExpr<KBoolSort> {
            result = processIte(chunk.uncheckedCast())
            return expr
        }

        override fun transform(expr: BvBitExtractDistinct): KExpr<KBoolSort> {
            result = processDistinct(chunk.uncheckedCast())
            return expr
        }

        override fun transform(expr: BvBitExtractNot): KExpr<KBoolSort> {
            result = processNot(chunk.uncheckedCast())
            return expr
        }

        override fun transform(expr: BvBitExtractOr): KExpr<KBoolSort> {
            result = processOr(chunk.uncheckedCast())
            return expr
        }

        override fun transform(expr: BvBitExtractXor): KExpr<KBoolSort> {
            result = processXor(chunk.uncheckedCast())
            return expr
        }

        private fun processRoot(chunk: List<BvBitExtractRoot>): KExpr<KBvSort> = with(ctx) {
            val groupedExtracts = groupBySubsequentIndices(chunk)
            val extracts = groupedExtracts.map { mkBvExtractExpr(it.first().idx, it.last().idx, chunk.first().expr) }
            extracts.reduceConcat()
        }

        private fun groupBySubsequentIndices(chunk: List<BvBitExtractRoot>): List<List<BvBitExtractRoot>> {
            var currentGroup = mutableListOf(chunk.first())
            val result = mutableListOf(currentGroup)

            for (i in 1 until chunk.size) {
                val element = chunk[i]
                if (element.idx == currentGroup.last().idx - 1) {
                    currentGroup += element
                    continue
                }
                currentGroup = mutableListOf(element)
                result += currentGroup
            }

            return result
        }

        private fun processEq(chunk: List<BvBitExtractEq>): KExpr<KBvSort> = with(ctx) {
            val lhsExpr = processBvArrayChunked(chunk.map { it.lhs })
            val rhsExpr = processBvArrayChunked(chunk.map { it.rhs })

            mkBvNotExpr(mkBvXorExpr(lhsExpr, rhsExpr))
        }

        private fun processDistinct(chunk: List<BvBitExtractDistinct>): KExpr<KBvSort> = with(ctx) {
            val args = processChunkList(chunk.map { it.args })
            when (args.size) {
                1 -> bvMaxValueUnsigned(args.single().sort.sizeBits)
                2 -> mkBvXorExpr(args.first(), args.last())
                else -> bvZero(args.single().sort.sizeBits)
            }.uncheckedCast()
        }

        private fun processIte(chunk: List<BvBitExtractIte>): KExpr<KBvSort> = with(ctx) {
            val trueExpr = processBvArrayChunked(chunk.map { it.trueBranch })
            val falseExpr = processBvArrayChunked(chunk.map { it.falseBranch })

            mkIte(chunk.first().condition, trueExpr, falseExpr)
        }

        private fun processNot(chunk: List<BvBitExtractNot>): KExpr<KBvSort> = with(ctx) {
            val arg = processBvArrayChunked(chunk.map { it.arg })
            mkBvNotExpr(arg)
        }

        private fun processOr(chunk: List<BvBitExtractOr>): KExpr<KBvSort> = with(ctx) {
            val args = processChunkList(chunk.map { it.args })
            args.reduce { acc, expr -> mkBvOrExpr(acc, expr) }
        }

        private fun processXor(chunk: List<BvBitExtractXor>): KExpr<KBvSort> = with(ctx) {
            val args = processChunkList(chunk.map { it.args })
            args.reduce { acc, expr -> mkBvXorExpr(acc, expr) }
        }

        private fun processChunkList(chunkArgs: List<List<KExpr<KBoolSort>>>): List<KExpr<KBvSort>> {
            val result = mutableListOf<KExpr<KBvSort>>()
            for (i in chunkArgs.first().indices) {
                result += processBvArrayChunked(chunkArgs.map { it[i] })
            }
            return result
        }
    }

    private fun <T : BvBitExtractExpr> applyBitExtractOperation(
        ctx: KContext,
        prototype: T,
        chunk: List<T>
    ): KExpr<KBvSort> {
        val operationApplier = BitExtractApplyOperation(ctx, chunk)
        prototype.accept(operationApplier)
        return operationApplier.result
    }
}
