package org.ksmt.solver.yices

import com.sri.yices.Constructor
import com.sri.yices.Terms
import com.sri.yices.Terms.Component
import com.sri.yices.Terms.NULL_TERM
import com.sri.yices.Types
import org.ksmt.KContext
import org.ksmt.decl.KConstDecl
import org.ksmt.decl.KDecl
import org.ksmt.expr.KExpr
import org.ksmt.solver.util.ExprConversionResult
import org.ksmt.solver.util.KExprConverterBase
import org.ksmt.sort.KArithSort
import org.ksmt.sort.KArray2Sort
import org.ksmt.sort.KArray3Sort
import org.ksmt.sort.KArrayNSort
import org.ksmt.sort.KArraySort
import org.ksmt.sort.KArraySortBase
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.KBv1Sort
import org.ksmt.sort.KBvSort
import org.ksmt.sort.KIntSort
import org.ksmt.sort.KRealSort
import org.ksmt.sort.KSort
import org.ksmt.utils.uncheckedCast

open class KYicesExprConverter(
    private val ctx: KContext,
    private val yicesCtx: KYicesContext
) : KExprConverterBase<YicesTerm>() {
    fun <T : KSort> YicesTerm.convert(): KExpr<T> = convertFromNative()

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

                mkRealNum(value)
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

            else -> error("Not supported term ${Terms.toString(expr)}")
        }
    }

    private fun convertProjection(expr: YicesTerm) = with(ctx) {
        val constructor: Constructor = Terms.constructor(expr)
        check(constructor == Constructor.BIT_TERM) { "Not supported term ${Terms.toString(expr)}" }

        expr.convert(arrayOf(Terms.projArg(expr))) { bv: KExpr<KBvSort> ->
            val bv1 = mkBvExtractExpr(Terms.projIndex(expr), Terms.projIndex(expr), bv)
            val condition = mkEq(mkBv(true, 1u), bv1)
            mkIte(condition, mkTrue(), mkFalse())
        }
    }

    private inline fun <K, T : KSort> YicesTerm.convertComponents(
        getComponent: (YicesTerm, Int) -> Component<K>,
        expectedTermSort: T,
        wrapConstant: KContext.(K, T) -> KExpr<T>,
        mkComponentTerm: KContext.(K, KExpr<T>) -> KExpr<T>,
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
            val expressions = mutableListOf<KExpr<T>>()
            simpleConstants.mapTo(expressions) { ctx.wrapConstant(it, expectedTermSort) }

            convertedTerms.zip(termConstants) { term, const ->
                val termWithCorrectSort = term.ensureSort(expectedTermSort)
                ctx.mkComponentTerm(const, termWithCorrectSort)
            }

            expressions.reduce { acc, e -> ctx.reduceComponentTerms(acc, e) }
        }
    }

    private fun convertSum(expr: YicesTerm): ExprConversionResult = expr.convertComponents(
        getComponent = { _, idx -> Terms.sumComponent(expr, idx) },
        expectedTermSort = ctx.realSort,
        wrapConstant = { value, _ -> mkRealNum(value) },
        mkComponentTerm = { const, term -> mkArithMul(mkRealNum(const), term) },
        reduceComponentTerms = { acc, term -> mkArithAdd(acc, term) }
    )

    private fun convertBvSum(expr: YicesTerm): ExprConversionResult = expr.convertComponents(
        getComponent = { _, idx -> Terms.sumbvComponent(expr, idx) },
        expectedTermSort = ctx.mkBvSort(Terms.bitSize(expr).toUInt()),
        wrapConstant = { value, sort -> mkBv(value, sort) },
        mkComponentTerm = { const, term -> mkBvMulExpr(mkBv(const, term.sort), term) },
        reduceComponentTerms = { acc, term -> mkBvAddExpr(acc, term) }
    )

    private fun convertProduct(expr: YicesTerm): ExprConversionResult =
        if (Terms.isBitvector(expr)) {
            convertBvProduct(expr)
        } else {
            convertRealProduct(expr)
        }

    private fun convertBvProduct(expr: YicesTerm): ExprConversionResult = expr.convertComponents(
        getComponent = { _, idx -> Terms.productComponent(expr, idx) },
        expectedTermSort = ctx.mkBvSort(Terms.bitSize(expr).toUInt()),
        wrapConstant = { _, _ -> error("Unexpected constant without term in product") },
        mkComponentTerm = { const, term -> mkBvPow(term, const) },
        reduceComponentTerms = { acc, term -> mkBvMulExpr(acc, term) }
    )

    private fun convertRealProduct(expr: YicesTerm): ExprConversionResult = expr.convertComponents(
        getComponent = { _, idx -> Terms.productComponent(expr, idx) },
        expectedTermSort = ctx.realSort,
        wrapConstant = { _, _ -> error("Unexpected constant without term in product") },
        mkComponentTerm = { const, term -> mkArithPower(term, mkRealNum(const)) },
        reduceComponentTerms = { acc, term -> mkArithMul(acc, term) }
    )

    private fun <S : KBvSort> KContext.mkBv(value: Array<Boolean>, sort: S): KExpr<S> =
        mkBv(value.toBooleanArray(), sort.sizeBits).uncheckedCast()

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
        val expectedSort = mergeSorts(trueBranch.sort, falseBranch.sort)
        return mkIte(
            condition,
            trueBranch.ensureSort(expectedSort),
            falseBranch.ensureSort(expectedSort)
        )
    }

    private fun converterMkEq(
        lhs: KExpr<KSort>,
        rhs: KExpr<KSort>
    ): KExpr<KBoolSort> = with(ctx) {
        val expectedSort = mergeSorts(lhs.sort, rhs.sort)
        return mkEq(lhs.ensureSort(expectedSort), rhs.ensureSort(expectedSort))
    }

    private fun converterMkDistinct(args: List<KExpr<KSort>>): KExpr<KBoolSort> = with(ctx) {
        val expectedSort = args.map { it.sort }.reduce(::mergeSorts)
        return mkDistinct(args.map { it.ensureSort(expectedSort) })
    }

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
            Constructor.ITE_TERM -> expr.convert(yicesArgs, ::converterMkIte)
            Constructor.APP_TERM -> expr.convertFunctionApp(yicesArgs)
            Constructor.UPDATE_TERM -> expr.convertFunctionStore(yicesArgs)
            Constructor.EQ_TERM -> expr.convert(yicesArgs, ::converterMkEq)
            Constructor.DISTINCT_TERM -> expr.convertList(yicesArgs, ::converterMkDistinct)
            Constructor.FORALL_TERM -> {
                expr.convert(yicesArgs.takeLast(1).toTypedArray()) { body: KExpr<KBoolSort> ->
                    val bounds = yicesArgs.dropLast(1).map { convertVar(it) }
                    ctx.mkUniversalQuantifier(body, bounds)
                }
            }

            Constructor.LAMBDA_TERM -> {
                expr.convert(yicesArgs.takeLast(1).toTypedArray()) { body: KExpr<KSort> ->
                    val bounds = yicesArgs.dropLast(1).map { convertVar(it) }
                    mkAnyArrayLambda(bounds, body)
                }
            }

            Constructor.NOT_TERM -> expr.convert(yicesArgs, ::mkNot)
            Constructor.OR_TERM -> expr.convertList(yicesArgs, ::mkOr)
            Constructor.XOR_TERM -> {
                expr.convertList(yicesArgs) { args: List<KExpr<KBoolSort>> ->
                    args.reduce(::mkXor)
                }
            }

            Constructor.BV_ARRAY -> {
                expr.convertList(yicesArgs) { args: List<KExpr<KBoolSort>> ->
                    val bvArgs = args.map { element: KExpr<KBoolSort> ->
                        mkIte(element, mkBv(true), mkBv(false))
                    }

                    bvArgs.reduce { acc: KExpr<out KBvSort>, t: KExpr<KBv1Sort> ->
                        mkBvConcatExpr(t, acc)
                    }
                }
            }

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
            array, indices,
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
        array, indices,
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
        indices: List<KExpr<KSort>>,
        array1: (KExpr<KArraySort<KSort, KSort>>, KExpr<KSort>) -> R,
        array2: (KExpr<KArray2Sort<KSort, KSort, KSort>>, KExpr<KSort>, KExpr<KSort>) -> R,
        array3: (KExpr<KArray3Sort<KSort, KSort, KSort, KSort>>, KExpr<KSort>, KExpr<KSort>, KExpr<KSort>) -> R,
        arrayN: (KExpr<KArrayNSort<KSort>>, List<KExpr<KSort>>) -> R
    ): R {
        val expectedIndicesSorts = array.sort.domainSorts.zip(indices) { domainSort, index ->
            mergeSorts(domainSort, index.sort)
        }
        val expectedArraySort = mkAnyArraySort(expectedIndicesSorts, array.sort.range)

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
}
