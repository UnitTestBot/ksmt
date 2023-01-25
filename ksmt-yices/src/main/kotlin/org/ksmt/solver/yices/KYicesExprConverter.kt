package org.ksmt.solver.yices

import com.sri.yices.Constructor
import com.sri.yices.Terms
import com.sri.yices.Types
import org.ksmt.KContext
import org.ksmt.decl.KConstDecl
import org.ksmt.expr.KExpr
import org.ksmt.solver.util.KExprConverterBase
import org.ksmt.decl.KDecl
import org.ksmt.decl.KFuncDecl
import org.ksmt.sort.KArithSort
import org.ksmt.sort.KArraySort
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.KBv1Sort
import org.ksmt.sort.KBvSort
import org.ksmt.sort.KIntSort
import org.ksmt.sort.KRealSort
import org.ksmt.sort.KSort
import org.ksmt.utils.asExpr
import org.ksmt.utils.mkConst
import org.ksmt.utils.mkFreshConst
import java.util.LinkedList
import java.util.Queue
import kotlin.collections.HashSet

open class KYicesExprConverter(
    private val ctx: KContext,
    private val yicesCtx: KYicesContext
) : KExprConverterBase<YicesTerm>() {
    private var argTypes: Queue<KSort> = LinkedList()

    fun <T : KSort> YicesTerm.convert(): KExpr<T> {
        /**
         * In Yices arrays are represented as functions,
         * so before the conversion we examine the expression
         * to distinguish between arrays and functions
         */
        KYicesExprPreprocessor().processExpr(this)
        return this.convertFromNative()
    }

    private fun castToInt(expr: KExpr<KRealSort>): KExpr<KIntSort> = ctx.mkRealToInt(expr)
    private fun castToReal(expr: KExpr<KIntSort>): KExpr<KRealSort> = ctx.mkIntToReal(expr)

    private fun <D: KSort> castArrayRangeToInt(
        expr: KExpr<KArraySort<D, KRealSort>>
    ): KExpr<KArraySort<D, KIntSort>> = with(ctx) {
        val indexDecl = expr.sort.domain.mkFreshConst("index")

        mkArrayLambda(
            indexDecl.decl,
            mkRealToInt(expr.select(indexDecl))
        )
    }

    private fun <D: KSort> castArrayRangeToReal(
        expr: KExpr<KArraySort<D, KIntSort>>
    ): KExpr<KArraySort<D, KRealSort>> = with(ctx) {
        val indexDecl = expr.sort.domain.mkFreshConst("index")

        mkArrayLambda(indexDecl.decl, mkIntToReal(expr.select(indexDecl)))
    }

    @Suppress("UNCHECKED_CAST", "ComplexMethod")
    override fun findConvertedNative(expr: YicesTerm): KExpr<*>? = with(ctx) {
        val convertedExpr = yicesCtx.findConvertedExpr(expr)

        /**
         * Yices doesn't distinguish between IntSort and RealSort,
         * therefore we need to cast terms to suitable sorts if needed
         */
        val argType = argTypes.poll() ?: return convertedExpr

        if (convertedExpr != null && convertedExpr.sort != argType) {
            val exprSort = convertedExpr.sort

            return when {
                argType is KIntSort && exprSort is KRealSort -> castToInt(convertedExpr as KExpr<KRealSort>)
                argType is KRealSort && exprSort is KIntSort -> castToReal(convertedExpr as KExpr<KIntSort>)
                argType is KArraySort<*, *>
                        && argType.range is KIntSort
                        && exprSort is KArraySort<*, *>
                        && exprSort.range is KRealSort ->
                    castArrayRangeToInt(convertedExpr as KExpr<KArraySort<*, KRealSort>>)

                argType is KArraySort<*, *>
                        && argType.range is KRealSort
                        && exprSort is KArraySort<*, *>
                        && exprSort.range is KIntSort ->
                    castArrayRangeToReal(convertedExpr as KExpr<KArraySort<*, KIntSort>>)

                else -> error("Invalid state")
            }
        }

        return convertedExpr
    }

    private fun setArithArgTypes(yicesArgs: Array<YicesTerm>) = with(ctx) {
        val argType = if (yicesArgs.any { Terms.typeOf(it) == Types.REAL }) realSort else intSort

        argTypes.addAll(List(yicesArgs.size) { argType })
    }

    private fun setArrayArgTypes(yicesArgs: Array<YicesTerm>) = with(ctx) {
        if (yicesArgs.isEmpty())
            return@with

        val arraySorts = yicesArgs.map {
            convertSort(Terms.typeOf(it)).toArraySort()
        }

        val domain = arraySorts.first().domain
        val range = if (arraySorts.any { it.range == realSort })
            realSort
        else
            arraySorts.first().range

        argTypes.addAll(List(yicesArgs.size) { mkArraySort(domain, range) })
    }

    override fun saveConvertedNative(native: YicesTerm, converted: KExpr<*>) {
        yicesCtx.convertExpr(native) { converted }
    }

    open fun convertDecl(decl: YicesTerm): KDecl<*> = yicesCtx.convertDecl(decl) {
        with(ctx) {
            val name = Terms.getName(decl) ?: error("Unexpected null name")

            if (!Terms.isFunction(decl))
                return@with mkConstDecl(name, convertSort(Terms.typeOf(decl)))

            val childrenTypes = Types.children(Terms.typeOf(decl)).map { convertSort(it) }

            mkFuncDecl(name, childrenTypes.last(), childrenTypes.dropLast(1))
        }
    }


    open fun convertSort(sort: YicesSort): KSort = yicesCtx.convertSort(sort) {
        with(ctx) {
            when (sort) {
                Types.INT -> intSort
                Types.REAL -> realSort
                Types.BOOL -> boolSort
                else -> {
                    if (Types.isBitvector(sort))
                        mkBvSort(Types.bvSize(sort).toUInt())
                    else if (Types.isUninterpreted(sort))
                        mkUninterpretedSort(Types.getName(sort) ?: error("Unexpected null name"))
                    else if (Types.isFunction(sort) && Types.numChildren(sort) == 2) {
                        val domain = convertSort(Types.child(sort, 0))
                        val range = convertSort(Types.child(sort, 1))

                        mkArraySort(domain, range)
                    } else
                        error("Not supported sort")
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

            Constructor.VARIABLE, Constructor.UNINTERPRETED_TERM -> convert {
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

    private fun convertSum(expr: YicesTerm) = with(ctx) {
        val (consts, children) = List(Terms.numChildren(expr)) { index ->
            val component = Terms.sumComponent(expr, index).apply {
                if (term == Terms.NULL_TERM)
                    term = Terms.ONE
            }

            Pair(
                mkRealNum(component.constValue),
                component.term
            )
        }.unzip()

        argTypes.addAll(List(children.size) { realSort })

        expr.convertList(children.toTypedArray()) { args: List<KExpr<KRealSort>> ->
            args.zip(consts)
                .map { mkArithMul(it.first, it.second) }
                .reduce { acc: KExpr<KRealSort>, t: KExpr<KRealSort> ->
                    mkArithAdd(acc, t)
                }
        }
    }

    private fun convertBvSum(expr: YicesTerm) = with(ctx) {
        val bvSize = Terms.bitSize(expr).toUInt()
        val (consts, children) = List(Terms.numChildren(expr)) { index ->
            val component = Terms.sumbvComponent(expr, index).apply {
                if (term == Terms.NULL_TERM)
                    term = Terms.bvConst(bvSize.toInt(), 1L)
            }

            Pair(
                mkBv(component.constValue.toBooleanArray(), bvSize),
                component.term
            )
        }.unzip()

        expr.convertList(children.toTypedArray()) { args: List<KExpr<KBvSort>> ->
            args.zip(consts)
                .map { mkBvMulExpr(it.first, it.second) }
                .reduce(::mkBvAddExpr)
        }
    }

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

    private fun convertProduct(expr: YicesTerm) = with(ctx) {
        val (consts, children) = List(Terms.numChildren(expr)) { index ->
            val component = Terms.productComponent(expr, index)

            Pair(
                component.constValue,
                component.term
            )
        }.unzip()

        check(children.isNotEmpty())

        if (Types.isBitvector(Terms.typeOf(children.first()))) {
            expr.convertList(children.toTypedArray()) { args: List<KExpr<KBvSort>> ->
                args.zip(consts)
                    .map { (base: KExpr<KBvSort>, exp: Int) ->
                        mkBvPow(base, exp)
                    }.reduce { acc: KExpr<KBvSort>, t: KExpr<KBvSort> ->
                        ctx.mkBvMulExpr(acc, t)
                    }
            }
        } else {
            argTypes.addAll(List(children.size) { realSort })
            expr.convertList(children.toTypedArray()) { args: List<KExpr<KRealSort>> ->
                args.zip(consts)
                    .map { (base: KExpr<KRealSort>, exp: Int) ->
                        mkArithPower(base, mkRealNum(exp))
                    }.reduce { acc: KExpr<KRealSort>, t: KExpr<KRealSort> ->
                        mkArithMul(acc, t)
                    }
            }
        }
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

    @Suppress("UNCHECKED_CAST", "LongMethod", "ComplexMethod")
    private fun convertComposite(expr: YicesTerm) = with(ctx) {
        val yicesArgs = Terms.children(expr).toTypedArray()
        val numChildren = Terms.numChildren(expr)

        check(yicesArgs.isNotEmpty())

        when (Terms.constructor(expr)) {
            Constructor.ITE_TERM -> {
                argTypes.add(boolSort)
                val branches = yicesArgs.drop(1).toTypedArray()
                if (Terms.isArithmetic(expr)) {
                    setArithArgTypes(branches)
                    expr.convert(yicesArgs, ::mkIte)
                } else if (Terms.isFunction(expr)) {
                    setArrayArgTypes(branches)
                    expr.convert(yicesArgs, ::mkIte)
                } else {
                    expr.convert(yicesArgs, ::mkIte)
                }
            }
            Constructor.APP_TERM -> {
                val convertedExprSort = yicesCtx.findConvertedExpr(yicesArgs.first())?.sort

                if (convertedExprSort is KArraySort<*, *>) {
                    argTypes.addAll(listOf(convertedExprSort, convertedExprSort.domain))
                    return expr.convert(yicesArgs, ::mkArraySelect)
                }

                if (!Terms.isAtomic(yicesArgs.first())) {
                    // first argument isn't converted
                    return expr.convertList<KBoolSort, _>(yicesArgs) { _: List<KExpr<KSort>> ->
                        error("Unexpected op call")
                    }
                }

                val funcDecl = convertDecl(yicesArgs.first()) as KFuncDecl<*>

                if (funcDecl is KConstDecl<*>) {
                    check(yicesArgs.size == 2)
                    argTypes.add(funcDecl.sort.toArraySort().domain)
                    expr.convert(yicesArgs.drop(1).toTypedArray()) { index: KExpr<KSort> ->
                        mkArraySelect(funcDecl.apply() as KExpr<KArraySort<*, *>>, index)
                    }
                } else {
                    argTypes.addAll(funcDecl.argSorts)
                    expr.convertList(yicesArgs.drop(1).toTypedArray()) { args: List<KExpr<KSort>> ->
                        mkApp(funcDecl, args)
                    }
                }
            }
            Constructor.UPDATE_TERM -> {
                val arrayType = convertSort(Terms.typeOf(yicesArgs.first())).toArraySort()
                val indexType = arrayType.domain
                val rangeType = arrayType.range
                argTypes.addAll(listOf(arrayType, indexType, rangeType))

                expr.convert(yicesArgs, ::mkArrayStore)
            }
            Constructor.EQ_TERM -> {
                if (Terms.isArithmetic(yicesArgs.first())) {
                    setArithArgTypes(yicesArgs)
                    expr.convert(yicesArgs, ::mkEq)
                } else if (Terms.isFunction(yicesArgs.first())) {
                    setArrayArgTypes(yicesArgs)
                    expr.convert(yicesArgs, ::mkEq)
                } else {
                    expr.convert(yicesArgs, ::mkEq)
                }
            }
            Constructor.DISTINCT_TERM -> {
                if (Terms.isArithmetic(yicesArgs.first())) {
                    setArithArgTypes(yicesArgs)
                    expr.convertList(yicesArgs, ::mkDistinct)
                } else if (Terms.isFunction(yicesArgs.first())) {
                    setArrayArgTypes(yicesArgs)
                    expr.convertList(yicesArgs, ::mkDistinct)
                } else {
                    expr.convertList(yicesArgs, ::mkDistinct)
                }
            }
            Constructor.FORALL_TERM -> {
                expr.convert(yicesArgs.takeLast(1).toTypedArray()) { body: KExpr<KBoolSort> ->
                    val bounds = yicesArgs.dropLast(1).map { convertDecl(it) }
                    ctx.mkUniversalQuantifier(body, bounds)
                }
            }
            Constructor.LAMBDA_TERM -> {
                check(numChildren == 2) { "Unexpected number of bounds" }

                expr.convert(yicesArgs.takeLast(1).toTypedArray()) { body: KExpr<KSort> ->
                    if (yicesCtx.findConvertedDecl(yicesArgs.first()) == null) {
                        mkArrayConst(mkArraySort(convertSort(Terms.typeOf(yicesArgs.first())), body.sort), body)
                    } else {
                        val index = convertDecl(yicesArgs.first())
                        mkArrayLambda(index, body)
                    }
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
            Constructor.ARITH_GE_ATOM -> {
                setArithArgTypes(yicesArgs)
                expr.convert<KBoolSort, KArithSort<*>, KArithSort<*>>(yicesArgs, ::mkArithGe)
            }
            Constructor.ABS -> {
                expr.convert(yicesArgs) { x: KExpr<KArithSort<*>> ->
                    val isReal = x.sort == realSort
                    val condition = if (isReal)
                        mkArithGe(x as KExpr<KRealSort>, mkRealNum(0))
                    else
                        mkArithGe(x as KExpr<KIntSort>, mkIntNum(0))

                    mkIte(condition, x, mkArithUnaryMinus(x))
                }
            }
            Constructor.CEIL -> {
                expr.convert(yicesArgs) { x: KExpr<KArithSort<*>> ->
                    if(x.sort == intSort)
                        x
                    else
                        mkCeil(x as KExpr<KRealSort>)
                }
            }
            Constructor.FLOOR -> {
                expr.convert(yicesArgs) { x: KExpr<KArithSort<*>> ->
                    if(x.sort == intSort)
                        x
                    else
                        mkFloor(x as KExpr<KRealSort>)
                }
            }
            Constructor.RDIV -> {
                argTypes.addAll(List(numChildren) { realSort })
                expr.convert<KArithSort<*>, KArithSort<*>, KArithSort<*>>(yicesArgs, ::mkArithDiv)
            }
            Constructor.IDIV -> {
                setArithArgTypes(yicesArgs)
                expr.convert(yicesArgs) { arg0: KExpr<KArithSort<*>>, arg1: KExpr<KArithSort<*>> ->
                    check(arg0.sort == arg1.sort)
                    if (arg0.sort is KIntSort)
                        mkArithDiv(arg0.asExpr(intSort), arg1.asExpr(intSort))
                    else
                        mkIDiv(arg0.asExpr(realSort), arg1.asExpr(realSort))
                }
            }
            Constructor.IMOD -> {
                setArithArgTypes(yicesArgs)
                expr.convert(yicesArgs) { arg0: KExpr<KArithSort<*>>, arg1: KExpr<KArithSort<*>> ->
                    check(arg0.sort == arg1.sort)
                    if (arg0.sort is KIntSort) {
                        mkIntMod(arg0.asExpr(intSort), arg1.asExpr(intSort))
                    } else {
                        val div = mkIntToReal(mkIDiv(arg0.asExpr(realSort), arg1.asExpr(realSort)))
                        val mul = mkArithMul(arg1, div.asExpr(realSort))

                        mkArithSub(arg0, mul)
                    }
                }
            }
            Constructor.IS_INT_ATOM -> {
                setArithArgTypes(yicesArgs)
                expr.convert(yicesArgs) { arg: KExpr<KArithSort<*>> ->
                    if (arg.sort is KIntSort)
                        true.expr
                    else
                        mkRealIsInt(arg.asExpr(realSort))
                }
            }
            Constructor.DIVIDES_ATOM -> {
                setArithArgTypes(yicesArgs)
                expr.convert(yicesArgs) { arg0: KExpr<KArithSort<*>>, arg1: KExpr<KArithSort<*>> ->
                    check(arg0.sort == arg1.sort)
                    if (arg0.sort is KIntSort)
                        mkIntRem(arg1.asExpr(intSort), arg0.asExpr(intSort)) eq mkIntNum(0)
                    else
                        mkRealIsInt(mkArithDiv(arg1.asExpr(realSort), arg0.asExpr(realSort)))
                }
            }
            Constructor.ARITH_ROOT_ATOM -> TODO("ARITH_ROOT conversion is not supported")
            Constructor.TUPLE_TERM -> TODO("Tuple conversion is not supported")
            Constructor.CONSTRUCTOR_ERROR -> error("Constructor error")
            else -> error("Unexpected constructor ${Terms.constructor(expr)}")
        }
    }

    private fun KSort.toArraySort(): KArraySort<*, *> {
        check(this is KArraySort<*, *>) { "Unexpected sort $this" }

        return this
    }

    @Suppress("LoopWithTooManyJumpStatements", "ComplexMethod", "NestedBlockDepth")
    private inner class KYicesExprPreprocessor {
        private val processedExpr = HashSet<YicesTerm>()
        private val exprStack = arrayListOf<YicesTerm>()
        private val isArrayFlagStack = arrayListOf<Boolean>()

        /**
         * Examine expression non-recursively to find array atomics
         */
        fun processExpr(yicesExpr: YicesTerm) {
            exprStack.add(yicesExpr)
            isArrayFlagStack.add(false)

            while (exprStack.isNotEmpty()) {
                val expr = exprStack.removeLast()
                val isArray = isArrayFlagStack.removeLast()
                val constructor: Constructor = Terms.constructor(expr)

                if (processedExpr.contains(expr))
                    continue

                if (Terms.isAtomic(expr)) {
                    if (isArray)
                        processAtomic(expr)

                    continue
                }

                val numChildren = Terms.numChildren(expr)
                check(numChildren > 0) { "Unexpected number of children" }

                val children = when {
                    Terms.isSum(expr) -> (0 until numChildren).mapNotNull { idx ->
                        Terms.sumComponent(expr, idx).term.takeIf { it != Terms.NULL_TERM }
                    }
                    Terms.isBvSum(expr) -> (0 until numChildren).mapNotNull { idx ->
                        Terms.sumbvComponent(expr, idx).term.takeIf { it != Terms.NULL_TERM }
                    }
                    Terms.isProduct(expr) -> List(numChildren) { idx -> Terms.productComponent(expr, idx).term }
                    Terms.isProjection(expr) -> listOf(Terms.projArg(expr))
                    Terms.isComposite(expr) -> Terms.children(expr).toList()
                    else -> error("Unexpected term ${Terms.toString(expr)}")
                }

                children.forEachIndexed { index, child: YicesTerm ->
                    if (Terms.isFunction(child)) {
                        if (constructor == Constructor.APP_TERM && index == 0 || constructor == Constructor.LAMBDA_TERM)
                            isArrayFlagStack.add(false)
                        else
                            isArrayFlagStack.add(true)
                    } else {
                        isArrayFlagStack.add(isArray)
                    }

                    exprStack.add(child)
                }

                processedExpr.add(expr)
            }
        }

        private fun processAtomic(expr: YicesTerm) {
            val constructor: Constructor = Terms.constructor(expr)
            val type = Terms.typeOf(expr)

            if (!(constructor == Constructor.VARIABLE || constructor == Constructor.UNINTERPRETED_TERM))
                return

            if (!Terms.isFunction(expr))
                return

            check(Types.numChildren(type) == 2) { "Unable to convert ${Terms.toString(expr)} into array" }

            val name = Terms.getName(expr) ?: error("Unexpected null name")
            val domain = convertSort(Types.child(type, 0))
            val range = convertSort(Types.child(type, 1))
            val array = ctx.mkArraySort(domain, range).mkConst(name)

            saveConvertedNative(expr, array)
        }
    }
}
