package org.ksmt.expr

import kotlinx.collections.immutable.PersistentMap
import kotlinx.collections.immutable.mutate
import kotlinx.collections.immutable.persistentHashMapOf
import org.ksmt.KContext
import org.ksmt.cache.hash
import org.ksmt.cache.structurallyEqual
import org.ksmt.decl.KArrayConstDecl
import org.ksmt.decl.KDecl
import org.ksmt.decl.KFuncDecl
import org.ksmt.expr.printer.ExpressionPrinter
import org.ksmt.expr.transformer.KTransformerBase
import org.ksmt.sort.KArraySort
import org.ksmt.sort.KArray2Sort
import org.ksmt.sort.KArray3Sort
import org.ksmt.sort.KArraySortBase
import org.ksmt.sort.KArrayNSort
import org.ksmt.sort.KSort
import org.ksmt.utils.asExpr
import org.ksmt.utils.uncheckedCast

internal enum class ArrayStoreType {
    INTERPRETED,
    UNINTERPRETED,
    NOT_INITIALIZED
}

sealed class KArrayStoreBase<A : KArraySortBase<R>, R : KSort>(
    ctx: KContext,
    val array: KExpr<A>,
    val value: KExpr<R>
) : KApp<A, KSort>(ctx) {
    override val sort: A = array.sort

    abstract val indices: List<KExpr<KSort>>

    override val args: List<KExpr<KSort>>
        get() = buildList {
            add(array)
            addAll(this@KArrayStoreBase.indices)
            add(value)
        }.uncheckedCast()

    private var type: ArrayStoreType = ArrayStoreType.NOT_INITIALIZED
    private var interpretedStores: PersistentMap<Any, KArrayStoreBase<A, R>>? = null
    private var nextUninterpretedArray: KExpr<A>? = null

    abstract fun indicesAreInterpreted(): Boolean

    abstract fun findArrayToSelectFrom(indices: List<KExpr<*>>): KExpr<A>

    fun analyzeStore() {
        if (type != ArrayStoreType.NOT_INITIALIZED) return

        if (!indicesAreInterpreted()) {
            type = ArrayStoreType.UNINTERPRETED
            return
        }
        type = ArrayStoreType.INTERPRETED

        if (array !is KArrayStoreBase<A, *> || array.type != ArrayStoreType.INTERPRETED) {
            nextUninterpretedArray = array
            interpretedStores = null
            return
        }

        interpretedStores = updateInterpretedStoresMap(array).uncheckedCast()
        nextUninterpretedArray = array.nextUninterpretedArray
    }

    internal abstract fun createLookupKey(): Any

    internal inline fun lookupArrayToSelectFrom(
        indicesAreEqual: () -> Boolean,
        selectIndicesAreInterpreted: () -> Boolean,
        selectIndicesLookupKey: () -> Any
    ): KExpr<A> {
        if (indicesAreEqual()) return this
        if (type != ArrayStoreType.INTERPRETED || !selectIndicesAreInterpreted()) return this
        return interpretedStores?.get(selectIndicesLookupKey()) ?: nextUninterpretedArray ?: this
    }

    private fun updateInterpretedStoresMap(
        nestedInterpretedStore: KArrayStoreBase<A, *>
    ) = nestedInterpretedStore.interpretedStores
        ?.put(createLookupKey(), this)
        ?: persistentHashMapOf<Any, KArrayStoreBase<A, *>>().mutate {
            it[nestedInterpretedStore.createLookupKey()] = nestedInterpretedStore
            it[createLookupKey()] = this
        }
}

class KArrayStore<D : KSort, R : KSort> internal constructor(
    ctx: KContext,
    array: KExpr<KArraySort<D, R>>,
    val index: KExpr<D>,
    value: KExpr<R>
) : KArrayStoreBase<KArraySort<D, R>, R>(ctx, array, value) {
    override val indices: List<KExpr<KSort>>
        get() = listOf(index).uncheckedCast()

    override val decl: KDecl<KArraySort<D, R>>
        get() = ctx.mkArrayStoreDecl(array.sort)

    override fun accept(transformer: KTransformerBase): KExpr<KArraySort<D, R>> = transformer.transform(this)

    override fun internHashCode(): Int = hash(array, index, value)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other, { array }, { index }, { value })

    override fun indicesAreInterpreted(): Boolean = index is KInterpretedValue<D>

    override fun createLookupKey(): Any = index

    override fun findArrayToSelectFrom(indices: List<KExpr<*>>): KExpr<KArraySort<D, R>> =
        findArrayToSelectFrom(indices.single().asExpr(index.sort))

    fun findArrayToSelectFrom(
        index: KExpr<D>
    ): KExpr<KArraySort<D, R>> = lookupArrayToSelectFrom(
        indicesAreEqual = { index == this.index },
        selectIndicesAreInterpreted = { index is KInterpretedValue<D> },
        selectIndicesLookupKey = { index }
    )
}

class KArray2Store<D0 : KSort, D1 : KSort, R : KSort> internal constructor(
    ctx: KContext,
    array: KExpr<KArray2Sort<D0, D1, R>>,
    val index0: KExpr<D0>,
    val index1: KExpr<D1>,
    value: KExpr<R>
) : KArrayStoreBase<KArray2Sort<D0, D1, R>, R>(ctx, array, value) {
    override val indices: List<KExpr<KSort>>
        get() = listOf(index0, index1).uncheckedCast()

    override val decl: KDecl<KArray2Sort<D0, D1, R>>
        get() = ctx.mkArrayStoreDecl(array.sort)

    override fun accept(transformer: KTransformerBase): KExpr<KArray2Sort<D0, D1, R>> =
        transformer.transform(this)

    override fun internHashCode(): Int =
        hash(array, index0, index1, value)

    override fun internEquals(other: Any): Boolean =
        structurallyEqual(other, { array }, { index0 }, { index1 }, { value })

    override fun indicesAreInterpreted(): Boolean = areInterpreted(index0, index1)

    override fun createLookupKey(): Any = lookupKey(index0, index1)

    override fun findArrayToSelectFrom(indices: List<KExpr<*>>): KExpr<KArray2Sort<D0, D1, R>> {
        require(indices.size == KArray2Sort.DOMAIN_SIZE) {
            "Array domain size mismatch: expected ${KArray2Sort.DOMAIN_SIZE}, provided: ${indices.size}"
        }
        val (i0, i1) = indices
        return findArrayToSelectFrom(i0.asExpr(index0.sort), i1.asExpr(index1.sort))
    }

    fun findArrayToSelectFrom(
        index0: KExpr<D0>,
        index1: KExpr<D1>,
    ): KExpr<KArray2Sort<D0, D1, R>> = lookupArrayToSelectFrom(
        indicesAreEqual = { index0 == this.index0 && index1 == this.index1 },
        selectIndicesAreInterpreted = { areInterpreted(index0, index1) },
        selectIndicesLookupKey = { lookupKey(index0, index1) }
    )

    private fun areInterpreted(i0: KExpr<D0>, i1: KExpr<D1>): Boolean =
        i0 is KInterpretedValue<D0> && i1 is KInterpretedValue<D1>

    private fun lookupKey(i0: KExpr<D0>, i1: KExpr<D1>) = Pair(i0, i1)
}

class KArray3Store<D0 : KSort, D1 : KSort, D2 : KSort, R : KSort> internal constructor(
    ctx: KContext,
    array: KExpr<KArray3Sort<D0, D1, D2, R>>,
    val index0: KExpr<D0>,
    val index1: KExpr<D1>,
    val index2: KExpr<D2>,
    value: KExpr<R>
) : KArrayStoreBase<KArray3Sort<D0, D1, D2, R>, R>(ctx, array, value) {
    override val indices: List<KExpr<KSort>>
        get() = listOf(index0, index1, index2).uncheckedCast()

    override val decl: KDecl<KArray3Sort<D0, D1, D2, R>>
        get() = ctx.mkArrayStoreDecl(array.sort)

    override fun accept(transformer: KTransformerBase): KExpr<KArray3Sort<D0, D1, D2, R>> =
        transformer.transform(this)

    override fun internHashCode(): Int =
        hash(array, index0, index1, index2, value)

    override fun internEquals(other: Any): Boolean =
        structurallyEqual(other, { array }, { index0 }, { index1 }, { index2 }, { value })

    override fun indicesAreInterpreted(): Boolean = areInterpreted(index0, index1, index2)

    override fun createLookupKey(): Any = lookupKey(index0, index1, index2)

    override fun findArrayToSelectFrom(indices: List<KExpr<*>>): KExpr<KArray3Sort<D0, D1, D2, R>> {
        require(indices.size == KArray3Sort.DOMAIN_SIZE) {
            "Array domain size mismatch: expected ${KArray3Sort.DOMAIN_SIZE}, provided: ${indices.size}"
        }
        val (i0, i1, i2) = indices
        return findArrayToSelectFrom(i0.asExpr(index0.sort), i1.asExpr(index1.sort), i2.asExpr(index2.sort))
    }

    fun findArrayToSelectFrom(
        index0: KExpr<D0>,
        index1: KExpr<D1>,
        index2: KExpr<D2>
    ): KExpr<KArray3Sort<D0, D1, D2, R>> = lookupArrayToSelectFrom(
        indicesAreEqual = { index0 == this.index0 && index1 == this.index1 && index2 == this.index2 },
        selectIndicesAreInterpreted = { areInterpreted(index0, index1, index2) },
        selectIndicesLookupKey = { lookupKey(index0, index1, index2) }
    )

    private fun areInterpreted(i0: KExpr<D0>, i1: KExpr<D1>, i2: KExpr<D2>): Boolean =
        i0 is KInterpretedValue<D0> && i1 is KInterpretedValue<D1> && i2 is KInterpretedValue<D2>

    private fun lookupKey(i0: KExpr<D0>, i1: KExpr<D1>, i2: KExpr<D2>) = Triple(i0, i1, i2)
}

class KArrayNStore<R : KSort> internal constructor(
    ctx: KContext,
    array: KExpr<KArrayNSort<R>>,
    override val indices: List<KExpr<KSort>>,
    value: KExpr<R>
) : KArrayStoreBase<KArrayNSort<R>, R>(ctx, array, value) {
    init {
        require(indices.size == array.sort.domainSorts.size) {
            "Array domain size mismatch: expected ${array.sort.domainSorts.size}, provided: ${indices.size}"
        }
    }

    override val decl: KDecl<KArrayNSort<R>>
        get() = ctx.mkArrayNStoreDecl(array.sort)

    override fun accept(transformer: KTransformerBase): KExpr<KArrayNSort<R>> =
        transformer.transform(this)

    override fun internHashCode(): Int = hash(array, indices, value)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other, { array }, { indices }, { value })

    override fun indicesAreInterpreted(): Boolean = indices.all { it is KInterpretedValue<*> }

    override fun createLookupKey(): Any = indices

    override fun findArrayToSelectFrom(indices: List<KExpr<*>>): KExpr<KArrayNSort<R>> {
        require(indices.size == this.indices.size) {
            "Array domain size mismatch: expected ${this.indices.size}, provided: ${indices.size}"
        }
        return lookupArrayToSelectFrom(
            indicesAreEqual = { indices == this.indices },
            selectIndicesAreInterpreted = { indices.all { it is KInterpretedValue<*> } },
            selectIndicesLookupKey = { indices }
        )
    }
}

sealed class KArraySelectBase<A : KArraySortBase<R>, R : KSort>(
    ctx: KContext,
    val array: KExpr<A>
) : KApp<R, KSort>(ctx) {
    override val sort: R = array.sort.range

    abstract val indices: List<KExpr<KSort>>

    override val args: List<KExpr<KSort>>
        get() = buildList {
            add(array)
            addAll(this@KArraySelectBase.indices)
        }.uncheckedCast()
}

class KArraySelect<D : KSort, R : KSort> internal constructor(
    ctx: KContext,
    array: KExpr<KArraySort<D, R>>,
    val index: KExpr<D>
) : KArraySelectBase<KArraySort<D, R>, R>(ctx, array) {
    override val indices: List<KExpr<KSort>>
        get() = listOf(index).uncheckedCast()

    override val decl: KDecl<R>
        get() = ctx.mkArraySelectDecl(array.sort)

    override fun accept(transformer: KTransformerBase): KExpr<R> = transformer.transform(this)

    override fun internHashCode(): Int = hash(array, index)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other, { array }, { index })
}

class KArray2Select<D0 : KSort, D1 : KSort, R : KSort> internal constructor(
    ctx: KContext,
    array: KExpr<KArray2Sort<D0, D1, R>>,
    val index0: KExpr<D0>,
    val index1: KExpr<D1>
) : KArraySelectBase<KArray2Sort<D0, D1, R>, R>(ctx, array) {
    override val indices: List<KExpr<KSort>>
        get() = listOf(index0, index1).uncheckedCast()

    override val decl: KDecl<R>
        get() = ctx.mkArraySelectDecl(array.sort)

    override fun accept(transformer: KTransformerBase): KExpr<R> = transformer.transform(this)

    override fun internHashCode(): Int = hash(array, index0, index1)
    override fun internEquals(other: Any): Boolean =
        structurallyEqual(other, { array }, { index0 }, { index1 })
}

class KArray3Select<D0 : KSort, D1 : KSort, D2 : KSort, R : KSort> internal constructor(
    ctx: KContext,
    array: KExpr<KArray3Sort<D0, D1, D2, R>>,
    val index0: KExpr<D0>,
    val index1: KExpr<D1>,
    val index2: KExpr<D2>
) : KArraySelectBase<KArray3Sort<D0, D1, D2, R>, R>(ctx, array) {
    override val indices: List<KExpr<KSort>>
        get() = listOf(index0, index1, index2).uncheckedCast()

    override val decl: KDecl<R>
        get() = ctx.mkArraySelectDecl(array.sort)

    override fun accept(transformer: KTransformerBase): KExpr<R> = transformer.transform(this)

    override fun internHashCode(): Int = hash(array, index0, index1, index2)
    override fun internEquals(other: Any): Boolean =
        structurallyEqual(other, { array }, { index0 }, { index1 }, { index2 })
}

class KArrayNSelect<R : KSort> internal constructor(
    ctx: KContext,
    array: KExpr<KArrayNSort<R>>,
    override val indices: List<KExpr<KSort>>
) : KArraySelectBase<KArrayNSort<R>, R>(ctx, array) {
    init {
        require(indices.size == array.sort.domainSorts.size) {
            "Array domain size mismatch: expected ${array.sort.domainSorts.size}, provided: ${indices.size}"
        }
    }

    override val decl: KDecl<R>
        get() = ctx.mkArrayNSelectDecl(array.sort)

    override fun accept(transformer: KTransformerBase): KExpr<R> = transformer.transform(this)

    override fun internHashCode(): Int = hash(array, indices)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other, { array }, { indices })
}

class KArrayConst<A : KArraySortBase<R>, R : KSort> internal constructor(
    ctx: KContext,
    override val sort: A,
    val value: KExpr<R>
) : KApp<A, R>(ctx) {

    override val decl: KArrayConstDecl<A, R>
        get() = ctx.mkArrayConstDecl(sort)

    override val args: List<KExpr<R>>
        get() = listOf(value)

    override fun accept(transformer: KTransformerBase): KExpr<A> =
        transformer.transform(this)

    override fun internHashCode(): Int = hash(sort, value)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other, { sort }, { value })
}

class KFunctionAsArray<A : KArraySortBase<R>, R : KSort> internal constructor(
    ctx: KContext,
    override val sort: A,
    val function: KFuncDecl<R>
) : KExpr<A>(ctx) {

    init {
        check(function.argSorts == sort.domainSorts) {
            "Function arguments sort mismatch"
        }
    }

    override fun print(printer: ExpressionPrinter): Unit = with(printer) {
        append("(asArray ")
        append(function.name)
        append(")")
    }

    override fun accept(transformer: KTransformerBase): KExpr<A> =
        transformer.transform(this)

    override fun internHashCode(): Int = hash(function)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other) { function }
}

/**
 * Array lambda binding.
 *
 * [Defined as in the Z3 solver](https://microsoft.github.io/z3guide/docs/logic/Lambdas)
 * */
sealed class KArrayLambdaBase<A : KArraySortBase<R>, R : KSort>(
    ctx: KContext,
    override val sort: A,
    val body: KExpr<R>
) : KExpr<A>(ctx) {
    abstract val indexVarDeclarations: List<KDecl<*>>

    override fun print(printer: ExpressionPrinter) {
        val str = buildString {
            append("(lambda (")

            indexVarDeclarations.forEach {
                append('(')
                append(it.name)
                append(' ')
                it.sort.print(this)
                append(')')
            }

            append(") ")

            body.print(this)

            append(')')
        }
        printer.append(str)
    }
}

class KArrayLambda<D : KSort, R : KSort> internal constructor(
    ctx: KContext,
    val indexVarDecl: KDecl<D>,
    body: KExpr<R>
) : KArrayLambdaBase<KArraySort<D, R>, R>(
    ctx,
    ctx.mkArraySort(indexVarDecl.sort, body.sort),
    body
) {
    override val indexVarDeclarations: List<KDecl<*>>
        get() = listOf(indexVarDecl)

    override fun accept(transformer: KTransformerBase): KExpr<KArraySort<D, R>> = transformer.transform(this)

    override fun internHashCode(): Int = hash(indexVarDecl, body)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other, { indexVarDecl }, { body })
}

class KArray2Lambda<D0 : KSort, D1 : KSort, R : KSort> internal constructor(
    ctx: KContext,
    val indexVar0Decl: KDecl<D0>,
    val indexVar1Decl: KDecl<D1>,
    body: KExpr<R>
) : KArrayLambdaBase<KArray2Sort<D0, D1, R>, R>(
    ctx,
    ctx.mkArraySort(indexVar0Decl.sort, indexVar1Decl.sort, body.sort),
    body
) {
    override val indexVarDeclarations: List<KDecl<*>>
        get() = listOf(indexVar0Decl, indexVar1Decl)

    override fun accept(transformer: KTransformerBase): KExpr<KArray2Sort<D0, D1, R>> =
        transformer.transform(this)

    override fun internHashCode(): Int = hash(indexVar0Decl, indexVar1Decl, body)
    override fun internEquals(other: Any): Boolean =
        structurallyEqual(other, { indexVar0Decl }, { indexVar1Decl }, { body })
}

class KArray3Lambda<D0 : KSort, D1 : KSort, D2 : KSort, R : KSort> internal constructor(
    ctx: KContext,
    val indexVar0Decl: KDecl<D0>,
    val indexVar1Decl: KDecl<D1>,
    val indexVar2Decl: KDecl<D2>,
    body: KExpr<R>
) : KArrayLambdaBase<KArray3Sort<D0, D1, D2, R>, R>(
    ctx,
    ctx.mkArraySort(indexVar0Decl.sort, indexVar1Decl.sort, indexVar2Decl.sort, body.sort),
    body
) {
    override val indexVarDeclarations: List<KDecl<*>>
        get() = listOf(indexVar0Decl, indexVar1Decl, indexVar2Decl)

    override fun accept(transformer: KTransformerBase): KExpr<KArray3Sort<D0, D1, D2, R>> =
        transformer.transform(this)

    override fun internHashCode(): Int = hash(indexVar0Decl, indexVar1Decl, indexVar2Decl, body)
    override fun internEquals(other: Any): Boolean =
        structurallyEqual(other, { indexVar0Decl }, { indexVar1Decl }, { indexVar2Decl }, { body })
}

class KArrayNLambda<R : KSort> internal constructor(
    ctx: KContext,
    override val indexVarDeclarations: List<KDecl<*>>,
    body: KExpr<R>
) : KArrayLambdaBase<KArrayNSort<R>, R>(
    ctx,
    ctx.mkArrayNSort(indexVarDeclarations.map { it.sort }, body.sort),
    body
) {
    override fun accept(transformer: KTransformerBase): KExpr<KArrayNSort<R>> = transformer.transform(this)

    override fun internHashCode(): Int = hash(indexVarDeclarations, body)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other, { indexVarDeclarations }, { body })
}
