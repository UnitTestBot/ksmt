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

    /**
     * Array store indices cache.
     *
     * Each node in array store chain contains a cache for only one index.
     * Cached index is specified by the [cacheKind] property.
     * Cached indices are rotated.
     * For example, in a case of 3 dimensional array nodes [cacheKind] will be 0, 1, 2, 0, 1, 2 ...
     *
     * If array store was not analyzed yet [cacheKind] is [NOT_INITIALIZED_CACHE_KIND].
     * */
    private var cacheKind: Int = NOT_INITIALIZED_CACHE_KIND

    /**
     * Table to perform fast index lookups.
     * Can be `null` in the following cases:
     * 1. Store was not analyzed.
     * 2. Store index is uninterpreted.
     * 3. Stores chain is small (see [LINEAR_LOOKUP_THRESHOLD]) and linear lookups are used.
     * */
    private var cacheLookupTable: PersistentMap<KExpr<*>, KArrayStoreBase<A, *>>? = null

    /**
     * Next array store node with uninterpreted index.
     * Can be `null` in the following cases:
     * 1. Store was not analyzed.
     * 2. Store index is uninterpreted.
     * */
    private var nextUninterpreted: KExpr<A>? = null

    private val arrayStoreChainSize: Int =
        if (array is KArrayStoreBase<*, *>) array.arrayStoreChainSize + 1 else 0

    abstract fun getNumIndices(): Int

    abstract fun getIndex(idx: Int): KExpr<*>

    /**
     * Find an array expression containing the value for the provided indices.
     *
     * Returns:
     * 1. Indices match the current store expression indices -> current store.
     * 2. We can't determine whether indices match or not -> current store.
     * 3. Indices definitely don't match current store indices
     * (e.g. both are interpreted values) -> apply this operation on a nested array.
     * */
    abstract fun findArrayToSelectFrom(indices: List<KExpr<*>>): KExpr<A>

    /**
     * Analyze store to provide faster index lookups when
     * store indices are interpreted values.
     * */
    @Suppress("ComplexMethod", "LoopWithTooManyJumpStatements")
    fun analyzeStore() {
        if (cacheKind != NOT_INITIALIZED_CACHE_KIND) return

        val numIndices = getNumIndices()
        val cacheEnabledNodesRate = (LINEAR_LOOKUP_THRESHOLD / numIndices).coerceAtLeast(1)

        // Rotate cached index
        cacheKind = if (array is KArrayStoreBase<*, *>) {
            val cacheKindRotation = numIndices * cacheEnabledNodesRate
            (array.cacheKind + 1) % cacheKindRotation
        } else {
            0 // We always start from 0 index
        }

        /**
         * To reduce memory usage we don't store cache in every node.
         * We guarantee that there is always a node with a cache within
         * at most [LINEAR_LOOKUP_THRESHOLD] nodes starting from the current one.
         * We ensure that if node has a cache for index i then it [cacheKind] is i.
         * */
        val cachedIndex = cacheKind % numIndices
        val nodeCacheEnabled = cachedIndex == cacheKind

        val index = getIndex(cachedIndex)

        /**
         * Index is not an interpreted value --> current array is uninterpreted.
         * */
        if (index !is KInterpretedValue<*>) return

        val parentStores = ArrayList<KArrayStoreBase<A, *>>(numIndices)
        var nestedArray = this.array

        /**
         * Search for the closest array store node that holds cache for [cacheKind].
         * */
        while (true) {

            /**
             * We found an uninterpreted array --> all stores are analyzed and we don't need to go deeper.
             * We consider not analyzed store as uninterpreted.
             * We don't initialize [cacheLookupTable] because array store chain is small
             * (less than number of array dimensions) and we can use linear lookup.
             * */
            if (nestedArray !is KArrayStoreBase<A, *> || nestedArray.cacheKind == NOT_INITIALIZED_CACHE_KIND) {
                nextUninterpreted = nestedArray
                return
            }

            val nestedIndex = nestedArray.getIndex(cachedIndex)

            /**
             * Array has an uninterpreted index and we can't use fast lookups.
             * We don't initialize [cacheLookupTable] because array store chain is small.
             * */
            if (nestedIndex !is KInterpretedValue<*>) {
                nextUninterpreted = nestedArray
                return
            }

            val kind = nestedArray.cacheKind
            val nestedArrayCachedIndex = kind % numIndices

            /**
             * Array index is interpreted but it doesn't contain required cache.
             * Continue to the next array in a chain.
             * */
            if (nestedArrayCachedIndex != cachedIndex) {
                parentStores.add(nestedArray)
                nestedArray = nestedArray.array
                continue
            }

            nextUninterpreted = nestedArray.nextUninterpreted
                ?: error("Interpreted array has no known next uninterpreted")

            /**
             * Whole store chain is interpreted, but we don't want to
             * store cache in a current node.
             * */
            if (!nodeCacheEnabled) return

            /**
             * We must find previous node with cache to
             * initialize cache in a current node.
             * Current [nestedArray] doesn't contain cache.
             * */
            if (kind != cachedIndex) {
                parentStores.add(nestedArray)
                nestedArray = nestedArray.array
                continue
            }

            // Find first array which is not in a current chain of arrays with interpreted indices.
            val nextUninterpretedArrayChainSize =
                (nextUninterpreted as? KArrayStoreBase<*, *>)?.arrayStoreChainSize ?: 0

            val interpretedArrayStoreChainSize = arrayStoreChainSize - nextUninterpretedArrayChainSize

            /**
             * We don't initialize map when interpreted stores chain is small enough.
             * On a small chains linear lookup will be faster and we can also save some memory.
             * */
            if (interpretedArrayStoreChainSize < LINEAR_LOOKUP_THRESHOLD) {
                return
            }

            cacheLookupTable = lookupTableAdd(
                nestedArray.cacheLookupTable, index, cachedIndex, nestedArray, parentStores
            )
            return
        }
    }

    fun searchForIndex(
        index: KExpr<*>,
        indexKind: Int,
    ): KExpr<A> {
        val currentIndex = getIndex(indexKind)

        // Current array contains a value for the selected index
        if (index == currentIndex) return this

        // Select index and store index are indistinguishable. We can't perform fast lookup.
        if (index !is KInterpretedValue<*> || currentIndex !is KInterpretedValue<*>) return this

        var lookupOwner: KArrayStoreBase<A, *> = this
        while (true) {
            /**
             * We found node that manages required cache.
             * Perform fast lookup.
             * */
            if (lookupOwner.cacheKind == indexKind) {
                return lookupOwner.lookupCachedIndex(index)
            }

            val lookupOwnerIndex = lookupOwner.getIndex(indexKind)

            /**
             * We can't proceed to the deeper node because store index is indistinguishable
             * from the select index.
             * */
            if (lookupOwnerIndex == index || lookupOwnerIndex !is KInterpretedValue<*>) return lookupOwner

            val nextArrayToLookup = lookupOwner.array

            /**
             * Nested array is uninterpreted and we can't go deeper.
             * We consider not analyzed store as uninterpreted.
             * */
            if (nextArrayToLookup !is KArrayStoreBase<A, *> || lookupOwner.cacheKind == NOT_INITIALIZED_CACHE_KIND) {
                return nextArrayToLookup
            }

            lookupOwner = nextArrayToLookup
        }
    }

    private fun lookupTableAdd(
        lookup: PersistentMap<KExpr<*>, KArrayStoreBase<A, *>>?,
        currentIndex: KExpr<*>,
        cachedIndexKind: Int,
        lookupOwnerStore: KArrayStoreBase<A, *>,
        parentInterpretedStores: List<KArrayStoreBase<A, *>>
    ): PersistentMap<KExpr<*>, KArrayStoreBase<A, *>> = if (lookup != null) {
        lookup.mutate { map ->
            for (store in parentInterpretedStores.asReversed()) {
                map[store.getIndex(cachedIndexKind)] = store
            }
            map[currentIndex] = this
        }
    } else {
        persistentHashMapOf<KExpr<*>, KArrayStoreBase<A, *>>().mutate { map ->
            map[currentIndex] = this
            for (store in parentInterpretedStores) {
                map.putIfAbsent(store.getIndex(cachedIndexKind), store)
            }

            var nestedArray: KExpr<A> = lookupOwnerStore
            while (nestedArray is KArrayStoreBase<A, *> && nestedArray != nextUninterpreted) {
                map.putIfAbsent(nestedArray.getIndex(cachedIndexKind), nestedArray)
                nestedArray = nestedArray.array
            }
        }
    }

    private fun lookupCachedIndex(index: KExpr<*>): KExpr<A> {
        /**
         * If [nextUninterpreted] is null then the current node is uninterpreted
         * and we must return it.
         * */
        val nextUninterpreted = this.nextUninterpreted ?: return this

        val lookupMap = cacheLookupTable
        return if (lookupMap != null) {
            lookupMap[index] ?: nextUninterpreted
        } else {
            var array: KExpr<A> = this
            while (array is KArrayStoreBase<A, *> && array != nextUninterpreted) {
                val storeIndex = array.getIndex(cacheKind)
                if (storeIndex == index) return array
                array = array.array
            }
            nextUninterpreted
        }
    }

    companion object {
        // On a small store chains linear lookup is faster than map lookup.
        private const val LINEAR_LOOKUP_THRESHOLD = 12

        private const val NOT_INITIALIZED_CACHE_KIND = -1

        /**
         * Select a single result array after performing lookup by multiple indices.
         * */
        internal inline fun <reified S : KArrayStoreBase<A, *>, A : KArraySortBase<*>> S.selectLookupResult(
            first: (S) -> KExpr<A>,
            second: (S) -> KExpr<A>
        ): KExpr<A> {
            val firstArray = first(this)
            // Array is not store expression --> we found the deepest array.
            if (firstArray !is S) return firstArray

            return second(firstArray)
        }
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

    override fun getIndex(idx: Int): KExpr<*> = index
    override fun getNumIndices(): Int = KArraySort.DOMAIN_SIZE

    override fun findArrayToSelectFrom(indices: List<KExpr<*>>): KExpr<KArraySort<D, R>> =
        findArrayToSelectFrom(indices.single().asExpr(index.sort))

    fun findArrayToSelectFrom(
        index: KExpr<D>
    ): KExpr<KArraySort<D, R>> = searchForIndex(index, indexKind = 0)
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

    override fun getIndex(idx: Int): KExpr<*> = if (idx == 0) index0 else index1
    override fun getNumIndices(): Int = KArray2Sort.DOMAIN_SIZE

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
    ): KExpr<KArray2Sort<D0, D1, R>> = selectLookupResult({ it.lookupIndex0(index0) }, { it.lookupIndex1(index1) })

    private fun lookupIndex0(index0: KExpr<D0>): KExpr<KArray2Sort<D0, D1, R>> =
        searchForIndex(index0, indexKind = 0)

    private fun lookupIndex1(index1: KExpr<D1>): KExpr<KArray2Sort<D0, D1, R>> =
        searchForIndex(index1, indexKind = 1)
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

    override fun getIndex(idx: Int): KExpr<*> = when (idx) {
        0 -> index0
        1 -> index1
        else -> index2
    }

    override fun getNumIndices(): Int = KArray3Sort.DOMAIN_SIZE

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
    ): KExpr<KArray3Sort<D0, D1, D2, R>> = selectLookupResult(
        { it.lookupIndex0(index0) },
        { array ->
            array.selectLookupResult(
                { it.lookupIndex1(index1) },
                { it.lookupIndex2(index2) },
            )
        }
    )

    private fun lookupIndex0(index0: KExpr<D0>): KExpr<KArray3Sort<D0, D1, D2, R>> =
        searchForIndex(index0, indexKind = 0)

    private fun lookupIndex1(index1: KExpr<D1>): KExpr<KArray3Sort<D0, D1, D2, R>> =
        searchForIndex(index1, indexKind = 1)

    private fun lookupIndex2(index2: KExpr<D2>): KExpr<KArray3Sort<D0, D1, D2, R>> =
        searchForIndex(index2, indexKind = 2)
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

    override fun getIndex(idx: Int): KExpr<*> = indices[idx]
    override fun getNumIndices(): Int = indices.size

    override fun findArrayToSelectFrom(indices: List<KExpr<*>>): KExpr<KArrayNSort<R>> {
        require(indices.size == this.indices.size) {
            "Array domain size mismatch: expected ${this.indices.size}, provided: ${indices.size}"
        }

        return indices.foldIndexed(this) { idx, array, index ->
            val next = array.lookupIndexI(index, idx)
            if (next !is KArrayNStore<R>) return next
            next
        }
    }

    private fun lookupIndexI(index: KExpr<*>, idx: Int): KExpr<KArrayNSort<R>> =
        searchForIndex(index, indexKind = idx)
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
