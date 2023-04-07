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

// [PersistentMap<KExpr<I>, S> | S] where S : KArrayStoreBase<*>
internal typealias ArrayStoreIndexLookup = Any

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

    val arrayStoreDepth: Int = if (array is KArrayStoreBase<*, *>) array.arrayStoreDepth + 1 else 0

    internal var arrayStoreCacheInitialized: Boolean = false

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
    abstract fun analyzeStore()

    companion object {
        // On a small store chains linear lookup is faster than map lookup.
        const val LINEAR_LOOKUP_THRESHOLD = 7

        inline fun <reified S : KArrayStoreBase<A, *>, I : KSort, A : KArraySortBase<*>> S.initializeLookupTable(
            nested: KExpr<A>,
            getIndex: (S) -> KExpr<I>,
            getLookup: (S) -> ArrayStoreIndexLookup?,
            getNextUninterpreted: (S) -> KExpr<A>?
        ): ArrayStoreIndexLookup? {
            val index = getIndex(this)

            /**
             * Index is not an interpreted value --> we can't perform fast lookups
             * because we can't skip current store expression.
             * */
            if (index !is KInterpretedValue<I>) return null

            // Nested array in not an array store expression --> we can't perform fast lookups
            if (nested !is S) return null

            val nestedLookup = getLookup(nested)
            if (nestedLookup == null) {
                val nestedIdx = getIndex(nested)

                /**
                 * Nested array has uninterpreted index. Therefore, current array
                 * is a first array with interpreted index.
                 *
                 * Note: we can't skip nested array in the indices lookup chain because it
                 * contains uninterpreted indices.
                 * */
                if (nestedIdx !is KInterpretedValue<I>) return null

                /**
                 * Nested array is the first array with interpreted index in a chain.
                 * */
                return nested
            }

            // Find first array which is not in a current chain of arrays with interpreted indices.
            val nestedNextUninterpreted = getNextUninterpreted(nested) as? S
            val arrayStoreChainStartDepth = nestedNextUninterpreted?.arrayStoreDepth ?: 0

            /**
             * We don't initialize map when interpreted stores chain is small enough.
             * On a small chains linear lookup will be faster and we can also save some memory.
             * */
            if (arrayStoreDepth - arrayStoreChainStartDepth < LINEAR_LOOKUP_THRESHOLD) {
                return nested
            }

            /**
             * We have a deep chain of interpreted stores.
             * We use persistent map to provide faster index lookups.
             * */
            return lookupTableAdd(nestedLookup, this, nested, index, getIndex, getLookup)
        }

        inline fun <reified S : KArrayStoreBase<A, *>, I : KSort, A : KArraySortBase<*>> S.initializeNextUninterpreted(
            nested: KExpr<A>,
            getIndex: (S) -> KExpr<I>,
            getNextUninterpreted: (S) -> KExpr<A>?
        ): KExpr<A>? {
            val index = getIndex(this)

            /**
             * Index is not an interpreted value --> current array is uninterpreted.
             * */
            if (index !is KInterpretedValue<I>) return null

            /**
             * Nested array is not a store expression.
             * Therefore, nested array is definitely uninterpreted.
             * */
            if (nested !is S) return nested

            val nestedNextUninterpreted = getNextUninterpreted(nested)

            /**
             * [nestedNextUninterpreted] is null only if nested array is uninterpreted.
             * */
            return nestedNextUninterpreted ?: nested
        }

        inline fun <reified S : KArrayStoreBase<A, *>, I : KSort, A : KArraySortBase<*>> S.lookupTableSearch(
            index: KExpr<I>,
            getIndex: (S) -> KExpr<I>,
            getLookup: (S) -> ArrayStoreIndexLookup?,
            getNextUninterpreted: (S) -> KExpr<A>?
        ): KExpr<A> {
            val currentIndex = getIndex(this)

            // Current array contains a value for the selected index
            if (index == currentIndex) return this

            // Select index and store index are indistinguishable. We can't perform fast lookup.
            if (index !is KInterpretedValue<I> || currentIndex !is KInterpretedValue<I>) return this

            val arrayToSelectFrom = getLookup(this)?.let {
                lookupTableSearch(it, index, getIndex, getLookup)
            }

            /**
             * Index is interpreted but not in the lookup table.
             * We can skip up to the next uninterpreted array.
             * */
            return arrayToSelectFrom ?: getNextUninterpreted(this) ?: this
        }

        @Suppress("LongParameterList")
        inline fun <reified S : KArrayStoreBase<*, *>, I : KSort> lookupTableAdd(
            lookup: ArrayStoreIndexLookup,
            current: S,
            nested: S,
            currentIndex: KExpr<I>,
            getIndex: (S) -> KExpr<I>,
            getLookup: (S) -> ArrayStoreIndexLookup?
        ): ArrayStoreIndexLookup = if (lookup is PersistentMap<*, *>) {
            @Suppress("UNCHECKED_CAST")
            lookup as PersistentMap<KExpr<I>, S>

            lookup.put(currentIndex, current)
        } else {
            // Create map based lookup from linear lookup
            persistentHashMapOf<KExpr<I>, S>().mutate { map ->
                map[currentIndex] = current
                var nestedLookup: S? = nested
                while (nestedLookup != null) {
                    map.putIfAbsent(getIndex(nestedLookup), nestedLookup)
                    nestedLookup = getLookup(nestedLookup) as? S
                }
            }
        }

        inline fun <reified S : KArrayStoreBase<A, *>, I : KSort, A : KArraySortBase<*>> lookupTableSearch(
            lookup: ArrayStoreIndexLookup,
            index: KExpr<I>,
            getIndex: (S) -> KExpr<I>,
            getLookup: (S) -> ArrayStoreIndexLookup?
        ): KExpr<A>? {
            if (lookup is PersistentMap<*, *>) {
                @Suppress("UNCHECKED_CAST")
                lookup as PersistentMap<KExpr<I>, S>

                return lookup[index]
            } else {
                // Linear lookup. We will check at most [LINEAR_LOOKUP_THRESHOLD] stores.
                var selectFrom: S? = lookup as S
                while (selectFrom != null) {
                    val elementIndex = getIndex(selectFrom)
                    if (index == elementIndex) return selectFrom
                    selectFrom = getLookup(selectFrom) as S?
                }

                return null
            }
        }

        /**
         * Select a single result array after performing lookup by multiple indices.
         * */
        inline fun <reified S : KArrayStoreBase<A, *>, A : KArraySortBase<*>> selectLookupResult(
            first: () -> KExpr<A>,
            second: () -> KExpr<A>
        ): KExpr<A> {
            val firstArray = first()
            // Array is not store expression --> we found the deepest array.
            if (firstArray !is S) return firstArray

            val secondArray = second()
            if (secondArray !is S) return secondArray

            // Prefer array that is deeper in a chain of stores.
            return if (firstArray.arrayStoreDepth <= secondArray.arrayStoreDepth) firstArray else secondArray
        }

        inline fun <reified S : KArrayStoreBase<A, *>, A : KArraySortBase<*>> selectLookupResult(
            size: Int,
            lookup: (Int) -> KExpr<A>
        ): KExpr<A> {
            val arrays = List(size) {
                val array = lookup(it)
                if (array !is S) return array
                array
            }

            return arrays.minBy { it.arrayStoreDepth }
        }

        internal inline fun <reified S : KArrayStoreBase<*, *>> S.ifCacheIsNotInitialized(body: () -> Unit) {
            if (arrayStoreCacheInitialized) return
            body()
            arrayStoreCacheInitialized = true
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

    private var indexLookupTable: ArrayStoreIndexLookup? = null
    private var nextUninterpretedArray: KExpr<KArraySort<D, R>>? = null

    override fun analyzeStore() = ifCacheIsNotInitialized {
        indexLookupTable = initializeLookupTable(
            array, { it.index }, { it.indexLookupTable }, { it.nextUninterpretedArray }
        )
        nextUninterpretedArray = initializeNextUninterpreted(array, { it.index }, { it.nextUninterpretedArray })
    }

    override fun findArrayToSelectFrom(indices: List<KExpr<*>>): KExpr<KArraySort<D, R>> =
        findArrayToSelectFrom(indices.single().asExpr(index.sort))

    fun findArrayToSelectFrom(
        index: KExpr<D>
    ): KExpr<KArraySort<D, R>> = lookupTableSearch(
        index, { it.index }, { it.indexLookupTable }, { it.nextUninterpretedArray }
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

    private var index0LookupTable: ArrayStoreIndexLookup? = null
    private var index1LookupTable: ArrayStoreIndexLookup? = null
    private var index0NextUninterpretedArray: KExpr<KArray2Sort<D0, D1, R>>? = null
    private var index1NextUninterpretedArray: KExpr<KArray2Sort<D0, D1, R>>? = null

    override fun analyzeStore() = ifCacheIsNotInitialized {
        index0LookupTable = initializeLookupTable(
            array, { it.index0 }, { it.index0LookupTable }, { it.index0NextUninterpretedArray }
        )
        index1LookupTable = initializeLookupTable(
            array, { it.index1 }, { it.index1LookupTable }, { it.index1NextUninterpretedArray }
        )

        index0NextUninterpretedArray = initializeNextUninterpreted(
            array, { it.index0 }, { it.index0NextUninterpretedArray }
        )
        index1NextUninterpretedArray = initializeNextUninterpreted(
            array, { it.index1 }, { it.index1NextUninterpretedArray }
        )
    }

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
    ): KExpr<KArray2Sort<D0, D1, R>> = selectLookupResult(
        {
            lookupTableSearch(
                index0, { it.index0 }, { it.index0LookupTable }, { it.index0NextUninterpretedArray }
            )
        },
        {
            lookupTableSearch(
                index1, { it.index1 }, { it.index1LookupTable }, { it.index1NextUninterpretedArray }
            )
        }
    )
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

    private var index0LookupTable: ArrayStoreIndexLookup? = null
    private var index1LookupTable: ArrayStoreIndexLookup? = null
    private var index2LookupTable: ArrayStoreIndexLookup? = null
    private var index0NextUninterpretedArray: KExpr<KArray3Sort<D0, D1, D2, R>>? = null
    private var index1NextUninterpretedArray: KExpr<KArray3Sort<D0, D1, D2, R>>? = null
    private var index2NextUninterpretedArray: KExpr<KArray3Sort<D0, D1, D2, R>>? = null

    override fun analyzeStore() = ifCacheIsNotInitialized {
        index0LookupTable = initializeLookupTable(
            array, { it.index0 }, { it.index0LookupTable }, { it.index0NextUninterpretedArray }
        )
        index1LookupTable = initializeLookupTable(
            array, { it.index1 }, { it.index1LookupTable }, { it.index1NextUninterpretedArray }
        )
        index2LookupTable = initializeLookupTable(
            array, { it.index2 }, { it.index2LookupTable }, { it.index2NextUninterpretedArray }
        )

        index0NextUninterpretedArray = initializeNextUninterpreted(
            array, { it.index0 }, { it.index0NextUninterpretedArray }
        )
        index1NextUninterpretedArray = initializeNextUninterpreted(
            array, { it.index1 }, { it.index1NextUninterpretedArray }
        )
        index2NextUninterpretedArray = initializeNextUninterpreted(
            array, { it.index2 }, { it.index2NextUninterpretedArray }
        )
    }

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
        {
            lookupTableSearch(
                index0, { it.index0 }, { it.index0LookupTable }, { it.index0NextUninterpretedArray }
            )
        },
        {
            selectLookupResult(
                {
                    lookupTableSearch(
                        index1, { it.index1 }, { it.index1LookupTable }, { it.index1NextUninterpretedArray }
                    )
                },
                {
                    lookupTableSearch(
                        index2, { it.index2 }, { it.index2LookupTable }, { it.index2NextUninterpretedArray }
                    )
                }
            )
        }
    )
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

    private var indexLookupTables: Array<ArrayStoreIndexLookup?>? = null
    private var nextUninterpretedArrays: Array<KExpr<KArrayNSort<R>>?>? = null

    override fun analyzeStore() = ifCacheIsNotInitialized {
        indexLookupTables = Array(indices.size) { idx ->
            initializeLookupTable(
                array,
                { it.indices[idx] },
                { it.indexLookupTables?.get(idx) },
                { it.nextUninterpretedArrays?.get(idx) }
            )
        }

        nextUninterpretedArrays = Array(indices.size) { idx ->
            initializeNextUninterpreted(array, { it.indices[idx] }, { it.nextUninterpretedArrays?.get(idx) })
        }
    }

    override fun findArrayToSelectFrom(indices: List<KExpr<*>>): KExpr<KArrayNSort<R>> {
        require(indices.size == this.indices.size) {
            "Array domain size mismatch: expected ${this.indices.size}, provided: ${indices.size}"
        }

        return selectLookupResult(indices.size) { idx ->
            lookupTableSearch(
                indices[idx].uncheckedCast(),
                { it.indices[idx] },
                { it.indexLookupTables?.get(idx) },
                { it.nextUninterpretedArrays?.get(idx) }
            )
        }
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
