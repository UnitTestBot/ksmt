package io.ksmt.solver.z3

import com.microsoft.z3.Context
import com.microsoft.z3.Z3Exception
import com.microsoft.z3.decRefUnsafe
import io.ksmt.KAst
import io.ksmt.KContext
import io.ksmt.decl.KDecl
import io.ksmt.expr.KExpr
import io.ksmt.expr.KUninterpretedSortValue
import io.ksmt.solver.KForkingSolver
import io.ksmt.solver.KForkingSolverManager
import io.ksmt.solver.KSolverException
import io.ksmt.solver.util.KExprLongInternalizerBase
import io.ksmt.sort.KSort
import io.ksmt.sort.KUninterpretedSort
import it.unimi.dsi.fastutil.longs.Long2ObjectOpenHashMap
import it.unimi.dsi.fastutil.longs.LongOpenHashSet
import it.unimi.dsi.fastutil.longs.LongSet
import it.unimi.dsi.fastutil.objects.Object2LongOpenHashMap
import java.util.concurrent.ConcurrentHashMap

/**
 * Responsible for creation and managing of [KZ3ForkingSolver].
 *
 * It's cheaper to create multiple copies of solvers with [KZ3ForkingSolver.fork]
 * instead of assertions transferring in [KZ3Solver] instances.
 *
 * All created solvers with one manager (via both [KZ3ForkingSolver.fork] and [mkForkingSolver])
 * use the same [Context], cache, and registered uninterpreted sort values.
 */
class KZ3ForkingSolverManager(private val ctx: KContext) : KForkingSolverManager<KZ3SolverConfiguration> {
    private val z3Context by lazy { Context() }
    private val solvers = ConcurrentHashMap.newKeySet<KZ3ForkingSolver>()

    // shared cache
    private val expressionsCache = ExpressionsCache().withNotInternalizedAsDefaultValue()
    private val expressionsReversedCache = ExpressionsReversedCache()
    private val sortsCache = SortsCache().withNotInternalizedAsDefaultValue()
    private val sortsReversedCache = SortsReversedCache()
    private val declsCache = DeclsCache().withNotInternalizedAsDefaultValue()
    private val declsReversedCache = DeclsReversedCache()

    private val tmpNativeObjectsCache = TmpNativeObjectsCache()
    private val converterNativeObjectsCache = ConverterNativeObjectsCache()

    private val uninterpretedSortValueInterpreter = UninterpretedSortValueInterpreterCache()
    private val uninterpretedSortValueDecls = UninterpretedSortValueDecls()
    private val uninterpretedSortValueInterpreters = UninterpretedSortValueInterpretersCache()

    internal fun KZ3Context.getExpressionsCache() = ensureContextMatches(nativeContext).let { expressionsCache }
    internal fun KZ3Context.getExpressionsReversedCache() = ensureContextMatches(nativeContext)
        .let { expressionsReversedCache }

    internal fun KZ3Context.getSortsCache() = ensureContextMatches(nativeContext).let { sortsCache }
    internal fun KZ3Context.getSortsReversedCache() = ensureContextMatches(nativeContext).let { sortsReversedCache }
    internal fun KZ3Context.getDeclsCache() = ensureContextMatches(nativeContext).let { declsCache }
    internal fun KZ3Context.getDeclsReversedCache() = ensureContextMatches(nativeContext).let { declsReversedCache }
    internal fun KZ3Context.getTmpNativeObjectsCache() = ensureContextMatches(nativeContext)
        .let { tmpNativeObjectsCache }

    internal fun KZ3Context.getConverterNativeObjectsCache() = ensureContextMatches(nativeContext)
        .let { converterNativeObjectsCache }

    internal fun KZ3Context.getUninterpretedSortValueInterpreter() = ensureContextMatches(nativeContext)
        .let { uninterpretedSortValueInterpreter }

    internal fun KZ3Context.getUninterpretedSortValueDecls() = ensureContextMatches(nativeContext)
        .let { uninterpretedSortValueDecls }

    internal fun KZ3Context.getUninterpretedSortValueInterpreters() = ensureContextMatches(nativeContext)
        .let { uninterpretedSortValueInterpreters }

    override fun mkForkingSolver(): KForkingSolver<KZ3SolverConfiguration> {
        return KZ3ForkingSolver(ctx, this, null).also { solvers += it }
    }

    internal fun mkForkingSolver(parent: KZ3ForkingSolver): KForkingSolver<KZ3SolverConfiguration> {
        return KZ3ForkingSolver(ctx, this, parent).also { solvers += it }
    }

    internal fun createZ3ForkingContext(parentCtx: KZ3ForkingContext? = null) = parentCtx?.fork(ctx, this)
        ?: KZ3ForkingContext(ctx, z3Context, this)

    /**
     * unregister [solver] for this manager
     */
    internal fun close(solver: KZ3ForkingSolver) {
        solvers -= solver
        closeContextIfStale()
    }

    override fun close() {
        solvers.forEach(KZ3ForkingSolver::close)
    }

    private fun closeContextIfStale() {
        if (solvers.isEmpty()) {
            val nCtx = z3Context.nCtx()

            expressionsReversedCache.keys.decRefAll(nCtx)
            expressionsReversedCache.clear()
            expressionsCache.clear()

            sortsReversedCache.keys.decRefAll(nCtx)
            sortsReversedCache.clear()
            sortsCache.clear()

            declsReversedCache.keys.decRefAll(nCtx)
            declsReversedCache.clear()
            declsCache.clear()

            uninterpretedSortValueInterpreters.decRefAll(nCtx)
            uninterpretedSortValueInterpreters.clear()
            uninterpretedSortValueInterpreter.clear()
            uninterpretedSortValueDecls.clear()

            converterNativeObjectsCache.decRefAll(nCtx)
            converterNativeObjectsCache.clear()
            tmpNativeObjectsCache.decRefAll(nCtx)
            tmpNativeObjectsCache.clear()

            try {
                ctx.close()
            } catch (e: Z3Exception) {
                throw KSolverException(e)
            }
        }
    }

    private fun <T : KAst> Object2LongOpenHashMap<T>.withNotInternalizedAsDefaultValue() = apply {
        defaultReturnValue(KExprLongInternalizerBase.NOT_INTERNALIZED)
    }

    private fun LongSet.decRefAll(nCtx: Long) = longIterator().forEachRemaining {
        decRefUnsafe(nCtx, it)
    }

    private fun ensureContextMatches(ctx: Context) {
        require(ctx == z3Context) { "Context is not registered by manager." }
    }
}

private typealias ExpressionsCache = Object2LongOpenHashMap<KExpr<*>>
private typealias ExpressionsReversedCache = Long2ObjectOpenHashMap<KExpr<*>>

private typealias SortsCache = Object2LongOpenHashMap<KSort>
private typealias SortsReversedCache = Long2ObjectOpenHashMap<KSort>

private typealias DeclsCache = Object2LongOpenHashMap<KDecl<*>>
private typealias DeclsReversedCache = Long2ObjectOpenHashMap<KDecl<*>>

private typealias TmpNativeObjectsCache = LongOpenHashSet
private typealias ConverterNativeObjectsCache = LongOpenHashSet

private typealias UninterpretedSortValueInterpreterCache = HashMap<KUninterpretedSort, Long>
private typealias UninterpretedSortValueDecls = Long2ObjectOpenHashMap<KUninterpretedSortValue>
private typealias UninterpretedSortValueInterpretersCache = LongOpenHashSet
