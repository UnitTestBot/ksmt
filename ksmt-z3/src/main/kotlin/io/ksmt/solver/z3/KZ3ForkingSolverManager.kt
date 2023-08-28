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
import java.util.IdentityHashMap
import java.util.concurrent.ConcurrentHashMap

class KZ3ForkingSolverManager(private val ctx: KContext) : KForkingSolverManager<KZ3SolverConfiguration> {
    private val solvers = ConcurrentHashMap.newKeySet<KZ3ForkingSolver>()

    /**
     * for each parent-to-child hierarchy created only one Context.
     * Each Context user is registered to control solver is alive
     */
    private val forkingSolverToContext = IdentityHashMap<KZ3ForkingSolver, Context>()
    private val contextReferences = IdentityHashMap<Context, Int>()

    // shared cache
    private val expressionsCache = IdentityHashMap<Context, ExpressionsCache>()
    private val expressionsReversedCache = IdentityHashMap<Context, ExpressionsReversedCache>()
    private val sortsCache = IdentityHashMap<Context, SortsCache>()
    private val sortsReversedCache = IdentityHashMap<Context, SortsReversedCache>()
    private val declsCache = IdentityHashMap<Context, DeclsCache>()
    private val declsReversedCache = IdentityHashMap<Context, DeclsReversedCache>()

    private val tmpNativeObjectsCache = IdentityHashMap<Context, TmpNativeObjectsCache>()
    private val converterNativeObjectsCache = IdentityHashMap<Context, ConverterNativeObjectsCache>()

    private val uninterpretedSortValueInterpreter = IdentityHashMap<Context, UninterpretedSortValueInterpreterCache>()
    private val uninterpretedSortValueDecls = IdentityHashMap<Context, UninterpretedSortValueDecls>()
    private val uninterpretedSortValueInterpreters = IdentityHashMap<Context, UninterpretedSortValueInterpretersCache>()
    private val registeredUninterpretedSortValues = IdentityHashMap<Context, RegisteredUninterpretedSortValues>()

    internal fun KZ3Context.findExpressionsCache() = expressionsCache.getValue(nativeContext)
    internal fun KZ3Context.findExpressionsReversedCache() = expressionsReversedCache.getValue(nativeContext)
    internal fun KZ3Context.findSortsCache() = sortsCache.getValue(nativeContext)
    internal fun KZ3Context.findSortsReversedCache() = sortsReversedCache.getValue(nativeContext)
    internal fun KZ3Context.findDeclsCache() = declsCache.getValue(nativeContext)
    internal fun KZ3Context.findDeclsReversedCache() = declsReversedCache.getValue(nativeContext)
    internal fun KZ3Context.findTmpNativeObjectsCache() = tmpNativeObjectsCache.getValue(nativeContext)
    internal fun KZ3Context.findConverterNativeObjectsCache() = converterNativeObjectsCache.getValue(nativeContext)
    internal fun KZ3Context.findUninterpretedSortValueInterpreter() =
        uninterpretedSortValueInterpreter.getValue(nativeContext)

    internal fun KZ3Context.findUninterpretedSortValueDecls() =
        uninterpretedSortValueDecls.getValue(nativeContext)

    internal fun KZ3Context.findUninterpretedSortValueInterpreters() =
        uninterpretedSortValueInterpreters.getValue(nativeContext)

    internal fun KZ3Context.findRegisteredUninterpretedSortValues() =
        registeredUninterpretedSortValues.getValue(nativeContext)

    internal fun KZ3ForkingSolver.registerContext(sharedContext: Context) {
        if (forkingSolverToContext.putIfAbsent(this, sharedContext) == null) {
            incRef(sharedContext)

            expressionsCache[sharedContext] = ExpressionsCache().withNotInternalizedAsDefaultValue()
            expressionsReversedCache[sharedContext] = ExpressionsReversedCache()
            sortsCache[sharedContext] = SortsCache().withNotInternalizedAsDefaultValue()
            sortsReversedCache[sharedContext] = SortsReversedCache()
            declsCache[sharedContext] = DeclsCache().withNotInternalizedAsDefaultValue()
            declsReversedCache[sharedContext] = DeclsReversedCache()
            tmpNativeObjectsCache[sharedContext] = TmpNativeObjectsCache()
            converterNativeObjectsCache[sharedContext] = ConverterNativeObjectsCache()
            uninterpretedSortValueInterpreter[sharedContext] = UninterpretedSortValueInterpreterCache()
            uninterpretedSortValueDecls[sharedContext] = UninterpretedSortValueDecls()
            uninterpretedSortValueInterpreters[sharedContext] = UninterpretedSortValueInterpretersCache()
            registeredUninterpretedSortValues[sharedContext] = RegisteredUninterpretedSortValues()
        }
    }

    private fun incRef(context: Context) {
        contextReferences[context] = contextReferences.getOrDefault(context, 0) + 1
    }

    private fun decRef(context: Context) {
        val referencesAfterDec = contextReferences.getValue(context) - 1
        if (referencesAfterDec == 0) {
            val nCtx = context.nCtx()
            contextReferences -= context

            expressionsReversedCache.remove(context)!!.keys.decRefAll(nCtx)
            expressionsCache -= context

            sortsReversedCache.remove(context)!!.keys.decRefAll(nCtx)
            sortsCache -= context

            declsReversedCache.remove(context)!!.keys.decRefAll(nCtx)
            declsCache -= context

            uninterpretedSortValueInterpreters.remove(context)!!.decRefAll(nCtx)
            uninterpretedSortValueInterpreter -= context
            uninterpretedSortValueDecls -= context
            registeredUninterpretedSortValues -= context

            converterNativeObjectsCache.remove(context)!!.decRefAll(nCtx)
            tmpNativeObjectsCache.remove(context)!!.decRefAll(nCtx)

            try {
                ctx.close()
            } catch (e: Z3Exception) {
                throw KSolverException(e)
            }
        } else {
            contextReferences[context] = referencesAfterDec
        }
    }

    override fun mkForkingSolver(): KForkingSolver<KZ3SolverConfiguration> {
        return KZ3ForkingSolver(ctx, this, null).also { solvers += it }
    }

    internal fun mkForkingSolver(parent: KZ3ForkingSolver): KForkingSolver<KZ3SolverConfiguration> {
        return KZ3ForkingSolver(ctx, this, parent).also {
            solvers += it
            forkingSolverToContext[it] = forkingSolverToContext[parent]
        }
    }

    /**
     * unregister [solver] for this manager
     */
    internal fun close(solver: KZ3ForkingSolver) {
        solvers -= solver
        val sharedContext = forkingSolverToContext.getValue(solver)
        forkingSolverToContext -= solver
        decRef(sharedContext)
    }

    override fun close() {
        solvers.forEach(KZ3ForkingSolver::close)
    }

    private fun <T : KAst> Object2LongOpenHashMap<T>.withNotInternalizedAsDefaultValue() = apply {
        defaultReturnValue(KExprLongInternalizerBase.NOT_INTERNALIZED)
    }

    private fun LongSet.decRefAll(nCtx: Long) =
        longIterator().forEachRemaining {
            decRefUnsafe(nCtx, it)
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
@Suppress("MaxLineLength")
private typealias RegisteredUninterpretedSortValues = HashMap<KUninterpretedSortValue, ExpressionUninterpretedValuesTracker.UninterpretedSortValueDescriptor>
