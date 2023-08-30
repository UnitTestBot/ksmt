package io.ksmt.solver.z3

import com.microsoft.z3.Context
import io.ksmt.KContext

class KZ3ForkingContext private constructor(
    ksmtCtx: KContext,
    private val ctx: Context,
    manager: KZ3ForkingSolverManager,
    parent: KZ3ForkingContext?,
) : KZ3Context(ksmtCtx, ctx) {

    constructor(
        ksmtCtx: KContext,
        ctx: Context,
        manager: KZ3ForkingSolverManager
    ) : this(ksmtCtx, ctx, manager, null)

    constructor(ksmtCtx: KContext, manager: KZ3ForkingSolverManager) : this(ksmtCtx, Context(), manager)

    // common for parent and child structures
    override val expressions = with(manager) { getExpressionsCache() }
    override val sorts = with(manager) { getSortsCache() }
    override val decls = with(manager) { getDeclsCache() }

    override val z3Expressions = with(manager) { getExpressionsReversedCache() }
    override val z3Sorts = with(manager) { getSortsReversedCache() }
    override val z3Decls = with(manager) { getDeclsReversedCache() }
    override val tmpNativeObjects = with(manager) { getTmpNativeObjectsCache() }
    override val converterNativeObjects = with(manager) { getConverterNativeObjectsCache() }

    override val uninterpretedSortValueInterpreter = with(manager) { getUninterpretedSortValueInterpreter() }
    override val uninterpretedSortValueDecls = with(manager) { getUninterpretedSortValueDecls() }
    override val uninterpretedSortValueInterpreters = with(manager) { getUninterpretedSortValueInterpreters() }

    override val uninterpretedValuesTracker: ExpressionUninterpretedValuesForkingTracker = parent
        ?.uninterpretedValuesTracker?.fork(this)
        ?: ExpressionUninterpretedValuesForkingTracker(ksmtCtx, this)

    internal fun fork(ksmtCtx: KContext, manager: KZ3ForkingSolverManager): KZ3ForkingContext {
        ensureActive()
        return KZ3ForkingContext(ksmtCtx, ctx, manager, this)
    }

    override fun close() {
        if (isClosed) return
        isClosed = true
    }
}
