package org.ksmt.decl

import org.ksmt.KContext
import org.ksmt.cache.CustomObjectEquality
import org.ksmt.cache.hash
import org.ksmt.cache.structurallyEqual
import org.ksmt.expr.KApp
import org.ksmt.expr.KExpr
import org.ksmt.sort.KSort

class KUninterpretedFuncDecl<T : KSort> internal constructor(
    ctx: KContext,
    name: String,
    resultSort: T,
    argSorts: List<KSort>,
) : KFuncDecl<T>(ctx, name, resultSort, argSorts), CustomObjectEquality {

    //  Contexts guarantee that any two equivalent declarations will be the same kotlin object
    override fun hashCode(): Int = System.identityHashCode(this)
    override fun equals(other: Any?): Boolean = this === other

    override fun apply(args: List<KExpr<*>>): KApp<T, *> = with(ctx) {
        checkArgSorts(args)
        return mkFunctionApp(this@KUninterpretedFuncDecl, args)
    }

    override fun customHashCode(): Int = hash(name, sort, argSorts)

    override fun customEquals(other: Any): Boolean = structurallyEqual(other, { name }, { sort }, { argSorts })
}

class KUninterpretedConstDecl<T : KSort> internal constructor(
    ctx: KContext,
    name: String,
    sort: T
) : KConstDecl<T>(ctx, name, sort), CustomObjectEquality {
    //  Contexts guarantee that any two equivalent declarations will be the same kotlin object
    override fun hashCode(): Int = System.identityHashCode(this)
    override fun equals(other: Any?): Boolean = this === other

    override fun apply(args: List<KExpr<*>>): KApp<T, *> {
        require(args.isEmpty()) { "Constant must have no arguments" }

        return ctx.mkConstApp(this@KUninterpretedConstDecl)
    }

    override fun customHashCode(): Int = hash(name, sort)

    override fun customEquals(other: Any): Boolean = structurallyEqual(other, { name }, { sort })
}
