package io.ksmt.utils

import io.ksmt.expr.KApp
import io.ksmt.expr.KExpr
import io.ksmt.sort.KSort
import kotlin.reflect.KProperty

fun <T : KSort> T.mkConst(name: String): KApp<T, *> = ctx.mkConst(name, this)

fun <T : KSort> T.mkFreshConst(name: String): KApp<T, *> = ctx.mkFreshConst(name, this)

inline operator fun <reified T : KSort> T.getValue(
    thisRef: Any?,
    property: KProperty<*>
): KApp<T, *> = mkConst(property.name)

fun <T : KSort> T.mkFreshConstDecl(name: String) = ctx.mkFreshConstDecl(name, this)

fun <T : KSort> T.mkConstDecl(name: String) = ctx.mkConstDecl(name, this)

fun <T : KSort> KExpr<*>.asExpr(sort: T): KExpr<T> = with(ctx) {
    check(this@asExpr.sort == sort) { "Sort mismatch" }

    @Suppress("UNCHECKED_CAST")
    this@asExpr as KExpr<T>
}

fun <T : KSort> T.sampleValue(): KExpr<T> =
    accept(ctx.defaultValueSampler).asExpr(this)
