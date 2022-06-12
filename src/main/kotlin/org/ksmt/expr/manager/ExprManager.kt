package org.ksmt.expr.manager

import org.ksmt.expr.KExpr
import java.lang.ref.WeakReference
import java.util.*

object ExprManager {
    private class ExprWrapper(val expr: WeakReference<KExpr<*>>) {
        override fun hashCode(): Int = expr.get()?.hashCode() ?: 0
        override fun equals(other: Any?): Boolean {
            if (this === other) return true
            other as ExprWrapper
            val rhs = other.expr.get() ?: return false
            val lhs = expr.get() ?: return false
            return lhs.equalTo(rhs)
        }
    }

    private val cache = WeakHashMap<ExprWrapper, WeakReference<KExpr<*>>>()
    private val wrapperKeeper = WeakHashMap<KExpr<*>, ExprWrapper>()

    @Suppress("UNCHECKED_CAST")
    fun <T : KExpr<*>> T.intern(): T {
        val nodeRef = WeakReference(this)
        val key = ExprWrapper(nodeRef as WeakReference<KExpr<*>>)
        val currentNode = cache[key]?.get()
        if (currentNode != null) return currentNode as T
        wrapperKeeper[this] = key
        cache[key] = nodeRef
        return this
    }
}
