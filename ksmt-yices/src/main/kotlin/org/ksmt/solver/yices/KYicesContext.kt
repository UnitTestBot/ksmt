package org.ksmt.solver.yices

import com.sri.yices.Yices
import org.ksmt.decl.KDecl
import org.ksmt.expr.KExpr
import org.ksmt.sort.KSort
import org.ksmt.utils.NativeLibraryLoader
import org.ksmt.utils.uncheckedCast

open class KYicesContext : AutoCloseable {
    private var isClosed = false

    protected val expressions = HashMap<KExpr<*>, YicesTerm>()
    private val yicesExpressions = HashMap<YicesTerm, KExpr<*>>()
    protected val sorts = HashMap<KSort, YicesSort>()
    private val yicesSorts = HashMap<YicesSort, KSort>()
    protected val decls = HashMap<KDecl<*>, YicesTerm>()
    private val yicesDecls = HashMap<YicesTerm, KDecl<*>>()
    private val transformed = HashMap<KExpr<*>, KExpr<*>>()

    val isActive: Boolean
        get() = !isClosed

    fun findInternalizedExpr(expr: KExpr<*>): YicesTerm? = expressions[expr]

    fun findConvertedExpr(expr: YicesTerm): KExpr<*>? = yicesExpressions[expr]

    fun findConvertedDecl(decl: YicesTerm): KDecl<*>? = yicesDecls[decl]

    fun <K: KExpr<*>> substituteDecls(expr: K, transform: (K) -> K): K =
        transformed.getOrPut(expr) { transform(expr) }.uncheckedCast()

    open fun internalizeExpr(expr: KExpr<*>, internalizer: (KExpr<*>) -> YicesTerm): YicesTerm =
        internalize(expressions, yicesExpressions, expr, internalizer)

    open fun internalizeSort(sort: KSort, internalizer: (KSort) -> YicesSort): YicesSort =
        internalize(sorts, yicesSorts, sort, internalizer)

    open fun internalizeDecl(decl: KDecl<*>, internalizer: (KDecl<*>) -> YicesTerm): YicesTerm =
        internalize(decls, yicesDecls, decl, internalizer)

    fun convertExpr(expr: YicesTerm, converter: (YicesTerm) -> KExpr<*>): KExpr<*> =
        convert(expressions, yicesExpressions, expr, converter)

    fun convertSort(sort: YicesSort, converter: (YicesSort) -> KSort): KSort =
        convert(sorts, yicesSorts, sort, converter)

    fun convertDecl(decl: YicesTerm, converter: (YicesTerm) -> KDecl<*>): KDecl<*> =
        convert(decls, yicesDecls, decl, converter)

    private inline fun <K, V> internalize(
        cache: MutableMap<K, V>,
        reverseCache: MutableMap<V, K>,
        key: K,
        internalizer: (K) -> V
    ): V = cache.getOrPut(key) {
        internalizer(key).also { reverseCache[it] = key }
    }

    private inline fun <K, V> convert(
        cache: MutableMap<K, V>,
        reverseCache: MutableMap<V, K>,
        key: V,
        converter: (V) -> K
    ): K {
        val current = reverseCache[key]

        if (current != null) return current

        val converted = converter(key)
        cache.getOrPut(converted) { key }
        reverseCache[key] = converted

        return converted
    }

    override fun close() {
        isClosed = true
    }

    companion object {
        init {
            if (!Yices.isReady()) {
                NativeLibraryLoader.load { os ->
                    when (os) {
                        NativeLibraryLoader.OS.LINUX -> listOf("libyices", "libyices2java")
                        NativeLibraryLoader.OS.WINDOWS -> listOf(
                            "libwinpthread-1", "libgcc_s_seh-1", "libstdc++-6",
                            "libgmp-10", "libyices", "libyices2java"
                        )

                        else -> emptyList()
                    }
                }
                Yices.init()
                Yices.setReadyFlag(true)
            }
        }
    }
}
