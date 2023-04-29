package io.ksmt.runner.core

import com.jetbrains.rd.util.lifetime.Lifetime

interface Lifetimed {
    val lifetime: Lifetime
    fun terminate()
}
