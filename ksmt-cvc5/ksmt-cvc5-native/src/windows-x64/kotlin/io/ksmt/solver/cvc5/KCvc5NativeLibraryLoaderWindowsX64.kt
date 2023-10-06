package io.ksmt.solver.cvc5

import io.ksmt.utils.library.NativeLibraryLoaderUtils
import io.ksmt.utils.library.NativeLibraryLoaderWindows
import io.ksmt.utils.library.NativeLibraryLoaderX64

@Suppress("unused")
class KCvc5NativeLibraryLoaderWindowsX64 :
    KCvc5NativeLibraryLoader,
    NativeLibraryLoaderWindows,
    NativeLibraryLoaderX64 {
    override fun load() {
        NativeLibraryLoaderUtils.loadLibrariesFromResources(this, libraries)
    }

    companion object {
        private val libraries = listOf("libcvc5", "libcvc5jni")
    }
}
