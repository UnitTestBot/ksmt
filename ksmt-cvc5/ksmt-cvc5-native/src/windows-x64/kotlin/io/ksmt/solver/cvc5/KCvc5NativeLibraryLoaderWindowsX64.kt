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
        // A single self-contained library: the cvc5 release build for this platform
        // links libcvc5 and libcvc5parser into libcvc5jni.
        private val libraries = listOf(
            "libcvc5jni"
        )
    }
}
