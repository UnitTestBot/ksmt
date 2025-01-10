package io.ksmt.solver.cvc5

import io.ksmt.utils.library.NativeLibraryLoaderLinux
import io.ksmt.utils.library.NativeLibraryLoaderUtils
import io.ksmt.utils.library.NativeLibraryLoaderX64

@Suppress("unused")
class KCvc5NativeLibraryLoaderLinuxX64 :
    KCvc5NativeLibraryLoader,
    NativeLibraryLoaderLinux,
    NativeLibraryLoaderX64 {
    override fun load() {
        NativeLibraryLoaderUtils.loadLibrariesFromResources(this, libraries)
    }

    // Names with extension used in libraries list
    override val osLibraryExt: String get() = ""

    companion object {
        private val libraries = listOf(
            "libcvc5.so.1",
            "libcvc5parser.so.1",
            "libcvc5jni.so",
        )
    }
}
