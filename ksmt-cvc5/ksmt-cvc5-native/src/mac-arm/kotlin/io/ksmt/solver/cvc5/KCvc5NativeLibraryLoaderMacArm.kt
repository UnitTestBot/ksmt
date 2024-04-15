package io.ksmt.solver.cvc5

import io.ksmt.utils.library.NativeLibraryLoaderArm
import io.ksmt.utils.library.NativeLibraryLoaderMac
import io.ksmt.utils.library.NativeLibraryLoaderUtils

@Suppress("unused")
class KCvc5NativeLibraryLoaderMacArm :
    KCvc5NativeLibraryLoader,
    NativeLibraryLoaderMac,
    NativeLibraryLoaderArm {
    override fun load() {
        NativeLibraryLoaderUtils.loadLibrariesFromResources(this, libraries)
    }

    companion object {
        private val libraries = listOf(
            "libpoly.0.1.13",
            "libpolyxx.0.1.13",
            "libcvc5.1",
            "libcvc5parser.1",
            "libcvc5jni",
        )
    }
}
