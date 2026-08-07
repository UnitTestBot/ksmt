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
        // A single self-contained library: the cvc5 release build for this platform
        // links libcvc5 and libcvc5parser into libcvc5jni.
        private val libraries = listOf(
            "libcvc5jni",
        )
    }
}
