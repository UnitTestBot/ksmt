package io.ksmt.solver.z3

import io.ksmt.utils.library.NativeLibraryLoaderArm
import io.ksmt.utils.library.NativeLibraryLoaderMac
import io.ksmt.utils.library.NativeLibraryLoaderUtils

@Suppress("unused")
class KZ3NativeLibraryLoaderMacArm :
    KZ3NativeLibraryLoader,
    NativeLibraryLoaderMac,
    NativeLibraryLoaderArm {
    override fun load() {
        NativeLibraryLoaderUtils.loadLibrariesFromResources(this, libraries)
    }

    companion object {
        private val libraries = listOf("libz3", "libz3java")
    }
}
