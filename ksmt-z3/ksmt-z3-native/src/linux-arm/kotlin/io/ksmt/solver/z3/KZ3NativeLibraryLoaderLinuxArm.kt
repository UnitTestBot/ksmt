package io.ksmt.solver.z3

import io.ksmt.utils.library.NativeLibraryLoaderArm
import io.ksmt.utils.library.NativeLibraryLoaderLinux
import io.ksmt.utils.library.NativeLibraryLoaderUtils

@Suppress("unused")
class KZ3NativeLibraryLoaderLinuxArm :
    KZ3NativeLibraryLoader,
    NativeLibraryLoaderLinux,
    NativeLibraryLoaderArm {

    override fun load() {
        NativeLibraryLoaderUtils.loadLibrariesFromResources(this, libraries)
    }

    companion object {
        private val libraries = listOf("libz3", "libz3java")
    }
}
