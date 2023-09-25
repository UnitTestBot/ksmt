package io.ksmt.solver.yices

import io.ksmt.utils.library.NativeLibraryLoaderLinux
import io.ksmt.utils.library.NativeLibraryLoaderUtils
import io.ksmt.utils.library.NativeLibraryLoaderX64

@Suppress("unused")
class KYicesNativeLibraryLoaderLinuxX64 :
    KYicesNativeLibraryLoader,
    NativeLibraryLoaderLinux,
    NativeLibraryLoaderX64 {
    override fun load() {
        NativeLibraryLoaderUtils.loadLibrariesFromResources(this, libraries)
    }

    companion object {
        private val libraries = listOf("libyices", "libyices2java")
    }
}
