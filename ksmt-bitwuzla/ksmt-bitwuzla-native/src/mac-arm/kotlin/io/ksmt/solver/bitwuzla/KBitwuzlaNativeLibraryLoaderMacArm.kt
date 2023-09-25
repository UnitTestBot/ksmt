package io.ksmt.solver.bitwuzla

import io.ksmt.utils.library.NativeLibraryLoaderArm
import io.ksmt.utils.library.NativeLibraryLoaderMac
import io.ksmt.utils.library.NativeLibraryLoaderUtils

@Suppress("unused")
class KBitwuzlaNativeLibraryLoaderMacArm :
    KBitwuzlaNativeLibraryLoader,
    NativeLibraryLoaderMac,
    NativeLibraryLoaderArm {
    override fun load() {
        NativeLibraryLoaderUtils.loadLibrariesFromResources(this, libraries)
    }

    companion object {
        private val libraries = listOf("libbitwuzla", "libbitwuzla_jni")
    }
}
