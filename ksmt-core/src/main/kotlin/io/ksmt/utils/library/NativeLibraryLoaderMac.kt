package io.ksmt.utils.library

interface NativeLibraryLoaderMac : NativeLibraryLoader {
    override val osName: String get() = NativeLibraryLoaderUtils.MAC_OS_NAME
    override val osLibraryExt: String get() = ".dylib"
}
