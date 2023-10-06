package io.ksmt.utils.library

interface NativeLibraryLoaderLinux : NativeLibraryLoader {
    override val osName: String get() = NativeLibraryLoaderUtils.LINUX_OS_NAME
    override val osLibraryExt: String get() = ".so"
}
