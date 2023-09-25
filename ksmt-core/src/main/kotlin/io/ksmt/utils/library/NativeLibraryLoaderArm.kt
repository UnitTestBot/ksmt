package io.ksmt.utils.library

interface NativeLibraryLoaderArm : NativeLibraryLoader {
    override val archName: String get() = NativeLibraryLoaderUtils.ARM_ARCH_NAME
}
