package org.ksmt.utils

import java.nio.file.Files
import kotlin.io.path.outputStream

object NativeLibraryLoader {
    enum class OS {
        LINUX, WINDOWS, MACOS
    }

    private val supportedArchs = setOf("amd64", "x86_64", "aarch64")

    fun load(libraries: (OS) -> List<String>) {
        val arch = System.getProperty("os.arch")

        require(arch in supportedArchs) { "Not supported arch: $arch" }

        val osName = System.getProperty("os.name").lowercase()
        val (os, libraryExt) = when {
            osName.startsWith("linux") -> OS.LINUX to ".so"
            osName.startsWith("windows") -> OS.WINDOWS to ".dll"
            osName.startsWith("mac") -> OS.MACOS to ".dylib"
            else -> error("Unknown OS: $osName")
        }

        val librariesToLoad = libraries(os)

        val folderName = constructFolderName(os, arch)

        for (libName in librariesToLoad) {
            val osLibName = libName + libraryExt
            val resourceName = "lib/x64/$folderName/$osLibName"
            val libUri = NativeLibraryLoader::class.java.classLoader
                .getResource(resourceName)
                ?.toURI()
                ?: error("Can't find native library $osLibName")

            if (libUri.scheme == "file") {
                System.load(libUri.path)
                continue
            }

            // use directory to preserve dll name on Windows
            val libUnpackDirectory = Files.createTempDirectory("ksmt")
            val libFile = libUnpackDirectory.resolve(libName + libraryExt)

            NativeLibraryLoader::class.java.classLoader
                .getResourceAsStream(resourceName)
                ?.use { libResourceStream ->
                    libFile.outputStream().use { libResourceStream.copyTo(it) }
                }

            System.load(libFile.toAbsolutePath().toString())

            // tmp files are not removed on Windows
            libFile.toFile().delete()
        }
    }

    private fun constructFolderName(os: OS, arch: String) = when (os) {
        OS.WINDOWS -> "windows"
        OS.LINUX -> "linux"
        OS.MACOS -> if (arch == "aarch64") "macArm" else "mac64"
    }
}
