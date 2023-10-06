package io.ksmt.utils.library

import java.io.IOException
import java.io.InputStream
import java.net.URL
import java.nio.file.Path
import kotlin.io.path.Path
import kotlin.io.path.createDirectories
import kotlin.io.path.createFile
import kotlin.io.path.createTempDirectory
import kotlin.io.path.deleteIfExists
import kotlin.io.path.div
import kotlin.io.path.exists
import kotlin.io.path.forEachDirectoryEntry
import kotlin.io.path.notExists
import kotlin.io.path.outputStream

object NativeLibraryLoaderUtils {
    const val LINUX_OS_NAME = "Linux"
    const val WINDOWS_OS_NAME = "Windows"
    const val MAC_OS_NAME = "Mac"

    const val X64_ARCH_NAME = "X64"
    const val ARM_ARCH_NAME = "Arm"

    val supportedArchs = setOf("amd64", "x86_64", "aarch64")

    /**
     * Load native libraries using [Loader].
     * The library loader for the specific architecture and OS
     * must have the same class name as [Loader] with suffix, containing OS and Arch.
     * For example, if [Loader] name is `a.b.MyLoader`, then the loader for the Windows OS
     * and x64 architecture must be named as `a.b.MyLoaderWindowsX64`.
     * Also, the resolved loader must be instantiatable and provide a constructor without arguments.
     * */
    inline fun <reified Loader : NativeLibraryLoader> load() {
        val osName = System.getProperty("os.name").lowercase()
        val resolvedOsName = when {
            osName.startsWith("linux") -> LINUX_OS_NAME
            osName.startsWith("windows") -> WINDOWS_OS_NAME
            osName.startsWith("mac") -> MAC_OS_NAME
            else -> error("Unknown OS: $osName")
        }

        val arch = System.getProperty("os.arch")
        require(arch in supportedArchs) { "Not supported arch: $arch" }
        val resolvedArchName = if (arch == "aarch64") ARM_ARCH_NAME else X64_ARCH_NAME

        val resolvedLoaderClassName = "${Loader::class.java.name}$resolvedOsName$resolvedArchName"

        @Suppress("SwallowedException", "TooGenericExceptionCaught")
        val resolvedLoaderClass = try {
            Class.forName(resolvedLoaderClassName)
        } catch (ex: Throwable) {
            error("No loader found for ${Loader::class.java.name} OS: $resolvedOsName Arch: $resolvedArchName")
        }

        val loader = resolvedLoaderClass.getDeclaredConstructor().newInstance() as Loader
        loader.load()
    }

    fun loadLibrariesFromResources(loader: NativeLibraryLoader, libraries: List<String>) {
        withLibraryUnpacker {
            for (libName in libraries) {
                val osLibName = loader.osSpecificLibraryName(libName)
                val resourceName = loader.resolveLibraryResourceName(libName)
                val libraryResource = findResource(resourceName)
                    ?: error("Can't find native library $osLibName")

                val libUri = libraryResource.toURI()
                if (libUri.scheme == "file") {
                    System.load(libUri.path)
                    continue
                }

                val libFile = libraryResource.openStream().use { libResourceStream ->
                    unpackLibrary(
                        name = osLibName,
                        libraryData = libResourceStream
                    )
                }

                System.load(libFile.toAbsolutePath().toString())
            }
        }
    }

    private fun NativeLibraryLoader.osSpecificLibraryName(libName: String): String =
        "$libName$osLibraryExt"

    private fun NativeLibraryLoader.resolveLibraryResourceName(libName: String): String =
        "lib/${osName.lowercase()}/${archName.lowercase()}/${libraryDir.lowercase()}/${osSpecificLibraryName(libName)}"

    private fun findResource(resourceName: String): URL? =
        NativeLibraryLoaderUtils::class.java.classLoader
            .getResource(resourceName)

    private inline fun withLibraryUnpacker(body: TmpDirLibraryUnpacker.() -> Unit) {
        val unpacker = TmpDirLibraryUnpacker()
        try {
            body(unpacker)
        } finally {
            unpacker.cleanup()
        }
    }

    private const val STALED_DIRECTORY_SUFFIX = ".ksmt-staled"

    private class TmpDirLibraryUnpacker {
        private val tmpDirectoryRoot = Path(System.getProperty("java.io.tmpdir", "."))
        private val unpackedLibrariesRoot = (tmpDirectoryRoot / "ksmt-unpacked-libraries").apply {
            createDirectories()
        }

        private val libUnpackDirectory = createTempDirectory(unpackedLibrariesRoot, "ksmt")
        private val unpackedFiles = mutableListOf<Path>()

        fun unpackLibrary(name: String, libraryData: InputStream): Path {
            val libFile = (libUnpackDirectory / name).also { unpackedFiles.add(it) }
            libFile.outputStream().use { libraryData.copyTo(it) }
            return libFile
        }

        /**
         * Cleanup unpacked files in two stages:
         * 1. Try to delete unpacked files.
         * This operation should succeed on Linux OS.
         * This operation will not succeed on Windows because process keep file handles
         * until the very end.
         * 2. If the previous operation was not successful we keep these files and try to delete them in the future.
         * Whenever a process with a KSMT library performs unpacked libraries cleanup
         * it also looks for staled files left over from other processes and tries to delete them.
         * */
        fun cleanup() {
            val notDeletedFiles = unpackedFiles.filterNot { it.safeDeleteFile() }

            var staledDirMarker: Path? = null
            if (notDeletedFiles.isNotEmpty() || !libUnpackDirectory.safeDeleteFile()) {
                staledDirMarker = markUnpackDirectoryAsStaled()
            }

            cleanupStaledDirectories(staledDirMarker)
        }

        private fun markUnpackDirectoryAsStaled(): Path {
            val markerName = "${libUnpackDirectory.fileName}$STALED_DIRECTORY_SUFFIX"
            val markerFile = libUnpackDirectory.resolveSibling(markerName)
            if (!markerFile.exists()) {
                markerFile.createFile()
            }
            return markerFile
        }

        private fun cleanupStaledDirectories(skipMarker: Path?) {
            unpackedLibrariesRoot.forEachDirectoryEntry("*$STALED_DIRECTORY_SUFFIX") { marker ->
                if (marker != skipMarker) {
                    val staledDirectoryName = marker.fileName.toString().removeSuffix(STALED_DIRECTORY_SUFFIX)
                    val staledDirectory = marker.resolveSibling(staledDirectoryName)
                    if (staledDirectory.notExists() || tryDeleteStaledDirectory(staledDirectory)) {
                        withNoFileExceptions { marker.deleteIfExists() }
                    }
                }
            }
        }

        private fun tryDeleteStaledDirectory(directory: Path): Boolean =
            withNoFileExceptions {
                directory.forEachDirectoryEntry {
                    if (!it.safeDeleteFile()) return false
                }
                return directory.safeDeleteFile()
            }

        private fun Path.safeDeleteFile(): Boolean = withNoFileExceptions {
            toFile().delete()
        }

        /**
         * Handle [SecurityException] because this exception may be thrown when file has no required permissions.
         * For example, we try do delete file without delete permission.
         *
         * Handle [IOException] because this exception may be thrown when file was deleted by another process.
         * */
        @Suppress("SwallowedException")
        private inline fun withNoFileExceptions(block: () -> Boolean): Boolean = try {
            block()
        } catch (ex: SecurityException) {
            false
        } catch (ex: IOException) {
            false
        }
    }
}
