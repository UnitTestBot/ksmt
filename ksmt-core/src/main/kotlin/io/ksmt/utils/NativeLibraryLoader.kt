package io.ksmt.utils

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

        val destinationFolder = if (arch == "aarch64") "arm" else "x64"

        withLibraryUnpacker {
            for (libName in librariesToLoad) {
                val osLibName = libName + libraryExt
                val resourceName = "lib/$destinationFolder/$osLibName"
                val libraryResource = findResource(resourceName)
                    ?: error("Can't find native library $osLibName")

                val libUri = libraryResource.toURI()
                if (libUri.scheme == "file") {
                    System.load(libUri.path)
                    continue
                }

                val libFile = libraryResource.openStream().use { libResourceStream ->
                    unpackLibrary(
                        name = libName + libraryExt,
                        libraryData = libResourceStream
                    )
                }

                System.load(libFile.toAbsolutePath().toString())
            }
        }
    }

    private fun findResource(resourceName: String): URL? =
        NativeLibraryLoader::class.java.classLoader
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

        private fun tryDeleteStaledDirectory(directory: Path): Boolean {
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
