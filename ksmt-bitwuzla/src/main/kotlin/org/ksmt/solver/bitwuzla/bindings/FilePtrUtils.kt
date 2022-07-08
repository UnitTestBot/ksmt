package org.ksmt.solver.bitwuzla.bindings

import com.sun.jna.Library
import com.sun.jna.Native
import com.sun.jna.Platform
import com.sun.jna.Pointer
import java.nio.file.Path
import kotlin.io.path.absolutePathString

object FilePtrUtils {
    fun openRead(path: Path): FilePtr = FilePtr(lib.fopen(path.absolutePathString(), "r"))
    fun openWrite(path: Path): FilePtr = FilePtr(lib.fopen(path.absolutePathString(), "w+"))
    fun stdout(): FilePtr = stdout
    fun flush(ptr: FilePtr) = lib.fflush(ptr.ptr)
    fun close(ptr: FilePtr) = lib.fclose(ptr.ptr)


    private interface CLib : Library {
        fun fopen(path: String, mode: String): Pointer
        fun fdopen(fd: Int, mode: String): Pointer
        fun fclose(ptr: Pointer)
        fun fflush(ptr: Pointer): Int
    }

    private interface MSVCRT : Library {
        fun fopen(path: String, mode: String): Pointer
        fun _fdopen(fd: Int, mode: String): Pointer
        fun fclose(ptr: Pointer)
        fun fflush(ptr: Pointer): Int
    }

    private class MSVCRTAdapter(private val msvcrt: MSVCRT) : CLib {
        override fun fopen(path: String, mode: String): Pointer = msvcrt.fopen(path, mode)
        override fun fdopen(fd: Int, mode: String): Pointer = msvcrt._fdopen(fd, mode)
        override fun fclose(ptr: Pointer) = msvcrt.fclose(ptr)
        override fun fflush(ptr: Pointer): Int = msvcrt.fflush(ptr)
    }

    private val lib: CLib by lazy {
        if (Platform.isWindows()) {
            MSVCRTAdapter(Native.load("msvcrt", MSVCRT::class.java))
        } else {
            Native.load("c", CLib::class.java)
        }
    }

    private val stdout: FilePtr by lazy {
        FilePtr(lib.fdopen(1, "w"), closable = false)
    }

}

class FilePtr internal constructor(val ptr: Pointer, private val closable: Boolean = true) : AutoCloseable {
    override fun close() {
        FilePtrUtils.flush(this)
        if (closable) {
            FilePtrUtils.close(this)
        }
    }
}
