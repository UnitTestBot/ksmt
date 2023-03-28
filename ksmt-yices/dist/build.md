### Build details

We build Yices with the following commands:
```shell
autoconf
./configure --enable-thread-safety --disable-mcsat
make MODE=release static-distribution 
```

To build Jni bindings `yices2java` we use the provided `Makefile` but we statically link `libgmp`.

#### Building on Windows
To produce Windows builds we use the same approach as in [`ksmt-bitwuzla`](../../ksmt-bitwuzla/dist/build.md).

Yices defines enum constants `STATUS_INTERRUPTED` and `STATUS_ERROR` (`/src/include/yices_types.h`) that conflict with those defined in the Windows stdlib.
We need to resolve this conflict before build.
The simplest and working approach is to replace all occurrences of these constants.

### Expected dynamic dependencies
To ensure that our distribution is portable, we verify that the produced binaries have no dependencies that might not be present on the user's machine.

#### Linux x64
```shell
$ ldd libyices.so libyices2java.so
libyices.so:
        linux-vdso.so.1 (0x00007ffeeecff000)
        libm.so.6 => /usr/lib/libm.so.6 (0x00007f59eacec000)
        libc.so.6 => /usr/lib/libc.so.6 (0x00007f59eab05000)
        /usr/lib64/ld-linux-x86-64.so.2 (0x00007f59eb03b000)
libyices2java.so:
        linux-vdso.so.1 (0x00007ffeafe26000)
        libyices.so.2.6 => not found
        libm.so.6 => /usr/lib/libm.so.6 (0x00007f38936e1000)
        libc.so.6 => /usr/lib/libc.so.6 (0x00007f38934fa000)
        /usr/lib64/ld-linux-x86-64.so.2 (0x00007f3893835000)
```
#### Windows x64
```shell
$ ldd libyices.dll libyices2java.dll
libyices.dll:
        ntdll.dll => /c/Windows/SYSTEM32/ntdll.dll (0x7ffb7a190000)
        KERNEL32.DLL => /c/Windows/System32/KERNEL32.DLL (0x7ffb78240000)
        KERNELBASE.dll => /c/Windows/System32/KERNELBASE.dll (0x7ffb77a70000)
        msvcrt.dll => /c/Windows/System32/msvcrt.dll (0x7ffb79410000)
libyices2java.dll:
        ntdll.dll => /c/Windows/SYSTEM32/ntdll.dll (0x7ffb7a190000)
        KERNEL32.DLL => /c/Windows/System32/KERNEL32.DLL (0x7ffb78240000)
        KERNELBASE.dll => /c/Windows/System32/KERNELBASE.dll (0x7ffb77a70000)
        msvcrt.dll => /c/Windows/System32/msvcrt.dll (0x7ffb79410000)
        libyices.dll => not found
        CRYPTBASE.DLL => /c/Windows/SYSTEM32/CRYPTBASE.DLL (0x7ffb77190000)
        bcryptPrimitives.dll => /c/Windows/System32/bcryptPrimitives.dll (0x7ffb77de0000)
```
#### MacOS aarch64
```shell
$ otool -L libyices.dylib libyices2java.dylib 
libyices.dylib:
        @rpath/libyices.dylib (compatibility version 2.6.0, current version 2.6.4)
        /usr/lib/libSystem.B.dylib (compatibility version 1.0.0, current version 1319.0.0)
libyices2java.dylib:
        @rpath/libyices2java.dylib (compatibility version 0.0.0, current version 0.0.0)
        @rpath/libyices.dylib (compatibility version 2.6.0, current version 2.6.4)
        /usr/lib/libc++.1.dylib (compatibility version 1.0.0, current version 1300.36.0)
        /usr/lib/libSystem.B.dylib (compatibility version 1.0.0, current version 1319.0.0)
```
