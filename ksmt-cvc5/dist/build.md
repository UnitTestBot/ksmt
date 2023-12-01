### Build details

We build cvc5 with the following configuration:
```shell
./configure.sh production --auto-download --ipo --no-static --no-cln --no-glpk --no-editline --cryptominisat --java-bindings
```
To make our distribution portable we statically link cvc5 against `libgmp`.

#### Building on Windows
To produce Windows builds we use the same approach as in [`ksmt-bitwuzla`](../../ksmt-bitwuzla/dist/build.md).
Also, we use addition configuration flag:
```shell
./configure.sh ... --win64-native
```

### Expected dynamic dependencies
To ensure that our distribution is portable, we verify that the produced binaries have no dependencies that might not be present on the user's machine.

#### Linux x64
```shell
$ ldd libcvc5.so libcvc5jni.so 
libcvc5.so:
        linux-vdso.so.1 (0x00007ffe22381000)
        libm.so.6 => /usr/lib/libm.so.6 (0x00007ff4c8bf4000)
        libc.so.6 => /usr/lib/libc.so.6 (0x00007ff4c7619000)
        /usr/lib64/ld-linux-x86-64.so.2 (0x00007ff4c8d0a000)
libcvc5jni.so:
        linux-vdso.so.1 (0x00007ffccbbb5000)
        libcvc5.so => not found
        libc.so.6 => /usr/lib/libc.so.6 (0x00007fa594c19000)
        libm.so.6 => /usr/lib/libm.so.6 (0x00007fa5961f3000)
        /usr/lib64/ld-linux-x86-64.so.2 (0x00007fa596389000)
```
#### Windows x64
```shell
$ ldd libcvc5.dll libcvc5jni.dll
libcvc5.dll:
        ntdll.dll => /c/Windows/SYSTEM32/ntdll.dll (0x7ffb7a190000)
        KERNEL32.DLL => /c/Windows/System32/KERNEL32.DLL (0x7ffb78240000)
        KERNELBASE.dll => /c/Windows/System32/KERNELBASE.dll (0x7ffb77a70000)
        msvcrt.dll => /c/Windows/System32/msvcrt.dll (0x7ffb79410000)
        CRYPTBASE.DLL => /c/Windows/SYSTEM32/CRYPTBASE.DLL (0x7ffb77190000)
        bcryptPrimitives.dll => /c/Windows/System32/bcryptPrimitives.dll (0x7ffb77de0000)
libcvc5jni.dll:
        ntdll.dll => /c/Windows/SYSTEM32/ntdll.dll (0x7ffb7a190000)
        KERNEL32.DLL => /c/Windows/System32/KERNEL32.DLL (0x7ffb78240000)
        KERNELBASE.dll => /c/Windows/System32/KERNELBASE.dll (0x7ffb77a70000)
        msvcrt.dll => /c/Windows/System32/msvcrt.dll (0x7ffb79410000)
        libcvc5.dll => not found
        CRYPTBASE.DLL => /c/Windows/SYSTEM32/CRYPTBASE.DLL (0x7ffb77190000)
        bcryptPrimitives.dll => /c/Windows/System32/bcryptPrimitives.dll (0x7ffb77de0000)
```
#### MacOS aarch64
```shell
$ otool -L libcvc5.dylib libcvc5jni.dylib 
libcvc5.dylib:
        @rpath/libcvc5.dylib (compatibility version 0.0.0, current version 0.0.0)
        /usr/lib/libc++.1.dylib (compatibility version 1.0.0, current version 1300.36.0)
        /usr/lib/libSystem.B.dylib (compatibility version 1.0.0, current version 1319.0.0)
libcvc5jni.dylib:
        @rpath/libcvc5jni.dylib (compatibility version 0.0.0, current version 0.0.0)
        @rpath/libcvc5.dylib (compatibility version 0.0.0, current version 0.0.0)
        /usr/lib/libc++.1.dylib (compatibility version 1.0.0, current version 1300.36.0)
        /usr/lib/libSystem.B.dylib (compatibility version 1.0.0, current version 1319.0.0)
```
