package org.ksmt.cache

import java.util.Objects

@Suppress("FunctionOnlyReturningConstant")
fun hash(): Int = 0
fun <A0> hash(a0: A0): Int = a0.hashCode()
fun <A0, A1> hash(a0: A0, a1: A1): Int = Objects.hash(a0, a1)
fun <A0, A1, A2> hash(a0: A0, a1: A1, a2: A2): Int = Objects.hash(a0, a1, a2)
fun <A0, A1, A2, A3> hash(a0: A0, a1: A1, a2: A2, a3: A3): Int = Objects.hash(a0, a1, a2, a3)
fun <A0, A1, A2, A3, A4> hash(a0: A0, a1: A1, a2: A2, a3: A3, a4: A4): Int = Objects.hash(a0, a1, a2, a3, a4)

inline fun <reified T> T.structurallyEqual(other: Any): Boolean = other is T

inline fun <reified T, A0> T.structurallyEqual(other: Any, a0: T.() -> A0): Boolean =
    other is T && this.a0() == other.a0()

inline fun <reified T, A0, A1> T.structurallyEqual(other: Any, a0: T.() -> A0, a1: T.() -> A1): Boolean =
    other is T && this.a0() == other.a0() && this.a1() == other.a1()

inline fun <reified T, A0, A1, A2> T.structurallyEqual(
    other: Any,
    a0: T.() -> A0,
    a1: T.() -> A1,
    a2: T.() -> A2
): Boolean = other is T
        && this.a0() == other.a0()
        && this.a1() == other.a1()
        && this.a2() == other.a2()

inline fun <reified T, A0, A1, A2, A3> T.structurallyEqual(
    other: Any,
    a0: T.() -> A0,
    a1: T.() -> A1,
    a2: T.() -> A2,
    a3: T.() -> A3
): Boolean = other is T
        && this.a0() == other.a0()
        && this.a1() == other.a1()
        && this.a2() == other.a2()
        && this.a3() == other.a3()

@Suppress("LongParameterList")
inline fun <reified T, A0, A1, A2, A3, A4> T.structurallyEqual(
    other: Any,
    a0: T.() -> A0,
    a1: T.() -> A1,
    a2: T.() -> A2,
    a3: T.() -> A3,
    a4: T.() -> A4
): Boolean = other is T
        && this.a0() == other.a0()
        && this.a1() == other.a1()
        && this.a2() == other.a2()
        && this.a3() == other.a3()
        && this.a4() == other.a4()
