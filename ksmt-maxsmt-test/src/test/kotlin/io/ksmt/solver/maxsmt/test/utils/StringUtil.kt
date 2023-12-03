package io.ksmt.solver.maxsmt.test.utils

fun getRandomString(length: Int): String {
    val charset = ('a'..'z') + ('0'..'9')

    return List(length) { charset.random() }
        .joinToString("")
}
