plugins {
    id("io.ksmt.ksmt-base")
}

dependencies {
    implementation(project(":ksmt-core"))

    implementation(project(":ksmt-z3"))
    implementation(project(":ksmt-bitwuzla"))
    implementation(project(":ksmt-yices"))
    implementation(project(":ksmt-cvc5"))

    implementation(kotlin("reflect"))
}

val generateSolverUtils = tasks.register("generateSolverUtils") {
    dependsOn.add(tasks.getByName("compileKotlin"))
    doLast {
        val generatedSolverUtilsPackage: String by project.ext
        val generatedSolverUtilsPath: String by project.ext
        javaexec {
            classpath = sourceSets["main"].compileClasspath + sourceSets["main"].runtimeClasspath
            mainClass.set("SolverUtilsGeneratorKt")
            args = listOf(generatedSolverUtilsPath, generatedSolverUtilsPackage)
        }
    }
}
