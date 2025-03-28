plugins {
    kotlin("jvm")
    application
}

group = "org.example"
version = "unspecified"

repositories {
    mavenCentral()
}

dependencies {
    implementation(project(":ksmt-core"))
    implementation(project(":ksmt-z3"))
    implementation(project(":ksmt-runner"))

    implementation("me.tongfei:progressbar:0.9.4")
}

tasks {
    val smt2FatJar = register<Jar>("smt2FatJar") {
        dependsOn.addAll(listOf("compileJava", "compileKotlin", "processResources"))

        archiveFileName.set("convert-smt2.jar")
        destinationDirectory.set(File("."))
        duplicatesStrategy = DuplicatesStrategy.EXCLUDE

        manifest {
            attributes(mapOf("Main-Class" to "io.ksmt.solver.neurosmt.smt2Converter.SMT2ConverterKt"))
        }

        val sourcesMain = sourceSets.main.get()
        val contents = configurations.runtimeClasspath.get()
            .map { if (it.isDirectory) it else zipTree(it) } + sourcesMain.output

        from(contents)
    }

    val ksmtFatJar = register<Jar>("ksmtFatJar") {
        dependsOn.addAll(listOf("compileJava", "compileKotlin", "processResources"))

        archiveFileName.set("convert-ksmt.jar")
        destinationDirectory.set(File("."))
        duplicatesStrategy = DuplicatesStrategy.EXCLUDE

        manifest {
            attributes(mapOf("Main-Class" to "io.ksmt.solver.neurosmt.ksmtBinaryConverter.KSMTBinaryConverterKt"))
        }

        val sourcesMain = sourceSets.main.get()
        val contents = configurations.runtimeClasspath.get()
            .map { if (it.isDirectory) it else zipTree(it) } + sourcesMain.output

        from(contents)
    }

    build {
        dependsOn(smt2FatJar, ksmtFatJar)
    }
}