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

    implementation("me.tongfei:progressbar:0.9.4")
}

application {
    mainClass.set("io.ksmt.solver.neurosmt.smt2converter.SMT2ConverterKt")
}

tasks {
    val fatJar = register<Jar>("fatJar") {
        dependsOn.addAll(listOf("compileJava", "compileKotlin", "processResources"))

        archiveFileName.set("convert-smt2.jar")
        destinationDirectory.set(File("."))
        duplicatesStrategy = DuplicatesStrategy.EXCLUDE

        manifest {
            attributes(mapOf("Main-Class" to application.mainClass))
        }

        val sourcesMain = sourceSets.main.get()
        val contents = configurations.runtimeClasspath.get()
            .map { if (it.isDirectory) it else zipTree(it) } + sourcesMain.output

        from(contents)
    }

    build {
        dependsOn(fatJar)
    }
}