plugins {
    id("org.ksmt.ksmt-base")
    `java-test-fixtures`
}

dependencies {
    testFixturesApi("org.junit.jupiter", "junit-jupiter-params", "5.8.2")
}

publishing {
    publications {
        create<MavenPublication>("maven") {
            from(components["java"].also { removeTestFixtures(it) })
            artifact(tasks["kotlinSourcesJar"])
        }
    }
}
