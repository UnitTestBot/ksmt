plugins {
    id("org.ksmt.ksmt-base")
    `java-test-fixtures`
}

dependencies {
    testFixturesImplementation("org.junit.jupiter", "junit-jupiter-params", "5.8.2")
}
