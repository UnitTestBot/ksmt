plugins {
    id("com.gradleup.nmcp.aggregation") version "0.1.3"
}

val mavenDeployUser = project.stringProperty("mavenDeployUser")
if (mavenDeployUser != null) {
    val mavenDeployPassword = project.stringProperty("mavenDeployPassword") ?: ""

    nmcpAggregation {
        centralPortal {
            username.set(mavenDeployUser)
            password.set(mavenDeployPassword)
            publishingType.set("USER_MANAGED")
        }

        // Publish all projects that apply the 'maven-publish' plugin
        publishAllProjectsProbablyBreakingProjectIsolation()
    }
}
