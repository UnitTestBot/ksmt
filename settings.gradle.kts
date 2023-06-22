rootProject.name = "ksmt"
include("ksmt-core")
include("ksmt-z3")
include("ksmt-bitwuzla")
include("ksmt-yices")
include("ksmt-runner")
include("ksmt-runner:solver-generator")
include("ksmt-test")
include("ksmt-cvc5")
include("ksmt-maxsmt")

pluginManagement {
    resolutionStrategy {
        eachPlugin {
            if (requested.id.name == "rdgen") {
                useModule("com.jetbrains.rd:rd-gen:${requested.version}")
            }
        }
    }
}
include("ksmt-maxsmt:src:test:kotlin")
findProject(":ksmt-maxsmt:src:test:kotlin")?.name = "kotlin"
include("ksmt-maxsmt:src:test")
findProject(":ksmt-maxsmt:src:test")?.name = "test"
