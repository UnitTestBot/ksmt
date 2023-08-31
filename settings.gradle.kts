rootProject.name = "ksmt"
include("ksmt-core")
include("ksmt-z3")
include("ksmt-bitwuzla")
include("ksmt-yices")
include("ksmt-runner")
include("ksmt-runner:solver-generator")
include("ksmt-test")
include("ksmt-cvc5")

pluginManagement {
    resolutionStrategy {
        eachPlugin {
            if (requested.id.name == "rdgen") {
                useModule("com.jetbrains.rd:rd-gen:${requested.version}")
            }
        }
    }
}
include("ksmt-neurosmt")
include("ksmt-neurosmt:utils")
// findProject(":ksmt-neurosmt:utils")?.name = "utils"
