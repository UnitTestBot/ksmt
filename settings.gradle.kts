rootProject.name = "ksmt"
include("ksmt-core")

include("ksmt-z3")
include("ksmt-z3:ksmt-z3-core")
include("ksmt-z3:ksmt-z3-native")

include("ksmt-bitwuzla")
include("ksmt-bitwuzla:ksmt-bitwuzla-core")
include("ksmt-bitwuzla:ksmt-bitwuzla-native")

include("ksmt-yices")
include("ksmt-yices:ksmt-yices-core")
include("ksmt-yices:ksmt-yices-native")

include("ksmt-cvc5")
include("ksmt-cvc5:ksmt-cvc5-core")
include("ksmt-cvc5:ksmt-cvc5-native")

include("ksmt-runner")
include("ksmt-runner:solver-generator")

include("ksmt-test")
include("ksmt-symfpu")

pluginManagement {
    resolutionStrategy {
        eachPlugin {
            if (requested.id.name == "rdgen") {
                useModule("com.jetbrains.rd:rd-gen:${requested.version}")
            }
        }
    }
}
