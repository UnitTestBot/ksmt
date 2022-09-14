import org.gradle.api.Project
import org.gradle.api.component.AdhocComponentWithVariants
import org.gradle.api.component.SoftwareComponent
import org.gradle.kotlin.dsl.get

fun Project.removeTestFixtures(softwareComponent: SoftwareComponent) {
    with(softwareComponent as AdhocComponentWithVariants) {
        withVariantsFromConfiguration(configurations["testFixturesApiElements"]) {
            skip()
        }
        withVariantsFromConfiguration(configurations["testFixturesRuntimeElements"]) {
            skip()
        }
    }
}
