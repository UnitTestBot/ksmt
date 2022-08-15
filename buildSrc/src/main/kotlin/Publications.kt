import org.gradle.api.Project
import org.gradle.api.component.AdhocComponentWithVariants
import org.gradle.api.component.SoftwareComponent
import org.gradle.kotlin.dsl.get

fun Project.removeTestFixtures(softwareComponent: SoftwareComponent) {
    val componenet = softwareComponent as AdhocComponentWithVariants
    componenet.withVariantsFromConfiguration(configurations["testFixturesApiElements"]) {
        skip()
    }
    componenet.withVariantsFromConfiguration(configurations["testFixturesRuntimeElements"]) {
        skip()
    }
}
