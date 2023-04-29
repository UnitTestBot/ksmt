package io.ksmt.runner.models

import com.jetbrains.rd.generator.nova.Enum
import com.jetbrains.rd.generator.nova.Struct
import com.jetbrains.rd.generator.nova.Toplevel
import com.jetbrains.rd.generator.nova.kotlin.Kotlin11Generator.Intrinsic
import com.jetbrains.rd.generator.nova.kotlin.Kotlin11Generator.Namespace
import com.jetbrains.rd.generator.nova.kotlin.KotlinIntrinsicMarshaller

const val ksmtPackage = "io.ksmt"
const val kastClassName = "$ksmtPackage.KAst"
const val serializerClassName = "$ksmtPackage.runner.serializer.AstSerializationCtx"

fun Toplevel.kastType(): Struct {
    val marshaller = KotlinIntrinsicMarshaller(
        "(ctx.serializers.get(${serializerClassName}.marshallerId)!! as IMarshaller<${kastClassName}>)"
    )
    return Struct.Open("KAst", this, null).apply {
        settings[Namespace] = ksmtPackage
        settings[Intrinsic] = marshaller
    }
}

fun Toplevel.solverStatusType() = Enum("KSolverStatus", this).apply {
    settings[Namespace] = "${ksmtPackage}.solver"
}
