@file:Suppress("EXPERIMENTAL_API_USAGE","EXPERIMENTAL_UNSIGNED_LITERALS","PackageDirectoryMismatch","UnusedImport","unused","LocalVariableName","CanBeVal","PropertyName","EnumEntryName","ClassName","ObjectPropertyName","UnnecessaryVariable","SpellCheckingInspection")
package org.ksmt.runner.generated.models

import com.jetbrains.rd.framework.*
import com.jetbrains.rd.framework.base.*
import com.jetbrains.rd.framework.impl.*

import com.jetbrains.rd.util.lifetime.*
import com.jetbrains.rd.util.reactive.*
import com.jetbrains.rd.util.string.*
import com.jetbrains.rd.util.*
import kotlin.reflect.KClass
import kotlin.jvm.JvmStatic



/**
 * #### Generated from [SolverProtocolModel.kt:13]
 */
class SolverProtocolRoot private constructor(
) : RdExtBase() {
    //companion
    
    companion object : ISerializersOwner {
        
        override fun registerSerializersCore(serializers: ISerializers)  {
            SolverProtocolRoot.register(serializers)
            SolverProtocolModel.register(serializers)
        }
        
        
        
        
        
        const val serializationHash = -4190574921788870705L
        
    }
    override val serializersOwner: ISerializersOwner get() = SolverProtocolRoot
    override val serializationHash: Long get() = SolverProtocolRoot.serializationHash
    
    //fields
    //methods
    //initializer
    //secondary constructor
    //equals trait
    //hash code trait
    //pretty print
    override fun print(printer: PrettyPrinter)  {
        printer.println("SolverProtocolRoot (")
        printer.print(")")
    }
    //deepClone
    override fun deepClone(): SolverProtocolRoot   {
        return SolverProtocolRoot(
        )
    }
    //contexts
}
