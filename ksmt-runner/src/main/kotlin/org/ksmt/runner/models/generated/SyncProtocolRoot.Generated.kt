@file:Suppress("EXPERIMENTAL_API_USAGE","EXPERIMENTAL_UNSIGNED_LITERALS","PackageDirectoryMismatch","UnusedImport","unused","LocalVariableName","CanBeVal","PropertyName","EnumEntryName","ClassName","ObjectPropertyName","UnnecessaryVariable","SpellCheckingInspection")
package org.ksmt.runner.models.generated

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
 * #### Generated from [SyncProtocolModel.kt:5]
 */
class SyncProtocolRoot private constructor(
) : RdExtBase() {
    //companion
    
    companion object : ISerializersOwner {
        
        override fun registerSerializersCore(serializers: ISerializers)  {
            SyncProtocolRoot.register(serializers)
            SyncProtocolModel.register(serializers)
        }
        
        
        
        
        
        const val serializationHash = 4402368171949397719L
        
    }
    override val serializersOwner: ISerializersOwner get() = SyncProtocolRoot
    override val serializationHash: Long get() = SyncProtocolRoot.serializationHash
    
    //fields
    //methods
    //initializer
    //secondary constructor
    //equals trait
    //hash code trait
    //pretty print
    override fun print(printer: PrettyPrinter)  {
        printer.println("SyncProtocolRoot (")
        printer.print(")")
    }
    //deepClone
    override fun deepClone(): SyncProtocolRoot   {
        return SyncProtocolRoot(
        )
    }
    //contexts
}
