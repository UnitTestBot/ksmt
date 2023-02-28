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
 * #### Generated from [TestProtocolModel.kt:12]
 */
class TestProtocolRoot private constructor(
) : RdExtBase() {
    //companion
    
    companion object : ISerializersOwner {
        
        override fun registerSerializersCore(serializers: ISerializers)  {
            TestProtocolRoot.register(serializers)
            TestProtocolModel.register(serializers)
        }
        
        
        
        
        
        const val serializationHash = -791875420344711602L
        
    }
    override val serializersOwner: ISerializersOwner get() = TestProtocolRoot
    override val serializationHash: Long get() = TestProtocolRoot.serializationHash
    
    //fields
    //methods
    //initializer
    //secondary constructor
    //equals trait
    //hash code trait
    //pretty print
    override fun print(printer: PrettyPrinter)  {
        printer.println("TestProtocolRoot (")
        printer.print(")")
    }
    //deepClone
    override fun deepClone(): TestProtocolRoot   {
        return TestProtocolRoot(
        )
    }
    //contexts
}
