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
 * #### Generated from [SyncProtocolModel.kt:7]
 */
class SyncProtocolModel private constructor(
    private val _synchronizationSignal: RdSignal<String>
) : RdExtBase() {
    //companion
    
    companion object : ISerializersOwner {
        
        override fun registerSerializersCore(serializers: ISerializers)  {
        }
        
        
        @JvmStatic
        @JvmName("internalCreateModel")
        @Deprecated("Use create instead", ReplaceWith("create(lifetime, protocol)"))
        internal fun createModel(lifetime: Lifetime, protocol: IProtocol): SyncProtocolModel  {
            @Suppress("DEPRECATION")
            return create(lifetime, protocol)
        }
        
        @JvmStatic
        @Deprecated("Use protocol.syncProtocolModel or revise the extension scope instead", ReplaceWith("protocol.syncProtocolModel"))
        fun create(lifetime: Lifetime, protocol: IProtocol): SyncProtocolModel  {
            SyncProtocolRoot.register(protocol.serializers)
            
            return SyncProtocolModel().apply {
                identify(protocol.identity, RdId.Null.mix("SyncProtocolModel"))
                bind(lifetime, protocol, "SyncProtocolModel")
            }
        }
        
        
        const val serializationHash = 5176282966920202414L
        
    }
    override val serializersOwner: ISerializersOwner get() = SyncProtocolModel
    override val serializationHash: Long get() = SyncProtocolModel.serializationHash
    
    //fields
    val synchronizationSignal: IAsyncSignal<String> get() = _synchronizationSignal
    //methods
    //initializer
    init {
        _synchronizationSignal.async = true
    }
    
    init {
        bindableChildren.add("synchronizationSignal" to _synchronizationSignal)
    }
    
    //secondary constructor
    private constructor(
    ) : this(
        RdSignal<String>(FrameworkMarshallers.String)
    )
    
    //equals trait
    //hash code trait
    //pretty print
    override fun print(printer: PrettyPrinter)  {
        printer.println("SyncProtocolModel (")
        printer.indent {
            print("synchronizationSignal = "); _synchronizationSignal.print(printer); println()
        }
        printer.print(")")
    }
    //deepClone
    override fun deepClone(): SyncProtocolModel   {
        return SyncProtocolModel(
            _synchronizationSignal.deepClonePolymorphic()
        )
    }
    //contexts
}
val IProtocol.syncProtocolModel get() = getOrCreateExtension(SyncProtocolModel::class) { @Suppress("DEPRECATION") SyncProtocolModel.create(lifetime, this) }

