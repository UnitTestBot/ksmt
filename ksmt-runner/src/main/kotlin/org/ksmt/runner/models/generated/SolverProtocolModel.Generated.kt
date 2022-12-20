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
 * #### Generated from [SolverProtocolModel.kt:16]
 */
class SolverProtocolModel private constructor(
    private val _initSolver: RdCall<CreateSolverParams, Unit>,
    private val _deleteSolver: RdCall<Unit, Unit>,
    private val _configure: RdCall<List<SolverConfigurationParam>, Unit>,
    private val _assert: RdCall<AssertParams, Unit>,
    private val _assertAndTrack: RdCall<AssertParams, AssertAndTrackResult>,
    private val _push: RdCall<Unit, Unit>,
    private val _pop: RdCall<PopParams, Unit>,
    private val _check: RdCall<CheckParams, CheckResult>,
    private val _checkWithAssumptions: RdCall<CheckWithAssumptionsParams, CheckResult>,
    private val _model: RdCall<Unit, ModelResult>,
    private val _unsatCore: RdCall<Unit, UnsatCoreResult>,
    private val _reasonOfUnknown: RdCall<Unit, ReasonUnknownResult>
) : RdExtBase() {
    //companion
    
    companion object : ISerializersOwner {
        
        override fun registerSerializersCore(serializers: ISerializers)  {
            serializers.register(CreateSolverParams)
            serializers.register(SolverConfigurationParam)
            serializers.register(AssertParams)
            serializers.register(AssertAndTrackResult)
            serializers.register(PopParams)
            serializers.register(CheckParams)
            serializers.register(CheckResult)
            serializers.register(CheckWithAssumptionsParams)
            serializers.register(UnsatCoreResult)
            serializers.register(ReasonUnknownResult)
            serializers.register(ModelFuncInterpEntry)
            serializers.register(ModelEntry)
            serializers.register(ModelUninterpretedSortUniverse)
            serializers.register(ModelResult)
            serializers.register(SolverType.marshaller)
            serializers.register(ConfigurationParamKind.marshaller)
        }
        
        
        @JvmStatic
        @JvmName("internalCreateModel")
        @Deprecated("Use create instead", ReplaceWith("create(lifetime, protocol)"))
        internal fun createModel(lifetime: Lifetime, protocol: IProtocol): SolverProtocolModel  {
            @Suppress("DEPRECATION")
            return create(lifetime, protocol)
        }
        
        @JvmStatic
        @Deprecated("Use protocol.solverProtocolModel or revise the extension scope instead", ReplaceWith("protocol.solverProtocolModel"))
        fun create(lifetime: Lifetime, protocol: IProtocol): SolverProtocolModel  {
            SolverProtocolRoot.register(protocol.serializers)
            
            return SolverProtocolModel().apply {
                identify(protocol.identity, RdId.Null.mix("SolverProtocolModel"))
                bind(lifetime, protocol, "SolverProtocolModel")
            }
        }
        
        private val __SolverConfigurationParamListSerializer = SolverConfigurationParam.list()
        
        const val serializationHash = 2419029814328334104L
        
    }
    override val serializersOwner: ISerializersOwner get() = SolverProtocolModel
    override val serializationHash: Long get() = SolverProtocolModel.serializationHash
    
    //fields
    
    /**
     * Initialize solver
     */
    val initSolver: RdCall<CreateSolverParams, Unit> get() = _initSolver
    
    /**
     * Delete solver
     */
    val deleteSolver: RdCall<Unit, Unit> get() = _deleteSolver
    
    /**
     * Configure solver with parameters
     */
    val configure: RdCall<List<SolverConfigurationParam>, Unit> get() = _configure
    
    /**
     * Assert expression
     */
    val assert: RdCall<AssertParams, Unit> get() = _assert
    
    /**
     * Assert and track expression
     */
    val assertAndTrack: RdCall<AssertParams, AssertAndTrackResult> get() = _assertAndTrack
    
    /**
     * Solver push
     */
    val push: RdCall<Unit, Unit> get() = _push
    
    /**
     * Solver pop
     */
    val pop: RdCall<PopParams, Unit> get() = _pop
    
    /**
     * Check SAT
     */
    val check: RdCall<CheckParams, CheckResult> get() = _check
    
    /**
     * Check SAT with assumptions
     */
    val checkWithAssumptions: RdCall<CheckWithAssumptionsParams, CheckResult> get() = _checkWithAssumptions
    
    /**
     * Get model
     */
    val model: RdCall<Unit, ModelResult> get() = _model
    
    /**
     * Get unsat core
     */
    val unsatCore: RdCall<Unit, UnsatCoreResult> get() = _unsatCore
    
    /**
     * Get reason of unknown
     */
    val reasonOfUnknown: RdCall<Unit, ReasonUnknownResult> get() = _reasonOfUnknown
    //methods
    //initializer
    init {
        _initSolver.async = true
        _deleteSolver.async = true
        _configure.async = true
        _assert.async = true
        _assertAndTrack.async = true
        _push.async = true
        _pop.async = true
        _check.async = true
        _checkWithAssumptions.async = true
        _model.async = true
        _unsatCore.async = true
        _reasonOfUnknown.async = true
    }
    
    init {
        bindableChildren.add("initSolver" to _initSolver)
        bindableChildren.add("deleteSolver" to _deleteSolver)
        bindableChildren.add("configure" to _configure)
        bindableChildren.add("assert" to _assert)
        bindableChildren.add("assertAndTrack" to _assertAndTrack)
        bindableChildren.add("push" to _push)
        bindableChildren.add("pop" to _pop)
        bindableChildren.add("check" to _check)
        bindableChildren.add("checkWithAssumptions" to _checkWithAssumptions)
        bindableChildren.add("model" to _model)
        bindableChildren.add("unsatCore" to _unsatCore)
        bindableChildren.add("reasonOfUnknown" to _reasonOfUnknown)
    }
    
    //secondary constructor
    private constructor(
    ) : this(
        RdCall<CreateSolverParams, Unit>(CreateSolverParams, FrameworkMarshallers.Void),
        RdCall<Unit, Unit>(FrameworkMarshallers.Void, FrameworkMarshallers.Void),
        RdCall<List<SolverConfigurationParam>, Unit>(__SolverConfigurationParamListSerializer, FrameworkMarshallers.Void),
        RdCall<AssertParams, Unit>(AssertParams, FrameworkMarshallers.Void),
        RdCall<AssertParams, AssertAndTrackResult>(AssertParams, AssertAndTrackResult),
        RdCall<Unit, Unit>(FrameworkMarshallers.Void, FrameworkMarshallers.Void),
        RdCall<PopParams, Unit>(PopParams, FrameworkMarshallers.Void),
        RdCall<CheckParams, CheckResult>(CheckParams, CheckResult),
        RdCall<CheckWithAssumptionsParams, CheckResult>(CheckWithAssumptionsParams, CheckResult),
        RdCall<Unit, ModelResult>(FrameworkMarshallers.Void, ModelResult),
        RdCall<Unit, UnsatCoreResult>(FrameworkMarshallers.Void, UnsatCoreResult),
        RdCall<Unit, ReasonUnknownResult>(FrameworkMarshallers.Void, ReasonUnknownResult)
    )
    
    //equals trait
    //hash code trait
    //pretty print
    override fun print(printer: PrettyPrinter)  {
        printer.println("SolverProtocolModel (")
        printer.indent {
            print("initSolver = "); _initSolver.print(printer); println()
            print("deleteSolver = "); _deleteSolver.print(printer); println()
            print("configure = "); _configure.print(printer); println()
            print("assert = "); _assert.print(printer); println()
            print("assertAndTrack = "); _assertAndTrack.print(printer); println()
            print("push = "); _push.print(printer); println()
            print("pop = "); _pop.print(printer); println()
            print("check = "); _check.print(printer); println()
            print("checkWithAssumptions = "); _checkWithAssumptions.print(printer); println()
            print("model = "); _model.print(printer); println()
            print("unsatCore = "); _unsatCore.print(printer); println()
            print("reasonOfUnknown = "); _reasonOfUnknown.print(printer); println()
        }
        printer.print(")")
    }
    //deepClone
    override fun deepClone(): SolverProtocolModel   {
        return SolverProtocolModel(
            _initSolver.deepClonePolymorphic(),
            _deleteSolver.deepClonePolymorphic(),
            _configure.deepClonePolymorphic(),
            _assert.deepClonePolymorphic(),
            _assertAndTrack.deepClonePolymorphic(),
            _push.deepClonePolymorphic(),
            _pop.deepClonePolymorphic(),
            _check.deepClonePolymorphic(),
            _checkWithAssumptions.deepClonePolymorphic(),
            _model.deepClonePolymorphic(),
            _unsatCore.deepClonePolymorphic(),
            _reasonOfUnknown.deepClonePolymorphic()
        )
    }
    //contexts
}
val IProtocol.solverProtocolModel get() = getOrCreateExtension(SolverProtocolModel::class) { @Suppress("DEPRECATION") SolverProtocolModel.create(lifetime, this) }



/**
 * #### Generated from [SolverProtocolModel.kt:42]
 */
data class AssertAndTrackResult (
    val expression: org.ksmt.KAst
) : IPrintable {
    //companion
    
    companion object : IMarshaller<AssertAndTrackResult> {
        override val _type: KClass<AssertAndTrackResult> = AssertAndTrackResult::class
        
        @Suppress("UNCHECKED_CAST")
        override fun read(ctx: SerializationCtx, buffer: AbstractBuffer): AssertAndTrackResult  {
            val expression = (ctx.serializers.get(org.ksmt.runner.serializer.AstSerializationCtx.marshallerId)!! as IMarshaller<org.ksmt.KAst>).read(ctx, buffer)
            return AssertAndTrackResult(expression)
        }
        
        override fun write(ctx: SerializationCtx, buffer: AbstractBuffer, value: AssertAndTrackResult)  {
            (ctx.serializers.get(org.ksmt.runner.serializer.AstSerializationCtx.marshallerId)!! as IMarshaller<org.ksmt.KAst>).write(ctx,buffer, value.expression)
        }
        
        
    }
    //fields
    //methods
    //initializer
    //secondary constructor
    //equals trait
    override fun equals(other: Any?): Boolean  {
        if (this === other) return true
        if (other == null || other::class != this::class) return false
        
        other as AssertAndTrackResult
        
        if (expression != other.expression) return false
        
        return true
    }
    //hash code trait
    override fun hashCode(): Int  {
        var __r = 0
        __r = __r*31 + expression.hashCode()
        return __r
    }
    //pretty print
    override fun print(printer: PrettyPrinter)  {
        printer.println("AssertAndTrackResult (")
        printer.indent {
            print("expression = "); expression.print(printer); println()
        }
        printer.print(")")
    }
    //deepClone
    //contexts
}


/**
 * #### Generated from [SolverProtocolModel.kt:38]
 */
data class AssertParams (
    val expression: org.ksmt.KAst
) : IPrintable {
    //companion
    
    companion object : IMarshaller<AssertParams> {
        override val _type: KClass<AssertParams> = AssertParams::class
        
        @Suppress("UNCHECKED_CAST")
        override fun read(ctx: SerializationCtx, buffer: AbstractBuffer): AssertParams  {
            val expression = (ctx.serializers.get(org.ksmt.runner.serializer.AstSerializationCtx.marshallerId)!! as IMarshaller<org.ksmt.KAst>).read(ctx, buffer)
            return AssertParams(expression)
        }
        
        override fun write(ctx: SerializationCtx, buffer: AbstractBuffer, value: AssertParams)  {
            (ctx.serializers.get(org.ksmt.runner.serializer.AstSerializationCtx.marshallerId)!! as IMarshaller<org.ksmt.KAst>).write(ctx,buffer, value.expression)
        }
        
        
    }
    //fields
    //methods
    //initializer
    //secondary constructor
    //equals trait
    override fun equals(other: Any?): Boolean  {
        if (this === other) return true
        if (other == null || other::class != this::class) return false
        
        other as AssertParams
        
        if (expression != other.expression) return false
        
        return true
    }
    //hash code trait
    override fun hashCode(): Int  {
        var __r = 0
        __r = __r*31 + expression.hashCode()
        return __r
    }
    //pretty print
    override fun print(printer: PrettyPrinter)  {
        printer.println("AssertParams (")
        printer.indent {
            print("expression = "); expression.print(printer); println()
        }
        printer.print(")")
    }
    //deepClone
    //contexts
}


/**
 * #### Generated from [SolverProtocolModel.kt:50]
 */
data class CheckParams (
    val timeout: Long
) : IPrintable {
    //companion
    
    companion object : IMarshaller<CheckParams> {
        override val _type: KClass<CheckParams> = CheckParams::class
        
        @Suppress("UNCHECKED_CAST")
        override fun read(ctx: SerializationCtx, buffer: AbstractBuffer): CheckParams  {
            val timeout = buffer.readLong()
            return CheckParams(timeout)
        }
        
        override fun write(ctx: SerializationCtx, buffer: AbstractBuffer, value: CheckParams)  {
            buffer.writeLong(value.timeout)
        }
        
        
    }
    //fields
    //methods
    //initializer
    //secondary constructor
    //equals trait
    override fun equals(other: Any?): Boolean  {
        if (this === other) return true
        if (other == null || other::class != this::class) return false
        
        other as CheckParams
        
        if (timeout != other.timeout) return false
        
        return true
    }
    //hash code trait
    override fun hashCode(): Int  {
        var __r = 0
        __r = __r*31 + timeout.hashCode()
        return __r
    }
    //pretty print
    override fun print(printer: PrettyPrinter)  {
        printer.println("CheckParams (")
        printer.indent {
            print("timeout = "); timeout.print(printer); println()
        }
        printer.print(")")
    }
    //deepClone
    //contexts
}


/**
 * #### Generated from [SolverProtocolModel.kt:54]
 */
data class CheckResult (
    val status: org.ksmt.solver.KSolverStatus
) : IPrintable {
    //companion
    
    companion object : IMarshaller<CheckResult> {
        override val _type: KClass<CheckResult> = CheckResult::class
        
        @Suppress("UNCHECKED_CAST")
        override fun read(ctx: SerializationCtx, buffer: AbstractBuffer): CheckResult  {
            val status = buffer.readEnum<org.ksmt.solver.KSolverStatus>()
            return CheckResult(status)
        }
        
        override fun write(ctx: SerializationCtx, buffer: AbstractBuffer, value: CheckResult)  {
            buffer.writeEnum(value.status)
        }
        
        
    }
    //fields
    //methods
    //initializer
    //secondary constructor
    //equals trait
    override fun equals(other: Any?): Boolean  {
        if (this === other) return true
        if (other == null || other::class != this::class) return false
        
        other as CheckResult
        
        if (status != other.status) return false
        
        return true
    }
    //hash code trait
    override fun hashCode(): Int  {
        var __r = 0
        __r = __r*31 + status.hashCode()
        return __r
    }
    //pretty print
    override fun print(printer: PrettyPrinter)  {
        printer.println("CheckResult (")
        printer.indent {
            print("status = "); status.print(printer); println()
        }
        printer.print(")")
    }
    //deepClone
    //contexts
}


/**
 * #### Generated from [SolverProtocolModel.kt:58]
 */
data class CheckWithAssumptionsParams (
    val assumptions: List<org.ksmt.KAst>,
    val timeout: Long
) : IPrintable {
    //companion
    
    companion object : IMarshaller<CheckWithAssumptionsParams> {
        override val _type: KClass<CheckWithAssumptionsParams> = CheckWithAssumptionsParams::class
        
        @Suppress("UNCHECKED_CAST")
        override fun read(ctx: SerializationCtx, buffer: AbstractBuffer): CheckWithAssumptionsParams  {
            val assumptions = buffer.readList { (ctx.serializers.get(org.ksmt.runner.serializer.AstSerializationCtx.marshallerId)!! as IMarshaller<org.ksmt.KAst>).read(ctx, buffer) }
            val timeout = buffer.readLong()
            return CheckWithAssumptionsParams(assumptions, timeout)
        }
        
        override fun write(ctx: SerializationCtx, buffer: AbstractBuffer, value: CheckWithAssumptionsParams)  {
            buffer.writeList(value.assumptions) { v -> (ctx.serializers.get(org.ksmt.runner.serializer.AstSerializationCtx.marshallerId)!! as IMarshaller<org.ksmt.KAst>).write(ctx,buffer, v) }
            buffer.writeLong(value.timeout)
        }
        
        
    }
    //fields
    //methods
    //initializer
    //secondary constructor
    //equals trait
    override fun equals(other: Any?): Boolean  {
        if (this === other) return true
        if (other == null || other::class != this::class) return false
        
        other as CheckWithAssumptionsParams
        
        if (assumptions != other.assumptions) return false
        if (timeout != other.timeout) return false
        
        return true
    }
    //hash code trait
    override fun hashCode(): Int  {
        var __r = 0
        __r = __r*31 + assumptions.hashCode()
        __r = __r*31 + timeout.hashCode()
        return __r
    }
    //pretty print
    override fun print(printer: PrettyPrinter)  {
        printer.println("CheckWithAssumptionsParams (")
        printer.indent {
            print("assumptions = "); assumptions.print(printer); println()
            print("timeout = "); timeout.print(printer); println()
        }
        printer.print(")")
    }
    //deepClone
    //contexts
}


/**
 * #### Generated from [SolverProtocolModel.kt:28]
 */
enum class ConfigurationParamKind {
    String, 
    Bool, 
    Int, 
    Double;
    
    companion object {
        val marshaller = FrameworkMarshallers.enum<ConfigurationParamKind>()
        
    }
}


/**
 * #### Generated from [SolverProtocolModel.kt:20]
 */
data class CreateSolverParams (
    val type: SolverType
) : IPrintable {
    //companion
    
    companion object : IMarshaller<CreateSolverParams> {
        override val _type: KClass<CreateSolverParams> = CreateSolverParams::class
        
        @Suppress("UNCHECKED_CAST")
        override fun read(ctx: SerializationCtx, buffer: AbstractBuffer): CreateSolverParams  {
            val type = buffer.readEnum<SolverType>()
            return CreateSolverParams(type)
        }
        
        override fun write(ctx: SerializationCtx, buffer: AbstractBuffer, value: CreateSolverParams)  {
            buffer.writeEnum(value.type)
        }
        
        
    }
    //fields
    //methods
    //initializer
    //secondary constructor
    //equals trait
    override fun equals(other: Any?): Boolean  {
        if (this === other) return true
        if (other == null || other::class != this::class) return false
        
        other as CreateSolverParams
        
        if (type != other.type) return false
        
        return true
    }
    //hash code trait
    override fun hashCode(): Int  {
        var __r = 0
        __r = __r*31 + type.hashCode()
        return __r
    }
    //pretty print
    override fun print(printer: PrettyPrinter)  {
        printer.println("CreateSolverParams (")
        printer.indent {
            print("type = "); type.print(printer); println()
        }
        printer.print(")")
    }
    //deepClone
    //contexts
}


/**
 * #### Generated from [SolverProtocolModel.kt:76]
 */
data class ModelEntry (
    val sort: org.ksmt.KAst,
    val vars: List<org.ksmt.KAst>,
    val entries: List<ModelFuncInterpEntry>,
    val default: org.ksmt.KAst?
) : IPrintable {
    //companion
    
    companion object : IMarshaller<ModelEntry> {
        override val _type: KClass<ModelEntry> = ModelEntry::class
        
        @Suppress("UNCHECKED_CAST")
        override fun read(ctx: SerializationCtx, buffer: AbstractBuffer): ModelEntry  {
            val sort = (ctx.serializers.get(org.ksmt.runner.serializer.AstSerializationCtx.marshallerId)!! as IMarshaller<org.ksmt.KAst>).read(ctx, buffer)
            val vars = buffer.readList { (ctx.serializers.get(org.ksmt.runner.serializer.AstSerializationCtx.marshallerId)!! as IMarshaller<org.ksmt.KAst>).read(ctx, buffer) }
            val entries = buffer.readList { ModelFuncInterpEntry.read(ctx, buffer) }
            val default = buffer.readNullable { (ctx.serializers.get(org.ksmt.runner.serializer.AstSerializationCtx.marshallerId)!! as IMarshaller<org.ksmt.KAst>).read(ctx, buffer) }
            return ModelEntry(sort, vars, entries, default)
        }
        
        override fun write(ctx: SerializationCtx, buffer: AbstractBuffer, value: ModelEntry)  {
            (ctx.serializers.get(org.ksmt.runner.serializer.AstSerializationCtx.marshallerId)!! as IMarshaller<org.ksmt.KAst>).write(ctx,buffer, value.sort)
            buffer.writeList(value.vars) { v -> (ctx.serializers.get(org.ksmt.runner.serializer.AstSerializationCtx.marshallerId)!! as IMarshaller<org.ksmt.KAst>).write(ctx,buffer, v) }
            buffer.writeList(value.entries) { v -> ModelFuncInterpEntry.write(ctx, buffer, v) }
            buffer.writeNullable(value.default) { (ctx.serializers.get(org.ksmt.runner.serializer.AstSerializationCtx.marshallerId)!! as IMarshaller<org.ksmt.KAst>).write(ctx,buffer, it) }
        }
        
        
    }
    //fields
    //methods
    //initializer
    //secondary constructor
    //equals trait
    override fun equals(other: Any?): Boolean  {
        if (this === other) return true
        if (other == null || other::class != this::class) return false
        
        other as ModelEntry
        
        if (sort != other.sort) return false
        if (vars != other.vars) return false
        if (entries != other.entries) return false
        if (default != other.default) return false
        
        return true
    }
    //hash code trait
    override fun hashCode(): Int  {
        var __r = 0
        __r = __r*31 + sort.hashCode()
        __r = __r*31 + vars.hashCode()
        __r = __r*31 + entries.hashCode()
        __r = __r*31 + if (default != null) default.hashCode() else 0
        return __r
    }
    //pretty print
    override fun print(printer: PrettyPrinter)  {
        printer.println("ModelEntry (")
        printer.indent {
            print("sort = "); sort.print(printer); println()
            print("vars = "); vars.print(printer); println()
            print("entries = "); entries.print(printer); println()
            print("default = "); default.print(printer); println()
        }
        printer.print(")")
    }
    //deepClone
    //contexts
}


/**
 * #### Generated from [SolverProtocolModel.kt:71]
 */
data class ModelFuncInterpEntry (
    val args: List<org.ksmt.KAst>,
    val value: org.ksmt.KAst
) : IPrintable {
    //companion
    
    companion object : IMarshaller<ModelFuncInterpEntry> {
        override val _type: KClass<ModelFuncInterpEntry> = ModelFuncInterpEntry::class
        
        @Suppress("UNCHECKED_CAST")
        override fun read(ctx: SerializationCtx, buffer: AbstractBuffer): ModelFuncInterpEntry  {
            val args = buffer.readList { (ctx.serializers.get(org.ksmt.runner.serializer.AstSerializationCtx.marshallerId)!! as IMarshaller<org.ksmt.KAst>).read(ctx, buffer) }
            val value = (ctx.serializers.get(org.ksmt.runner.serializer.AstSerializationCtx.marshallerId)!! as IMarshaller<org.ksmt.KAst>).read(ctx, buffer)
            return ModelFuncInterpEntry(args, value)
        }
        
        override fun write(ctx: SerializationCtx, buffer: AbstractBuffer, value: ModelFuncInterpEntry)  {
            buffer.writeList(value.args) { v -> (ctx.serializers.get(org.ksmt.runner.serializer.AstSerializationCtx.marshallerId)!! as IMarshaller<org.ksmt.KAst>).write(ctx,buffer, v) }
            (ctx.serializers.get(org.ksmt.runner.serializer.AstSerializationCtx.marshallerId)!! as IMarshaller<org.ksmt.KAst>).write(ctx,buffer, value.value)
        }
        
        
    }
    //fields
    //methods
    //initializer
    //secondary constructor
    //equals trait
    override fun equals(other: Any?): Boolean  {
        if (this === other) return true
        if (other == null || other::class != this::class) return false
        
        other as ModelFuncInterpEntry
        
        if (args != other.args) return false
        if (value != other.value) return false
        
        return true
    }
    //hash code trait
    override fun hashCode(): Int  {
        var __r = 0
        __r = __r*31 + args.hashCode()
        __r = __r*31 + value.hashCode()
        return __r
    }
    //pretty print
    override fun print(printer: PrettyPrinter)  {
        printer.println("ModelFuncInterpEntry (")
        printer.indent {
            print("args = "); args.print(printer); println()
            print("value = "); value.print(printer); println()
        }
        printer.print(")")
    }
    //deepClone
    //contexts
}


/**
 * #### Generated from [SolverProtocolModel.kt:88]
 */
data class ModelResult (
    val declarations: List<org.ksmt.KAst>,
    val interpretations: List<ModelEntry>,
    val uninterpretedSortUniverse: List<ModelUninterpretedSortUniverse>
) : IPrintable {
    //companion
    
    companion object : IMarshaller<ModelResult> {
        override val _type: KClass<ModelResult> = ModelResult::class
        
        @Suppress("UNCHECKED_CAST")
        override fun read(ctx: SerializationCtx, buffer: AbstractBuffer): ModelResult  {
            val declarations = buffer.readList { (ctx.serializers.get(org.ksmt.runner.serializer.AstSerializationCtx.marshallerId)!! as IMarshaller<org.ksmt.KAst>).read(ctx, buffer) }
            val interpretations = buffer.readList { ModelEntry.read(ctx, buffer) }
            val uninterpretedSortUniverse = buffer.readList { ModelUninterpretedSortUniverse.read(ctx, buffer) }
            return ModelResult(declarations, interpretations, uninterpretedSortUniverse)
        }
        
        override fun write(ctx: SerializationCtx, buffer: AbstractBuffer, value: ModelResult)  {
            buffer.writeList(value.declarations) { v -> (ctx.serializers.get(org.ksmt.runner.serializer.AstSerializationCtx.marshallerId)!! as IMarshaller<org.ksmt.KAst>).write(ctx,buffer, v) }
            buffer.writeList(value.interpretations) { v -> ModelEntry.write(ctx, buffer, v) }
            buffer.writeList(value.uninterpretedSortUniverse) { v -> ModelUninterpretedSortUniverse.write(ctx, buffer, v) }
        }
        
        
    }
    //fields
    //methods
    //initializer
    //secondary constructor
    //equals trait
    override fun equals(other: Any?): Boolean  {
        if (this === other) return true
        if (other == null || other::class != this::class) return false
        
        other as ModelResult
        
        if (declarations != other.declarations) return false
        if (interpretations != other.interpretations) return false
        if (uninterpretedSortUniverse != other.uninterpretedSortUniverse) return false
        
        return true
    }
    //hash code trait
    override fun hashCode(): Int  {
        var __r = 0
        __r = __r*31 + declarations.hashCode()
        __r = __r*31 + interpretations.hashCode()
        __r = __r*31 + uninterpretedSortUniverse.hashCode()
        return __r
    }
    //pretty print
    override fun print(printer: PrettyPrinter)  {
        printer.println("ModelResult (")
        printer.indent {
            print("declarations = "); declarations.print(printer); println()
            print("interpretations = "); interpretations.print(printer); println()
            print("uninterpretedSortUniverse = "); uninterpretedSortUniverse.print(printer); println()
        }
        printer.print(")")
    }
    //deepClone
    //contexts
}


/**
 * #### Generated from [SolverProtocolModel.kt:83]
 */
data class ModelUninterpretedSortUniverse (
    val sort: org.ksmt.KAst,
    val universe: List<org.ksmt.KAst>
) : IPrintable {
    //companion
    
    companion object : IMarshaller<ModelUninterpretedSortUniverse> {
        override val _type: KClass<ModelUninterpretedSortUniverse> = ModelUninterpretedSortUniverse::class
        
        @Suppress("UNCHECKED_CAST")
        override fun read(ctx: SerializationCtx, buffer: AbstractBuffer): ModelUninterpretedSortUniverse  {
            val sort = (ctx.serializers.get(org.ksmt.runner.serializer.AstSerializationCtx.marshallerId)!! as IMarshaller<org.ksmt.KAst>).read(ctx, buffer)
            val universe = buffer.readList { (ctx.serializers.get(org.ksmt.runner.serializer.AstSerializationCtx.marshallerId)!! as IMarshaller<org.ksmt.KAst>).read(ctx, buffer) }
            return ModelUninterpretedSortUniverse(sort, universe)
        }
        
        override fun write(ctx: SerializationCtx, buffer: AbstractBuffer, value: ModelUninterpretedSortUniverse)  {
            (ctx.serializers.get(org.ksmt.runner.serializer.AstSerializationCtx.marshallerId)!! as IMarshaller<org.ksmt.KAst>).write(ctx,buffer, value.sort)
            buffer.writeList(value.universe) { v -> (ctx.serializers.get(org.ksmt.runner.serializer.AstSerializationCtx.marshallerId)!! as IMarshaller<org.ksmt.KAst>).write(ctx,buffer, v) }
        }
        
        
    }
    //fields
    //methods
    //initializer
    //secondary constructor
    //equals trait
    override fun equals(other: Any?): Boolean  {
        if (this === other) return true
        if (other == null || other::class != this::class) return false
        
        other as ModelUninterpretedSortUniverse
        
        if (sort != other.sort) return false
        if (universe != other.universe) return false
        
        return true
    }
    //hash code trait
    override fun hashCode(): Int  {
        var __r = 0
        __r = __r*31 + sort.hashCode()
        __r = __r*31 + universe.hashCode()
        return __r
    }
    //pretty print
    override fun print(printer: PrettyPrinter)  {
        printer.println("ModelUninterpretedSortUniverse (")
        printer.indent {
            print("sort = "); sort.print(printer); println()
            print("universe = "); universe.print(printer); println()
        }
        printer.print(")")
    }
    //deepClone
    //contexts
}


/**
 * #### Generated from [SolverProtocolModel.kt:46]
 */
data class PopParams (
    val levels: UInt
) : IPrintable {
    //companion
    
    companion object : IMarshaller<PopParams> {
        override val _type: KClass<PopParams> = PopParams::class
        
        @Suppress("UNCHECKED_CAST")
        override fun read(ctx: SerializationCtx, buffer: AbstractBuffer): PopParams  {
            val levels = buffer.readUInt()
            return PopParams(levels)
        }
        
        override fun write(ctx: SerializationCtx, buffer: AbstractBuffer, value: PopParams)  {
            buffer.writeUInt(value.levels)
        }
        
        
    }
    //fields
    //methods
    //initializer
    //secondary constructor
    //equals trait
    override fun equals(other: Any?): Boolean  {
        if (this === other) return true
        if (other == null || other::class != this::class) return false
        
        other as PopParams
        
        if (levels != other.levels) return false
        
        return true
    }
    //hash code trait
    override fun hashCode(): Int  {
        var __r = 0
        __r = __r*31 + levels.hashCode()
        return __r
    }
    //pretty print
    override fun print(printer: PrettyPrinter)  {
        printer.println("PopParams (")
        printer.indent {
            print("levels = "); levels.print(printer); println()
        }
        printer.print(")")
    }
    //deepClone
    //contexts
}


/**
 * #### Generated from [SolverProtocolModel.kt:67]
 */
data class ReasonUnknownResult (
    val reasonUnknown: String
) : IPrintable {
    //companion
    
    companion object : IMarshaller<ReasonUnknownResult> {
        override val _type: KClass<ReasonUnknownResult> = ReasonUnknownResult::class
        
        @Suppress("UNCHECKED_CAST")
        override fun read(ctx: SerializationCtx, buffer: AbstractBuffer): ReasonUnknownResult  {
            val reasonUnknown = buffer.readString()
            return ReasonUnknownResult(reasonUnknown)
        }
        
        override fun write(ctx: SerializationCtx, buffer: AbstractBuffer, value: ReasonUnknownResult)  {
            buffer.writeString(value.reasonUnknown)
        }
        
        
    }
    //fields
    //methods
    //initializer
    //secondary constructor
    //equals trait
    override fun equals(other: Any?): Boolean  {
        if (this === other) return true
        if (other == null || other::class != this::class) return false
        
        other as ReasonUnknownResult
        
        if (reasonUnknown != other.reasonUnknown) return false
        
        return true
    }
    //hash code trait
    override fun hashCode(): Int  {
        var __r = 0
        __r = __r*31 + reasonUnknown.hashCode()
        return __r
    }
    //pretty print
    override fun print(printer: PrettyPrinter)  {
        printer.println("ReasonUnknownResult (")
        printer.indent {
            print("reasonUnknown = "); reasonUnknown.print(printer); println()
        }
        printer.print(")")
    }
    //deepClone
    //contexts
}


/**
 * #### Generated from [SolverProtocolModel.kt:27]
 */
data class SolverConfigurationParam (
    val kind: ConfigurationParamKind,
    val name: String,
    val value: String
) : IPrintable {
    //companion
    
    companion object : IMarshaller<SolverConfigurationParam> {
        override val _type: KClass<SolverConfigurationParam> = SolverConfigurationParam::class
        
        @Suppress("UNCHECKED_CAST")
        override fun read(ctx: SerializationCtx, buffer: AbstractBuffer): SolverConfigurationParam  {
            val kind = buffer.readEnum<ConfigurationParamKind>()
            val name = buffer.readString()
            val value = buffer.readString()
            return SolverConfigurationParam(kind, name, value)
        }
        
        override fun write(ctx: SerializationCtx, buffer: AbstractBuffer, value: SolverConfigurationParam)  {
            buffer.writeEnum(value.kind)
            buffer.writeString(value.name)
            buffer.writeString(value.value)
        }
        
        
    }
    //fields
    //methods
    //initializer
    //secondary constructor
    //equals trait
    override fun equals(other: Any?): Boolean  {
        if (this === other) return true
        if (other == null || other::class != this::class) return false
        
        other as SolverConfigurationParam
        
        if (kind != other.kind) return false
        if (name != other.name) return false
        if (value != other.value) return false
        
        return true
    }
    //hash code trait
    override fun hashCode(): Int  {
        var __r = 0
        __r = __r*31 + kind.hashCode()
        __r = __r*31 + name.hashCode()
        __r = __r*31 + value.hashCode()
        return __r
    }
    //pretty print
    override fun print(printer: PrettyPrinter)  {
        printer.println("SolverConfigurationParam (")
        printer.indent {
            print("kind = "); kind.print(printer); println()
            print("name = "); name.print(printer); println()
            print("value = "); value.print(printer); println()
        }
        printer.print(")")
    }
    //deepClone
    //contexts
}


/**
 * #### Generated from [SolverProtocolModel.kt:21]
 */
enum class SolverType {
    Z3, 
    Bitwuzla;
    
    companion object {
        val marshaller = FrameworkMarshallers.enum<SolverType>()
        
    }
}


/**
 * #### Generated from [SolverProtocolModel.kt:63]
 */
data class UnsatCoreResult (
    val core: List<org.ksmt.KAst>
) : IPrintable {
    //companion
    
    companion object : IMarshaller<UnsatCoreResult> {
        override val _type: KClass<UnsatCoreResult> = UnsatCoreResult::class
        
        @Suppress("UNCHECKED_CAST")
        override fun read(ctx: SerializationCtx, buffer: AbstractBuffer): UnsatCoreResult  {
            val core = buffer.readList { (ctx.serializers.get(org.ksmt.runner.serializer.AstSerializationCtx.marshallerId)!! as IMarshaller<org.ksmt.KAst>).read(ctx, buffer) }
            return UnsatCoreResult(core)
        }
        
        override fun write(ctx: SerializationCtx, buffer: AbstractBuffer, value: UnsatCoreResult)  {
            buffer.writeList(value.core) { v -> (ctx.serializers.get(org.ksmt.runner.serializer.AstSerializationCtx.marshallerId)!! as IMarshaller<org.ksmt.KAst>).write(ctx,buffer, v) }
        }
        
        
    }
    //fields
    //methods
    //initializer
    //secondary constructor
    //equals trait
    override fun equals(other: Any?): Boolean  {
        if (this === other) return true
        if (other == null || other::class != this::class) return false
        
        other as UnsatCoreResult
        
        if (core != other.core) return false
        
        return true
    }
    //hash code trait
    override fun hashCode(): Int  {
        var __r = 0
        __r = __r*31 + core.hashCode()
        return __r
    }
    //pretty print
    override fun print(printer: PrettyPrinter)  {
        printer.println("UnsatCoreResult (")
        printer.indent {
            print("core = "); core.print(printer); println()
        }
        printer.print(")")
    }
    //deepClone
    //contexts
}
