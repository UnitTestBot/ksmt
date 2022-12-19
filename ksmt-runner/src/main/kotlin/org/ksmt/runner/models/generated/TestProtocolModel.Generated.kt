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
 * #### Generated from [TestProtocolModel.kt:15]
 */
class TestProtocolModel private constructor(
    private val _create: RdCall<Unit, Unit>,
    private val _delete: RdCall<Unit, Unit>,
    private val _parseFile: RdCall<String, List<Long>>,
    private val _convertAssertions: RdCall<List<Long>, TestConversionResult>,
    private val _internalizeAndConvertBitwuzla: RdCall<TestInternalizeAndConvertParams, TestConversionResult>,
    private val _createSolver: RdCall<Unit, Int>,
    private val _assert: RdCall<TestAssertParams, Unit>,
    private val _check: RdCall<Int, TestCheckResult>,
    private val _exprToString: RdCall<Long, String>,
    private val _getReasonUnknown: RdCall<Int, String>,
    private val _addEqualityCheck: RdCall<EqualityCheckParams, Unit>,
    private val _checkEqualities: RdCall<Int, TestCheckResult>,
    private val _findFirstFailedEquality: RdCall<Int, Int?>,
    private val _mkTrueExpr: RdCall<Unit, Long>
) : RdExtBase() {
    //companion
    
    companion object : ISerializersOwner {
        
        override fun registerSerializersCore(serializers: ISerializers)  {
            serializers.register(EqualityCheckParams)
            serializers.register(TestAssertParams)
            serializers.register(TestCheckResult)
            serializers.register(TestConversionResult)
            serializers.register(TestInternalizeAndConvertParams)
        }
        
        
        @JvmStatic
        @JvmName("internalCreateModel")
        @Deprecated("Use create instead", ReplaceWith("create(lifetime, protocol)"))
        internal fun createModel(lifetime: Lifetime, protocol: IProtocol): TestProtocolModel  {
            @Suppress("DEPRECATION")
            return create(lifetime, protocol)
        }
        
        @JvmStatic
        @Deprecated("Use protocol.testProtocolModel or revise the extension scope instead", ReplaceWith("protocol.testProtocolModel"))
        fun create(lifetime: Lifetime, protocol: IProtocol): TestProtocolModel  {
            TestProtocolRoot.register(protocol.serializers)
            
            return TestProtocolModel().apply {
                identify(protocol.identity, RdId.Null.mix("TestProtocolModel"))
                bind(lifetime, protocol, "TestProtocolModel")
            }
        }
        
        private val __LongListSerializer = FrameworkMarshallers.Long.list()
        private val __IntNullableSerializer = FrameworkMarshallers.Int.nullable()
        
        const val serializationHash = -213203582714111536L
        
    }
    override val serializersOwner: ISerializersOwner get() = TestProtocolModel
    override val serializationHash: Long get() = TestProtocolModel.serializationHash
    
    //fields
    
    /**
     * Create context
     */
    val create: RdCall<Unit, Unit> get() = _create
    
    /**
     * Delete context
     */
    val delete: RdCall<Unit, Unit> get() = _delete
    
    /**
     * Parse smt-lib2 file
     */
    val parseFile: RdCall<String, List<Long>> get() = _parseFile
    
    /**
     * Convert native solver expression into KSMT
     */
    val convertAssertions: RdCall<List<Long>, TestConversionResult> get() = _convertAssertions
    
    /**
     * Internalize and convert expressions using Bitwuzla converter/internalizer
     */
    val internalizeAndConvertBitwuzla: RdCall<TestInternalizeAndConvertParams, TestConversionResult> get() = _internalizeAndConvertBitwuzla
    
    /**
     * Create solver
     */
    val createSolver: RdCall<Unit, Int> get() = _createSolver
    
    /**
     * Assert expr
     */
    val assert: RdCall<TestAssertParams, Unit> get() = _assert
    
    /**
     * Check-sat
     */
    val check: RdCall<Int, TestCheckResult> get() = _check
    
    /**
     * Expression to string
     */
    val exprToString: RdCall<Long, String> get() = _exprToString
    
    /**
     * Get reason unknown
     */
    val getReasonUnknown: RdCall<Int, String> get() = _getReasonUnknown
    
    /**
     * Add equality check
     */
    val addEqualityCheck: RdCall<EqualityCheckParams, Unit> get() = _addEqualityCheck
    
    /**
     * Check added equalities
     */
    val checkEqualities: RdCall<Int, TestCheckResult> get() = _checkEqualities
    
    /**
     * Find first failed equality check
     */
    val findFirstFailedEquality: RdCall<Int, Int?> get() = _findFirstFailedEquality
    
    /**
     * Create true expression
     */
    val mkTrueExpr: RdCall<Unit, Long> get() = _mkTrueExpr
    //methods
    //initializer
    init {
        _create.async = true
        _delete.async = true
        _parseFile.async = true
        _convertAssertions.async = true
        _internalizeAndConvertBitwuzla.async = true
        _createSolver.async = true
        _assert.async = true
        _check.async = true
        _exprToString.async = true
        _getReasonUnknown.async = true
        _addEqualityCheck.async = true
        _checkEqualities.async = true
        _findFirstFailedEquality.async = true
        _mkTrueExpr.async = true
    }
    
    init {
        bindableChildren.add("create" to _create)
        bindableChildren.add("delete" to _delete)
        bindableChildren.add("parseFile" to _parseFile)
        bindableChildren.add("convertAssertions" to _convertAssertions)
        bindableChildren.add("internalizeAndConvertBitwuzla" to _internalizeAndConvertBitwuzla)
        bindableChildren.add("createSolver" to _createSolver)
        bindableChildren.add("assert" to _assert)
        bindableChildren.add("check" to _check)
        bindableChildren.add("exprToString" to _exprToString)
        bindableChildren.add("getReasonUnknown" to _getReasonUnknown)
        bindableChildren.add("addEqualityCheck" to _addEqualityCheck)
        bindableChildren.add("checkEqualities" to _checkEqualities)
        bindableChildren.add("findFirstFailedEquality" to _findFirstFailedEquality)
        bindableChildren.add("mkTrueExpr" to _mkTrueExpr)
    }
    
    //secondary constructor
    private constructor(
    ) : this(
        RdCall<Unit, Unit>(FrameworkMarshallers.Void, FrameworkMarshallers.Void),
        RdCall<Unit, Unit>(FrameworkMarshallers.Void, FrameworkMarshallers.Void),
        RdCall<String, List<Long>>(FrameworkMarshallers.String, __LongListSerializer),
        RdCall<List<Long>, TestConversionResult>(__LongListSerializer, TestConversionResult),
        RdCall<TestInternalizeAndConvertParams, TestConversionResult>(TestInternalizeAndConvertParams, TestConversionResult),
        RdCall<Unit, Int>(FrameworkMarshallers.Void, FrameworkMarshallers.Int),
        RdCall<TestAssertParams, Unit>(TestAssertParams, FrameworkMarshallers.Void),
        RdCall<Int, TestCheckResult>(FrameworkMarshallers.Int, TestCheckResult),
        RdCall<Long, String>(FrameworkMarshallers.Long, FrameworkMarshallers.String),
        RdCall<Int, String>(FrameworkMarshallers.Int, FrameworkMarshallers.String),
        RdCall<EqualityCheckParams, Unit>(EqualityCheckParams, FrameworkMarshallers.Void),
        RdCall<Int, TestCheckResult>(FrameworkMarshallers.Int, TestCheckResult),
        RdCall<Int, Int?>(FrameworkMarshallers.Int, __IntNullableSerializer),
        RdCall<Unit, Long>(FrameworkMarshallers.Void, FrameworkMarshallers.Long)
    )
    
    //equals trait
    //hash code trait
    //pretty print
    override fun print(printer: PrettyPrinter)  {
        printer.println("TestProtocolModel (")
        printer.indent {
            print("create = "); _create.print(printer); println()
            print("delete = "); _delete.print(printer); println()
            print("parseFile = "); _parseFile.print(printer); println()
            print("convertAssertions = "); _convertAssertions.print(printer); println()
            print("internalizeAndConvertBitwuzla = "); _internalizeAndConvertBitwuzla.print(printer); println()
            print("createSolver = "); _createSolver.print(printer); println()
            print("assert = "); _assert.print(printer); println()
            print("check = "); _check.print(printer); println()
            print("exprToString = "); _exprToString.print(printer); println()
            print("getReasonUnknown = "); _getReasonUnknown.print(printer); println()
            print("addEqualityCheck = "); _addEqualityCheck.print(printer); println()
            print("checkEqualities = "); _checkEqualities.print(printer); println()
            print("findFirstFailedEquality = "); _findFirstFailedEquality.print(printer); println()
            print("mkTrueExpr = "); _mkTrueExpr.print(printer); println()
        }
        printer.print(")")
    }
    //deepClone
    override fun deepClone(): TestProtocolModel   {
        return TestProtocolModel(
            _create.deepClonePolymorphic(),
            _delete.deepClonePolymorphic(),
            _parseFile.deepClonePolymorphic(),
            _convertAssertions.deepClonePolymorphic(),
            _internalizeAndConvertBitwuzla.deepClonePolymorphic(),
            _createSolver.deepClonePolymorphic(),
            _assert.deepClonePolymorphic(),
            _check.deepClonePolymorphic(),
            _exprToString.deepClonePolymorphic(),
            _getReasonUnknown.deepClonePolymorphic(),
            _addEqualityCheck.deepClonePolymorphic(),
            _checkEqualities.deepClonePolymorphic(),
            _findFirstFailedEquality.deepClonePolymorphic(),
            _mkTrueExpr.deepClonePolymorphic()
        )
    }
    //contexts
}
val IProtocol.testProtocolModel get() = getOrCreateExtension(TestProtocolModel::class) { @Suppress("DEPRECATION") TestProtocolModel.create(lifetime, this) }



/**
 * #### Generated from [TestProtocolModel.kt:19]
 */
data class EqualityCheckParams (
    val solver: Int,
    val `actual`: org.ksmt.KAst,
    val expected: Long
) : IPrintable {
    //companion
    
    companion object : IMarshaller<EqualityCheckParams> {
        override val _type: KClass<EqualityCheckParams> = EqualityCheckParams::class
        
        @Suppress("UNCHECKED_CAST")
        override fun read(ctx: SerializationCtx, buffer: AbstractBuffer): EqualityCheckParams  {
            val solver = buffer.readInt()
            val `actual` = (ctx.serializers.get(org.ksmt.runner.serializer.AstSerializationCtx.marshallerId)!! as IMarshaller<org.ksmt.KAst>).read(ctx, buffer)
            val expected = buffer.readLong()
            return EqualityCheckParams(solver, `actual`, expected)
        }
        
        override fun write(ctx: SerializationCtx, buffer: AbstractBuffer, value: EqualityCheckParams)  {
            buffer.writeInt(value.solver)
            (ctx.serializers.get(org.ksmt.runner.serializer.AstSerializationCtx.marshallerId)!! as IMarshaller<org.ksmt.KAst>).write(ctx,buffer, value.`actual`)
            buffer.writeLong(value.expected)
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
        
        other as EqualityCheckParams
        
        if (solver != other.solver) return false
        if (`actual` != other.`actual`) return false
        if (expected != other.expected) return false
        
        return true
    }
    //hash code trait
    override fun hashCode(): Int  {
        var __r = 0
        __r = __r*31 + solver.hashCode()
        __r = __r*31 + `actual`.hashCode()
        __r = __r*31 + expected.hashCode()
        return __r
    }
    //pretty print
    override fun print(printer: PrettyPrinter)  {
        printer.println("EqualityCheckParams (")
        printer.indent {
            print("solver = "); solver.print(printer); println()
            print("actual = "); `actual`.print(printer); println()
            print("expected = "); expected.print(printer); println()
        }
        printer.print(")")
    }
    //deepClone
    //contexts
}


/**
 * #### Generated from [TestProtocolModel.kt:25]
 */
data class TestAssertParams (
    val solver: Int,
    val expr: Long
) : IPrintable {
    //companion
    
    companion object : IMarshaller<TestAssertParams> {
        override val _type: KClass<TestAssertParams> = TestAssertParams::class
        
        @Suppress("UNCHECKED_CAST")
        override fun read(ctx: SerializationCtx, buffer: AbstractBuffer): TestAssertParams  {
            val solver = buffer.readInt()
            val expr = buffer.readLong()
            return TestAssertParams(solver, expr)
        }
        
        override fun write(ctx: SerializationCtx, buffer: AbstractBuffer, value: TestAssertParams)  {
            buffer.writeInt(value.solver)
            buffer.writeLong(value.expr)
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
        
        other as TestAssertParams
        
        if (solver != other.solver) return false
        if (expr != other.expr) return false
        
        return true
    }
    //hash code trait
    override fun hashCode(): Int  {
        var __r = 0
        __r = __r*31 + solver.hashCode()
        __r = __r*31 + expr.hashCode()
        return __r
    }
    //pretty print
    override fun print(printer: PrettyPrinter)  {
        printer.println("TestAssertParams (")
        printer.indent {
            print("solver = "); solver.print(printer); println()
            print("expr = "); expr.print(printer); println()
        }
        printer.print(")")
    }
    //deepClone
    //contexts
}


/**
 * #### Generated from [TestProtocolModel.kt:30]
 */
data class TestCheckResult (
    val status: org.ksmt.solver.KSolverStatus
) : IPrintable {
    //companion
    
    companion object : IMarshaller<TestCheckResult> {
        override val _type: KClass<TestCheckResult> = TestCheckResult::class
        
        @Suppress("UNCHECKED_CAST")
        override fun read(ctx: SerializationCtx, buffer: AbstractBuffer): TestCheckResult  {
            val status = buffer.readEnum<org.ksmt.solver.KSolverStatus>()
            return TestCheckResult(status)
        }
        
        override fun write(ctx: SerializationCtx, buffer: AbstractBuffer, value: TestCheckResult)  {
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
        
        other as TestCheckResult
        
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
        printer.println("TestCheckResult (")
        printer.indent {
            print("status = "); status.print(printer); println()
        }
        printer.print(")")
    }
    //deepClone
    //contexts
}


/**
 * #### Generated from [TestProtocolModel.kt:34]
 */
data class TestConversionResult (
    val expressions: List<org.ksmt.KAst>
) : IPrintable {
    //companion
    
    companion object : IMarshaller<TestConversionResult> {
        override val _type: KClass<TestConversionResult> = TestConversionResult::class
        
        @Suppress("UNCHECKED_CAST")
        override fun read(ctx: SerializationCtx, buffer: AbstractBuffer): TestConversionResult  {
            val expressions = buffer.readList { (ctx.serializers.get(org.ksmt.runner.serializer.AstSerializationCtx.marshallerId)!! as IMarshaller<org.ksmt.KAst>).read(ctx, buffer) }
            return TestConversionResult(expressions)
        }
        
        override fun write(ctx: SerializationCtx, buffer: AbstractBuffer, value: TestConversionResult)  {
            buffer.writeList(value.expressions) { v -> (ctx.serializers.get(org.ksmt.runner.serializer.AstSerializationCtx.marshallerId)!! as IMarshaller<org.ksmt.KAst>).write(ctx,buffer, v) }
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
        
        other as TestConversionResult
        
        if (expressions != other.expressions) return false
        
        return true
    }
    //hash code trait
    override fun hashCode(): Int  {
        var __r = 0
        __r = __r*31 + expressions.hashCode()
        return __r
    }
    //pretty print
    override fun print(printer: PrettyPrinter)  {
        printer.println("TestConversionResult (")
        printer.indent {
            print("expressions = "); expressions.print(printer); println()
        }
        printer.print(")")
    }
    //deepClone
    //contexts
}


/**
 * #### Generated from [TestProtocolModel.kt:38]
 */
data class TestInternalizeAndConvertParams (
    val expressions: List<org.ksmt.KAst>
) : IPrintable {
    //companion
    
    companion object : IMarshaller<TestInternalizeAndConvertParams> {
        override val _type: KClass<TestInternalizeAndConvertParams> = TestInternalizeAndConvertParams::class
        
        @Suppress("UNCHECKED_CAST")
        override fun read(ctx: SerializationCtx, buffer: AbstractBuffer): TestInternalizeAndConvertParams  {
            val expressions = buffer.readList { (ctx.serializers.get(org.ksmt.runner.serializer.AstSerializationCtx.marshallerId)!! as IMarshaller<org.ksmt.KAst>).read(ctx, buffer) }
            return TestInternalizeAndConvertParams(expressions)
        }
        
        override fun write(ctx: SerializationCtx, buffer: AbstractBuffer, value: TestInternalizeAndConvertParams)  {
            buffer.writeList(value.expressions) { v -> (ctx.serializers.get(org.ksmt.runner.serializer.AstSerializationCtx.marshallerId)!! as IMarshaller<org.ksmt.KAst>).write(ctx,buffer, v) }
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
        
        other as TestInternalizeAndConvertParams
        
        if (expressions != other.expressions) return false
        
        return true
    }
    //hash code trait
    override fun hashCode(): Int  {
        var __r = 0
        __r = __r*31 + expressions.hashCode()
        return __r
    }
    //pretty print
    override fun print(printer: PrettyPrinter)  {
        printer.println("TestInternalizeAndConvertParams (")
        printer.indent {
            print("expressions = "); expressions.print(printer); println()
        }
        printer.print(")")
    }
    //deepClone
    //contexts
}
