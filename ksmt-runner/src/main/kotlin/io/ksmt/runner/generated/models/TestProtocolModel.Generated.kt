@file:Suppress("EXPERIMENTAL_API_USAGE","EXPERIMENTAL_UNSIGNED_LITERALS","PackageDirectoryMismatch","UnusedImport","unused","LocalVariableName","CanBeVal","PropertyName","EnumEntryName","ClassName","ObjectPropertyName","UnnecessaryVariable","SpellCheckingInspection")
package io.ksmt.runner.generated.models

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
    private val _internalizeAndConvertYices: RdCall<TestInternalizeAndConvertParams, TestConversionResult>,
    private val _internalizeAndConvertCvc5: RdCall<TestInternalizeAndConvertParams, TestConversionResult>,
    private val _createSolver: RdCall<Int, Int>,
    private val _assert: RdCall<TestAssertParams, Unit>,
    private val _assertSoft: RdCall<TestSoftConstraint, Unit>,
    private val _check: RdCall<Int, TestCheckResult>,
    private val _checkMaxSMT: RdCall<TestCheckMaxSMTParams, TestCheckMaxSMTResult>,
    private val _checkSubOptMaxSMT: RdCall<TestCheckMaxSMTParams, TestCheckMaxSMTResult>,
    private val _collectMaxSMTStatistics: RdCall<Unit, TestCollectMaxSMTStatisticsResult>,
    private val _exprToString: RdCall<Long, String>,
    private val _getReasonUnknown: RdCall<Int, String>,
    private val _addEqualityCheck: RdCall<EqualityCheckParams, Unit>,
    private val _addEqualityCheckAssumption: RdCall<EqualityCheckAssumptionsParams, Unit>,
    private val _checkEqualities: RdCall<Int, TestCheckResult>,
    private val _findFirstFailedEquality: RdCall<Int, Int?>,
    private val _mkTrueExpr: RdCall<Unit, Long>
) : RdExtBase() {
    //companion
    
    companion object : ISerializersOwner {
        
        override fun registerSerializersCore(serializers: ISerializers)  {
            serializers.register(TestSoftConstraint)
            serializers.register(EqualityCheckParams)
            serializers.register(EqualityCheckAssumptionsParams)
            serializers.register(TestAssertParams)
            serializers.register(TestCheckResult)
            serializers.register(TestCheckMaxSMTParams)
            serializers.register(TestCheckMaxSMTResult)
            serializers.register(TestCollectMaxSMTStatisticsResult)
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
        
        const val serializationHash = -5506113420646493100L
        
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
     * Internalize and convert expressions using Yices converter/internalizer
     */
    val internalizeAndConvertYices: RdCall<TestInternalizeAndConvertParams, TestConversionResult> get() = _internalizeAndConvertYices
    
    /**
     * Internalize and convert expressions using cvc5 converter/internalizer
     */
    val internalizeAndConvertCvc5: RdCall<TestInternalizeAndConvertParams, TestConversionResult> get() = _internalizeAndConvertCvc5
    
    /**
     * Create solver
     */
    val createSolver: RdCall<Int, Int> get() = _createSolver
    
    /**
     * Assert expr
     */
    val assert: RdCall<TestAssertParams, Unit> get() = _assert
    
    /**
     * Assert expression softly
     */
    val assertSoft: RdCall<TestSoftConstraint, Unit> get() = _assertSoft
    
    /**
     * Check-sat
     */
    val check: RdCall<Int, TestCheckResult> get() = _check
    
    /**
     * Check MaxSMT
     */
    val checkMaxSMT: RdCall<TestCheckMaxSMTParams, TestCheckMaxSMTResult> get() = _checkMaxSMT
    
    /**
     * Check SubOptMaxSMT
     */
    val checkSubOptMaxSMT: RdCall<TestCheckMaxSMTParams, TestCheckMaxSMTResult> get() = _checkSubOptMaxSMT
    
    /**
     * Collect MaxSMT statistics
     */
    val collectMaxSMTStatistics: RdCall<Unit, TestCollectMaxSMTStatisticsResult> get() = _collectMaxSMTStatistics
    
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
     * Add assumptions for the subsequent equality check
     */
    val addEqualityCheckAssumption: RdCall<EqualityCheckAssumptionsParams, Unit> get() = _addEqualityCheckAssumption
    
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
        _internalizeAndConvertYices.async = true
        _internalizeAndConvertCvc5.async = true
        _createSolver.async = true
        _assert.async = true
        _check.async = true
        _exprToString.async = true
        _getReasonUnknown.async = true
        _addEqualityCheck.async = true
        _addEqualityCheckAssumption.async = true
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
        bindableChildren.add("internalizeAndConvertYices" to _internalizeAndConvertYices)
        bindableChildren.add("internalizeAndConvertCvc5" to _internalizeAndConvertCvc5)
        bindableChildren.add("createSolver" to _createSolver)
        bindableChildren.add("assert" to _assert)
        bindableChildren.add("assertSoft" to _assertSoft)
        bindableChildren.add("check" to _check)
        bindableChildren.add("checkMaxSMT" to _checkMaxSMT)
        bindableChildren.add("checkSubOptMaxSMT" to _checkSubOptMaxSMT)
        bindableChildren.add("collectMaxSMTStatistics" to _collectMaxSMTStatistics)
        bindableChildren.add("exprToString" to _exprToString)
        bindableChildren.add("getReasonUnknown" to _getReasonUnknown)
        bindableChildren.add("addEqualityCheck" to _addEqualityCheck)
        bindableChildren.add("addEqualityCheckAssumption" to _addEqualityCheckAssumption)
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
        RdCall<TestInternalizeAndConvertParams, TestConversionResult>(TestInternalizeAndConvertParams, TestConversionResult),
        RdCall<TestInternalizeAndConvertParams, TestConversionResult>(TestInternalizeAndConvertParams, TestConversionResult),
        RdCall<Int, Int>(FrameworkMarshallers.Int, FrameworkMarshallers.Int),
        RdCall<TestAssertParams, Unit>(TestAssertParams, FrameworkMarshallers.Void),
        RdCall<TestSoftConstraint, Unit>(TestSoftConstraint, FrameworkMarshallers.Void),
        RdCall<Int, TestCheckResult>(FrameworkMarshallers.Int, TestCheckResult),
        RdCall<TestCheckMaxSMTParams, TestCheckMaxSMTResult>(TestCheckMaxSMTParams, TestCheckMaxSMTResult),
        RdCall<TestCheckMaxSMTParams, TestCheckMaxSMTResult>(TestCheckMaxSMTParams, TestCheckMaxSMTResult),
        RdCall<Unit, TestCollectMaxSMTStatisticsResult>(FrameworkMarshallers.Void, TestCollectMaxSMTStatisticsResult),
        RdCall<Long, String>(FrameworkMarshallers.Long, FrameworkMarshallers.String),
        RdCall<Int, String>(FrameworkMarshallers.Int, FrameworkMarshallers.String),
        RdCall<EqualityCheckParams, Unit>(EqualityCheckParams, FrameworkMarshallers.Void),
        RdCall<EqualityCheckAssumptionsParams, Unit>(EqualityCheckAssumptionsParams, FrameworkMarshallers.Void),
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
            print("internalizeAndConvertYices = "); _internalizeAndConvertYices.print(printer); println()
            print("internalizeAndConvertCvc5 = "); _internalizeAndConvertCvc5.print(printer); println()
            print("createSolver = "); _createSolver.print(printer); println()
            print("assert = "); _assert.print(printer); println()
            print("assertSoft = "); _assertSoft.print(printer); println()
            print("check = "); _check.print(printer); println()
            print("checkMaxSMT = "); _checkMaxSMT.print(printer); println()
            print("checkSubOptMaxSMT = "); _checkSubOptMaxSMT.print(printer); println()
            print("collectMaxSMTStatistics = "); _collectMaxSMTStatistics.print(printer); println()
            print("exprToString = "); _exprToString.print(printer); println()
            print("getReasonUnknown = "); _getReasonUnknown.print(printer); println()
            print("addEqualityCheck = "); _addEqualityCheck.print(printer); println()
            print("addEqualityCheckAssumption = "); _addEqualityCheckAssumption.print(printer); println()
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
            _internalizeAndConvertYices.deepClonePolymorphic(),
            _internalizeAndConvertCvc5.deepClonePolymorphic(),
            _createSolver.deepClonePolymorphic(),
            _assert.deepClonePolymorphic(),
            _assertSoft.deepClonePolymorphic(),
            _check.deepClonePolymorphic(),
            _checkMaxSMT.deepClonePolymorphic(),
            _checkSubOptMaxSMT.deepClonePolymorphic(),
            _collectMaxSMTStatistics.deepClonePolymorphic(),
            _exprToString.deepClonePolymorphic(),
            _getReasonUnknown.deepClonePolymorphic(),
            _addEqualityCheck.deepClonePolymorphic(),
            _addEqualityCheckAssumption.deepClonePolymorphic(),
            _checkEqualities.deepClonePolymorphic(),
            _findFirstFailedEquality.deepClonePolymorphic(),
            _mkTrueExpr.deepClonePolymorphic()
        )
    }
    //contexts
}
val IProtocol.testProtocolModel get() = getOrCreateExtension(TestProtocolModel::class) { @Suppress("DEPRECATION") TestProtocolModel.create(lifetime, this) }



/**
 * #### Generated from [TestProtocolModel.kt:30]
 */
data class EqualityCheckAssumptionsParams (
    val solver: Int,
    val assumption: io.ksmt.KAst
) : IPrintable {
    //companion
    
    companion object : IMarshaller<EqualityCheckAssumptionsParams> {
        override val _type: KClass<EqualityCheckAssumptionsParams> = EqualityCheckAssumptionsParams::class
        
        @Suppress("UNCHECKED_CAST")
        override fun read(ctx: SerializationCtx, buffer: AbstractBuffer): EqualityCheckAssumptionsParams  {
            val solver = buffer.readInt()
            val assumption = (ctx.serializers.get(io.ksmt.runner.serializer.AstSerializationCtx.marshallerId)!! as IMarshaller<io.ksmt.KAst>).read(ctx, buffer)
            return EqualityCheckAssumptionsParams(solver, assumption)
        }
        
        override fun write(ctx: SerializationCtx, buffer: AbstractBuffer, value: EqualityCheckAssumptionsParams)  {
            buffer.writeInt(value.solver)
            (ctx.serializers.get(io.ksmt.runner.serializer.AstSerializationCtx.marshallerId)!! as IMarshaller<io.ksmt.KAst>).write(ctx,buffer, value.assumption)
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
        
        other as EqualityCheckAssumptionsParams
        
        if (solver != other.solver) return false
        if (assumption != other.assumption) return false
        
        return true
    }
    //hash code trait
    override fun hashCode(): Int  {
        var __r = 0
        __r = __r*31 + solver.hashCode()
        __r = __r*31 + assumption.hashCode()
        return __r
    }
    //pretty print
    override fun print(printer: PrettyPrinter)  {
        printer.println("EqualityCheckAssumptionsParams (")
        printer.indent {
            print("solver = "); solver.print(printer); println()
            print("assumption = "); assumption.print(printer); println()
        }
        printer.print(")")
    }
    //deepClone
    //contexts
}


/**
 * #### Generated from [TestProtocolModel.kt:24]
 */
data class EqualityCheckParams (
    val solver: Int,
    val `actual`: io.ksmt.KAst,
    val expected: Long
) : IPrintable {
    //companion
    
    companion object : IMarshaller<EqualityCheckParams> {
        override val _type: KClass<EqualityCheckParams> = EqualityCheckParams::class
        
        @Suppress("UNCHECKED_CAST")
        override fun read(ctx: SerializationCtx, buffer: AbstractBuffer): EqualityCheckParams  {
            val solver = buffer.readInt()
            val `actual` = (ctx.serializers.get(io.ksmt.runner.serializer.AstSerializationCtx.marshallerId)!! as IMarshaller<io.ksmt.KAst>).read(ctx, buffer)
            val expected = buffer.readLong()
            return EqualityCheckParams(solver, `actual`, expected)
        }
        
        override fun write(ctx: SerializationCtx, buffer: AbstractBuffer, value: EqualityCheckParams)  {
            buffer.writeInt(value.solver)
            (ctx.serializers.get(io.ksmt.runner.serializer.AstSerializationCtx.marshallerId)!! as IMarshaller<io.ksmt.KAst>).write(ctx,buffer, value.`actual`)
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
 * #### Generated from [TestProtocolModel.kt:35]
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
 * #### Generated from [TestProtocolModel.kt:44]
 */
data class TestCheckMaxSMTParams (
    val timeout: Long,
    val collectStatistics: Boolean
) : IPrintable {
    //companion
    
    companion object : IMarshaller<TestCheckMaxSMTParams> {
        override val _type: KClass<TestCheckMaxSMTParams> = TestCheckMaxSMTParams::class
        
        @Suppress("UNCHECKED_CAST")
        override fun read(ctx: SerializationCtx, buffer: AbstractBuffer): TestCheckMaxSMTParams  {
            val timeout = buffer.readLong()
            val collectStatistics = buffer.readBool()
            return TestCheckMaxSMTParams(timeout, collectStatistics)
        }
        
        override fun write(ctx: SerializationCtx, buffer: AbstractBuffer, value: TestCheckMaxSMTParams)  {
            buffer.writeLong(value.timeout)
            buffer.writeBool(value.collectStatistics)
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
        
        other as TestCheckMaxSMTParams
        
        if (timeout != other.timeout) return false
        if (collectStatistics != other.collectStatistics) return false
        
        return true
    }
    //hash code trait
    override fun hashCode(): Int  {
        var __r = 0
        __r = __r*31 + timeout.hashCode()
        __r = __r*31 + collectStatistics.hashCode()
        return __r
    }
    //pretty print
    override fun print(printer: PrettyPrinter)  {
        printer.println("TestCheckMaxSMTParams (")
        printer.indent {
            print("timeout = "); timeout.print(printer); println()
            print("collectStatistics = "); collectStatistics.print(printer); println()
        }
        printer.print(")")
    }
    //deepClone
    //contexts
}


/**
 * #### Generated from [TestProtocolModel.kt:49]
 */
data class TestCheckMaxSMTResult (
    val satSoftConstraints: List<TestSoftConstraint>,
    val hardConstraintsSatStatus: io.ksmt.solver.KSolverStatus,
    val maxSMTSucceeded: Boolean
) : IPrintable {
    //companion
    
    companion object : IMarshaller<TestCheckMaxSMTResult> {
        override val _type: KClass<TestCheckMaxSMTResult> = TestCheckMaxSMTResult::class
        
        @Suppress("UNCHECKED_CAST")
        override fun read(ctx: SerializationCtx, buffer: AbstractBuffer): TestCheckMaxSMTResult  {
            val satSoftConstraints = buffer.readList { TestSoftConstraint.read(ctx, buffer) }
            val hardConstraintsSatStatus = buffer.readEnum<io.ksmt.solver.KSolverStatus>()
            val maxSMTSucceeded = buffer.readBool()
            return TestCheckMaxSMTResult(satSoftConstraints, hardConstraintsSatStatus, maxSMTSucceeded)
        }
        
        override fun write(ctx: SerializationCtx, buffer: AbstractBuffer, value: TestCheckMaxSMTResult)  {
            buffer.writeList(value.satSoftConstraints) { v -> TestSoftConstraint.write(ctx, buffer, v) }
            buffer.writeEnum(value.hardConstraintsSatStatus)
            buffer.writeBool(value.maxSMTSucceeded)
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
        
        other as TestCheckMaxSMTResult
        
        if (satSoftConstraints != other.satSoftConstraints) return false
        if (hardConstraintsSatStatus != other.hardConstraintsSatStatus) return false
        if (maxSMTSucceeded != other.maxSMTSucceeded) return false
        
        return true
    }
    //hash code trait
    override fun hashCode(): Int  {
        var __r = 0
        __r = __r*31 + satSoftConstraints.hashCode()
        __r = __r*31 + hardConstraintsSatStatus.hashCode()
        __r = __r*31 + maxSMTSucceeded.hashCode()
        return __r
    }
    //pretty print
    override fun print(printer: PrettyPrinter)  {
        printer.println("TestCheckMaxSMTResult (")
        printer.indent {
            print("satSoftConstraints = "); satSoftConstraints.print(printer); println()
            print("hardConstraintsSatStatus = "); hardConstraintsSatStatus.print(printer); println()
            print("maxSMTSucceeded = "); maxSMTSucceeded.print(printer); println()
        }
        printer.print(")")
    }
    //deepClone
    //contexts
}


/**
 * #### Generated from [TestProtocolModel.kt:40]
 */
data class TestCheckResult (
    val status: io.ksmt.solver.KSolverStatus
) : IPrintable {
    //companion
    
    companion object : IMarshaller<TestCheckResult> {
        override val _type: KClass<TestCheckResult> = TestCheckResult::class
        
        @Suppress("UNCHECKED_CAST")
        override fun read(ctx: SerializationCtx, buffer: AbstractBuffer): TestCheckResult  {
            val status = buffer.readEnum<io.ksmt.solver.KSolverStatus>()
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
 * #### Generated from [TestProtocolModel.kt:55]
 */
data class TestCollectMaxSMTStatisticsResult (
    val timeoutMs: Long,
    val elapsedTimeMs: Long,
    val timeInSolverQueriesMs: Long,
    val queriesToSolverNumber: Int
) : IPrintable {
    //companion
    
    companion object : IMarshaller<TestCollectMaxSMTStatisticsResult> {
        override val _type: KClass<TestCollectMaxSMTStatisticsResult> = TestCollectMaxSMTStatisticsResult::class
        
        @Suppress("UNCHECKED_CAST")
        override fun read(ctx: SerializationCtx, buffer: AbstractBuffer): TestCollectMaxSMTStatisticsResult  {
            val timeoutMs = buffer.readLong()
            val elapsedTimeMs = buffer.readLong()
            val timeInSolverQueriesMs = buffer.readLong()
            val queriesToSolverNumber = buffer.readInt()
            return TestCollectMaxSMTStatisticsResult(timeoutMs, elapsedTimeMs, timeInSolverQueriesMs, queriesToSolverNumber)
        }
        
        override fun write(ctx: SerializationCtx, buffer: AbstractBuffer, value: TestCollectMaxSMTStatisticsResult)  {
            buffer.writeLong(value.timeoutMs)
            buffer.writeLong(value.elapsedTimeMs)
            buffer.writeLong(value.timeInSolverQueriesMs)
            buffer.writeInt(value.queriesToSolverNumber)
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
        
        other as TestCollectMaxSMTStatisticsResult
        
        if (timeoutMs != other.timeoutMs) return false
        if (elapsedTimeMs != other.elapsedTimeMs) return false
        if (timeInSolverQueriesMs != other.timeInSolverQueriesMs) return false
        if (queriesToSolverNumber != other.queriesToSolverNumber) return false
        
        return true
    }
    //hash code trait
    override fun hashCode(): Int  {
        var __r = 0
        __r = __r*31 + timeoutMs.hashCode()
        __r = __r*31 + elapsedTimeMs.hashCode()
        __r = __r*31 + timeInSolverQueriesMs.hashCode()
        __r = __r*31 + queriesToSolverNumber.hashCode()
        return __r
    }
    //pretty print
    override fun print(printer: PrettyPrinter)  {
        printer.println("TestCollectMaxSMTStatisticsResult (")
        printer.indent {
            print("timeoutMs = "); timeoutMs.print(printer); println()
            print("elapsedTimeMs = "); elapsedTimeMs.print(printer); println()
            print("timeInSolverQueriesMs = "); timeInSolverQueriesMs.print(printer); println()
            print("queriesToSolverNumber = "); queriesToSolverNumber.print(printer); println()
        }
        printer.print(")")
    }
    //deepClone
    //contexts
}


/**
 * #### Generated from [TestProtocolModel.kt:62]
 */
data class TestConversionResult (
    val expressions: List<io.ksmt.KAst>
) : IPrintable {
    //companion
    
    companion object : IMarshaller<TestConversionResult> {
        override val _type: KClass<TestConversionResult> = TestConversionResult::class
        
        @Suppress("UNCHECKED_CAST")
        override fun read(ctx: SerializationCtx, buffer: AbstractBuffer): TestConversionResult  {
            val expressions = buffer.readList { (ctx.serializers.get(io.ksmt.runner.serializer.AstSerializationCtx.marshallerId)!! as IMarshaller<io.ksmt.KAst>).read(ctx, buffer) }
            return TestConversionResult(expressions)
        }
        
        override fun write(ctx: SerializationCtx, buffer: AbstractBuffer, value: TestConversionResult)  {
            buffer.writeList(value.expressions) { v -> (ctx.serializers.get(io.ksmt.runner.serializer.AstSerializationCtx.marshallerId)!! as IMarshaller<io.ksmt.KAst>).write(ctx,buffer, v) }
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
 * #### Generated from [TestProtocolModel.kt:66]
 */
data class TestInternalizeAndConvertParams (
    val expressions: List<io.ksmt.KAst>
) : IPrintable {
    //companion
    
    companion object : IMarshaller<TestInternalizeAndConvertParams> {
        override val _type: KClass<TestInternalizeAndConvertParams> = TestInternalizeAndConvertParams::class
        
        @Suppress("UNCHECKED_CAST")
        override fun read(ctx: SerializationCtx, buffer: AbstractBuffer): TestInternalizeAndConvertParams  {
            val expressions = buffer.readList { (ctx.serializers.get(io.ksmt.runner.serializer.AstSerializationCtx.marshallerId)!! as IMarshaller<io.ksmt.KAst>).read(ctx, buffer) }
            return TestInternalizeAndConvertParams(expressions)
        }
        
        override fun write(ctx: SerializationCtx, buffer: AbstractBuffer, value: TestInternalizeAndConvertParams)  {
            buffer.writeList(value.expressions) { v -> (ctx.serializers.get(io.ksmt.runner.serializer.AstSerializationCtx.marshallerId)!! as IMarshaller<io.ksmt.KAst>).write(ctx,buffer, v) }
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


/**
 * #### Generated from [TestProtocolModel.kt:19]
 */
data class TestSoftConstraint (
    val expression: io.ksmt.KAst,
    val weight: UInt
) : IPrintable {
    //companion
    
    companion object : IMarshaller<TestSoftConstraint> {
        override val _type: KClass<TestSoftConstraint> = TestSoftConstraint::class
        
        @Suppress("UNCHECKED_CAST")
        override fun read(ctx: SerializationCtx, buffer: AbstractBuffer): TestSoftConstraint  {
            val expression = (ctx.serializers.get(io.ksmt.runner.serializer.AstSerializationCtx.marshallerId)!! as IMarshaller<io.ksmt.KAst>).read(ctx, buffer)
            val weight = buffer.readUInt()
            return TestSoftConstraint(expression, weight)
        }
        
        override fun write(ctx: SerializationCtx, buffer: AbstractBuffer, value: TestSoftConstraint)  {
            (ctx.serializers.get(io.ksmt.runner.serializer.AstSerializationCtx.marshallerId)!! as IMarshaller<io.ksmt.KAst>).write(ctx,buffer, value.expression)
            buffer.writeUInt(value.weight)
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
        
        other as TestSoftConstraint
        
        if (expression != other.expression) return false
        if (weight != other.weight) return false
        
        return true
    }
    //hash code trait
    override fun hashCode(): Int  {
        var __r = 0
        __r = __r*31 + expression.hashCode()
        __r = __r*31 + weight.hashCode()
        return __r
    }
    //pretty print
    override fun print(printer: PrettyPrinter)  {
        printer.println("TestSoftConstraint (")
        printer.indent {
            print("expression = "); expression.print(printer); println()
            print("weight = "); weight.print(printer); println()
        }
        printer.print(")")
    }
    //deepClone
    //contexts
}
