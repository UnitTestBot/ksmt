package org.ksmt.runner.serializer

enum class ExprKind {
    FunctionApp,
    Const,
    AndExpr,
    OrExpr,
    NotExpr,
    ImpliesExpr,
    XorExpr,
    True,
    False,
    EqExpr,
    DistinctExpr,
    IteExpr,
    BitVec1Value,
    BitVec8Value,
    BitVec16Value,
    BitVec32Value,
    BitVec64Value,
    BitVecCustomValue,
    BvNotExpr,
    BvReductionAndExpr,
    BvReductionOrExpr,
    BvAndExpr,
    BvOrExpr,
    BvXorExpr,
    BvNAndExpr,
    BvNorExpr,
    BvXNorExpr,
    BvNegationExpr,
    BvAddExpr,
    BvSubExpr,
    BvMulExpr,
    BvUnsignedDivExpr,
    BvSignedDivExpr,
    BvUnsignedRemExpr,
    BvSignedRemExpr,
    BvSignedModExpr,

    BvUnsignedLessExpr,

    BvSignedLessExpr,

    BvUnsignedLessOrEqualExpr,

    BvSignedLessOrEqualExpr,

    BvUnsignedGreaterOrEqualExpr,

    BvSignedGreaterOrEqualExpr,

    BvUnsignedGreaterExpr,

    BvSignedGreaterExpr,
    BvConcatExpr,
    BvExtractExpr,
    BvSignExtensionExpr,
    BvZeroExtensionExpr,
    BvRepeatExpr,
    BvShiftLeftExpr,

    BvLogicalShiftRightExpr,

    BvArithShiftRightExpr,
    BvRotateLeftExpr,

    BvRotateLeftIndexedExpr,
    BvRotateRightExpr,

    BvRotateRightIndexedExpr,
    Bv2IntExpr,

    BvAddNoOverflowExpr,

    BvAddNoUnderflowExpr,

    BvSubNoOverflowExpr,

    BvSubNoUnderflowExpr,

    BvDivNoOverflowExpr,

    BvNegNoOverflowExpr,

    BvMulNoOverflowExpr,

    BvMulNoUnderflowExpr,
    Fp16Value,
    Fp32Value,
    Fp64Value,
    Fp128Value,
    FpCustomSizeValue,

    FpRoundingModeExpr,
    FpAbsExpr,
    FpNegationExpr,
    FpAddExpr,
    FpSubExpr,
    FpMulExpr,
    FpDivExpr,
    FpFusedMulAddExpr,
    FpSqrtExpr,
    FpRemExpr,

    FpRoundToIntegralExpr,
    FpMinExpr,
    FpMaxExpr,

    FpLessOrEqualExpr,
    FpLessExpr,

    FpGreaterOrEqualExpr,
    FpGreaterExpr,
    FpEqualExpr,

    FpIsNormalExpr,

    FpIsSubnormalExpr,
    FpIsZeroExpr,

    FpIsInfiniteExpr,
    FpIsNaNExpr,

    FpIsNegativeExpr,

    FpIsPositiveExpr,
    FpToBvExpr,
    FpToRealExpr,
    FpToIEEEBvExpr,
    FpFromBvExpr,
    FpToFpExpr,
    RealToFpExpr,
    BvToFpExpr,

    ArrayStore,
    ArraySelect,

    ArrayConst,

    FunctionAsArray,

    ArrayLambda,
    AddArithExpr,
    MulArithExpr,
    SubArithExpr,

    UnaryMinusArithExpr,
    DivArithExpr,
    PowerArithExpr,

    LtArithExpr,

    LeArithExpr,

    GtArithExpr,

    GeArithExpr,
    ModIntExpr,
    RemIntExpr,
    ToRealIntExpr,
    Int32NumExpr,
    Int64NumExpr,
    IntBigNumExpr,
    ToIntRealExpr,
    IsIntRealExpr,
    RealNumExpr,
    ExistentialQuantifier,
    UniversalQuantifier,
}
