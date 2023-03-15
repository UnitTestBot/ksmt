package org.ksmt

import org.ksmt.KContext.AstManagementMode.GC
import org.ksmt.KContext.AstManagementMode.NO_GC
import org.ksmt.KContext.OperationMode.CONCURRENT
import org.ksmt.KContext.OperationMode.SINGLE_THREAD
import org.ksmt.KContext.SimplificationMode.NO_SIMPLIFY
import org.ksmt.KContext.SimplificationMode.SIMPLIFY
import org.ksmt.cache.AstInterner
import org.ksmt.cache.KInternedObject
import org.ksmt.cache.mkAstInterner
import org.ksmt.cache.mkCache
import org.ksmt.decl.KAndDecl
import org.ksmt.decl.KArithAddDecl
import org.ksmt.decl.KArithDivDecl
import org.ksmt.decl.KArithGeDecl
import org.ksmt.decl.KArithGtDecl
import org.ksmt.decl.KArithLeDecl
import org.ksmt.decl.KArithLtDecl
import org.ksmt.decl.KArithMulDecl
import org.ksmt.decl.KArithPowerDecl
import org.ksmt.decl.KArithSubDecl
import org.ksmt.decl.KArithUnaryMinusDecl
import org.ksmt.decl.KArray2SelectDecl
import org.ksmt.decl.KArray2StoreDecl
import org.ksmt.decl.KArray3SelectDecl
import org.ksmt.decl.KArray3StoreDecl
import org.ksmt.decl.KArrayConstDecl
import org.ksmt.decl.KArrayNSelectDecl
import org.ksmt.decl.KArrayNStoreDecl
import org.ksmt.decl.KArraySelectDecl
import org.ksmt.decl.KArrayStoreDecl
import org.ksmt.decl.KBitVec16ValueDecl
import org.ksmt.decl.KBitVec1ValueDecl
import org.ksmt.decl.KBitVec32ValueDecl
import org.ksmt.decl.KBitVec64ValueDecl
import org.ksmt.decl.KBitVec8ValueDecl
import org.ksmt.decl.KBitVecCustomSizeValueDecl
import org.ksmt.decl.KBv2IntDecl
import org.ksmt.decl.KBvAddDecl
import org.ksmt.decl.KBvAddNoOverflowDecl
import org.ksmt.decl.KBvAddNoUnderflowDecl
import org.ksmt.decl.KBvAndDecl
import org.ksmt.decl.KBvArithShiftRightDecl
import org.ksmt.decl.KBvConcatDecl
import org.ksmt.decl.KBvDivNoOverflowDecl
import org.ksmt.decl.KBvExtractDecl
import org.ksmt.decl.KBvLogicalShiftRightDecl
import org.ksmt.decl.KBvMulDecl
import org.ksmt.decl.KBvMulNoOverflowDecl
import org.ksmt.decl.KBvMulNoUnderflowDecl
import org.ksmt.decl.KBvNAndDecl
import org.ksmt.decl.KBvNegNoOverflowDecl
import org.ksmt.decl.KBvNegationDecl
import org.ksmt.decl.KBvNorDecl
import org.ksmt.decl.KBvNotDecl
import org.ksmt.decl.KBvOrDecl
import org.ksmt.decl.KBvReductionAndDecl
import org.ksmt.decl.KBvReductionOrDecl
import org.ksmt.decl.KBvRepeatDecl
import org.ksmt.decl.KBvRotateLeftDecl
import org.ksmt.decl.KBvRotateLeftIndexedDecl
import org.ksmt.decl.KBvRotateRightDecl
import org.ksmt.decl.KBvRotateRightIndexedDecl
import org.ksmt.decl.KBvShiftLeftDecl
import org.ksmt.decl.KBvSignedDivDecl
import org.ksmt.decl.KBvSignedGreaterDecl
import org.ksmt.decl.KBvSignedGreaterOrEqualDecl
import org.ksmt.decl.KBvSignedLessDecl
import org.ksmt.decl.KBvSignedLessOrEqualDecl
import org.ksmt.decl.KBvSignedModDecl
import org.ksmt.decl.KBvSignedRemDecl
import org.ksmt.decl.KBvSubDecl
import org.ksmt.decl.KBvSubNoOverflowDecl
import org.ksmt.decl.KBvSubNoUnderflowDecl
import org.ksmt.decl.KBvToFpDecl
import org.ksmt.decl.KBvUnsignedDivDecl
import org.ksmt.decl.KBvUnsignedGreaterDecl
import org.ksmt.decl.KBvUnsignedGreaterOrEqualDecl
import org.ksmt.decl.KBvUnsignedLessDecl
import org.ksmt.decl.KBvUnsignedLessOrEqualDecl
import org.ksmt.decl.KBvUnsignedRemDecl
import org.ksmt.decl.KBvXNorDecl
import org.ksmt.decl.KBvXorDecl
import org.ksmt.decl.KConstDecl
import org.ksmt.decl.KDecl
import org.ksmt.decl.KDistinctDecl
import org.ksmt.decl.KEqDecl
import org.ksmt.decl.KFalseDecl
import org.ksmt.decl.KFp128Decl
import org.ksmt.decl.KFp16Decl
import org.ksmt.decl.KFp32Decl
import org.ksmt.decl.KFp64Decl
import org.ksmt.decl.KFpAbsDecl
import org.ksmt.decl.KFpAddDecl
import org.ksmt.decl.KFpCustomSizeDecl
import org.ksmt.decl.KFpDecl
import org.ksmt.decl.KFpDivDecl
import org.ksmt.decl.KFpEqualDecl
import org.ksmt.decl.KFpFromBvDecl
import org.ksmt.decl.KFpFusedMulAddDecl
import org.ksmt.decl.KFpGreaterDecl
import org.ksmt.decl.KFpGreaterOrEqualDecl
import org.ksmt.decl.KFpIsInfiniteDecl
import org.ksmt.decl.KFpIsNaNDecl
import org.ksmt.decl.KFpIsNegativeDecl
import org.ksmt.decl.KFpIsNormalDecl
import org.ksmt.decl.KFpIsPositiveDecl
import org.ksmt.decl.KFpIsSubnormalDecl
import org.ksmt.decl.KFpIsZeroDecl
import org.ksmt.decl.KFpLessDecl
import org.ksmt.decl.KFpLessOrEqualDecl
import org.ksmt.decl.KFpMaxDecl
import org.ksmt.decl.KFpMinDecl
import org.ksmt.decl.KFpMulDecl
import org.ksmt.decl.KFpNegationDecl
import org.ksmt.decl.KFpRemDecl
import org.ksmt.decl.KFpRoundToIntegralDecl
import org.ksmt.decl.KFpRoundingModeDecl
import org.ksmt.decl.KFpSqrtDecl
import org.ksmt.decl.KFpSubDecl
import org.ksmt.decl.KFpToBvDecl
import org.ksmt.decl.KFpToFpDecl
import org.ksmt.decl.KFpToIEEEBvDecl
import org.ksmt.decl.KFpToRealDecl
import org.ksmt.decl.KFuncDecl
import org.ksmt.decl.KImpliesDecl
import org.ksmt.decl.KIntModDecl
import org.ksmt.decl.KIntNumDecl
import org.ksmt.decl.KIntRemDecl
import org.ksmt.decl.KIntToRealDecl
import org.ksmt.decl.KIteDecl
import org.ksmt.decl.KNotDecl
import org.ksmt.decl.KOrDecl
import org.ksmt.decl.KRealIsIntDecl
import org.ksmt.decl.KRealNumDecl
import org.ksmt.decl.KRealToFpDecl
import org.ksmt.decl.KRealToIntDecl
import org.ksmt.decl.KSignExtDecl
import org.ksmt.decl.KTrueDecl
import org.ksmt.decl.KUninterpretedConstDecl
import org.ksmt.decl.KUninterpretedFuncDecl
import org.ksmt.decl.KXorDecl
import org.ksmt.decl.KZeroExtDecl
import org.ksmt.expr.KAddArithExpr
import org.ksmt.expr.KAndBinaryExpr
import org.ksmt.expr.KAndExpr
import org.ksmt.expr.KAndNaryExpr
import org.ksmt.expr.KApp
import org.ksmt.expr.KArray2Lambda
import org.ksmt.expr.KArray2Select
import org.ksmt.expr.KArray2Store
import org.ksmt.expr.KArray3Lambda
import org.ksmt.expr.KArray3Select
import org.ksmt.expr.KArray3Store
import org.ksmt.expr.KArrayConst
import org.ksmt.expr.KArrayLambda
import org.ksmt.expr.KArrayNLambda
import org.ksmt.expr.KArrayNSelect
import org.ksmt.expr.KArrayNStore
import org.ksmt.expr.KArraySelect
import org.ksmt.expr.KArrayStore
import org.ksmt.expr.KBitVec16Value
import org.ksmt.expr.KBitVec1Value
import org.ksmt.expr.KBitVec32Value
import org.ksmt.expr.KBitVec64Value
import org.ksmt.expr.KBitVec8Value
import org.ksmt.expr.KBitVecCustomValue
import org.ksmt.expr.KBitVecNumberValue
import org.ksmt.expr.KBitVecValue
import org.ksmt.expr.KBv2IntExpr
import org.ksmt.expr.KBvAddExpr
import org.ksmt.expr.KBvAddNoOverflowExpr
import org.ksmt.expr.KBvAddNoUnderflowExpr
import org.ksmt.expr.KBvAndExpr
import org.ksmt.expr.KBvArithShiftRightExpr
import org.ksmt.expr.KBvConcatExpr
import org.ksmt.expr.KBvDivNoOverflowExpr
import org.ksmt.expr.KBvExtractExpr
import org.ksmt.expr.KBvLogicalShiftRightExpr
import org.ksmt.expr.KBvMulExpr
import org.ksmt.expr.KBvMulNoOverflowExpr
import org.ksmt.expr.KBvMulNoUnderflowExpr
import org.ksmt.expr.KBvNAndExpr
import org.ksmt.expr.KBvNegNoOverflowExpr
import org.ksmt.expr.KBvNegationExpr
import org.ksmt.expr.KBvNorExpr
import org.ksmt.expr.KBvNotExpr
import org.ksmt.expr.KBvOrExpr
import org.ksmt.expr.KBvReductionAndExpr
import org.ksmt.expr.KBvReductionOrExpr
import org.ksmt.expr.KBvRepeatExpr
import org.ksmt.expr.KBvRotateLeftExpr
import org.ksmt.expr.KBvRotateLeftIndexedExpr
import org.ksmt.expr.KBvRotateRightExpr
import org.ksmt.expr.KBvRotateRightIndexedExpr
import org.ksmt.expr.KBvShiftLeftExpr
import org.ksmt.expr.KBvSignExtensionExpr
import org.ksmt.expr.KBvSignedDivExpr
import org.ksmt.expr.KBvSignedGreaterExpr
import org.ksmt.expr.KBvSignedGreaterOrEqualExpr
import org.ksmt.expr.KBvSignedLessExpr
import org.ksmt.expr.KBvSignedLessOrEqualExpr
import org.ksmt.expr.KBvSignedModExpr
import org.ksmt.expr.KBvSignedRemExpr
import org.ksmt.expr.KBvSubExpr
import org.ksmt.expr.KBvSubNoOverflowExpr
import org.ksmt.expr.KBvSubNoUnderflowExpr
import org.ksmt.expr.KBvToFpExpr
import org.ksmt.expr.KBvUnsignedDivExpr
import org.ksmt.expr.KBvUnsignedGreaterExpr
import org.ksmt.expr.KBvUnsignedGreaterOrEqualExpr
import org.ksmt.expr.KBvUnsignedLessExpr
import org.ksmt.expr.KBvUnsignedLessOrEqualExpr
import org.ksmt.expr.KBvUnsignedRemExpr
import org.ksmt.expr.KBvXNorExpr
import org.ksmt.expr.KBvXorExpr
import org.ksmt.expr.KBvZeroExtensionExpr
import org.ksmt.expr.KConst
import org.ksmt.expr.KDistinctExpr
import org.ksmt.expr.KDivArithExpr
import org.ksmt.expr.KEqExpr
import org.ksmt.expr.KExistentialQuantifier
import org.ksmt.expr.KExpr
import org.ksmt.expr.KFalse
import org.ksmt.expr.KFp128Value
import org.ksmt.expr.KFp16Value
import org.ksmt.expr.KFp32Value
import org.ksmt.expr.KFp64Value
import org.ksmt.expr.KFpAbsExpr
import org.ksmt.expr.KFpAddExpr
import org.ksmt.expr.KFpCustomSizeValue
import org.ksmt.expr.KFpDivExpr
import org.ksmt.expr.KFpEqualExpr
import org.ksmt.expr.KFpFromBvExpr
import org.ksmt.expr.KFpFusedMulAddExpr
import org.ksmt.expr.KFpGreaterExpr
import org.ksmt.expr.KFpGreaterOrEqualExpr
import org.ksmt.expr.KFpIsInfiniteExpr
import org.ksmt.expr.KFpIsNaNExpr
import org.ksmt.expr.KFpIsNegativeExpr
import org.ksmt.expr.KFpIsNormalExpr
import org.ksmt.expr.KFpIsPositiveExpr
import org.ksmt.expr.KFpIsSubnormalExpr
import org.ksmt.expr.KFpIsZeroExpr
import org.ksmt.expr.KFpLessExpr
import org.ksmt.expr.KFpLessOrEqualExpr
import org.ksmt.expr.KFpMaxExpr
import org.ksmt.expr.KFpMinExpr
import org.ksmt.expr.KFpMulExpr
import org.ksmt.expr.KFpNegationExpr
import org.ksmt.expr.KFpRemExpr
import org.ksmt.expr.KFpRoundToIntegralExpr
import org.ksmt.expr.KFpRoundingMode
import org.ksmt.expr.KFpRoundingModeExpr
import org.ksmt.expr.KFpSqrtExpr
import org.ksmt.expr.KFpSubExpr
import org.ksmt.expr.KFpToBvExpr
import org.ksmt.expr.KFpToFpExpr
import org.ksmt.expr.KFpToIEEEBvExpr
import org.ksmt.expr.KFpToRealExpr
import org.ksmt.expr.KFpValue
import org.ksmt.expr.KFunctionApp
import org.ksmt.expr.KFunctionAsArray
import org.ksmt.expr.KGeArithExpr
import org.ksmt.expr.KGtArithExpr
import org.ksmt.expr.KImpliesExpr
import org.ksmt.expr.KInt32NumExpr
import org.ksmt.expr.KInt64NumExpr
import org.ksmt.expr.KIntBigNumExpr
import org.ksmt.expr.KIntNumExpr
import org.ksmt.expr.KIsIntRealExpr
import org.ksmt.expr.KIteExpr
import org.ksmt.expr.KLeArithExpr
import org.ksmt.expr.KLtArithExpr
import org.ksmt.expr.KModIntExpr
import org.ksmt.expr.KMulArithExpr
import org.ksmt.expr.KNotExpr
import org.ksmt.expr.KOrBinaryExpr
import org.ksmt.expr.KOrExpr
import org.ksmt.expr.KOrNaryExpr
import org.ksmt.expr.KPowerArithExpr
import org.ksmt.expr.KRealNumExpr
import org.ksmt.expr.KRealToFpExpr
import org.ksmt.expr.KRemIntExpr
import org.ksmt.expr.KSubArithExpr
import org.ksmt.expr.KToIntRealExpr
import org.ksmt.expr.KToRealIntExpr
import org.ksmt.expr.KTrue
import org.ksmt.expr.KUnaryMinusArithExpr
import org.ksmt.expr.KUniversalQuantifier
import org.ksmt.expr.KXorExpr
import org.ksmt.expr.rewrite.simplify.simplifyAnd
import org.ksmt.expr.rewrite.simplify.simplifyAndNoFlat
import org.ksmt.expr.rewrite.simplify.simplifyArithAdd
import org.ksmt.expr.rewrite.simplify.simplifyArithDiv
import org.ksmt.expr.rewrite.simplify.simplifyArithGe
import org.ksmt.expr.rewrite.simplify.simplifyArithGt
import org.ksmt.expr.rewrite.simplify.simplifyArithLe
import org.ksmt.expr.rewrite.simplify.simplifyArithLt
import org.ksmt.expr.rewrite.simplify.simplifyArithMul
import org.ksmt.expr.rewrite.simplify.simplifyArithPower
import org.ksmt.expr.rewrite.simplify.simplifyArithSub
import org.ksmt.expr.rewrite.simplify.simplifyArithUnaryMinus
import org.ksmt.expr.rewrite.simplify.simplifyArraySelect
import org.ksmt.expr.rewrite.simplify.simplifyArrayNSelect
import org.ksmt.expr.rewrite.simplify.simplifyArrayStore
import org.ksmt.expr.rewrite.simplify.simplifyArrayNStore
import org.ksmt.expr.rewrite.simplify.simplifyBv2IntExpr
import org.ksmt.expr.rewrite.simplify.simplifyBvAddExpr
import org.ksmt.expr.rewrite.simplify.simplifyBvAddNoOverflowExpr
import org.ksmt.expr.rewrite.simplify.simplifyBvAddNoUnderflowExpr
import org.ksmt.expr.rewrite.simplify.simplifyBvAndExpr
import org.ksmt.expr.rewrite.simplify.simplifyBvArithShiftRightExpr
import org.ksmt.expr.rewrite.simplify.simplifyBvConcatExpr
import org.ksmt.expr.rewrite.simplify.simplifyBvDivNoOverflowExpr
import org.ksmt.expr.rewrite.simplify.simplifyBvExtractExpr
import org.ksmt.expr.rewrite.simplify.simplifyBvLogicalShiftRightExpr
import org.ksmt.expr.rewrite.simplify.simplifyBvMulExpr
import org.ksmt.expr.rewrite.simplify.simplifyBvMulNoOverflowExpr
import org.ksmt.expr.rewrite.simplify.simplifyBvMulNoUnderflowExpr
import org.ksmt.expr.rewrite.simplify.simplifyBvNAndExpr
import org.ksmt.expr.rewrite.simplify.simplifyBvNegationExpr
import org.ksmt.expr.rewrite.simplify.simplifyBvNegationNoOverflowExpr
import org.ksmt.expr.rewrite.simplify.simplifyBvNorExpr
import org.ksmt.expr.rewrite.simplify.simplifyBvNotExpr
import org.ksmt.expr.rewrite.simplify.simplifyBvOrExpr
import org.ksmt.expr.rewrite.simplify.simplifyBvReductionAndExpr
import org.ksmt.expr.rewrite.simplify.simplifyBvReductionOrExpr
import org.ksmt.expr.rewrite.simplify.simplifyBvRepeatExpr
import org.ksmt.expr.rewrite.simplify.simplifyBvRotateLeftExpr
import org.ksmt.expr.rewrite.simplify.simplifyBvRotateLeftIndexedExpr
import org.ksmt.expr.rewrite.simplify.simplifyBvRotateRightExpr
import org.ksmt.expr.rewrite.simplify.simplifyBvRotateRightIndexedExpr
import org.ksmt.expr.rewrite.simplify.simplifyBvShiftLeftExpr
import org.ksmt.expr.rewrite.simplify.simplifyBvSignExtensionExpr
import org.ksmt.expr.rewrite.simplify.simplifyBvSignedDivExpr
import org.ksmt.expr.rewrite.simplify.simplifyBvSignedGreaterExpr
import org.ksmt.expr.rewrite.simplify.simplifyBvSignedGreaterOrEqualExpr
import org.ksmt.expr.rewrite.simplify.simplifyBvSignedLessExpr
import org.ksmt.expr.rewrite.simplify.simplifyBvSignedLessOrEqualExpr
import org.ksmt.expr.rewrite.simplify.simplifyBvSignedModExpr
import org.ksmt.expr.rewrite.simplify.simplifyBvSignedRemExpr
import org.ksmt.expr.rewrite.simplify.simplifyBvSubExpr
import org.ksmt.expr.rewrite.simplify.simplifyBvSubNoOverflowExpr
import org.ksmt.expr.rewrite.simplify.simplifyBvSubNoUnderflowExpr
import org.ksmt.expr.rewrite.simplify.simplifyBvToFpExpr
import org.ksmt.expr.rewrite.simplify.simplifyBvUnsignedDivExpr
import org.ksmt.expr.rewrite.simplify.simplifyBvUnsignedGreaterExpr
import org.ksmt.expr.rewrite.simplify.simplifyBvUnsignedGreaterOrEqualExpr
import org.ksmt.expr.rewrite.simplify.simplifyBvUnsignedLessExpr
import org.ksmt.expr.rewrite.simplify.simplifyBvUnsignedLessOrEqualExpr
import org.ksmt.expr.rewrite.simplify.simplifyBvUnsignedRemExpr
import org.ksmt.expr.rewrite.simplify.simplifyBvXNorExpr
import org.ksmt.expr.rewrite.simplify.simplifyBvXorExpr
import org.ksmt.expr.rewrite.simplify.simplifyBvZeroExtensionExpr
import org.ksmt.expr.rewrite.simplify.simplifyDistinct
import org.ksmt.expr.rewrite.simplify.simplifyEq
import org.ksmt.expr.rewrite.simplify.simplifyFpAbsExpr
import org.ksmt.expr.rewrite.simplify.simplifyFpAddExpr
import org.ksmt.expr.rewrite.simplify.simplifyFpDivExpr
import org.ksmt.expr.rewrite.simplify.simplifyFpEqualExpr
import org.ksmt.expr.rewrite.simplify.simplifyFpFromBvExpr
import org.ksmt.expr.rewrite.simplify.simplifyFpFusedMulAddExpr
import org.ksmt.expr.rewrite.simplify.simplifyFpGreaterExpr
import org.ksmt.expr.rewrite.simplify.simplifyFpGreaterOrEqualExpr
import org.ksmt.expr.rewrite.simplify.simplifyFpIsInfiniteExpr
import org.ksmt.expr.rewrite.simplify.simplifyFpIsNaNExpr
import org.ksmt.expr.rewrite.simplify.simplifyFpIsNegativeExpr
import org.ksmt.expr.rewrite.simplify.simplifyFpIsNormalExpr
import org.ksmt.expr.rewrite.simplify.simplifyFpIsPositiveExpr
import org.ksmt.expr.rewrite.simplify.simplifyFpIsSubnormalExpr
import org.ksmt.expr.rewrite.simplify.simplifyFpIsZeroExpr
import org.ksmt.expr.rewrite.simplify.simplifyFpLessExpr
import org.ksmt.expr.rewrite.simplify.simplifyFpLessOrEqualExpr
import org.ksmt.expr.rewrite.simplify.simplifyFpMaxExpr
import org.ksmt.expr.rewrite.simplify.simplifyFpMinExpr
import org.ksmt.expr.rewrite.simplify.simplifyFpMulExpr
import org.ksmt.expr.rewrite.simplify.simplifyFpNegationExpr
import org.ksmt.expr.rewrite.simplify.simplifyFpRemExpr
import org.ksmt.expr.rewrite.simplify.simplifyFpRoundToIntegralExpr
import org.ksmt.expr.rewrite.simplify.simplifyFpSqrtExpr
import org.ksmt.expr.rewrite.simplify.simplifyFpSubExpr
import org.ksmt.expr.rewrite.simplify.simplifyFpToBvExpr
import org.ksmt.expr.rewrite.simplify.simplifyFpToFpExpr
import org.ksmt.expr.rewrite.simplify.simplifyFpToIEEEBvExpr
import org.ksmt.expr.rewrite.simplify.simplifyFpToRealExpr
import org.ksmt.expr.rewrite.simplify.simplifyImplies
import org.ksmt.expr.rewrite.simplify.simplifyIntMod
import org.ksmt.expr.rewrite.simplify.simplifyIntRem
import org.ksmt.expr.rewrite.simplify.simplifyIntToReal
import org.ksmt.expr.rewrite.simplify.simplifyIte
import org.ksmt.expr.rewrite.simplify.simplifyNot
import org.ksmt.expr.rewrite.simplify.simplifyOr
import org.ksmt.expr.rewrite.simplify.simplifyOrNoFlat
import org.ksmt.expr.rewrite.simplify.simplifyRealIsInt
import org.ksmt.expr.rewrite.simplify.simplifyRealToFpExpr
import org.ksmt.expr.rewrite.simplify.simplifyRealToInt
import org.ksmt.expr.rewrite.simplify.simplifyXor
import org.ksmt.sort.KArithSort
import org.ksmt.sort.KArray2Sort
import org.ksmt.sort.KArray3Sort
import org.ksmt.sort.KArrayNSort
import org.ksmt.sort.KArraySort
import org.ksmt.sort.KArraySortBase
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.KBv16Sort
import org.ksmt.sort.KBv1Sort
import org.ksmt.sort.KBv32Sort
import org.ksmt.sort.KBv64Sort
import org.ksmt.sort.KBv8Sort
import org.ksmt.sort.KBvCustomSizeSort
import org.ksmt.sort.KBvSort
import org.ksmt.sort.KFp128Sort
import org.ksmt.sort.KFp16Sort
import org.ksmt.sort.KFp32Sort
import org.ksmt.sort.KFp64Sort
import org.ksmt.sort.KFpCustomSizeSort
import org.ksmt.sort.KFpRoundingModeSort
import org.ksmt.sort.KFpSort
import org.ksmt.sort.KIntSort
import org.ksmt.sort.KRealSort
import org.ksmt.sort.KSort
import org.ksmt.sort.KUninterpretedSort
import org.ksmt.utils.FpUtils.biasFpExponent
import org.ksmt.utils.FpUtils.fpInfExponentBiased
import org.ksmt.utils.FpUtils.fpInfSignificand
import org.ksmt.utils.FpUtils.fpNaNExponentBiased
import org.ksmt.utils.FpUtils.fpNaNSignificand
import org.ksmt.utils.FpUtils.fpZeroExponentBiased
import org.ksmt.utils.FpUtils.fpZeroSignificand
import org.ksmt.utils.FpUtils.isNaN
import org.ksmt.utils.FpUtils.unbiasFpExponent
import org.ksmt.utils.booleanSignBit
import org.ksmt.utils.cast
import org.ksmt.utils.extractExponent
import org.ksmt.utils.extractSignificand
import org.ksmt.utils.getHalfPrecisionExponent
import org.ksmt.utils.halfPrecisionSignificand
import org.ksmt.utils.normalizeValue
import org.ksmt.utils.signBit
import org.ksmt.utils.toBigInteger
import org.ksmt.utils.toULongValue
import org.ksmt.utils.toUnsignedBigInteger
import org.ksmt.utils.uncheckedCast
import java.lang.Double.longBitsToDouble
import java.lang.Float.intBitsToFloat
import java.math.BigInteger

@Suppress("TooManyFunctions", "LargeClass", "unused")
open class KContext(
    private val operationMode: OperationMode = CONCURRENT,
    private val astManagementMode: AstManagementMode = GC,
    private val simplificationMode: SimplificationMode = SIMPLIFY
) : AutoCloseable {

    /**
     * Allow or disallow concurrent execution of
     * [KContext] operations (e.g. expression creation)
     *
     * [SINGLE_THREAD] --- disallow concurrent execution and maximize
     * performance within a single thread.
     * [CONCURRENT] --- allow concurrent execution with about 10% lower
     * single thread performance compared to the [SINGLE_THREAD] mode.
     * */
    enum class OperationMode {
        SINGLE_THREAD,
        CONCURRENT
    }

    /**
     * Enable or disable Garbage Collection of unused KSMT expressions.
     *
     * [NO_GC] --- all managed KSMT expressions will be kept in memory
     * as long as their [KContext] is reachable.
     * [GC] --- allow managed KSMT expressions to be garbage collected when they become unreachable.
     * Enabling this option will result in a performance degradation of about 10%.
     *
     * Note: using [GC] only makes sense when working with a long-lived [KContext].
     * */
    enum class AstManagementMode {
        GC,
        NO_GC
    }

    /**
     * Enable or disable expression simplification during expression creation.
     *
     * [SIMPLIFY] --- apply cheap simplifications during expression creation.
     * [NO_SIMPLIFY] --- create expressions `as is` without any simplifications.
     * */
    enum class SimplificationMode {
        SIMPLIFY,
        NO_SIMPLIFY
    }

    /**
     * KContext and all created expressions are only valid as long as
     * the context is active (not closed).
     * @see ensureContextActive
     * */
    var isActive: Boolean = true
        private set

    override fun close() {
        isActive = false
    }

    /*
    * sorts
    * */

    val boolSort: KBoolSort = KBoolSort(this)

    fun mkBoolSort(): KBoolSort = boolSort

    private val arraySortCache = mkCache<KArraySort<*, *>, KArraySort<*, *>>(operationMode)
    private val array2SortCache = mkCache<KArray2Sort<*, *, *>, KArray2Sort<*, *, *>>(operationMode)
    private val array3SortCache = mkCache<KArray3Sort<*, *, *, *>, KArray3Sort<*, *, *, *>>(operationMode)
    private val arrayNSortCache = mkCache<KArrayNSort<*>, KArrayNSort<*>>(operationMode)

    fun <D : KSort, R : KSort> mkArraySort(domain: D, range: R): KArraySort<D, R> =
        ensureContextActive {
            ensureContextMatch(domain, range)
            val sort = KArraySort(this, domain, range)
            (arraySortCache.putIfAbsent(sort, sort) ?: sort).uncheckedCast()
        }

    fun <D0 : KSort, D1 : KSort, R : KSort> mkArraySort(domain0: D0, domain1: D1, range: R): KArray2Sort<D0, D1, R> =
        ensureContextActive {
            ensureContextMatch(domain0, domain1, range)
            val sort = KArray2Sort(this, domain0, domain1, range)
            (array2SortCache.putIfAbsent(sort, sort) ?: sort).uncheckedCast()
        }

    fun <D0 : KSort, D1 : KSort, D2 : KSort, R : KSort> mkArraySort(
        domain0: D0, domain1: D1, domain2: D2, range: R
    ): KArray3Sort<D0, D1, D2, R> =
        ensureContextActive {
            ensureContextMatch(domain0, domain1, domain2, range)
            val sort = KArray3Sort(this, domain0, domain1, domain2, range)
            (array3SortCache.putIfAbsent(sort, sort) ?: sort).uncheckedCast()
        }

    fun <R : KSort> mkArrayNSort(domain: List<KSort>, range: R): KArrayNSort<R> =
        ensureContextActive {
            ensureContextMatch(range)
            ensureContextMatch(domain)

            val sort = KArrayNSort(this, domain, range)
            (arrayNSortCache.putIfAbsent(sort, sort) ?: sort).uncheckedCast()
        }

    private val intSortCache by lazy {
        ensureContextActive { KIntSort(this) }
    }

    fun mkIntSort(): KIntSort = intSortCache

    private val realSortCache by lazy {
        ensureContextActive { KRealSort(this) }
    }

    fun mkRealSort(): KRealSort = realSortCache

    // bit-vec
    private val bv1SortCache: KBv1Sort by lazy { KBv1Sort(this) }
    private val bv8SortCache: KBv8Sort by lazy { KBv8Sort(this) }
    private val bv16SortCache: KBv16Sort by lazy { KBv16Sort(this) }
    private val bv32SortCache: KBv32Sort by lazy { KBv32Sort(this) }
    private val bv64SortCache: KBv64Sort by lazy { KBv64Sort(this) }
    private val bvCustomSizeSortCache = mkCache<UInt, KBvSort>(operationMode)

    fun mkBv1Sort(): KBv1Sort = bv1SortCache
    fun mkBv8Sort(): KBv8Sort = bv8SortCache
    fun mkBv16Sort(): KBv16Sort = bv16SortCache
    fun mkBv32Sort(): KBv32Sort = bv32SortCache
    fun mkBv64Sort(): KBv64Sort = bv64SortCache

    fun mkBvSort(sizeBits: UInt): KBvSort = ensureContextActive {
        when (sizeBits.toInt()) {
            1 -> mkBv1Sort()
            Byte.SIZE_BITS -> mkBv8Sort()
            Short.SIZE_BITS -> mkBv16Sort()
            Int.SIZE_BITS -> mkBv32Sort()
            Long.SIZE_BITS -> mkBv64Sort()
            else -> bvCustomSizeSortCache.getOrPut(sizeBits) {
                KBvCustomSizeSort(this, sizeBits)
            }
        }
    }

    fun mkUninterpretedSort(name: String): KUninterpretedSort =
        ensureContextActive {
            KUninterpretedSort(name, this)
        }

    // floating point
    private val fp16SortCache: KFp16Sort by lazy { KFp16Sort(this) }
    private val fp32SortCache: KFp32Sort by lazy { KFp32Sort(this) }
    private val fp64SortCache: KFp64Sort by lazy { KFp64Sort(this) }
    private val fp128SortCache: KFp128Sort by lazy { KFp128Sort(this) }
    private val fpCustomSizeSortCache = mkCache<Pair<UInt, UInt>, KFpSort>(operationMode)

    fun mkFp16Sort(): KFp16Sort = fp16SortCache
    fun mkFp32Sort(): KFp32Sort = fp32SortCache
    fun mkFp64Sort(): KFp64Sort = fp64SortCache
    fun mkFp128Sort(): KFp128Sort = fp128SortCache

    fun mkFpSort(exponentBits: UInt, significandBits: UInt): KFpSort =
        ensureContextActive {
            val eb = exponentBits
            val sb = significandBits
            when {
                eb == KFp16Sort.exponentBits && sb == KFp16Sort.significandBits -> mkFp16Sort()
                eb == KFp32Sort.exponentBits && sb == KFp32Sort.significandBits -> mkFp32Sort()
                eb == KFp64Sort.exponentBits && sb == KFp64Sort.significandBits -> mkFp64Sort()
                eb == KFp128Sort.exponentBits && sb == KFp128Sort.significandBits -> mkFp128Sort()
                else -> fpCustomSizeSortCache.getOrPut(eb to sb) {
                    KFpCustomSizeSort(this, eb, sb)
                }
            }
        }

    private val roundingModeSortCache by lazy {
        ensureContextActive { KFpRoundingModeSort(this) }
    }

    fun mkFpRoundingModeSort(): KFpRoundingModeSort = roundingModeSortCache

    // utils
    val intSort: KIntSort
        get() = mkIntSort()

    val realSort: KRealSort
        get() = mkRealSort()

    val bv1Sort: KBv1Sort
        get() = mkBv1Sort()

    val bv8Sort: KBv8Sort
        get() = mkBv8Sort()

    val bv16Sort: KBv16Sort
        get() = mkBv16Sort()

    val bv32Sort: KBv32Sort
        get() = mkBv32Sort()

    val bv64Sort: KBv64Sort
        get() = mkBv64Sort()

    val fp16Sort: KFp16Sort
        get() = mkFp16Sort()

    val fp32Sort: KFp32Sort
        get() = mkFp32Sort()

    val fp64Sort: KFp64Sort
        get() = mkFp64Sort()

    /*
    * expressions
    * */
    // bool
    private val andNaryCache = mkAstInterner<KAndNaryExpr>()
    private val andBinaryCache = mkAstInterner<KAndBinaryExpr>()

    open fun mkAnd(args: List<KExpr<KBoolSort>>): KExpr<KBoolSort> =
        mkSimplified(args, KContext::simplifyAnd, ::mkAndNoSimplify)

    open fun mkAndNoFlat(args: List<KExpr<KBoolSort>>): KExpr<KBoolSort> =
        mkSimplified(args, KContext::simplifyAndNoFlat, ::mkAndNoSimplify)

    open fun mkAnd(lhs: KExpr<KBoolSort>, rhs: KExpr<KBoolSort>): KExpr<KBoolSort> =
        mkSimplified(lhs, rhs, KContext::simplifyAnd, ::mkAndNoSimplify)

    open fun mkAndNoFlat(lhs: KExpr<KBoolSort>, rhs: KExpr<KBoolSort>): KExpr<KBoolSort> =
        mkSimplified(lhs, rhs, KContext::simplifyAndNoFlat, ::mkAndNoSimplify)

    open fun mkAndNoSimplify(args: List<KExpr<KBoolSort>>): KAndExpr =
        if (args.size == 2) {
            mkAndNoSimplify(args.first(), args.last())
        } else {
            andNaryCache.createIfContextActive {
                ensureContextMatch(args)
                KAndNaryExpr(this, args)
            }
        }

    open fun mkAndNoSimplify(lhs: KExpr<KBoolSort>, rhs: KExpr<KBoolSort>): KAndBinaryExpr =
        andBinaryCache.createIfContextActive {
            ensureContextMatch(lhs, rhs)
            KAndBinaryExpr(this, lhs, rhs)
        }

    private val orNaryCache = mkAstInterner<KOrNaryExpr>()
    private val orBinaryCache = mkAstInterner<KOrBinaryExpr>()

    open fun mkOr(args: List<KExpr<KBoolSort>>): KExpr<KBoolSort> =
        mkSimplified(args, KContext::simplifyOr, ::mkOrNoSimplify)

    open fun mkOrNoFlat(args: List<KExpr<KBoolSort>>): KExpr<KBoolSort> =
        mkSimplified(args, KContext::simplifyOrNoFlat, ::mkOrNoSimplify)

    open fun mkOr(lhs: KExpr<KBoolSort>, rhs: KExpr<KBoolSort>): KExpr<KBoolSort> =
        mkSimplified(lhs, rhs, KContext::simplifyOr, ::mkOrNoSimplify)

    open fun mkOrNoFlat(lhs: KExpr<KBoolSort>, rhs: KExpr<KBoolSort>): KExpr<KBoolSort> =
        mkSimplified(lhs, rhs, KContext::simplifyOrNoFlat, ::mkOrNoSimplify)

    open fun mkOrNoSimplify(args: List<KExpr<KBoolSort>>): KOrExpr =
        if (args.size == 2) {
            mkOrNoSimplify(args.first(), args.last())
        } else {
            orNaryCache.createIfContextActive {
                ensureContextMatch(args)
                KOrNaryExpr(this, args)
            }
        }

    open fun mkOrNoSimplify(lhs: KExpr<KBoolSort>, rhs: KExpr<KBoolSort>): KOrBinaryExpr =
        orBinaryCache.createIfContextActive {
            ensureContextMatch(lhs, rhs)
            KOrBinaryExpr(this, lhs, rhs)
        }

    private val notCache = mkAstInterner<KNotExpr>()

    open fun mkNot(arg: KExpr<KBoolSort>): KExpr<KBoolSort> =
        mkSimplified(arg, KContext::simplifyNot, ::mkNotNoSimplify)

    open fun mkNotNoSimplify(arg: KExpr<KBoolSort>): KNotExpr = notCache.createIfContextActive {
        ensureContextMatch(arg)
        KNotExpr(this, arg)
    }

    private val impliesCache = mkAstInterner<KImpliesExpr>()

    open fun mkImplies(p: KExpr<KBoolSort>, q: KExpr<KBoolSort>): KExpr<KBoolSort> =
        mkSimplified(p, q, KContext::simplifyImplies, ::mkImpliesNoSimplify)

    open fun mkImpliesNoSimplify(
        p: KExpr<KBoolSort>,
        q: KExpr<KBoolSort>
    ): KImpliesExpr = impliesCache.createIfContextActive {
        ensureContextMatch(p, q)
        KImpliesExpr(this, p, q)
    }

    private val xorCache = mkAstInterner<KXorExpr>()

    open fun mkXor(a: KExpr<KBoolSort>, b: KExpr<KBoolSort>): KExpr<KBoolSort> =
        mkSimplified(a, b, KContext::simplifyXor, ::mkXorNoSimplify)

    open fun mkXorNoSimplify(a: KExpr<KBoolSort>, b: KExpr<KBoolSort>): KXorExpr =
        xorCache.createIfContextActive {
            ensureContextMatch(a, b)
            KXorExpr(this, a, b)
        }

    val trueExpr: KTrue = KTrue(this)
    val falseExpr: KFalse = KFalse(this)

    fun mkTrue(): KTrue = trueExpr
    fun mkFalse(): KFalse = falseExpr

    private val eqCache = mkAstInterner<KEqExpr<out KSort>>()

    open fun <T : KSort> mkEq(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<KBoolSort> =
        mkSimplified(lhs, rhs, KContext::simplifyEq, ::mkEqNoSimplify)

    open fun <T : KSort> mkEqNoSimplify(lhs: KExpr<T>, rhs: KExpr<T>): KEqExpr<T> =
        eqCache.createIfContextActive {
            ensureContextMatch(lhs, rhs)
            KEqExpr(this, lhs, rhs)
        }.cast()

    private val distinctCache = mkAstInterner<KDistinctExpr<out KSort>>()

    open fun <T : KSort> mkDistinct(args: List<KExpr<T>>): KExpr<KBoolSort> =
        mkSimplified(args, KContext::simplifyDistinct, ::mkDistinctNoSimplify)

    open fun <T : KSort> mkDistinctNoSimplify(args: List<KExpr<T>>): KDistinctExpr<T> =
        distinctCache.createIfContextActive {
            ensureContextMatch(args)
            KDistinctExpr(this, args)
        }.cast()

    private val iteCache = mkAstInterner<KIteExpr<out KSort>>()

    open fun <T : KSort> mkIte(
        condition: KExpr<KBoolSort>,
        trueBranch: KExpr<T>,
        falseBranch: KExpr<T>
    ): KExpr<T> = mkSimplified(condition, trueBranch, falseBranch, KContext::simplifyIte, ::mkIteNoSimplify)

    open fun <T : KSort> mkIteNoSimplify(
        condition: KExpr<KBoolSort>,
        trueBranch: KExpr<T>,
        falseBranch: KExpr<T>
    ): KIteExpr<T> = iteCache.createIfContextActive {
        ensureContextMatch(condition, trueBranch, falseBranch)
        KIteExpr(this, condition, trueBranch, falseBranch)
    }.cast()

    infix fun <T : KSort> KExpr<T>.eq(other: KExpr<T>) = mkEq(this, other)
    operator fun KExpr<KBoolSort>.not() = mkNot(this)
    infix fun KExpr<KBoolSort>.and(other: KExpr<KBoolSort>) = mkAnd(this, other)
    infix fun KExpr<KBoolSort>.or(other: KExpr<KBoolSort>) = mkOr(this, other)
    infix fun KExpr<KBoolSort>.xor(other: KExpr<KBoolSort>) = mkXor(this, other)
    infix fun KExpr<KBoolSort>.implies(other: KExpr<KBoolSort>) = mkImplies(this, other)
    infix fun <T : KSort> KExpr<T>.neq(other: KExpr<T>) = !(this eq other)

    fun mkAnd(vararg args: KExpr<KBoolSort>): KExpr<KBoolSort> = mkAnd(args.toList())
    fun mkOr(vararg args: KExpr<KBoolSort>): KExpr<KBoolSort> = mkOr(args.toList())

    fun mkBool(value: Boolean): KExpr<KBoolSort> =
        if (value) trueExpr else falseExpr

    val Boolean.expr: KExpr<KBoolSort>
        get() = mkBool(this)

    // functions
    /*
    * For builtin declarations e.g. KAndDecl, mkApp must return the same object as a corresponding builder.
    * For example, mkApp(KAndDecl, a, b) and mkAnd(a, b) must end up with the same KAndExpr object.
    * To achieve such behaviour we override apply for all builtin declarations.
    */
    open fun <T : KSort> mkApp(decl: KDecl<T>, args: List<KExpr<*>>) = with(decl) { apply(args) }

    private val functionAppCache = mkAstInterner<KFunctionApp<out KSort>>()

    internal fun <T : KSort> mkFunctionApp(decl: KDecl<T>, args: List<KExpr<*>>): KApp<T, *> =
        if (args.isEmpty()) {
            mkConstApp(decl)
        } else {
            functionAppCache.createIfContextActive {
                ensureContextMatch(decl)
                ensureContextMatch(args)
                KFunctionApp(this, decl, args.uncheckedCast())
            }.cast()
        }

    private val constAppCache = mkAstInterner<KConst<out KSort>>()

    open fun <T : KSort> mkConstApp(decl: KDecl<T>): KConst<T> = constAppCache.createIfContextActive {
        ensureContextMatch(decl)
        KConst(this, decl)
    }.cast()

    fun <T : KSort> mkConst(name: String, sort: T): KApp<T, *> = with(mkConstDecl(name, sort)) { apply() }

    fun <T : KSort> mkFreshConst(name: String, sort: T): KApp<T, *> = with(mkFreshConstDecl(name, sort)) { apply() }

    // array
    private val arrayStoreCache = mkAstInterner<KArrayStore<out KSort, out KSort>>()
    private val array2StoreCache = mkAstInterner<KArray2Store<out KSort, out KSort, out KSort>>()
    private val array3StoreCache = mkAstInterner<KArray3Store<out KSort, out KSort, out KSort, out KSort>>()
    private val arrayNStoreCache = mkAstInterner<KArrayNStore<out KSort>>()

    open fun <D : KSort, R : KSort> mkArrayStore(
        array: KExpr<KArraySort<D, R>>,
        index: KExpr<D>,
        value: KExpr<R>
    ): KExpr<KArraySort<D, R>> =
        mkSimplified(array, index, value, KContext::simplifyArrayStore, ::mkArrayStoreNoSimplify)

    open fun <D0 : KSort, D1 : KSort, R : KSort> mkArrayStore(
        array: KExpr<KArray2Sort<D0, D1, R>>,
        index0: KExpr<D0>,
        index1: KExpr<D1>,
        value: KExpr<R>
    ): KExpr<KArray2Sort<D0, D1, R>> =
        mkSimplified(array, index0, index1, value, KContext::simplifyArrayStore, ::mkArrayStoreNoSimplify)

    open fun <D0 : KSort, D1 : KSort, D2 : KSort, R : KSort> mkArrayStore(
        array: KExpr<KArray3Sort<D0, D1, D2, R>>,
        index0: KExpr<D0>,
        index1: KExpr<D1>,
        index2: KExpr<D2>,
        value: KExpr<R>
    ): KExpr<KArray3Sort<D0, D1, D2, R>> =
        mkSimplified(array, index0, index1, index2, value, KContext::simplifyArrayStore, ::mkArrayStoreNoSimplify)

    open fun <R : KSort> mkArrayNStore(
        array: KExpr<KArrayNSort<R>>,
        indices: List<KExpr<*>>,
        value: KExpr<R>
    ): KExpr<KArrayNSort<R>> =
        mkSimplified(array, indices, value, KContext::simplifyArrayNStore, ::mkArrayNStoreNoSimplify)

    open fun <D : KSort, R : KSort> mkArrayStoreNoSimplify(
        array: KExpr<KArraySort<D, R>>,
        index: KExpr<D>,
        value: KExpr<R>
    ): KArrayStore<D, R> = arrayStoreCache.createIfContextActive {
        ensureContextMatch(array, index, value)
        KArrayStore(this, array, index, value)
    }.cast()

    open fun <D0 : KSort, D1 : KSort, R : KSort> mkArrayStoreNoSimplify(
        array: KExpr<KArray2Sort<D0, D1, R>>,
        index0: KExpr<D0>,
        index1: KExpr<D1>,
        value: KExpr<R>
    ): KArray2Store<D0, D1, R> = array2StoreCache.createIfContextActive {
        ensureContextMatch(array, index0, index1, value)
        KArray2Store(this, array, index0, index1, value)
    }.cast()

    open fun <D0 : KSort, D1 : KSort, D2 : KSort, R : KSort> mkArrayStoreNoSimplify(
        array: KExpr<KArray3Sort<D0, D1, D2, R>>,
        index0: KExpr<D0>,
        index1: KExpr<D1>,
        index2: KExpr<D2>,
        value: KExpr<R>
    ): KArray3Store<D0, D1, D2, R> = array3StoreCache.createIfContextActive {
        ensureContextMatch(array, index0, index1, index2, value)
        KArray3Store(this, array, index0, index1, index2, value)
    }.cast()

    open fun <R : KSort> mkArrayNStoreNoSimplify(
        array: KExpr<KArrayNSort<R>>,
        indices: List<KExpr<*>>,
        value: KExpr<R>
    ): KArrayNStore<R> = arrayNStoreCache.createIfContextActive {
        ensureContextMatch(indices)
        ensureContextMatch(array, value)

        KArrayNStore(this, array, indices.uncheckedCast(), value)
    }.cast()

    private val arraySelectCache = mkAstInterner<KArraySelect<out KSort, out KSort>>()
    private val array2SelectCache = mkAstInterner<KArray2Select<out KSort, out KSort, out KSort>>()
    private val array3SelectCache = mkAstInterner<KArray3Select<out KSort, out KSort, out KSort, out KSort>>()
    private val arrayNSelectCache = mkAstInterner<KArrayNSelect<out KSort>>()

    open fun <D : KSort, R : KSort> mkArraySelect(
        array: KExpr<KArraySort<D, R>>,
        index: KExpr<D>
    ): KExpr<R> = mkSimplified(array, index, KContext::simplifyArraySelect, ::mkArraySelectNoSimplify)

    open fun <D0 : KSort, D1 : KSort, R : KSort> mkArraySelect(
        array: KExpr<KArray2Sort<D0, D1, R>>,
        index0: KExpr<D0>,
        index1: KExpr<D1>
    ): KExpr<R> = mkSimplified(array, index0, index1, KContext::simplifyArraySelect, ::mkArraySelectNoSimplify)

    open fun <D0 : KSort, D1 : KSort, D2 : KSort, R : KSort> mkArraySelect(
        array: KExpr<KArray3Sort<D0, D1, D2, R>>,
        index0: KExpr<D0>,
        index1: KExpr<D1>,
        index2: KExpr<D2>
    ): KExpr<R> = mkSimplified(array, index0, index1, index2, KContext::simplifyArraySelect, ::mkArraySelectNoSimplify)

    open fun <R : KSort> mkArrayNSelect(
        array: KExpr<KArrayNSort<R>>,
        indices: List<KExpr<*>>
    ): KExpr<R> = mkSimplified(array, indices, KContext::simplifyArrayNSelect, ::mkArrayNSelectNoSimplify)

    open fun <D : KSort, R : KSort> mkArraySelectNoSimplify(
        array: KExpr<KArraySort<D, R>>,
        index: KExpr<D>
    ): KArraySelect<D, R> = arraySelectCache.createIfContextActive {
        ensureContextMatch(array, index)
        KArraySelect(this, array, index)
    }.cast()

    open fun <D0 : KSort, D1 : KSort, R : KSort> mkArraySelectNoSimplify(
        array: KExpr<KArray2Sort<D0, D1, R>>,
        index0: KExpr<D0>,
        index1: KExpr<D1>
    ): KArray2Select<D0, D1, R> = array2SelectCache.createIfContextActive {
        ensureContextMatch(array, index0, index1)
        KArray2Select(this, array, index0, index1)
    }.cast()

    open fun <D0 : KSort, D1 : KSort, D2 : KSort, R : KSort> mkArraySelectNoSimplify(
        array: KExpr<KArray3Sort<D0, D1, D2, R>>,
        index0: KExpr<D0>,
        index1: KExpr<D1>,
        index2: KExpr<D2>
    ): KArray3Select<D0, D1, D2, R> = array3SelectCache.createIfContextActive {
        ensureContextMatch(array, index0, index1, index2)
        KArray3Select(this, array, index0, index1, index2)
    }.cast()

    open fun <R : KSort> mkArrayNSelectNoSimplify(
        array: KExpr<KArrayNSort<R>>,
        indices: List<KExpr<*>>
    ): KArrayNSelect<R> = arrayNSelectCache.createIfContextActive {
        ensureContextMatch(array)
        ensureContextMatch(indices)

        KArrayNSelect(this, array, indices.uncheckedCast())
    }.cast()

    private val arrayConstCache = mkAstInterner<KArrayConst<out KArraySortBase<out KSort>, out KSort>>()

    open fun <A : KArraySortBase<R>, R : KSort> mkArrayConst(
        arraySort: A,
        value: KExpr<R>
    ): KArrayConst<A, R> = arrayConstCache.createIfContextActive {
        ensureContextMatch(arraySort, value)
        KArrayConst(this, arraySort, value)
    }.cast()

    private val functionAsArrayCache = mkAstInterner<KFunctionAsArray<out KArraySortBase<out KSort>, out KSort>>()

    open fun <A : KArraySortBase<R>, R : KSort> mkFunctionAsArray(
        sort: A, function: KFuncDecl<R>
    ): KFunctionAsArray<A, R> =
        functionAsArrayCache.createIfContextActive {
            ensureContextMatch(function)
            KFunctionAsArray(this, sort, function)
        }.cast()

    private val arrayLambdaCache = mkAstInterner<KArrayLambda<out KSort, out KSort>>()
    private val array2LambdaCache = mkAstInterner<KArray2Lambda<out KSort, out KSort, out KSort>>()
    private val array3LambdaCache = mkAstInterner<KArray3Lambda<out KSort, out KSort, out KSort, out KSort>>()
    private val arrayNLambdaCache = mkAstInterner<KArrayNLambda<out KSort>>()

    open fun <D : KSort, R : KSort> mkArrayLambda(
        indexVar: KDecl<D>, body: KExpr<R>
    ): KArrayLambda<D, R> = arrayLambdaCache.createIfContextActive {
        ensureContextMatch(indexVar, body)
        KArrayLambda(this, indexVar, body)
    }.cast()

    open fun <D0 : KSort, D1 : KSort, R : KSort> mkArrayLambda(
        indexVar0: KDecl<D0>, indexVar1: KDecl<D1>, body: KExpr<R>
    ): KArray2Lambda<D0, D1, R> = array2LambdaCache.createIfContextActive {
        ensureContextMatch(indexVar0, indexVar1, body)
        KArray2Lambda(this, indexVar0, indexVar1, body)
    }.cast()

    open fun <D0 : KSort, D1 : KSort, D2 : KSort, R : KSort> mkArrayLambda(
        indexVar0: KDecl<D0>, indexVar1: KDecl<D1>, indexVar2: KDecl<D2>, body: KExpr<R>
    ): KArray3Lambda<D0, D1, D2, R> = array3LambdaCache.createIfContextActive {
        ensureContextMatch(indexVar0, indexVar1, indexVar2, body)
        KArray3Lambda(this, indexVar0, indexVar1, indexVar2, body)
    }.cast()

    open fun <R : KSort> mkArrayNLambda(
        indices: List<KDecl<*>>, body: KExpr<R>
    ): KArrayNLambda<R> = arrayNLambdaCache.createIfContextActive {
        ensureContextMatch(indices)
        ensureContextMatch(body)

        KArrayNLambda(this, indices, body)
    }.cast()

    fun <D : KSort, R : KSort> KExpr<KArraySort<D, R>>.store(index: KExpr<D>, value: KExpr<R>) =
        mkArrayStore(this, index, value)

    fun <D0 : KSort, D1 : KSort, R : KSort> KExpr<KArray2Sort<D0, D1, R>>.store(
        index0: KExpr<D0>, index1: KExpr<D1>, value: KExpr<R>
    ) = mkArrayStore(this, index0, index1, value)

    fun <D0 : KSort, D1 : KSort, D2 : KSort, R : KSort> KExpr<KArray3Sort<D0, D1, D2, R>>.store(
        index0: KExpr<D0>, index1: KExpr<D1>, index2: KExpr<D2>, value: KExpr<R>
    ) = mkArrayStore(this, index0, index1, index2, value)

    fun <D : KSort, R : KSort> KExpr<KArraySort<D, R>>.select(index: KExpr<D>) = mkArraySelect(this, index)

    fun <D0 : KSort, D1 : KSort, R : KSort> KExpr<KArray2Sort<D0, D1, R>>.select(
        index0: KExpr<D0>,
        index1: KExpr<D1>
    ) = mkArraySelect(this, index0, index1)

    fun <D0 : KSort, D1 : KSort, D2 : KSort, R : KSort> KExpr<KArray3Sort<D0, D1, D2, R>>.select(
        index0: KExpr<D0>,
        index1: KExpr<D1>,
        index2: KExpr<D2>
    ) = mkArraySelect(this, index0, index1, index2)

    // arith
    private val arithAddCache = mkAstInterner<KAddArithExpr<out KArithSort>>()

    open fun <T : KArithSort> mkArithAdd(args: List<KExpr<T>>): KExpr<T> =
        mkSimplified(args, KContext::simplifyArithAdd, ::mkArithAddNoSimplify)

    open fun <T : KArithSort> mkArithAddNoSimplify(args: List<KExpr<T>>): KAddArithExpr<T> =
        arithAddCache.createIfContextActive {
            ensureContextMatch(args)
            KAddArithExpr(this, args)
        }.cast()

    private val arithMulCache = mkAstInterner<KMulArithExpr<out KArithSort>>()

    open fun <T : KArithSort> mkArithMul(args: List<KExpr<T>>): KExpr<T> =
        mkSimplified(args, KContext::simplifyArithMul, ::mkArithMulNoSimplify)

    open fun <T : KArithSort> mkArithMulNoSimplify(args: List<KExpr<T>>): KMulArithExpr<T> =
        arithMulCache.createIfContextActive {
            ensureContextMatch(args)
            KMulArithExpr(this, args)
        }.cast()

    private val arithSubCache = mkAstInterner<KSubArithExpr<out KArithSort>>()

    open fun <T : KArithSort> mkArithSub(args: List<KExpr<T>>): KExpr<T> =
        mkSimplified(args, KContext::simplifyArithSub, ::mkArithSubNoSimplify)

    open fun <T : KArithSort> mkArithSubNoSimplify(args: List<KExpr<T>>): KSubArithExpr<T> =
        arithSubCache.createIfContextActive {
            ensureContextMatch(args)
            KSubArithExpr(this, args)
        }.cast()

    @Suppress("MemberVisibilityCanBePrivate")
    fun <T : KArithSort> mkArithAdd(vararg args: KExpr<T>) = mkArithAdd(args.toList())

    @Suppress("MemberVisibilityCanBePrivate")
    fun <T : KArithSort> mkArithMul(vararg args: KExpr<T>) = mkArithMul(args.toList())

    @Suppress("MemberVisibilityCanBePrivate")
    fun <T : KArithSort> mkArithSub(vararg args: KExpr<T>) = mkArithSub(args.toList())

    private val arithUnaryMinusCache = mkAstInterner<KUnaryMinusArithExpr<out KArithSort>>()

    open fun <T : KArithSort> mkArithUnaryMinus(arg: KExpr<T>): KExpr<T> =
        mkSimplified(arg, KContext::simplifyArithUnaryMinus, ::mkArithUnaryMinusNoSimplify)

    open fun <T : KArithSort> mkArithUnaryMinusNoSimplify(arg: KExpr<T>): KUnaryMinusArithExpr<T> =
        arithUnaryMinusCache.createIfContextActive {
            ensureContextMatch(arg)
            KUnaryMinusArithExpr(this, arg)
        }.cast()

    private val arithDivCache = mkAstInterner<KDivArithExpr<out KArithSort>>()

    open fun <T : KArithSort> mkArithDiv(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<T> =
        mkSimplified(lhs, rhs, KContext::simplifyArithDiv, ::mkArithDivNoSimplify)

    open fun <T : KArithSort> mkArithDivNoSimplify(lhs: KExpr<T>, rhs: KExpr<T>): KDivArithExpr<T> =
        arithDivCache.createIfContextActive {
            ensureContextMatch(lhs, rhs)
            KDivArithExpr(this, lhs, rhs)
        }.cast()

    private val arithPowerCache = mkAstInterner<KPowerArithExpr<out KArithSort>>()

    open fun <T : KArithSort> mkArithPower(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<T> =
        mkSimplified(lhs, rhs, KContext::simplifyArithPower, ::mkArithPowerNoSimplify)

    open fun <T : KArithSort> mkArithPowerNoSimplify(lhs: KExpr<T>, rhs: KExpr<T>): KPowerArithExpr<T> =
        arithPowerCache.createIfContextActive {
            ensureContextMatch(lhs, rhs)
            KPowerArithExpr(this, lhs, rhs)
        }.cast()

    private val arithLtCache = mkAstInterner<KLtArithExpr<out KArithSort>>()

    open fun <T : KArithSort> mkArithLt(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<KBoolSort> =
        mkSimplified(lhs, rhs, KContext::simplifyArithLt, ::mkArithLtNoSimplify)

    open fun <T : KArithSort> mkArithLtNoSimplify(lhs: KExpr<T>, rhs: KExpr<T>): KLtArithExpr<T> =
        arithLtCache.createIfContextActive {
            ensureContextMatch(lhs, rhs)
            KLtArithExpr(this, lhs, rhs)
        }.cast()

    private val arithLeCache = mkAstInterner<KLeArithExpr<out KArithSort>>()

    open fun <T : KArithSort> mkArithLe(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<KBoolSort> =
        mkSimplified(lhs, rhs, KContext::simplifyArithLe, ::mkArithLeNoSimplify)

    open fun <T : KArithSort> mkArithLeNoSimplify(lhs: KExpr<T>, rhs: KExpr<T>): KLeArithExpr<T> =
        arithLeCache.createIfContextActive {
            ensureContextMatch(lhs, rhs)
            KLeArithExpr(this, lhs, rhs)
        }.cast()

    private val arithGtCache = mkAstInterner<KGtArithExpr<out KArithSort>>()

    open fun <T : KArithSort> mkArithGt(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<KBoolSort> =
        mkSimplified(lhs, rhs, KContext::simplifyArithGt, ::mkArithGtNoSimplify)

    open fun <T : KArithSort> mkArithGtNoSimplify(lhs: KExpr<T>, rhs: KExpr<T>): KGtArithExpr<T> =
        arithGtCache.createIfContextActive {
            ensureContextMatch(lhs, rhs)
            KGtArithExpr(this, lhs, rhs)
        }.cast()

    private val arithGeCache = mkAstInterner<KGeArithExpr<out KArithSort>>()

    open fun <T : KArithSort> mkArithGe(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<KBoolSort> =
        mkSimplified(lhs, rhs, KContext::simplifyArithGe, ::mkArithGeNoSimplify)

    open fun <T : KArithSort> mkArithGeNoSimplify(lhs: KExpr<T>, rhs: KExpr<T>): KGeArithExpr<T> =
        arithGeCache.createIfContextActive {
            ensureContextMatch(lhs, rhs)
            KGeArithExpr(this, lhs, rhs)
        }.cast()

    operator fun <T : KArithSort> KExpr<T>.plus(other: KExpr<T>) = mkArithAdd(this, other)
    operator fun <T : KArithSort> KExpr<T>.times(other: KExpr<T>) = mkArithMul(this, other)
    operator fun <T : KArithSort> KExpr<T>.minus(other: KExpr<T>) = mkArithSub(this, other)
    operator fun <T : KArithSort> KExpr<T>.unaryMinus() = mkArithUnaryMinus(this)
    operator fun <T : KArithSort> KExpr<T>.div(other: KExpr<T>) = mkArithDiv(this, other)
    fun <T : KArithSort> KExpr<T>.power(other: KExpr<T>) = mkArithPower(this, other)

    infix fun <T : KArithSort> KExpr<T>.lt(other: KExpr<T>) = mkArithLt(this, other)
    infix fun <T : KArithSort> KExpr<T>.le(other: KExpr<T>) = mkArithLe(this, other)
    infix fun <T : KArithSort> KExpr<T>.gt(other: KExpr<T>) = mkArithGt(this, other)
    infix fun <T : KArithSort> KExpr<T>.ge(other: KExpr<T>) = mkArithGe(this, other)

    // integer
    private val intModCache = mkAstInterner<KModIntExpr>()

    open fun mkIntMod(lhs: KExpr<KIntSort>, rhs: KExpr<KIntSort>): KExpr<KIntSort> =
        mkSimplified(lhs, rhs, KContext::simplifyIntMod, ::mkIntModNoSimplify)

    open fun mkIntModNoSimplify(
        lhs: KExpr<KIntSort>,
        rhs: KExpr<KIntSort>
    ): KModIntExpr = intModCache.createIfContextActive {
        ensureContextMatch(lhs, rhs)
        KModIntExpr(this, lhs, rhs)
    }

    private val intRemCache = mkAstInterner<KRemIntExpr>()

    open fun mkIntRem(lhs: KExpr<KIntSort>, rhs: KExpr<KIntSort>): KExpr<KIntSort> =
        mkSimplified(lhs, rhs, KContext::simplifyIntRem, ::mkIntRemNoSimplify)

    open fun mkIntRemNoSimplify(
        lhs: KExpr<KIntSort>,
        rhs: KExpr<KIntSort>
    ): KRemIntExpr = intRemCache.createIfContextActive {
        ensureContextMatch(lhs, rhs)
        KRemIntExpr(this, lhs, rhs)
    }

    private val intToRealCache = mkAstInterner<KToRealIntExpr>()

    open fun mkIntToReal(arg: KExpr<KIntSort>): KExpr<KRealSort> =
        mkSimplified(arg, KContext::simplifyIntToReal, ::mkIntToRealNoSimplify)

    open fun mkIntToRealNoSimplify(arg: KExpr<KIntSort>): KToRealIntExpr = intToRealCache.createIfContextActive {
        ensureContextMatch(arg)
        KToRealIntExpr(this, arg)
    }

    private val int32NumCache = mkAstInterner<KInt32NumExpr>()
    private val int64NumCache = mkAstInterner<KInt64NumExpr>()
    private val intBigNumCache = mkAstInterner<KIntBigNumExpr>()

    fun mkIntNum(value: Int): KIntNumExpr = int32NumCache.createIfContextActive {
        KInt32NumExpr(this, value)
    }

    fun mkIntNum(value: Long): KIntNumExpr = if (value.toInt().toLong() == value) {
        mkIntNum(value.toInt())
    } else {
        int64NumCache.createIfContextActive {
            KInt64NumExpr(this, value)
        }
    }

    fun mkIntNum(value: BigInteger): KIntNumExpr = if (value.toLong().toBigInteger() == value) {
        mkIntNum(value.toLong())
    } else {
        intBigNumCache.createIfContextActive {
            KIntBigNumExpr(this, value)
        }
    }

    fun mkIntNum(value: String): KIntNumExpr =
        mkIntNum(value.toBigInteger())

    infix fun KExpr<KIntSort>.mod(rhs: KExpr<KIntSort>) = mkIntMod(this, rhs)
    infix fun KExpr<KIntSort>.rem(rhs: KExpr<KIntSort>) = mkIntRem(this, rhs)
    fun KExpr<KIntSort>.toRealExpr() = mkIntToReal(this)

    val Int.expr
        get() = mkIntNum(this)
    val Long.expr
        get() = mkIntNum(this)
    val BigInteger.expr
        get() = mkIntNum(this)

    // real
    private val realToIntCache = mkAstInterner<KToIntRealExpr>()

    open fun mkRealToInt(arg: KExpr<KRealSort>): KExpr<KIntSort> =
        mkSimplified(arg, KContext::simplifyRealToInt, ::mkRealToIntNoSimplify)

    open fun mkRealToIntNoSimplify(arg: KExpr<KRealSort>): KToIntRealExpr = realToIntCache.createIfContextActive {
        ensureContextMatch(arg)
        KToIntRealExpr(this, arg)
    }

    private val realIsIntCache = mkAstInterner<KIsIntRealExpr>()

    open fun mkRealIsInt(arg: KExpr<KRealSort>): KExpr<KBoolSort> =
        mkSimplified(arg, KContext::simplifyRealIsInt, ::mkRealIsIntNoSimplify)

    open fun mkRealIsIntNoSimplify(arg: KExpr<KRealSort>): KIsIntRealExpr = realIsIntCache.createIfContextActive {
        ensureContextMatch(arg)
        KIsIntRealExpr(this, arg)
    }

    private val realNumCache = mkAstInterner<KRealNumExpr>()

    fun mkRealNum(numerator: KIntNumExpr, denominator: KIntNumExpr): KRealNumExpr =
        realNumCache.createIfContextActive {
            ensureContextMatch(numerator, denominator)
            KRealNumExpr(this, numerator, denominator)
        }

    @Suppress("MemberVisibilityCanBePrivate")
    fun mkRealNum(numerator: KIntNumExpr) = mkRealNum(numerator, 1.expr)
    fun mkRealNum(numerator: Int) = mkRealNum(mkIntNum(numerator))
    fun mkRealNum(numerator: Int, denominator: Int) = mkRealNum(mkIntNum(numerator), mkIntNum(denominator))
    fun mkRealNum(numerator: Long) = mkRealNum(mkIntNum(numerator))
    fun mkRealNum(numerator: Long, denominator: Long) = mkRealNum(mkIntNum(numerator), mkIntNum(denominator))
    fun mkRealNum(value: String): KRealNumExpr {
        val parts = value.split('/')

        return when (parts.size) {
            1 -> mkRealNum(mkIntNum(parts[0]))
            2 -> mkRealNum(mkIntNum(parts[0]), mkIntNum(parts[1]))
            else -> error("incorrect real num format")
        }
    }

    fun KExpr<KRealSort>.toIntExpr() = mkRealToInt(this)
    fun KExpr<KRealSort>.isIntExpr() = mkRealIsInt(this)

    // bitvectors
    private val bv1Cache = mkAstInterner<KBitVec1Value>()
    private val bv8Cache = mkAstInterner<KBitVec8Value>()
    private val bv16Cache = mkAstInterner<KBitVec16Value>()
    private val bv32Cache = mkAstInterner<KBitVec32Value>()
    private val bv64Cache = mkAstInterner<KBitVec64Value>()
    private val bvCache = mkAstInterner<KBitVecCustomValue>()

    fun mkBv(value: Boolean): KBitVec1Value = bv1Cache.createIfContextActive { KBitVec1Value(this, value) }
    fun mkBv(value: Boolean, sizeBits: UInt): KBitVecValue<KBvSort> {
        val intValue = (if (value) 1 else 0) as Number
        return mkBv(intValue, sizeBits)
    }

    fun <T : KBvSort> mkBv(value: Boolean, sort: T): KBitVecValue<T> =
        mkBv(value, sort.sizeBits).cast()

    fun Boolean.toBv(): KBitVec1Value = mkBv(this)
    fun Boolean.toBv(sizeBits: UInt): KBitVecValue<KBvSort> = mkBv(this, sizeBits)
    fun <T : KBvSort> Boolean.toBv(sort: T): KBitVecValue<T> = mkBv(this, sort)

    fun mkBv(value: Byte): KBitVec8Value = bv8Cache.createIfContextActive { KBitVec8Value(this, value) }
    fun mkBv(value: Byte, sizeBits: UInt): KBitVecValue<KBvSort> = mkBv(value as Number, sizeBits)
    fun <T : KBvSort> mkBv(value: Byte, sort: T): KBitVecValue<T> = mkBv(value as Number, sort)
    fun Byte.toBv(): KBitVec8Value = mkBv(this)
    fun Byte.toBv(sizeBits: UInt): KBitVecValue<KBvSort> = mkBv(this, sizeBits)
    fun <T : KBvSort> Byte.toBv(sort: T): KBitVecValue<T> = mkBv(this, sort)
    fun UByte.toBv(): KBitVec8Value = mkBv(toByte())

    fun mkBv(value: Short): KBitVec16Value = bv16Cache.createIfContextActive { KBitVec16Value(this, value) }
    fun mkBv(value: Short, sizeBits: UInt): KBitVecValue<KBvSort> = mkBv(value as Number, sizeBits)
    fun <T : KBvSort> mkBv(value: Short, sort: T): KBitVecValue<T> = mkBv(value as Number, sort)
    fun Short.toBv(): KBitVec16Value = mkBv(this)
    fun Short.toBv(sizeBits: UInt): KBitVecValue<KBvSort> = mkBv(this, sizeBits)
    fun <T : KBvSort> Short.toBv(sort: T): KBitVecValue<T> = mkBv(this, sort)
    fun UShort.toBv(): KBitVec16Value = mkBv(toShort())

    fun mkBv(value: Int): KBitVec32Value = bv32Cache.createIfContextActive { KBitVec32Value(this, value) }
    fun mkBv(value: Int, sizeBits: UInt): KBitVecValue<KBvSort> = mkBv(value as Number, sizeBits)
    fun <T : KBvSort> mkBv(value: Int, sort: T): KBitVecValue<T> = mkBv(value as Number, sort)
    fun Int.toBv(): KBitVec32Value = mkBv(this)
    fun Int.toBv(sizeBits: UInt): KBitVecValue<KBvSort> = mkBv(this, sizeBits)
    fun <T : KBvSort> Int.toBv(sort: T): KBitVecValue<T> = mkBv(this, sort)
    fun UInt.toBv(): KBitVec32Value = mkBv(toInt())

    fun mkBv(value: Long): KBitVec64Value = bv64Cache.createIfContextActive { KBitVec64Value(this, value) }
    fun mkBv(value: Long, sizeBits: UInt): KBitVecValue<KBvSort> = mkBv(value as Number, sizeBits)
    fun <T : KBvSort> mkBv(value: Long, sort: T): KBitVecValue<T> = mkBv(value as Number, sort)
    fun Long.toBv(): KBitVec64Value = mkBv(this)
    fun Long.toBv(sizeBits: UInt): KBitVecValue<KBvSort> = mkBv(this, sizeBits)
    fun <T : KBvSort> Long.toBv(sort: T): KBitVecValue<T> = mkBv(this, sort)
    fun ULong.toBv(): KBitVec64Value = mkBv(toLong())

    fun mkBv(value: BigInteger, sizeBits: UInt): KBitVecValue<KBvSort> = mkBv(value as Number, sizeBits)
    fun <T : KBvSort> mkBv(value: BigInteger, sort: T): KBitVecValue<T> = mkBv(value as Number, sort)

    /**
     * Constructs a bit vector from the given [value] containing of [sizeBits] bits.
     *
     * Note: if [sizeBits] is less than is required to represent the [value],
     * the last [sizeBits] bits of the [value] will be taken.
     *
     * At the same time, if [sizeBits] is greater than it is required,
     * binary representation of the [value] will be padded from the start with its sign bit.
     */
    private fun mkBv(value: Number, sizeBits: UInt): KBitVecValue<KBvSort> {
        val bigIntValue = value.toBigInteger().normalizeValue(sizeBits)
        return mkBvFromUnsignedBigInteger(bigIntValue, sizeBits)
    }

    private fun <T : KBvSort> mkBv(value: Number, sort: T): KBitVecValue<T> =
        mkBv(value, sort.sizeBits).cast()

    /**
     * Constructs a bit vector from the given [value] containing of [sizeBits] bits.
     * Binary representation of the [value] will be padded from the start with 0.
     */
    private fun mkBvUnsigned(value: Number, sizeBits: UInt): KBitVecValue<KBvSort> {
        val bigIntValue = value.toUnsignedBigInteger().normalizeValue(sizeBits)
        return mkBv(bigIntValue, sizeBits)
    }

    private fun Number.toBv(sizeBits: UInt) = mkBv(this, sizeBits)

    fun mkBv(value: String, sizeBits: UInt): KBitVecValue<KBvSort> =
        mkBv(value.toBigInteger(radix = 2), sizeBits)

    private fun mkBvFromUnsignedBigInteger(
        value: BigInteger,
        sizeBits: UInt
    ): KBitVecValue<KBvSort> {
        require(value.signum() >= 0) {
            "Unsigned value required, but $value provided"
        }
        return when (sizeBits.toInt()) {
            1 -> mkBv(value != BigInteger.ZERO).cast()
            Byte.SIZE_BITS -> mkBv(value.toByte()).cast()
            Short.SIZE_BITS -> mkBv(value.toShort()).cast()
            Int.SIZE_BITS -> mkBv(value.toInt()).cast()
            Long.SIZE_BITS -> mkBv(value.toLong()).cast()
            else -> bvCache.createIfContextActive {
                KBitVecCustomValue(this, value, sizeBits)
            }
        }
    }

    private val bvNotExprCache = mkAstInterner<KBvNotExpr<out KBvSort>>()

    open fun <T : KBvSort> mkBvNotExpr(value: KExpr<T>): KExpr<T> =
        mkSimplified(value, KContext::simplifyBvNotExpr, ::mkBvNotExprNoSimplify)

    open fun <T : KBvSort> mkBvNotExprNoSimplify(value: KExpr<T>): KBvNotExpr<T> =
        bvNotExprCache.createIfContextActive {
            ensureContextMatch(value)
            KBvNotExpr(this, value)
        }.cast()

    private val bvRedAndExprCache = mkAstInterner<KBvReductionAndExpr<out KBvSort>>()

    open fun <T : KBvSort> mkBvReductionAndExpr(value: KExpr<T>): KExpr<KBv1Sort> =
        mkSimplified(value, KContext::simplifyBvReductionAndExpr, ::mkBvReductionAndExprNoSimplify)

    open fun <T : KBvSort> mkBvReductionAndExprNoSimplify(value: KExpr<T>): KBvReductionAndExpr<T> =
        bvRedAndExprCache.createIfContextActive {
            ensureContextMatch(value)
            KBvReductionAndExpr(this, value)
        }.cast()

    fun <T : KBvSort> KExpr<T>.reductionAnd() = mkBvReductionAndExpr(this)

    private val bvRedOrExprCache = mkAstInterner<KBvReductionOrExpr<out KBvSort>>()

    open fun <T : KBvSort> mkBvReductionOrExpr(value: KExpr<T>): KExpr<KBv1Sort> =
        mkSimplified(value, KContext::simplifyBvReductionOrExpr, ::mkBvReductionOrExprNoSimplify)

    open fun <T : KBvSort> mkBvReductionOrExprNoSimplify(value: KExpr<T>): KBvReductionOrExpr<T> =
        bvRedOrExprCache.createIfContextActive {
            ensureContextMatch(value)
            KBvReductionOrExpr(this, value)
        }.cast()

    fun <T : KBvSort> KExpr<T>.reductionOr() = mkBvReductionOrExpr(this)

    private val bvAndExprCache = mkAstInterner<KBvAndExpr<out KBvSort>>()

    open fun <T : KBvSort> mkBvAndExpr(arg0: KExpr<T>, arg1: KExpr<T>): KExpr<T> =
        mkSimplified(arg0, arg1, KContext::simplifyBvAndExpr, ::mkBvAndExprNoSimplify)

    open fun <T : KBvSort> mkBvAndExprNoSimplify(arg0: KExpr<T>, arg1: KExpr<T>): KBvAndExpr<T> =
        bvAndExprCache.createIfContextActive {
            ensureContextMatch(arg0, arg1)
            KBvAndExpr(this, arg0, arg1)
        }.cast()

    private val bvOrExprCache = mkAstInterner<KBvOrExpr<out KBvSort>>()

    open fun <T : KBvSort> mkBvOrExpr(arg0: KExpr<T>, arg1: KExpr<T>): KExpr<T> =
        mkSimplified(arg0, arg1, KContext::simplifyBvOrExpr, ::mkBvOrExprNoSimplify)

    open fun <T : KBvSort> mkBvOrExprNoSimplify(arg0: KExpr<T>, arg1: KExpr<T>): KBvOrExpr<T> =
        bvOrExprCache.createIfContextActive {
            ensureContextMatch(arg0, arg1)
            KBvOrExpr(this, arg0, arg1)
        }.cast()

    private val bvXorExprCache = mkAstInterner<KBvXorExpr<out KBvSort>>()

    open fun <T : KBvSort> mkBvXorExpr(arg0: KExpr<T>, arg1: KExpr<T>): KExpr<T> =
        mkSimplified(arg0, arg1, KContext::simplifyBvXorExpr, ::mkBvXorExprNoSimplify)

    open fun <T : KBvSort> mkBvXorExprNoSimplify(arg0: KExpr<T>, arg1: KExpr<T>): KBvXorExpr<T> =
        bvXorExprCache.createIfContextActive {
            ensureContextMatch(arg0, arg1)
            KBvXorExpr(this, arg0, arg1)
        }.cast()

    private val bvNAndExprCache = mkAstInterner<KBvNAndExpr<out KBvSort>>()

    open fun <T : KBvSort> mkBvNAndExpr(arg0: KExpr<T>, arg1: KExpr<T>): KExpr<T> =
        mkSimplified(arg0, arg1, KContext::simplifyBvNAndExpr, ::mkBvNAndExprNoSimplify)

    open fun <T : KBvSort> mkBvNAndExprNoSimplify(arg0: KExpr<T>, arg1: KExpr<T>): KBvNAndExpr<T> =
        bvNAndExprCache.createIfContextActive {
            ensureContextMatch(arg0, arg1)
            KBvNAndExpr(this, arg0, arg1)
        }.cast()

    private val bvNorExprCache = mkAstInterner<KBvNorExpr<out KBvSort>>()

    open fun <T : KBvSort> mkBvNorExpr(arg0: KExpr<T>, arg1: KExpr<T>): KExpr<T> =
        mkSimplified(arg0, arg1, KContext::simplifyBvNorExpr, ::mkBvNorExprNoSimplify)

    open fun <T : KBvSort> mkBvNorExprNoSimplify(arg0: KExpr<T>, arg1: KExpr<T>): KBvNorExpr<T> =
        bvNorExprCache.createIfContextActive {
            ensureContextMatch(arg0, arg1)
            KBvNorExpr(this, arg0, arg1)
        }.cast()

    private val bvXNorExprCache = mkAstInterner<KBvXNorExpr<out KBvSort>>()

    open fun <T : KBvSort> mkBvXNorExpr(arg0: KExpr<T>, arg1: KExpr<T>): KExpr<T> =
        mkSimplified(arg0, arg1, KContext::simplifyBvXNorExpr, ::mkBvXNorExprNoSimplify)

    open fun <T : KBvSort> mkBvXNorExprNoSimplify(arg0: KExpr<T>, arg1: KExpr<T>): KBvXNorExpr<T> =
        bvXNorExprCache.createIfContextActive {
            ensureContextMatch(arg0, arg1)
            KBvXNorExpr(this, arg0, arg1)
        }.cast()

    private val bvNegationExprCache = mkAstInterner<KBvNegationExpr<out KBvSort>>()

    open fun <T : KBvSort> mkBvNegationExpr(value: KExpr<T>): KExpr<T> =
        mkSimplified(value, KContext::simplifyBvNegationExpr, ::mkBvNegationExprNoSimplify)

    open fun <T : KBvSort> mkBvNegationExprNoSimplify(value: KExpr<T>): KBvNegationExpr<T> =
        bvNegationExprCache.createIfContextActive {
            ensureContextMatch(value)
            KBvNegationExpr(this, value)
        }.cast()

    private val bvAddExprCache = mkAstInterner<KBvAddExpr<out KBvSort>>()

    open fun <T : KBvSort> mkBvAddExpr(arg0: KExpr<T>, arg1: KExpr<T>): KExpr<T> =
        mkSimplified(arg0, arg1, KContext::simplifyBvAddExpr, ::mkBvAddExprNoSimplify)

    open fun <T : KBvSort> mkBvAddExprNoSimplify(arg0: KExpr<T>, arg1: KExpr<T>): KBvAddExpr<T> =
        bvAddExprCache.createIfContextActive {
            ensureContextMatch(arg0, arg1)
            KBvAddExpr(this, arg0, arg1)
        }.cast()

    private val bvSubExprCache = mkAstInterner<KBvSubExpr<out KBvSort>>()

    open fun <T : KBvSort> mkBvSubExpr(arg0: KExpr<T>, arg1: KExpr<T>): KExpr<T> =
        mkSimplified(arg0, arg1, KContext::simplifyBvSubExpr, ::mkBvSubExprNoSimplify)

    open fun <T : KBvSort> mkBvSubExprNoSimplify(arg0: KExpr<T>, arg1: KExpr<T>): KBvSubExpr<T> =
        bvSubExprCache.createIfContextActive {
            ensureContextMatch(arg0, arg1)
            KBvSubExpr(this, arg0, arg1)
        }.cast()

    private val bvMulExprCache = mkAstInterner<KBvMulExpr<out KBvSort>>()

    open fun <T : KBvSort> mkBvMulExpr(arg0: KExpr<T>, arg1: KExpr<T>): KExpr<T> =
        mkSimplified(arg0, arg1, KContext::simplifyBvMulExpr, ::mkBvMulExprNoSimplify)

    open fun <T : KBvSort> mkBvMulExprNoSimplify(arg0: KExpr<T>, arg1: KExpr<T>): KBvMulExpr<T> =
        bvMulExprCache.createIfContextActive {
            ensureContextMatch(arg0, arg1)
            KBvMulExpr(this, arg0, arg1)
        }.cast()

    private val bvUnsignedDivExprCache = mkAstInterner<KBvUnsignedDivExpr<out KBvSort>>()

    open fun <T : KBvSort> mkBvUnsignedDivExpr(arg0: KExpr<T>, arg1: KExpr<T>): KExpr<T> =
        mkSimplified(arg0, arg1, KContext::simplifyBvUnsignedDivExpr, ::mkBvUnsignedDivExprNoSimplify)

    open fun <T : KBvSort> mkBvUnsignedDivExprNoSimplify(arg0: KExpr<T>, arg1: KExpr<T>): KBvUnsignedDivExpr<T> =
        bvUnsignedDivExprCache.createIfContextActive {
            ensureContextMatch(arg0, arg1)
            KBvUnsignedDivExpr(this, arg0, arg1)
        }.cast()

    private val bvSignedDivExprCache = mkAstInterner<KBvSignedDivExpr<out KBvSort>>()

    open fun <T : KBvSort> mkBvSignedDivExpr(arg0: KExpr<T>, arg1: KExpr<T>): KExpr<T> =
        mkSimplified(arg0, arg1, KContext::simplifyBvSignedDivExpr, ::mkBvSignedDivExprNoSimplify)

    open fun <T : KBvSort> mkBvSignedDivExprNoSimplify(arg0: KExpr<T>, arg1: KExpr<T>): KBvSignedDivExpr<T> =
        bvSignedDivExprCache.createIfContextActive {
            ensureContextMatch(arg0, arg1)
            KBvSignedDivExpr(this, arg0, arg1)
        }.cast()

    private val bvUnsignedRemExprCache = mkAstInterner<KBvUnsignedRemExpr<out KBvSort>>()

    open fun <T : KBvSort> mkBvUnsignedRemExpr(arg0: KExpr<T>, arg1: KExpr<T>): KExpr<T> =
        mkSimplified(arg0, arg1, KContext::simplifyBvUnsignedRemExpr, ::mkBvUnsignedRemExprNoSimplify)

    open fun <T : KBvSort> mkBvUnsignedRemExprNoSimplify(arg0: KExpr<T>, arg1: KExpr<T>): KBvUnsignedRemExpr<T> =
        bvUnsignedRemExprCache.createIfContextActive {
            ensureContextMatch(arg0, arg1)
            KBvUnsignedRemExpr(this, arg0, arg1)
        }.cast()

    private val bvSignedRemExprCache = mkAstInterner<KBvSignedRemExpr<out KBvSort>>()

    open fun <T : KBvSort> mkBvSignedRemExpr(arg0: KExpr<T>, arg1: KExpr<T>): KExpr<T> =
        mkSimplified(arg0, arg1, KContext::simplifyBvSignedRemExpr, ::mkBvSignedRemExprNoSimplify)

    open fun <T : KBvSort> mkBvSignedRemExprNoSimplify(arg0: KExpr<T>, arg1: KExpr<T>): KBvSignedRemExpr<T> =
        bvSignedRemExprCache.createIfContextActive {
            ensureContextMatch(arg0, arg1)
            KBvSignedRemExpr(this, arg0, arg1)
        }.cast()

    private val bvSignedModExprCache = mkAstInterner<KBvSignedModExpr<out KBvSort>>()

    open fun <T : KBvSort> mkBvSignedModExpr(arg0: KExpr<T>, arg1: KExpr<T>): KExpr<T> =
        mkSimplified(arg0, arg1, KContext::simplifyBvSignedModExpr, ::mkBvSignedModExprNoSimplify)

    open fun <T : KBvSort> mkBvSignedModExprNoSimplify(arg0: KExpr<T>, arg1: KExpr<T>): KBvSignedModExpr<T> =
        bvSignedModExprCache.createIfContextActive {
            ensureContextMatch(arg0, arg1)
            KBvSignedModExpr(this, arg0, arg1)
        }.cast()

    private val bvUnsignedLessExprCache = mkAstInterner<KBvUnsignedLessExpr<out KBvSort>>()

    open fun <T : KBvSort> mkBvUnsignedLessExpr(arg0: KExpr<T>, arg1: KExpr<T>): KExpr<KBoolSort> =
        mkSimplified(arg0, arg1, KContext::simplifyBvUnsignedLessExpr, ::mkBvUnsignedLessExprNoSimplify)

    open fun <T : KBvSort> mkBvUnsignedLessExprNoSimplify(arg0: KExpr<T>, arg1: KExpr<T>): KBvUnsignedLessExpr<T> =
        bvUnsignedLessExprCache.createIfContextActive {
            ensureContextMatch(arg0, arg1)
            KBvUnsignedLessExpr(this, arg0, arg1)
        }.cast()

    private val bvSignedLessExprCache = mkAstInterner<KBvSignedLessExpr<out KBvSort>>()

    open fun <T : KBvSort> mkBvSignedLessExpr(arg0: KExpr<T>, arg1: KExpr<T>): KExpr<KBoolSort> =
        mkSimplified(arg0, arg1, KContext::simplifyBvSignedLessExpr, ::mkBvSignedLessExprNoSimplify)

    open fun <T : KBvSort> mkBvSignedLessExprNoSimplify(arg0: KExpr<T>, arg1: KExpr<T>): KBvSignedLessExpr<T> =
        bvSignedLessExprCache.createIfContextActive {
            ensureContextMatch(arg0, arg1)
            KBvSignedLessExpr(this, arg0, arg1)
        }.cast()

    private val bvSignedLessOrEqualExprCache = mkAstInterner<KBvSignedLessOrEqualExpr<out KBvSort>>()

    open fun <T : KBvSort> mkBvSignedLessOrEqualExpr(arg0: KExpr<T>, arg1: KExpr<T>): KExpr<KBoolSort> =
        mkSimplified(arg0, arg1, KContext::simplifyBvSignedLessOrEqualExpr, ::mkBvSignedLessOrEqualExprNoSimplify)

    open fun <T : KBvSort> mkBvSignedLessOrEqualExprNoSimplify(
        arg0: KExpr<T>, arg1: KExpr<T>
    ): KBvSignedLessOrEqualExpr<T> =
        bvSignedLessOrEqualExprCache.createIfContextActive {
            ensureContextMatch(arg0, arg1)
            KBvSignedLessOrEqualExpr(this, arg0, arg1)
        }.cast()

    private val bvUnsignedLessOrEqualExprCache = mkAstInterner<KBvUnsignedLessOrEqualExpr<out KBvSort>>()

    open fun <T : KBvSort> mkBvUnsignedLessOrEqualExpr(arg0: KExpr<T>, arg1: KExpr<T>): KExpr<KBoolSort> =
        mkSimplified(arg0, arg1, KContext::simplifyBvUnsignedLessOrEqualExpr, ::mkBvUnsignedLessOrEqualExprNoSimplify)

    open fun <T : KBvSort> mkBvUnsignedLessOrEqualExprNoSimplify(
        arg0: KExpr<T>,
        arg1: KExpr<T>
    ): KBvUnsignedLessOrEqualExpr<T> = bvUnsignedLessOrEqualExprCache.createIfContextActive {
        ensureContextMatch(arg0, arg1)
        KBvUnsignedLessOrEqualExpr(this, arg0, arg1)
    }.cast()

    private val bvUnsignedGreaterOrEqualExprCache = mkAstInterner<KBvUnsignedGreaterOrEqualExpr<out KBvSort>>()

    open fun <T : KBvSort> mkBvUnsignedGreaterOrEqualExpr(arg0: KExpr<T>, arg1: KExpr<T>): KExpr<KBoolSort> =
        mkSimplified(
            arg0,
            arg1,
            KContext::simplifyBvUnsignedGreaterOrEqualExpr,
            ::mkBvUnsignedGreaterOrEqualExprNoSimplify
        )

    open fun <T : KBvSort> mkBvUnsignedGreaterOrEqualExprNoSimplify(
        arg0: KExpr<T>,
        arg1: KExpr<T>
    ): KBvUnsignedGreaterOrEqualExpr<T> = bvUnsignedGreaterOrEqualExprCache.createIfContextActive {
        ensureContextMatch(arg0, arg1)
        KBvUnsignedGreaterOrEqualExpr(this, arg0, arg1)
    }.cast()

    private val bvSignedGreaterOrEqualExprCache = mkAstInterner<KBvSignedGreaterOrEqualExpr<out KBvSort>>()

    open fun <T : KBvSort> mkBvSignedGreaterOrEqualExpr(arg0: KExpr<T>, arg1: KExpr<T>): KExpr<KBoolSort> =
        mkSimplified(arg0, arg1, KContext::simplifyBvSignedGreaterOrEqualExpr, ::mkBvSignedGreaterOrEqualExprNoSimplify)

    open fun <T : KBvSort> mkBvSignedGreaterOrEqualExprNoSimplify(
        arg0: KExpr<T>,
        arg1: KExpr<T>
    ): KBvSignedGreaterOrEqualExpr<T> = bvSignedGreaterOrEqualExprCache.createIfContextActive {
        ensureContextMatch(arg0, arg1)
        KBvSignedGreaterOrEqualExpr(this, arg0, arg1)
    }.cast()

    private val bvUnsignedGreaterExprCache = mkAstInterner<KBvUnsignedGreaterExpr<out KBvSort>>()

    open fun <T : KBvSort> mkBvUnsignedGreaterExpr(arg0: KExpr<T>, arg1: KExpr<T>): KExpr<KBoolSort> =
        mkSimplified(arg0, arg1, KContext::simplifyBvUnsignedGreaterExpr, ::mkBvUnsignedGreaterExprNoSimplify)

    open fun <T : KBvSort> mkBvUnsignedGreaterExprNoSimplify(
        arg0: KExpr<T>, arg1: KExpr<T>
    ): KBvUnsignedGreaterExpr<T> =
        bvUnsignedGreaterExprCache.createIfContextActive {
            ensureContextMatch(arg0, arg1)
            KBvUnsignedGreaterExpr(this, arg0, arg1)
        }.cast()

    private val bvSignedGreaterExprCache = mkAstInterner<KBvSignedGreaterExpr<out KBvSort>>()

    open fun <T : KBvSort> mkBvSignedGreaterExpr(arg0: KExpr<T>, arg1: KExpr<T>): KExpr<KBoolSort> =
        mkSimplified(arg0, arg1, KContext::simplifyBvSignedGreaterExpr, ::mkBvSignedGreaterExprNoSimplify)

    open fun <T : KBvSort> mkBvSignedGreaterExprNoSimplify(arg0: KExpr<T>, arg1: KExpr<T>): KBvSignedGreaterExpr<T> =
        bvSignedGreaterExprCache.createIfContextActive {
            ensureContextMatch(arg0, arg1)
            KBvSignedGreaterExpr(this, arg0, arg1)
        }.cast()

    private val concatExprCache = mkAstInterner<KBvConcatExpr>()

    open fun <T : KBvSort, S : KBvSort> mkBvConcatExpr(arg0: KExpr<T>, arg1: KExpr<S>): KExpr<KBvSort> =
        mkSimplified(arg0, arg1, KContext::simplifyBvConcatExpr, ::mkBvConcatExprNoSimplify)

    open fun <T : KBvSort, S : KBvSort> mkBvConcatExprNoSimplify(arg0: KExpr<T>, arg1: KExpr<S>): KBvConcatExpr =
        concatExprCache.createIfContextActive {
            ensureContextMatch(arg0, arg1)
            KBvConcatExpr(this, arg0.cast(), arg1.cast())
        }

    private val extractExprCache = mkAstInterner<KBvExtractExpr>()

    open fun <T : KBvSort> mkBvExtractExpr(high: Int, low: Int, value: KExpr<T>): KExpr<KBvSort> =
        mkSimplified(high, low, value, KContext::simplifyBvExtractExpr, ::mkBvExtractExprNoSimplify)

    open fun <T : KBvSort> mkBvExtractExprNoSimplify(high: Int, low: Int, value: KExpr<T>): KBvExtractExpr =
        extractExprCache.createIfContextActive {
            ensureContextMatch(value)
            KBvExtractExpr(this, high, low, value.cast())
        }

    private val signExtensionExprCache = mkAstInterner<KBvSignExtensionExpr>()

    open fun <T : KBvSort> mkBvSignExtensionExpr(extensionSize: Int, value: KExpr<T>): KExpr<KBvSort> =
        mkSimplified(extensionSize, value, KContext::simplifyBvSignExtensionExpr, ::mkBvSignExtensionExprNoSimplify)

    open fun <T : KBvSort> mkBvSignExtensionExprNoSimplify(
        extensionSize: Int,
        value: KExpr<T>
    ): KBvSignExtensionExpr = signExtensionExprCache.createIfContextActive {
        ensureContextMatch(value)
        KBvSignExtensionExpr(this, extensionSize, value.cast())
    }

    private val zeroExtensionExprCache = mkAstInterner<KBvZeroExtensionExpr>()

    open fun <T : KBvSort> mkBvZeroExtensionExpr(extensionSize: Int, value: KExpr<T>): KExpr<KBvSort> =
        mkSimplified(extensionSize, value, KContext::simplifyBvZeroExtensionExpr, ::mkBvZeroExtensionExprNoSimplify)

    open fun <T : KBvSort> mkBvZeroExtensionExprNoSimplify(
        extensionSize: Int,
        value: KExpr<T>
    ): KBvZeroExtensionExpr = zeroExtensionExprCache.createIfContextActive {
        ensureContextMatch(value)
        KBvZeroExtensionExpr(this, extensionSize, value.cast())
    }

    private val repeatExprCache = mkAstInterner<KBvRepeatExpr>()

    open fun <T : KBvSort> mkBvRepeatExpr(repeatNumber: Int, value: KExpr<T>): KExpr<KBvSort> =
        mkSimplified(repeatNumber, value, KContext::simplifyBvRepeatExpr, ::mkBvRepeatExprNoSimplify)

    open fun <T : KBvSort> mkBvRepeatExprNoSimplify(
        repeatNumber: Int,
        value: KExpr<T>
    ): KBvRepeatExpr = repeatExprCache.createIfContextActive {
        ensureContextMatch(value)
        KBvRepeatExpr(this, repeatNumber, value.cast())
    }

    private val bvShiftLeftExprCache = mkAstInterner<KBvShiftLeftExpr<out KBvSort>>()

    open fun <T : KBvSort> mkBvShiftLeftExpr(arg: KExpr<T>, shift: KExpr<T>): KExpr<T> =
        mkSimplified(arg, shift, KContext::simplifyBvShiftLeftExpr, ::mkBvShiftLeftExprNoSimplify)

    open fun <T : KBvSort> mkBvShiftLeftExprNoSimplify(arg: KExpr<T>, shift: KExpr<T>): KBvShiftLeftExpr<T> =
        bvShiftLeftExprCache.createIfContextActive {
            ensureContextMatch(arg, shift)
            KBvShiftLeftExpr(this, arg, shift)
        }.cast()

    private val bvLogicalShiftRightExprCache = mkAstInterner<KBvLogicalShiftRightExpr<out KBvSort>>()

    open fun <T : KBvSort> mkBvLogicalShiftRightExpr(arg: KExpr<T>, shift: KExpr<T>): KExpr<T> =
        mkSimplified(arg, shift, KContext::simplifyBvLogicalShiftRightExpr, ::mkBvLogicalShiftRightExprNoSimplify)

    open fun <T : KBvSort> mkBvLogicalShiftRightExprNoSimplify(
        arg: KExpr<T>, shift: KExpr<T>
    ): KBvLogicalShiftRightExpr<T> =
        bvLogicalShiftRightExprCache.createIfContextActive {
            ensureContextMatch(arg, shift)
            KBvLogicalShiftRightExpr(this, arg, shift)
        }.cast()

    private val bvArithShiftRightExprCache = mkAstInterner<KBvArithShiftRightExpr<out KBvSort>>()

    open fun <T : KBvSort> mkBvArithShiftRightExpr(arg: KExpr<T>, shift: KExpr<T>): KExpr<T> =
        mkSimplified(arg, shift, KContext::simplifyBvArithShiftRightExpr, ::mkBvArithShiftRightExprNoSimplify)

    open fun <T : KBvSort> mkBvArithShiftRightExprNoSimplify(
        arg: KExpr<T>, shift: KExpr<T>
    ): KBvArithShiftRightExpr<T> =
        bvArithShiftRightExprCache.createIfContextActive {
            ensureContextMatch(arg, shift)
            KBvArithShiftRightExpr(this, arg, shift)
        }.cast()

    private val bvRotateLeftExprCache = mkAstInterner<KBvRotateLeftExpr<out KBvSort>>()

    open fun <T : KBvSort> mkBvRotateLeftExpr(arg: KExpr<T>, rotation: KExpr<T>): KExpr<T> =
        mkSimplified(arg, rotation, KContext::simplifyBvRotateLeftExpr, ::mkBvRotateLeftExprNoSimplify)

    open fun <T : KBvSort> mkBvRotateLeftExprNoSimplify(arg: KExpr<T>, rotation: KExpr<T>): KBvRotateLeftExpr<T> =
        bvRotateLeftExprCache.createIfContextActive {
            ensureContextMatch(arg, rotation)
            KBvRotateLeftExpr(this, arg, rotation)
        }.cast()

    private val bvRotateLeftIndexedExprCache = mkAstInterner<KBvRotateLeftIndexedExpr<out KBvSort>>()

    open fun <T : KBvSort> mkBvRotateLeftIndexedExpr(rotation: Int, value: KExpr<T>): KExpr<T> =
        mkSimplified(rotation, value, KContext::simplifyBvRotateLeftIndexedExpr, ::mkBvRotateLeftIndexedExprNoSimplify)

    open fun <T : KBvSort> mkBvRotateLeftIndexedExprNoSimplify(
        rotation: Int, value: KExpr<T>
    ): KBvRotateLeftIndexedExpr<T> =
        bvRotateLeftIndexedExprCache.createIfContextActive {
            ensureContextMatch(value)
            KBvRotateLeftIndexedExpr(this, rotation, value)
        }.cast()

    private val bvRotateRightIndexedExprCache = mkAstInterner<KBvRotateRightIndexedExpr<out KBvSort>>()

    open fun <T : KBvSort> mkBvRotateRightIndexedExpr(rotation: Int, value: KExpr<T>): KExpr<T> =
        mkSimplified(
            rotation,
            value,
            KContext::simplifyBvRotateRightIndexedExpr,
            ::mkBvRotateRightIndexedExprNoSimplify
        )

    open fun <T : KBvSort> mkBvRotateRightIndexedExprNoSimplify(
        rotation: Int,
        value: KExpr<T>
    ): KBvRotateRightIndexedExpr<T> =
        bvRotateRightIndexedExprCache.createIfContextActive {
            ensureContextMatch(value)
            KBvRotateRightIndexedExpr(this, rotation, value)
        }.cast()

    private val bvRotateRightExprCache = mkAstInterner<KBvRotateRightExpr<out KBvSort>>()

    open fun <T : KBvSort> mkBvRotateRightExpr(arg: KExpr<T>, rotation: KExpr<T>): KExpr<T> =
        mkSimplified(arg, rotation, KContext::simplifyBvRotateRightExpr, ::mkBvRotateRightExprNoSimplify)

    open fun <T : KBvSort> mkBvRotateRightExprNoSimplify(arg: KExpr<T>, rotation: KExpr<T>): KBvRotateRightExpr<T> =
        bvRotateRightExprCache.createIfContextActive {
            ensureContextMatch(arg, rotation)
            KBvRotateRightExpr(this, arg, rotation)
        }.cast()

    private val bv2IntExprCache = mkAstInterner<KBv2IntExpr>()

    open fun <T : KBvSort> mkBv2IntExpr(value: KExpr<T>, isSigned: Boolean): KExpr<KIntSort> =
        mkSimplified(value, isSigned, KContext::simplifyBv2IntExpr, ::mkBv2IntExprNoSimplify)

    open fun <T : KBvSort> mkBv2IntExprNoSimplify(value: KExpr<T>, isSigned: Boolean): KBv2IntExpr =
        bv2IntExprCache.createIfContextActive {
            ensureContextMatch(value)
            KBv2IntExpr(this, value.cast(), isSigned)
        }

    private val bvAddNoOverflowExprCache = mkAstInterner<KBvAddNoOverflowExpr<out KBvSort>>()

    open fun <T : KBvSort> mkBvAddNoOverflowExpr(arg0: KExpr<T>, arg1: KExpr<T>, isSigned: Boolean): KExpr<KBoolSort> =
        mkSimplified(arg0, arg1, isSigned, KContext::simplifyBvAddNoOverflowExpr, ::mkBvAddNoOverflowExprNoSimplify)

    open fun <T : KBvSort> mkBvAddNoOverflowExprNoSimplify(
        arg0: KExpr<T>,
        arg1: KExpr<T>,
        isSigned: Boolean
    ): KBvAddNoOverflowExpr<T> = bvAddNoOverflowExprCache.createIfContextActive {
        ensureContextMatch(arg0, arg1)
        KBvAddNoOverflowExpr(this, arg0, arg1, isSigned)
    }.cast()

    private val bvAddNoUnderflowExprCache = mkAstInterner<KBvAddNoUnderflowExpr<out KBvSort>>()

    open fun <T : KBvSort> mkBvAddNoUnderflowExpr(arg0: KExpr<T>, arg1: KExpr<T>): KExpr<KBoolSort> =
        mkSimplified(arg0, arg1, KContext::simplifyBvAddNoUnderflowExpr, ::mkBvAddNoUnderflowExprNoSimplify)

    open fun <T : KBvSort> mkBvAddNoUnderflowExprNoSimplify(arg0: KExpr<T>, arg1: KExpr<T>): KBvAddNoUnderflowExpr<T> =
        bvAddNoUnderflowExprCache.createIfContextActive {
            ensureContextMatch(arg0, arg1)
            KBvAddNoUnderflowExpr(this, arg0, arg1)
        }.cast()

    private val bvSubNoOverflowExprCache = mkAstInterner<KBvSubNoOverflowExpr<out KBvSort>>()

    open fun <T : KBvSort> mkBvSubNoOverflowExpr(arg0: KExpr<T>, arg1: KExpr<T>): KExpr<KBoolSort> =
        mkSimplified(arg0, arg1, KContext::simplifyBvSubNoOverflowExpr, ::mkBvSubNoOverflowExprNoSimplify)

    open fun <T : KBvSort> mkBvSubNoOverflowExprNoSimplify(arg0: KExpr<T>, arg1: KExpr<T>): KBvSubNoOverflowExpr<T> =
        bvSubNoOverflowExprCache.createIfContextActive {
            ensureContextMatch(arg0, arg1)
            KBvSubNoOverflowExpr(this, arg0, arg1)
        }.cast()

    private val bvSubNoUnderflowExprCache = mkAstInterner<KBvSubNoUnderflowExpr<out KBvSort>>()

    open fun <T : KBvSort> mkBvSubNoUnderflowExpr(arg0: KExpr<T>, arg1: KExpr<T>, isSigned: Boolean): KExpr<KBoolSort> =
        mkSimplified(arg0, arg1, isSigned, KContext::simplifyBvSubNoUnderflowExpr, ::mkBvSubNoUnderflowExprNoSimplify)

    open fun <T : KBvSort> mkBvSubNoUnderflowExprNoSimplify(
        arg0: KExpr<T>,
        arg1: KExpr<T>,
        isSigned: Boolean
    ): KBvSubNoUnderflowExpr<T> = bvSubNoUnderflowExprCache.createIfContextActive {
        ensureContextMatch(arg0, arg1)
        KBvSubNoUnderflowExpr(this, arg0, arg1, isSigned)
    }.cast()

    private val bvDivNoOverflowExprCache = mkAstInterner<KBvDivNoOverflowExpr<out KBvSort>>()

    open fun <T : KBvSort> mkBvDivNoOverflowExpr(arg0: KExpr<T>, arg1: KExpr<T>): KExpr<KBoolSort> =
        mkSimplified(arg0, arg1, KContext::simplifyBvDivNoOverflowExpr, ::mkBvDivNoOverflowExprNoSimplify)

    open fun <T : KBvSort> mkBvDivNoOverflowExprNoSimplify(arg0: KExpr<T>, arg1: KExpr<T>): KBvDivNoOverflowExpr<T> =
        bvDivNoOverflowExprCache.createIfContextActive {
            ensureContextMatch(arg0, arg1)
            KBvDivNoOverflowExpr(this, arg0, arg1)
        }.cast()

    private val bvNegNoOverflowExprCache = mkAstInterner<KBvNegNoOverflowExpr<out KBvSort>>()

    open fun <T : KBvSort> mkBvNegationNoOverflowExpr(value: KExpr<T>): KExpr<KBoolSort> =
        mkSimplified(value, KContext::simplifyBvNegationNoOverflowExpr, ::mkBvNegationNoOverflowExprNoSimplify)

    open fun <T : KBvSort> mkBvNegationNoOverflowExprNoSimplify(value: KExpr<T>): KBvNegNoOverflowExpr<T> =
        bvNegNoOverflowExprCache.createIfContextActive {
            ensureContextMatch(value)
            KBvNegNoOverflowExpr(this, value)
        }.cast()

    private val bvMulNoOverflowExprCache = mkAstInterner<KBvMulNoOverflowExpr<out KBvSort>>()

    open fun <T : KBvSort> mkBvMulNoOverflowExpr(arg0: KExpr<T>, arg1: KExpr<T>, isSigned: Boolean): KExpr<KBoolSort> =
        mkSimplified(arg0, arg1, isSigned, KContext::simplifyBvMulNoOverflowExpr, ::mkBvMulNoOverflowExprNoSimplify)

    open fun <T : KBvSort> mkBvMulNoOverflowExprNoSimplify(
        arg0: KExpr<T>,
        arg1: KExpr<T>,
        isSigned: Boolean
    ): KBvMulNoOverflowExpr<T> = bvMulNoOverflowExprCache.createIfContextActive {
        ensureContextMatch(arg0, arg1)
        KBvMulNoOverflowExpr(this, arg0, arg1, isSigned)
    }.cast()

    private val bvMulNoUnderflowExprCache = mkAstInterner<KBvMulNoUnderflowExpr<out KBvSort>>()

    open fun <T : KBvSort> mkBvMulNoUnderflowExpr(arg0: KExpr<T>, arg1: KExpr<T>): KExpr<KBoolSort> =
        mkSimplified(arg0, arg1, KContext::simplifyBvMulNoUnderflowExpr, ::mkBvMulNoUnderflowExprNoSimplify)

    open fun <T : KBvSort> mkBvMulNoUnderflowExprNoSimplify(arg0: KExpr<T>, arg1: KExpr<T>): KBvMulNoUnderflowExpr<T> =
        bvMulNoUnderflowExprCache.createIfContextActive {
            ensureContextMatch(arg0, arg1)
            KBvMulNoUnderflowExpr(this, arg0, arg1)
        }.cast()

    // fp values
    private val fp16Cache = mkAstInterner<KFp16Value>()
    private val fp32Cache = mkAstInterner<KFp32Value>()
    private val fp64Cache = mkAstInterner<KFp64Value>()
    private val fp128Cache = mkAstInterner<KFp128Value>()
    private val fpCustomSizeCache = mkAstInterner<KFpCustomSizeValue>()

    /**
     * Creates FP16 from the [value].
     *
     * Important: we suppose that [value] has biased exponent, but FP16 will be created from the unbiased one.
     * So, at first, we'll subtract [KFp16Sort.exponentShiftSize] from the [value]'s exponent,
     * take required for FP16 bits, and this will be **unbiased** FP16 exponent.
     * The same is true for other methods but [mkFpCustomSize].
     * */
    fun mkFp16(value: Float): KFp16Value {
        val exponent = value.getHalfPrecisionExponent(isBiased = false)
        val significand = value.halfPrecisionSignificand
        val normalizedValue = constructFp16Number(exponent.toLong(), significand.toLong(), value.signBit)
        return if (KFp16Value(this, normalizedValue).isNaN()) {
            mkFp16NaN()
        } else {
            mkFp16WithoutNaNCheck(normalizedValue)
        }
    }

    fun mkFp16NaN(): KFp16Value = mkFp16WithoutNaNCheck(Float.NaN)
    private fun mkFp16WithoutNaNCheck(value: Float): KFp16Value =
        fp16Cache.createIfContextActive { KFp16Value(this, value) }

    fun mkFp32(value: Float): KFp32Value = if (value.isNaN()) mkFp32NaN() else mkFp32WithoutNaNCheck(value)
    fun mkFp32NaN(): KFp32Value = mkFp32WithoutNaNCheck(Float.NaN)
    private fun mkFp32WithoutNaNCheck(value: Float): KFp32Value =
        fp32Cache.createIfContextActive { KFp32Value(this, value) }

    fun mkFp64(value: Double): KFp64Value = if (value.isNaN()) mkFp64NaN() else mkFp64WithoutNaNCheck(value)
    fun mkFp64NaN(): KFp64Value = mkFp64WithoutNaNCheck(Double.NaN)
    private fun mkFp64WithoutNaNCheck(value: Double): KFp64Value =
        fp64Cache.createIfContextActive { KFp64Value(this, value) }

    fun mkFp128Biased(significand: KBitVecValue<*>, biasedExponent: KBitVecValue<*>, signBit: Boolean): KFp128Value =
        if (KFp128Value(this, significand, biasedExponent, signBit).isNaN()) {
            mkFp128NaN()
        } else {
            mkFp128BiasedWithoutNaNCheck(significand, biasedExponent, signBit)
        }

    fun mkFp128NaN(): KFp128Value = mkFpNaN(mkFp128Sort()).cast()
    private fun mkFp128BiasedWithoutNaNCheck(
        significand: KBitVecValue<*>,
        biasedExponent: KBitVecValue<*>,
        signBit: Boolean
    ): KFp128Value = fp128Cache.createIfContextActive {
        ensureContextMatch(significand, biasedExponent)
        KFp128Value(this, significand, biasedExponent, signBit)
    }

    fun mkFp128(significand: KBitVecValue<*>, unbiasedExponent: KBitVecValue<*>, signBit: Boolean): KFp128Value =
        mkFp128Biased(
            significand = significand,
            biasedExponent = biasFp128Exponent(unbiasedExponent),
            signBit = signBit
        )

    fun mkFp128(significand: Long, unbiasedExponent: Long, signBit: Boolean): KFp128Value =
        mkFp128(
            significand = mkBvUnsigned(significand, KFp128Sort.significandBits - 1u),
            unbiasedExponent = mkBvUnsigned(unbiasedExponent, KFp128Sort.exponentBits),
            signBit = signBit
        )

    val Float.expr
        get() = mkFp32(this)

    val Double.expr
        get() = mkFp64(this)

    /**
     * Creates FP with a custom size.
     * Important: [unbiasedExponent] here is an **unbiased** value.
     */
    fun <T : KFpSort> mkFpCustomSize(
        exponentSize: UInt,
        significandSize: UInt,
        unbiasedExponent: KBitVecValue<*>,
        significand: KBitVecValue<*>,
        signBit: Boolean
    ): KFpValue<T> {
        val intSignBit = if (signBit) 1 else 0

        return when (mkFpSort(exponentSize, significandSize)) {
            is KFp16Sort -> {
                val number = constructFp16Number(unbiasedExponent.longValue(), significand.longValue(), intSignBit)

                mkFp16(number).cast()
            }

            is KFp32Sort -> {
                val number = constructFp32Number(unbiasedExponent.longValue(), significand.longValue(), intSignBit)

                mkFp32(number).cast()
            }

            is KFp64Sort -> {
                val number = constructFp64Number(unbiasedExponent.longValue(), significand.longValue(), intSignBit)

                mkFp64(number).cast()
            }

            is KFp128Sort -> mkFp128(significand, unbiasedExponent, signBit).cast()
            else -> {
                val biasedExponent = biasFpCustomSizeExponent(unbiasedExponent, exponentSize)
                mkFpCustomSizeBiased(significandSize, exponentSize, significand, biasedExponent, signBit)
            }
        }
    }

    /**
     * Creates FP with a custom size.
     * Important: [biasedExponent] here is an **biased** value.
     */
    fun <T : KFpSort> mkFpCustomSizeBiased(
        significandSize: UInt,
        exponentSize: UInt,
        significand: KBitVecValue<*>,
        biasedExponent: KBitVecValue<*>,
        signBit: Boolean
    ): KFpValue<T> {
        val intSignBit = if (signBit) 1 else 0

        return when (val sort = mkFpSort(exponentSize, significandSize)) {
            is KFp16Sort -> {
                val number = constructFp16NumberBiased(biasedExponent.longValue(), significand.longValue(), intSignBit)

                mkFp16(number).cast()
            }

            is KFp32Sort -> {
                val number = constructFp32NumberBiased(biasedExponent.longValue(), significand.longValue(), intSignBit)

                mkFp32(number).cast()
            }

            is KFp64Sort -> {
                val number = constructFp64NumberBiased(biasedExponent.longValue(), significand.longValue(), intSignBit)

                mkFp64(number).cast()
            }

            is KFp128Sort -> mkFp128Biased(significand, biasedExponent, signBit).cast()
            else -> {
                val valueForNaNCheck = KFpCustomSizeValue(
                    this, significandSize, exponentSize, significand, biasedExponent, signBit
                )
                if (valueForNaNCheck.isNaN()) {
                    mkFpNaN(sort)
                } else {
                    mkFpCustomSizeBiasedWithoutNaNCheck(sort, significand, biasedExponent, signBit)
                }.cast()
            }
        }
    }

    private fun <T : KFpSort> mkFpCustomSizeBiasedWithoutNaNCheck(
        sort: T,
        significand: KBitVecValue<*>,
        biasedExponent: KBitVecValue<*>,
        signBit: Boolean
    ): KFpValue<T> {
        val intSignBit = if (signBit) 1 else 0

        return when (sort) {
            is KFp16Sort -> {
                val number = constructFp16NumberBiased(biasedExponent.longValue(), significand.longValue(), intSignBit)

                mkFp16WithoutNaNCheck(number).cast()
            }

            is KFp32Sort -> {
                val number = constructFp32NumberBiased(biasedExponent.longValue(), significand.longValue(), intSignBit)

                mkFp32WithoutNaNCheck(number).cast()
            }

            is KFp64Sort -> {
                val number = constructFp64NumberBiased(biasedExponent.longValue(), significand.longValue(), intSignBit)

                mkFp64WithoutNaNCheck(number).cast()
            }

            is KFp128Sort -> mkFp128BiasedWithoutNaNCheck(significand, biasedExponent, signBit).cast()
            else -> {
                fpCustomSizeCache.createIfContextActive {
                    ensureContextMatch(significand, biasedExponent)
                    KFpCustomSizeValue(
                        this, sort.significandBits, sort.exponentBits, significand, biasedExponent, signBit
                    )
                }.cast()
            }
        }
    }

    fun <T : KFpSort> mkFpCustomSize(
        unbiasedExponent: KBitVecValue<out KBvSort>,
        significand: KBitVecValue<out KBvSort>,
        signBit: Boolean
    ): KFpValue<T> = mkFpCustomSize(
        unbiasedExponent.sort.sizeBits,
        significand.sort.sizeBits + 1u, // +1 for sign bit
        unbiasedExponent,
        significand,
        signBit
    )

    private fun KBitVecValue<*>.longValue() = when (this) {
        is KBitVecNumberValue<*, *> -> numberValue.toULongValue().toLong()
        is KBitVecCustomValue -> value.longValueExact()
        is KBitVec1Value -> if (value) 1L else 0L
        else -> stringValue.toULong(radix = 2).toLong()
    }

    @Suppress("MagicNumber")
    private fun constructFp16Number(exponent: Long, significand: Long, intSignBit: Int): Float {
        /**
         * Transform fp16 exponent into fp32 exponent.
         * Transform fp16 top and bot exponent to fp32 top and bot exponent to
         * preserve representation of special values (NaN, Inf, Zero)
         */
        val unbiasedFp16Exponent = exponent.toInt()
        val unbiasedFp32Exponent = when {
            unbiasedFp16Exponent <= -KFp16Sort.exponentShiftSize -> -KFp32Sort.exponentShiftSize
            unbiasedFp16Exponent >= KFp16Sort.exponentShiftSize + 1 -> KFp32Sort.exponentShiftSize + 1
            else -> unbiasedFp16Exponent
        }

        // get fp16 significand part -- last teb bits (eleventh stored implicitly)
        val significandBits = significand.toInt() and 0b1111_1111_11

        // Add the bias for fp32 and apply the mask to avoid overflow of the eight bits
        val biasedFloatExponent = (unbiasedFp32Exponent + KFp32Sort.exponentShiftSize) and 0xff

        val bits = (intSignBit shl 31) or (biasedFloatExponent shl 23) or (significandBits shl 13)

        return intBitsToFloat(bits)
    }

    @Suppress("MagicNumber")
    private fun constructFp16NumberBiased(biasedExponent: Long, significand: Long, intSignBit: Int): Float {
        val unbiasedExponent = biasedExponent - KFp16Sort.exponentShiftSize
        return constructFp16Number(unbiasedExponent, significand, intSignBit)
    }

    @Suppress("MagicNumber")
    private fun constructFp32Number(exponent: Long, significand: Long, intSignBit: Int): Float {
        // `and 0xff` here is to avoid overloading when we have a number greater than 255,
        // and the result of the addition will affect the sign bit
        val biasedExponent = (exponent.toInt() + KFp32Sort.exponentShiftSize) and 0xff
        val intValue = (intSignBit shl 31) or (biasedExponent shl 23) or significand.toInt()

        return intBitsToFloat(intValue)
    }

    @Suppress("MagicNumber")
    private fun constructFp32NumberBiased(biasedExponent: Long, significand: Long, intSignBit: Int): Float {
        // `and 0xff` here is to normalize exponent value
        val normalizedBiasedExponent = (biasedExponent.toInt()) and 0xff
        val intValue = (intSignBit shl 31) or (normalizedBiasedExponent shl 23) or significand.toInt()

        return intBitsToFloat(intValue)
    }

    @Suppress("MagicNumber")
    private fun constructFp64Number(exponent: Long, significand: Long, intSignBit: Int): Double {
        // `and 0b111_1111_1111` here is to avoid overloading when we have a number greater than 255,
        // and the result of the addition will affect the sign bit
        val biasedExponent = (exponent + KFp64Sort.exponentShiftSize) and 0b111_1111_1111
        val longValue = (intSignBit.toLong() shl 63) or (biasedExponent shl 52) or significand

        return longBitsToDouble(longValue)
    }

    @Suppress("MagicNumber")
    private fun constructFp64NumberBiased(biasedExponent: Long, significand: Long, intSignBit: Int): Double {
        // `and 0b111_1111_1111` here is to normalize exponent value
        val normalizedBiasedExponent = biasedExponent and 0b111_1111_1111
        val longValue = (intSignBit.toLong() shl 63) or (normalizedBiasedExponent shl 52) or significand

        return longBitsToDouble(longValue)
    }

    private fun biasFp128Exponent(exponent: KBitVecValue<*>): KBitVecValue<*> =
        biasFpExponent(exponent, KFp128Sort.exponentBits)

    private fun unbiasFp128Exponent(exponent: KBitVecValue<*>): KBitVecValue<*> =
        unbiasFpExponent(exponent, KFp128Sort.exponentBits)

    private fun biasFpCustomSizeExponent(exponent: KBitVecValue<*>, exponentSize: UInt): KBitVecValue<*> =
        biasFpExponent(exponent, exponentSize)

    private fun unbiasFpCustomSizeExponent(exponent: KBitVecValue<*>, exponentSize: UInt): KBitVecValue<*> =
        unbiasFpExponent(exponent, exponentSize)

    fun <T : KFpSort> mkFp(value: Float, sort: T): KFpValue<T> {
        if (sort == mkFp32Sort()) {
            return mkFp32(value).cast()
        }

        val significand = mkBvUnsigned(value.extractSignificand(sort), sort.significandBits - 1u)
        val exponent = mkBvUnsigned(value.extractExponent(sort, isBiased = false), sort.exponentBits)
        val sign = value.booleanSignBit

        return mkFpCustomSize(
            sort.exponentBits,
            sort.significandBits,
            exponent,
            significand,
            sign
        )
    }

    fun <T : KFpSort> mkFp(value: Double, sort: T): KFpValue<T> {
        if (sort == mkFp64Sort()) {
            return mkFp64(value).cast()
        }

        val significand = mkBvUnsigned(value.extractSignificand(sort), sort.significandBits - 1u)
        val exponent = mkBvUnsigned(value.extractExponent(sort, isBiased = false), sort.exponentBits)
        val sign = value.booleanSignBit

        return mkFpCustomSize(sort.exponentBits, sort.significandBits, exponent, significand, sign)
    }

    fun Double.toFp(sort: KFpSort = mkFp64Sort()): KFpValue<KFpSort> = mkFp(this, sort)

    fun Float.toFp(sort: KFpSort = mkFp32Sort()): KFpValue<KFpSort> = mkFp(this, sort)

    fun <T : KFpSort> mkFp(significand: Int, unbiasedExponent: Int, signBit: Boolean, sort: T): KFpValue<T> =
        mkFpCustomSize(
            exponentSize = sort.exponentBits,
            significandSize = sort.significandBits,
            unbiasedExponent = mkBvUnsigned(unbiasedExponent, sort.exponentBits),
            significand = mkBvUnsigned(significand, sort.significandBits - 1u),
            signBit = signBit
        )

    fun <T : KFpSort> mkFp(significand: Long, unbiasedExponent: Long, signBit: Boolean, sort: T): KFpValue<T> =
        mkFpCustomSize(
            exponentSize = sort.exponentBits,
            significandSize = sort.significandBits,
            unbiasedExponent = mkBvUnsigned(unbiasedExponent, sort.exponentBits),
            significand = mkBvUnsigned(significand, sort.significandBits - 1u),
            signBit = signBit
        )

    fun <T : KFpSort> mkFp(
        significand: KBitVecValue<*>,
        unbiasedExponent: KBitVecValue<*>,
        signBit: Boolean,
        sort: T
    ): KFpValue<T> = mkFpCustomSize(
        exponentSize = sort.exponentBits,
        significandSize = sort.significandBits,
        unbiasedExponent = unbiasedExponent,
        significand = significand,
        signBit = signBit
    )

    fun <T : KFpSort> mkFpBiased(significand: Int, biasedExponent: Int, signBit: Boolean, sort: T): KFpValue<T> =
        mkFpCustomSizeBiased(
            exponentSize = sort.exponentBits,
            significandSize = sort.significandBits,
            biasedExponent = mkBvUnsigned(biasedExponent, sort.exponentBits),
            significand = mkBvUnsigned(significand, sort.significandBits - 1u),
            signBit = signBit
        )

    fun <T : KFpSort> mkFpBiased(significand: Long, biasedExponent: Long, signBit: Boolean, sort: T): KFpValue<T> =
        mkFpCustomSizeBiased(
            exponentSize = sort.exponentBits,
            significandSize = sort.significandBits,
            biasedExponent = mkBvUnsigned(biasedExponent, sort.exponentBits),
            significand = mkBvUnsigned(significand, sort.significandBits - 1u),
            signBit = signBit
        )

    fun <T : KFpSort> mkFpBiased(
        significand: KBitVecValue<*>,
        biasedExponent: KBitVecValue<*>,
        signBit: Boolean,
        sort: T
    ): KFpValue<T> = mkFpCustomSizeBiased(
        exponentSize = sort.exponentBits,
        significandSize = sort.significandBits,
        biasedExponent = biasedExponent,
        significand = significand,
        signBit = signBit
    )

    /**
     * Special Fp values
     * */
    @Suppress("MagicNumber")
    fun <T : KFpSort> mkFpZero(signBit: Boolean, sort: T): KFpValue<T> = when (sort) {
        is KFp16Sort -> mkFp16(if (signBit) -0.0f else 0.0f).cast()
        is KFp32Sort -> mkFp32(if (signBit) -0.0f else 0.0f).cast()
        is KFp64Sort -> mkFp64(if (signBit) -0.0 else 0.0).cast()
        else -> mkFpCustomSizeBiased(
            exponentSize = sort.exponentBits,
            significandSize = sort.significandBits,
            biasedExponent = fpZeroExponentBiased(sort),
            significand = fpZeroSignificand(sort),
            signBit = signBit
        )
    }

    fun <T : KFpSort> mkFpInf(signBit: Boolean, sort: T): KFpValue<T> = when (sort) {
        is KFp16Sort -> mkFp16(if (signBit) Float.NEGATIVE_INFINITY else Float.POSITIVE_INFINITY).cast()
        is KFp32Sort -> mkFp32(if (signBit) Float.NEGATIVE_INFINITY else Float.POSITIVE_INFINITY).cast()
        is KFp64Sort -> mkFp64(if (signBit) Double.NEGATIVE_INFINITY else Double.POSITIVE_INFINITY).cast()
        else -> mkFpCustomSizeBiased(
            exponentSize = sort.exponentBits,
            significandSize = sort.significandBits,
            biasedExponent = fpInfExponentBiased(sort),
            significand = fpInfSignificand(sort),
            signBit = signBit
        )
    }

    fun <T : KFpSort> mkFpNaN(sort: T): KFpValue<T> = when (sort) {
        is KFp16Sort -> mkFp16NaN().cast()
        is KFp32Sort -> mkFp32NaN().cast()
        is KFp64Sort -> mkFp64NaN().cast()
        else -> mkFpCustomSizeBiasedWithoutNaNCheck(
            sort = sort,
            biasedExponent = fpNaNExponentBiased(sort),
            significand = fpNaNSignificand(sort),
            signBit = false
        )
    }

    private val roundingModeCache = mkAstInterner<KFpRoundingModeExpr>()

    fun mkFpRoundingModeExpr(
        value: KFpRoundingMode
    ): KFpRoundingModeExpr = roundingModeCache.createIfContextActive {
        KFpRoundingModeExpr(this, value)
    }

    private val fpAbsExprCache = mkAstInterner<KFpAbsExpr<out KFpSort>>()

    open fun <T : KFpSort> mkFpAbsExpr(value: KExpr<T>): KExpr<T> =
        mkSimplified(value, KContext::simplifyFpAbsExpr, ::mkFpAbsExprNoSimplify)

    open fun <T : KFpSort> mkFpAbsExprNoSimplify(
        value: KExpr<T>
    ): KFpAbsExpr<T> = fpAbsExprCache.createIfContextActive {
        ensureContextMatch(value)
        KFpAbsExpr(this, value)
    }.cast()

    private val fpNegationExprCache = mkAstInterner<KFpNegationExpr<out KFpSort>>()

    open fun <T : KFpSort> mkFpNegationExpr(value: KExpr<T>): KExpr<T> =
        mkSimplified(value, KContext::simplifyFpNegationExpr, ::mkFpNegationExprNoSimplify)

    open fun <T : KFpSort> mkFpNegationExprNoSimplify(
        value: KExpr<T>
    ): KFpNegationExpr<T> = fpNegationExprCache.createIfContextActive {
        ensureContextMatch(value)
        KFpNegationExpr(this, value)
    }.cast()

    private val fpAddExprCache = mkAstInterner<KFpAddExpr<out KFpSort>>()

    open fun <T : KFpSort> mkFpAddExpr(
        roundingMode: KExpr<KFpRoundingModeSort>, arg0: KExpr<T>, arg1: KExpr<T>
    ): KExpr<T> =
        mkSimplified(roundingMode, arg0, arg1, KContext::simplifyFpAddExpr, ::mkFpAddExprNoSimplify)

    open fun <T : KFpSort> mkFpAddExprNoSimplify(
        roundingMode: KExpr<KFpRoundingModeSort>,
        arg0: KExpr<T>,
        arg1: KExpr<T>
    ): KFpAddExpr<T> = fpAddExprCache.createIfContextActive {
        ensureContextMatch(roundingMode, arg0, arg1)
        KFpAddExpr(this, roundingMode, arg0, arg1)
    }.cast()

    private val fpSubExprCache = mkAstInterner<KFpSubExpr<out KFpSort>>()

    open fun <T : KFpSort> mkFpSubExpr(
        roundingMode: KExpr<KFpRoundingModeSort>, arg0: KExpr<T>, arg1: KExpr<T>
    ): KExpr<T> =
        mkSimplified(roundingMode, arg0, arg1, KContext::simplifyFpSubExpr, ::mkFpSubExprNoSimplify)

    open fun <T : KFpSort> mkFpSubExprNoSimplify(
        roundingMode: KExpr<KFpRoundingModeSort>,
        arg0: KExpr<T>,
        arg1: KExpr<T>
    ): KFpSubExpr<T> = fpSubExprCache.createIfContextActive {
        ensureContextMatch(roundingMode, arg0, arg1)
        KFpSubExpr(this, roundingMode, arg0, arg1)
    }.cast()

    private val fpMulExprCache = mkAstInterner<KFpMulExpr<out KFpSort>>()

    open fun <T : KFpSort> mkFpMulExpr(
        roundingMode: KExpr<KFpRoundingModeSort>, arg0: KExpr<T>, arg1: KExpr<T>
    ): KExpr<T> =
        mkSimplified(roundingMode, arg0, arg1, KContext::simplifyFpMulExpr, ::mkFpMulExprNoSimplify)

    open fun <T : KFpSort> mkFpMulExprNoSimplify(
        roundingMode: KExpr<KFpRoundingModeSort>,
        arg0: KExpr<T>,
        arg1: KExpr<T>
    ): KFpMulExpr<T> = fpMulExprCache.createIfContextActive {
        ensureContextMatch(roundingMode, arg0, arg1)
        KFpMulExpr(this, roundingMode, arg0, arg1)
    }.cast()

    private val fpDivExprCache = mkAstInterner<KFpDivExpr<out KFpSort>>()

    open fun <T : KFpSort> mkFpDivExpr(
        roundingMode: KExpr<KFpRoundingModeSort>, arg0: KExpr<T>, arg1: KExpr<T>
    ): KExpr<T> =
        mkSimplified(roundingMode, arg0, arg1, KContext::simplifyFpDivExpr, ::mkFpDivExprNoSimplify)

    open fun <T : KFpSort> mkFpDivExprNoSimplify(
        roundingMode: KExpr<KFpRoundingModeSort>,
        arg0: KExpr<T>,
        arg1: KExpr<T>
    ): KFpDivExpr<T> = fpDivExprCache.createIfContextActive {
        ensureContextMatch(roundingMode, arg0, arg1)
        KFpDivExpr(this, roundingMode, arg0, arg1)
    }.cast()

    private val fpFusedMulAddExprCache = mkAstInterner<KFpFusedMulAddExpr<out KFpSort>>()

    open fun <T : KFpSort> mkFpFusedMulAddExpr(
        roundingMode: KExpr<KFpRoundingModeSort>,
        arg0: KExpr<T>,
        arg1: KExpr<T>,
        arg2: KExpr<T>
    ): KExpr<T> = mkSimplified(
        roundingMode,
        arg0,
        arg1,
        arg2,
        KContext::simplifyFpFusedMulAddExpr,
        ::mkFpFusedMulAddExprNoSimplify
    )

    open fun <T : KFpSort> mkFpFusedMulAddExprNoSimplify(
        roundingMode: KExpr<KFpRoundingModeSort>,
        arg0: KExpr<T>,
        arg1: KExpr<T>,
        arg2: KExpr<T>
    ): KFpFusedMulAddExpr<T> = fpFusedMulAddExprCache.createIfContextActive {
        ensureContextMatch(roundingMode, arg0, arg1, arg2)
        KFpFusedMulAddExpr(this, roundingMode, arg0, arg1, arg2)
    }.cast()

    private val fpSqrtExprCache = mkAstInterner<KFpSqrtExpr<out KFpSort>>()

    open fun <T : KFpSort> mkFpSqrtExpr(roundingMode: KExpr<KFpRoundingModeSort>, value: KExpr<T>): KExpr<T> =
        mkSimplified(roundingMode, value, KContext::simplifyFpSqrtExpr, ::mkFpSqrtExprNoSimplify)

    open fun <T : KFpSort> mkFpSqrtExprNoSimplify(
        roundingMode: KExpr<KFpRoundingModeSort>,
        value: KExpr<T>
    ): KFpSqrtExpr<T> = fpSqrtExprCache.createIfContextActive {
        ensureContextMatch(roundingMode, value)
        KFpSqrtExpr(this, roundingMode, value)
    }.cast()

    private val fpRemExprCache = mkAstInterner<KFpRemExpr<out KFpSort>>()

    open fun <T : KFpSort> mkFpRemExpr(arg0: KExpr<T>, arg1: KExpr<T>): KExpr<T> =
        mkSimplified(arg0, arg1, KContext::simplifyFpRemExpr, ::mkFpRemExprNoSimplify)

    open fun <T : KFpSort> mkFpRemExprNoSimplify(arg0: KExpr<T>, arg1: KExpr<T>): KFpRemExpr<T> =
        fpRemExprCache.createIfContextActive {
            ensureContextMatch(arg0, arg1)
            KFpRemExpr(this, arg0, arg1)
        }.cast()

    private val fpRoundToIntegralExprCache = mkAstInterner<KFpRoundToIntegralExpr<out KFpSort>>()

    open fun <T : KFpSort> mkFpRoundToIntegralExpr(
        roundingMode: KExpr<KFpRoundingModeSort>, value: KExpr<T>
    ): KExpr<T> =
        mkSimplified(roundingMode, value, KContext::simplifyFpRoundToIntegralExpr, ::mkFpRoundToIntegralExprNoSimplify)

    open fun <T : KFpSort> mkFpRoundToIntegralExprNoSimplify(
        roundingMode: KExpr<KFpRoundingModeSort>,
        value: KExpr<T>
    ): KFpRoundToIntegralExpr<T> = fpRoundToIntegralExprCache.createIfContextActive {
        ensureContextMatch(roundingMode, value)
        KFpRoundToIntegralExpr(this, roundingMode, value)
    }.cast()

    private val fpMinExprCache = mkAstInterner<KFpMinExpr<out KFpSort>>()

    open fun <T : KFpSort> mkFpMinExpr(arg0: KExpr<T>, arg1: KExpr<T>): KExpr<T> =
        mkSimplified(arg0, arg1, KContext::simplifyFpMinExpr, ::mkFpMinExprNoSimplify)

    open fun <T : KFpSort> mkFpMinExprNoSimplify(arg0: KExpr<T>, arg1: KExpr<T>): KFpMinExpr<T> =
        fpMinExprCache.createIfContextActive {
            ensureContextMatch(arg0, arg1)
            KFpMinExpr(this, arg0, arg1)
        }.cast()

    private val fpMaxExprCache = mkAstInterner<KFpMaxExpr<out KFpSort>>()

    open fun <T : KFpSort> mkFpMaxExpr(arg0: KExpr<T>, arg1: KExpr<T>): KExpr<T> =
        mkSimplified(arg0, arg1, KContext::simplifyFpMaxExpr, ::mkFpMaxExprNoSimplify)

    open fun <T : KFpSort> mkFpMaxExprNoSimplify(arg0: KExpr<T>, arg1: KExpr<T>): KFpMaxExpr<T> =
        fpMaxExprCache.createIfContextActive {
            ensureContextMatch(arg0, arg1)
            KFpMaxExpr(this, arg0, arg1)
        }.cast()

    private val fpLessOrEqualExprCache = mkAstInterner<KFpLessOrEqualExpr<out KFpSort>>()

    open fun <T : KFpSort> mkFpLessOrEqualExpr(arg0: KExpr<T>, arg1: KExpr<T>): KExpr<KBoolSort> =
        mkSimplified(arg0, arg1, KContext::simplifyFpLessOrEqualExpr, ::mkFpLessOrEqualExprNoSimplify)

    open fun <T : KFpSort> mkFpLessOrEqualExprNoSimplify(arg0: KExpr<T>, arg1: KExpr<T>): KFpLessOrEqualExpr<T> =
        fpLessOrEqualExprCache.createIfContextActive {
            ensureContextMatch(arg0, arg1)
            KFpLessOrEqualExpr(this, arg0, arg1)
        }.cast()

    private val fpLessExprCache = mkAstInterner<KFpLessExpr<out KFpSort>>()

    open fun <T : KFpSort> mkFpLessExpr(arg0: KExpr<T>, arg1: KExpr<T>): KExpr<KBoolSort> =
        mkSimplified(arg0, arg1, KContext::simplifyFpLessExpr, ::mkFpLessExprNoSimplify)

    open fun <T : KFpSort> mkFpLessExprNoSimplify(arg0: KExpr<T>, arg1: KExpr<T>): KFpLessExpr<T> =
        fpLessExprCache.createIfContextActive {
            ensureContextMatch(arg0, arg1)
            KFpLessExpr(this, arg0, arg1)
        }.cast()

    private val fpGreaterOrEqualExprCache = mkAstInterner<KFpGreaterOrEqualExpr<out KFpSort>>()

    open fun <T : KFpSort> mkFpGreaterOrEqualExpr(arg0: KExpr<T>, arg1: KExpr<T>): KExpr<KBoolSort> =
        mkSimplified(arg0, arg1, KContext::simplifyFpGreaterOrEqualExpr, ::mkFpGreaterOrEqualExprNoSimplify)

    open fun <T : KFpSort> mkFpGreaterOrEqualExprNoSimplify(arg0: KExpr<T>, arg1: KExpr<T>): KFpGreaterOrEqualExpr<T> =
        fpGreaterOrEqualExprCache.createIfContextActive {
            ensureContextMatch(arg0, arg1)
            KFpGreaterOrEqualExpr(this, arg0, arg1)
        }.cast()

    private val fpGreaterExprCache = mkAstInterner<KFpGreaterExpr<out KFpSort>>()

    open fun <T : KFpSort> mkFpGreaterExpr(arg0: KExpr<T>, arg1: KExpr<T>): KExpr<KBoolSort> =
        mkSimplified(arg0, arg1, KContext::simplifyFpGreaterExpr, ::mkFpGreaterExprNoSimplify)

    open fun <T : KFpSort> mkFpGreaterExprNoSimplify(arg0: KExpr<T>, arg1: KExpr<T>): KFpGreaterExpr<T> =
        fpGreaterExprCache.createIfContextActive {
            ensureContextMatch(arg0, arg1)
            KFpGreaterExpr(this, arg0, arg1)
        }.cast()

    private val fpEqualExprCache = mkAstInterner<KFpEqualExpr<out KFpSort>>()

    open fun <T : KFpSort> mkFpEqualExpr(arg0: KExpr<T>, arg1: KExpr<T>): KExpr<KBoolSort> =
        mkSimplified(arg0, arg1, KContext::simplifyFpEqualExpr, ::mkFpEqualExprNoSimplify)

    open fun <T : KFpSort> mkFpEqualExprNoSimplify(arg0: KExpr<T>, arg1: KExpr<T>): KFpEqualExpr<T> =
        fpEqualExprCache.createIfContextActive {
            ensureContextMatch(arg0, arg1)
            KFpEqualExpr(this, arg0, arg1)
        }.cast()

    private val fpIsNormalExprCache = mkAstInterner<KFpIsNormalExpr<out KFpSort>>()

    open fun <T : KFpSort> mkFpIsNormalExpr(value: KExpr<T>): KExpr<KBoolSort> =
        mkSimplified(value, KContext::simplifyFpIsNormalExpr, ::mkFpIsNormalExprNoSimplify)

    open fun <T : KFpSort> mkFpIsNormalExprNoSimplify(value: KExpr<T>): KFpIsNormalExpr<T> =
        fpIsNormalExprCache.createIfContextActive {
            ensureContextMatch(value)
            KFpIsNormalExpr(this, value)
        }.cast()

    private val fpIsSubnormalExprCache = mkAstInterner<KFpIsSubnormalExpr<out KFpSort>>()

    open fun <T : KFpSort> mkFpIsSubnormalExpr(value: KExpr<T>): KExpr<KBoolSort> =
        mkSimplified(value, KContext::simplifyFpIsSubnormalExpr, ::mkFpIsSubnormalExprNoSimplify)

    open fun <T : KFpSort> mkFpIsSubnormalExprNoSimplify(value: KExpr<T>): KFpIsSubnormalExpr<T> =
        fpIsSubnormalExprCache.createIfContextActive {
            ensureContextMatch(value)
            KFpIsSubnormalExpr(this, value)
        }.cast()

    private val fpIsZeroExprCache = mkAstInterner<KFpIsZeroExpr<out KFpSort>>()

    open fun <T : KFpSort> mkFpIsZeroExpr(value: KExpr<T>): KExpr<KBoolSort> =
        mkSimplified(value, KContext::simplifyFpIsZeroExpr, ::mkFpIsZeroExprNoSimplify)

    open fun <T : KFpSort> mkFpIsZeroExprNoSimplify(value: KExpr<T>): KFpIsZeroExpr<T> =
        fpIsZeroExprCache.createIfContextActive {
            ensureContextMatch(value)
            KFpIsZeroExpr(this, value)
        }.cast()

    private val fpIsInfiniteExprCache = mkAstInterner<KFpIsInfiniteExpr<out KFpSort>>()

    open fun <T : KFpSort> mkFpIsInfiniteExpr(value: KExpr<T>): KExpr<KBoolSort> =
        mkSimplified(value, KContext::simplifyFpIsInfiniteExpr, ::mkFpIsInfiniteExprNoSimplify)

    open fun <T : KFpSort> mkFpIsInfiniteExprNoSimplify(value: KExpr<T>): KFpIsInfiniteExpr<T> =
        fpIsInfiniteExprCache.createIfContextActive {
            ensureContextMatch(value)
            KFpIsInfiniteExpr(this, value)
        }.cast()

    private val fpIsNaNExprCache = mkAstInterner<KFpIsNaNExpr<out KFpSort>>()

    open fun <T : KFpSort> mkFpIsNaNExpr(value: KExpr<T>): KExpr<KBoolSort> =
        mkSimplified(value, KContext::simplifyFpIsNaNExpr, ::mkFpIsNaNExprNoSimplify)

    open fun <T : KFpSort> mkFpIsNaNExprNoSimplify(value: KExpr<T>): KFpIsNaNExpr<T> =
        fpIsNaNExprCache.createIfContextActive {
            ensureContextMatch(value)
            KFpIsNaNExpr(this, value)
        }.cast()

    private val fpIsNegativeExprCache = mkAstInterner<KFpIsNegativeExpr<out KFpSort>>()

    open fun <T : KFpSort> mkFpIsNegativeExpr(value: KExpr<T>): KExpr<KBoolSort> =
        mkSimplified(value, KContext::simplifyFpIsNegativeExpr, ::mkFpIsNegativeExprNoSimplify)

    open fun <T : KFpSort> mkFpIsNegativeExprNoSimplify(value: KExpr<T>): KFpIsNegativeExpr<T> =
        fpIsNegativeExprCache.createIfContextActive {
            ensureContextMatch(value)
            KFpIsNegativeExpr(this, value)
        }.cast()

    private val fpIsPositiveExprCache = mkAstInterner<KFpIsPositiveExpr<out KFpSort>>()

    open fun <T : KFpSort> mkFpIsPositiveExpr(value: KExpr<T>): KExpr<KBoolSort> =
        mkSimplified(value, KContext::simplifyFpIsPositiveExpr, ::mkFpIsPositiveExprNoSimplify)

    open fun <T : KFpSort> mkFpIsPositiveExprNoSimplify(value: KExpr<T>): KFpIsPositiveExpr<T> =
        fpIsPositiveExprCache.createIfContextActive {
            ensureContextMatch(value)
            KFpIsPositiveExpr(this, value)
        }.cast()

    private val fpToBvExprCache = mkAstInterner<KFpToBvExpr<out KFpSort>>()

    open fun <T : KFpSort> mkFpToBvExpr(
        roundingMode: KExpr<KFpRoundingModeSort>,
        value: KExpr<T>,
        bvSize: Int,
        isSigned: Boolean
    ): KExpr<KBvSort> =
        mkSimplified(roundingMode, value, bvSize, isSigned, KContext::simplifyFpToBvExpr, ::mkFpToBvExprNoSimplify)

    open fun <T : KFpSort> mkFpToBvExprNoSimplify(
        roundingMode: KExpr<KFpRoundingModeSort>,
        value: KExpr<T>,
        bvSize: Int,
        isSigned: Boolean
    ): KFpToBvExpr<T> = fpToBvExprCache.createIfContextActive {
        ensureContextMatch(roundingMode, value)
        KFpToBvExpr(this, roundingMode, value, bvSize, isSigned)
    }.cast()

    private val fpToRealExprCache = mkAstInterner<KFpToRealExpr<out KFpSort>>()

    open fun <T : KFpSort> mkFpToRealExpr(value: KExpr<T>): KExpr<KRealSort> =
        mkSimplified(value, KContext::simplifyFpToRealExpr, ::mkFpToRealExprNoSimplify)

    open fun <T : KFpSort> mkFpToRealExprNoSimplify(value: KExpr<T>): KFpToRealExpr<T> =
        fpToRealExprCache.createIfContextActive {
            ensureContextMatch(value)
            KFpToRealExpr(this, value)
        }.cast()

    private val fpToIEEEBvExprCache = mkAstInterner<KFpToIEEEBvExpr<out KFpSort>>()

    open fun <T : KFpSort> mkFpToIEEEBvExpr(value: KExpr<T>): KExpr<KBvSort> =
        mkSimplified(value, KContext::simplifyFpToIEEEBvExpr, ::mkFpToIEEEBvExprNoSimplify)

    open fun <T : KFpSort> mkFpToIEEEBvExprNoSimplify(value: KExpr<T>): KFpToIEEEBvExpr<T> =
        fpToIEEEBvExprCache.createIfContextActive {
            ensureContextMatch(value)
            KFpToIEEEBvExpr(this, value)
        }.cast()

    private val fpFromBvExprCache = mkAstInterner<KFpFromBvExpr<out KFpSort>>()

    open fun <T : KFpSort> mkFpFromBvExpr(
        sign: KExpr<KBv1Sort>,
        biasedExponent: KExpr<out KBvSort>,
        significand: KExpr<out KBvSort>
    ): KExpr<T> = mkSimplified(
        sign,
        biasedExponent,
        significand,
        KContext::simplifyFpFromBvExpr,
        ::mkFpFromBvExprNoSimplify
    )

    open fun <T : KFpSort> mkFpFromBvExprNoSimplify(
        sign: KExpr<KBv1Sort>,
        biasedExponent: KExpr<out KBvSort>,
        significand: KExpr<out KBvSort>,
    ): KFpFromBvExpr<T> = fpFromBvExprCache.createIfContextActive {
        ensureContextMatch(sign, biasedExponent, significand)

        val exponentBits = biasedExponent.sort.sizeBits
        // +1 it required since bv doesn't contain `hidden bit`
        val significandBits = significand.sort.sizeBits + 1u
        val sort = mkFpSort(exponentBits, significandBits)

        KFpFromBvExpr(this, sort, sign, biasedExponent, significand)
    }.cast()

    private val fpToFpExprCache = mkAstInterner<KFpToFpExpr<out KFpSort>>()
    private val realToFpExprCache = mkAstInterner<KRealToFpExpr<out KFpSort>>()
    private val bvToFpExprCache = mkAstInterner<KBvToFpExpr<out KFpSort>>()

    open fun <T : KFpSort> mkFpToFpExpr(
        sort: T,
        roundingMode: KExpr<KFpRoundingModeSort>,
        value: KExpr<out KFpSort>
    ): KExpr<T> = mkSimplified(sort, roundingMode, value, KContext::simplifyFpToFpExpr, ::mkFpToFpExprNoSimplify)

    open fun <T : KFpSort> mkFpToFpExprNoSimplify(
        sort: T,
        roundingMode: KExpr<KFpRoundingModeSort>,
        value: KExpr<out KFpSort>
    ): KFpToFpExpr<T> = fpToFpExprCache.createIfContextActive {
        ensureContextMatch(sort, roundingMode, value)
        KFpToFpExpr(this, sort, roundingMode, value)
    }.cast()

    open fun <T : KFpSort> mkRealToFpExpr(
        sort: T,
        roundingMode: KExpr<KFpRoundingModeSort>,
        value: KExpr<KRealSort>
    ): KExpr<T> = mkSimplified(sort, roundingMode, value, KContext::simplifyRealToFpExpr, ::mkRealToFpExprNoSimplify)

    open fun <T : KFpSort> mkRealToFpExprNoSimplify(
        sort: T,
        roundingMode: KExpr<KFpRoundingModeSort>,
        value: KExpr<KRealSort>
    ): KRealToFpExpr<T> = realToFpExprCache.createIfContextActive {
        ensureContextMatch(sort, roundingMode, value)
        KRealToFpExpr(this, sort, roundingMode, value)
    }.cast()

    open fun <T : KFpSort> mkBvToFpExpr(
        sort: T,
        roundingMode: KExpr<KFpRoundingModeSort>,
        value: KExpr<KBvSort>,
        signed: Boolean
    ): KExpr<T> =
        mkSimplified(sort, roundingMode, value, signed, KContext::simplifyBvToFpExpr, ::mkBvToFpExprNoSimplify)

    open fun <T : KFpSort> mkBvToFpExprNoSimplify(
        sort: T,
        roundingMode: KExpr<KFpRoundingModeSort>,
        value: KExpr<KBvSort>,
        signed: Boolean
    ): KBvToFpExpr<T> = bvToFpExprCache.createIfContextActive {
        ensureContextMatch(sort, roundingMode, value)
        KBvToFpExpr(this, sort, roundingMode, value, signed)
    }.cast()

    // quantifiers
    private val existentialQuantifierCache = mkAstInterner<KExistentialQuantifier>()

    open fun mkExistentialQuantifier(body: KExpr<KBoolSort>, bounds: List<KDecl<*>>) =
        existentialQuantifierCache.createIfContextActive {
            ensureContextMatch(body)
            ensureContextMatch(bounds)
            KExistentialQuantifier(this, body, bounds)
        }

    private val universalQuantifierCache = mkAstInterner<KUniversalQuantifier>()

    open fun mkUniversalQuantifier(body: KExpr<KBoolSort>, bounds: List<KDecl<*>>) =
        universalQuantifierCache.createIfContextActive {
            ensureContextMatch(body)
            ensureContextMatch(bounds)
            KUniversalQuantifier(this, body, bounds)
        }

    private val uninterpretedSortDefaultValueCache =
        mkCache<KUninterpretedSort, KExpr<KUninterpretedSort>>(operationMode)

    open fun uninterpretedSortDefaultValue(sort: KUninterpretedSort): KExpr<KUninterpretedSort> =
        uninterpretedSortDefaultValueCache.computeIfAbsent(sort) {
            ensureContextMatch(it)
            mkFreshConst("${it.name}_default_value", it)
        }

    /*
    * declarations
    * */

    // functions
    private val funcDeclCache = mkAstInterner<KUninterpretedFuncDecl<out KSort>>()

    fun <T : KSort> mkFuncDecl(
        name: String,
        sort: T,
        args: List<KSort>
    ): KFuncDecl<T> = if (args.isEmpty()) {
        mkConstDecl(name, sort)
    } else {
        funcDeclCache.createIfContextActive {
            ensureContextMatch(sort)
            ensureContextMatch(args)
            KUninterpretedFuncDecl(this, name, sort, args)
        }.cast()
    }

    private val constDeclCache = mkAstInterner<KUninterpretedConstDecl<out KSort>>()

    @Suppress("MemberVisibilityCanBePrivate")
    fun <T : KSort> mkConstDecl(name: String, sort: T): KUninterpretedConstDecl<T> =
        constDeclCache.createIfContextActive {
            ensureContextMatch(sort)
            KUninterpretedConstDecl(this, name, sort)
        }.cast()

    /* Since any two KUninterpretedFuncDecl are only equivalent if they are the same kotlin object,
     * we can guarantee that the returned func decl is not equal to any other declaration.
    */
    private var freshConstIdx = 0
    fun <T : KSort> mkFreshFuncDecl(name: String, sort: T, args: List<KSort>): KFuncDecl<T> {
        if (args.isEmpty()) return mkFreshConstDecl(name, sort)

        ensureContextMatch(sort)
        ensureContextMatch(args)

        return KUninterpretedFuncDecl(this, "$name!fresh!${freshConstIdx++}", sort, args)
    }

    fun <T : KSort> mkFreshConstDecl(name: String, sort: T): KConstDecl<T> {
        ensureContextMatch(sort)

        return KUninterpretedConstDecl(this, "$name!fresh!${freshConstIdx++}", sort)
    }

    // bool
    fun mkFalseDecl(): KFalseDecl = KFalseDecl(this)

    fun mkTrueDecl(): KTrueDecl = KTrueDecl(this)

    fun mkAndDecl(): KAndDecl = KAndDecl(this)

    fun mkOrDecl(): KOrDecl = KOrDecl(this)

    fun mkNotDecl(): KNotDecl = KNotDecl(this)

    fun mkImpliesDecl(): KImpliesDecl = KImpliesDecl(this)

    fun mkXorDecl(): KXorDecl = KXorDecl(this)

    fun <T : KSort> mkEqDecl(arg: T): KEqDecl<T> = KEqDecl(this, arg)

    fun <T : KSort> mkDistinctDecl(arg: T): KDistinctDecl<T> = KDistinctDecl(this, arg)

    fun <T : KSort> mkIteDecl(arg: T): KIteDecl<T> = KIteDecl(this, arg)

    // array
    fun <D : KSort, R : KSort> mkArraySelectDecl(array: KArraySort<D, R>): KArraySelectDecl<D, R> =
        KArraySelectDecl(this, array)

    fun <D0 : KSort, D1 : KSort, R : KSort> mkArraySelectDecl(
        array: KArray2Sort<D0, D1, R>
    ): KArray2SelectDecl<D0, D1, R> = KArray2SelectDecl(this, array)

    fun <D0 : KSort, D1 : KSort, D2 : KSort, R : KSort> mkArraySelectDecl(
        array: KArray3Sort<D0, D1, D2, R>
    ): KArray3SelectDecl<D0, D1, D2, R> = KArray3SelectDecl(this, array)

    fun <R : KSort> mkArrayNSelectDecl(array: KArrayNSort<R>): KArrayNSelectDecl<R> =
        KArrayNSelectDecl(this, array)

    fun <D : KSort, R : KSort> mkArrayStoreDecl(array: KArraySort<D, R>): KArrayStoreDecl<D, R> =
        KArrayStoreDecl(this, array)

    fun <D0 : KSort, D1 : KSort, R : KSort> mkArrayStoreDecl(
        array: KArray2Sort<D0, D1, R>
    ): KArray2StoreDecl<D0, D1, R> = KArray2StoreDecl(this, array)

    fun <D0 : KSort, D1 : KSort, D2 : KSort, R : KSort> mkArrayStoreDecl(
        array: KArray3Sort<D0, D1, D2, R>
    ): KArray3StoreDecl<D0, D1, D2, R> = KArray3StoreDecl(this, array)

    fun <R : KSort> mkArrayNStoreDecl(array: KArrayNSort<R>): KArrayNStoreDecl<R> =
        KArrayNStoreDecl(this, array)

    fun <A : KArraySortBase<R>, R : KSort> mkArrayConstDecl(array: A): KArrayConstDecl<A, R> =
        KArrayConstDecl(this, array)

    // arith
    fun <T : KArithSort> mkArithAddDecl(arg: T): KArithAddDecl<T> =
        KArithAddDecl(this, arg)

    fun <T : KArithSort> mkArithSubDecl(arg: T): KArithSubDecl<T> =
        KArithSubDecl(this, arg)

    fun <T : KArithSort> mkArithMulDecl(arg: T): KArithMulDecl<T> =
        KArithMulDecl(this, arg)

    fun <T : KArithSort> mkArithDivDecl(arg: T): KArithDivDecl<T> =
        KArithDivDecl(this, arg)

    fun <T : KArithSort> mkArithPowerDecl(arg: T): KArithPowerDecl<T> =
        KArithPowerDecl(this, arg)

    fun <T : KArithSort> mkArithUnaryMinusDecl(arg: T): KArithUnaryMinusDecl<T> =
        KArithUnaryMinusDecl(this, arg)

    fun <T : KArithSort> mkArithGeDecl(arg: T): KArithGeDecl<T> =
        KArithGeDecl(this, arg)

    fun <T : KArithSort> mkArithGtDecl(arg: T): KArithGtDecl<T> =
        KArithGtDecl(this, arg)

    fun <T : KArithSort> mkArithLeDecl(arg: T): KArithLeDecl<T> =
        KArithLeDecl(this, arg)

    fun <T : KArithSort> mkArithLtDecl(arg: T): KArithLtDecl<T> =
        KArithLtDecl(this, arg)

    // int
    fun mkIntModDecl(): KIntModDecl = KIntModDecl(this)

    fun mkIntToRealDecl(): KIntToRealDecl = KIntToRealDecl(this)

    fun mkIntRemDecl(): KIntRemDecl = KIntRemDecl(this)

    fun mkIntNumDecl(value: String): KIntNumDecl = KIntNumDecl(this, value)

    // real
    fun mkRealIsIntDecl(): KRealIsIntDecl = KRealIsIntDecl(this)

    fun mkRealToIntDecl(): KRealToIntDecl = KRealToIntDecl(this)

    fun mkRealNumDecl(value: String): KRealNumDecl = KRealNumDecl(this, value)

    fun mkBvDecl(value: Boolean): KDecl<KBv1Sort> =
        KBitVec1ValueDecl(this, value)

    fun mkBvDecl(value: Byte): KDecl<KBv8Sort> =
        KBitVec8ValueDecl(this, value)

    fun mkBvDecl(value: Short): KDecl<KBv16Sort> =
        KBitVec16ValueDecl(this, value)

    fun mkBvDecl(value: Int): KDecl<KBv32Sort> =
        KBitVec32ValueDecl(this, value)

    fun mkBvDecl(value: Long): KDecl<KBv64Sort> =
        KBitVec64ValueDecl(this, value)

    fun mkBvDecl(value: BigInteger, size: UInt): KDecl<KBvSort> =
        mkBvDeclFromUnsignedBigInteger(value.normalizeValue(size), size)

    fun mkBvDecl(value: String, sizeBits: UInt): KDecl<KBvSort> =
        mkBvDecl(value.toBigInteger(radix = 2), sizeBits)

    private fun mkBvDeclFromUnsignedBigInteger(
        value: BigInteger,
        sizeBits: UInt
    ): KDecl<KBvSort> {
        require(value.signum() >= 0) {
            "Unsigned value required, but $value provided"
        }
        return when (sizeBits.toInt()) {
            1 -> mkBvDecl(value != BigInteger.ZERO).cast()
            Byte.SIZE_BITS -> mkBvDecl(value.toByte()).cast()
            Short.SIZE_BITS -> mkBvDecl(value.toShort()).cast()
            Int.SIZE_BITS -> mkBvDecl(value.toInt()).cast()
            Long.SIZE_BITS -> mkBvDecl(value.toLong()).cast()
            else -> KBitVecCustomSizeValueDecl(this, value, sizeBits)
        }
    }

    fun <T : KBvSort> mkBvNotDecl(sort: T): KBvNotDecl<T> =
        KBvNotDecl(this, sort)

    fun <T : KBvSort> mkBvReductionAndDecl(sort: T): KBvReductionAndDecl<T> =
        KBvReductionAndDecl(this, sort)

    fun <T : KBvSort> mkBvReductionOrDecl(sort: T): KBvReductionOrDecl<T> =
        KBvReductionOrDecl(this, sort)

    fun <T : KBvSort> mkBvAndDecl(arg0: T, arg1: T): KBvAndDecl<T> =
        KBvAndDecl(this, arg0, arg1)

    fun <T : KBvSort> mkBvOrDecl(arg0: T, arg1: T): KBvOrDecl<T> =
        KBvOrDecl(this, arg0, arg1)

    fun <T : KBvSort> mkBvXorDecl(arg0: T, arg1: T): KBvXorDecl<T> =
        KBvXorDecl(this, arg0, arg1)

    fun <T : KBvSort> mkBvNAndDecl(arg0: T, arg1: T): KBvNAndDecl<T> =
        KBvNAndDecl(this, arg0, arg1)

    fun <T : KBvSort> mkBvNorDecl(arg0: T, arg1: T): KBvNorDecl<T> =
        KBvNorDecl(this, arg0, arg1)

    fun <T : KBvSort> mkBvXNorDecl(arg0: T, arg1: T): KBvXNorDecl<T> =
        KBvXNorDecl(this, arg0, arg1)

    fun <T : KBvSort> mkBvNegationDecl(sort: T): KBvNegationDecl<T> =
        KBvNegationDecl(this, sort)

    fun <T : KBvSort> mkBvAddDecl(arg0: T, arg1: T): KBvAddDecl<T> =
        KBvAddDecl(this, arg0, arg1)

    fun <T : KBvSort> mkBvSubDecl(arg0: T, arg1: T): KBvSubDecl<T> =
        KBvSubDecl(this, arg0, arg1)

    fun <T : KBvSort> mkBvMulDecl(arg0: T, arg1: T): KBvMulDecl<T> =
        KBvMulDecl(this, arg0, arg1)

    fun <T : KBvSort> mkBvUnsignedDivDecl(arg0: T, arg1: T): KBvUnsignedDivDecl<T> =
        KBvUnsignedDivDecl(this, arg0, arg1)

    fun <T : KBvSort> mkBvSignedDivDecl(arg0: T, arg1: T): KBvSignedDivDecl<T> =
        KBvSignedDivDecl(this, arg0, arg1)

    fun <T : KBvSort> mkBvUnsignedRemDecl(arg0: T, arg1: T): KBvUnsignedRemDecl<T> =
        KBvUnsignedRemDecl(this, arg0, arg1)

    fun <T : KBvSort> mkBvSignedRemDecl(arg0: T, arg1: T): KBvSignedRemDecl<T> =
        KBvSignedRemDecl(this, arg0, arg1)

    fun <T : KBvSort> mkBvSignedModDecl(arg0: T, arg1: T): KBvSignedModDecl<T> =
        KBvSignedModDecl(this, arg0, arg1)

    fun <T : KBvSort> mkBvUnsignedLessDecl(arg0: T, arg1: T): KBvUnsignedLessDecl<T> =
        KBvUnsignedLessDecl(this, arg0, arg1)

    fun <T : KBvSort> mkBvSignedLessDecl(arg0: T, arg1: T): KBvSignedLessDecl<T> =
        KBvSignedLessDecl(this, arg0, arg1)

    fun <T : KBvSort> mkBvSignedLessOrEqualDecl(arg0: T, arg1: T): KBvSignedLessOrEqualDecl<T> =
        KBvSignedLessOrEqualDecl(this, arg0, arg1)

    fun <T : KBvSort> mkBvUnsignedLessOrEqualDecl(arg0: T, arg1: T): KBvUnsignedLessOrEqualDecl<T> =
        KBvUnsignedLessOrEqualDecl(this, arg0, arg1)

    fun <T : KBvSort> mkBvUnsignedGreaterOrEqualDecl(arg0: T, arg1: T): KBvUnsignedGreaterOrEqualDecl<T> =
        KBvUnsignedGreaterOrEqualDecl(this, arg0, arg1)

    fun <T : KBvSort> mkBvSignedGreaterOrEqualDecl(arg0: T, arg1: T): KBvSignedGreaterOrEqualDecl<T> =
        KBvSignedGreaterOrEqualDecl(this, arg0, arg1)

    fun <T : KBvSort> mkBvUnsignedGreaterDecl(arg0: T, arg1: T): KBvUnsignedGreaterDecl<T> =
        KBvUnsignedGreaterDecl(this, arg0, arg1)

    fun <T : KBvSort> mkBvSignedGreaterDecl(arg0: T, arg1: T): KBvSignedGreaterDecl<T> =
        KBvSignedGreaterDecl(this, arg0, arg1)

    fun mkBvConcatDecl(arg0: KBvSort, arg1: KBvSort): KBvConcatDecl =
        KBvConcatDecl(this, arg0, arg1)

    fun mkBvExtractDecl(high: Int, low: Int, value: KExpr<KBvSort>): KBvExtractDecl =
        KBvExtractDecl(this, high, low, value)

    fun mkBvSignExtensionDecl(i: Int, value: KBvSort): KSignExtDecl =
        KSignExtDecl(this, i, value)

    fun mkBvZeroExtensionDecl(i: Int, value: KBvSort): KZeroExtDecl =
        KZeroExtDecl(this, i, value)

    fun mkBvRepeatDecl(i: Int, value: KBvSort): KBvRepeatDecl =
        KBvRepeatDecl(this, i, value)

    fun <T : KBvSort> mkBvShiftLeftDecl(arg0: T, arg1: T): KBvShiftLeftDecl<T> =
        KBvShiftLeftDecl(this, arg0, arg1)

    fun <T : KBvSort> mkBvLogicalShiftRightDecl(arg0: T, arg1: T): KBvLogicalShiftRightDecl<T> =
        KBvLogicalShiftRightDecl(this, arg0, arg1)

    fun <T : KBvSort> mkBvArithShiftRightDecl(arg0: T, arg1: T): KBvArithShiftRightDecl<T> =
        KBvArithShiftRightDecl(this, arg0, arg1)

    fun <T : KBvSort> mkBvRotateLeftDecl(arg0: T, arg1: T): KBvRotateLeftDecl<T> =
        KBvRotateLeftDecl(this, arg0, arg1)

    fun <T : KBvSort> mkBvRotateLeftIndexedDecl(i: Int, valueSort: T): KBvRotateLeftIndexedDecl<T> =
        KBvRotateLeftIndexedDecl(this, i, valueSort)

    fun <T : KBvSort> mkBvRotateRightDecl(arg0: T, arg1: T): KBvRotateRightDecl<T> =
        KBvRotateRightDecl(this, arg0, arg1)

    fun <T : KBvSort> mkBvRotateRightIndexedDecl(i: Int, valueSort: T): KBvRotateRightIndexedDecl<T> =
        KBvRotateRightIndexedDecl(this, i, valueSort)

    fun mkBv2IntDecl(value: KBvSort, isSigned: Boolean): KBv2IntDecl =
        KBv2IntDecl(this, value, isSigned)

    fun <T : KBvSort> mkBvAddNoOverflowDecl(arg0: T, arg1: T, isSigned: Boolean): KBvAddNoOverflowDecl<T> =
        KBvAddNoOverflowDecl(this, arg0, arg1, isSigned)

    fun <T : KBvSort> mkBvAddNoUnderflowDecl(arg0: T, arg1: T): KBvAddNoUnderflowDecl<T> =
        KBvAddNoUnderflowDecl(this, arg0, arg1)

    fun <T : KBvSort> mkBvSubNoOverflowDecl(arg0: T, arg1: T): KBvSubNoOverflowDecl<T> =
        KBvSubNoOverflowDecl(this, arg0, arg1)

    fun <T : KBvSort> mkBvSubNoUnderflowDecl(arg0: T, arg1: T, isSigned: Boolean): KBvSubNoUnderflowDecl<T> =
        KBvSubNoUnderflowDecl(this, arg0, arg1, isSigned)

    fun <T : KBvSort> mkBvDivNoOverflowDecl(arg0: T, arg1: T): KBvDivNoOverflowDecl<T> =
        KBvDivNoOverflowDecl(this, arg0, arg1)

    fun <T : KBvSort> mkBvNegationNoOverflowDecl(value: T): KBvNegNoOverflowDecl<T> =
        KBvNegNoOverflowDecl(this, value)

    fun <T : KBvSort> mkBvMulNoOverflowDecl(arg0: T, arg1: T, isSigned: Boolean): KBvMulNoOverflowDecl<T> =
        KBvMulNoOverflowDecl(this, arg0, arg1, isSigned)

    fun <T : KBvSort> mkBvMulNoUnderflowDecl(arg0: T, arg1: T): KBvMulNoUnderflowDecl<T> =
        KBvMulNoUnderflowDecl(this, arg0, arg1)

    // FP
    fun mkFp16Decl(value: Float): KFp16Decl = KFp16Decl(this, value)

    fun mkFp32Decl(value: Float): KFp32Decl = KFp32Decl(this, value)

    fun mkFp64Decl(value: Double): KFp64Decl = KFp64Decl(this, value)

    fun mkFp128Decl(significandBits: KBitVecValue<*>, unbiasedExponent: KBitVecValue<*>, signBit: Boolean): KFp128Decl =
        KFp128Decl(this, significandBits, unbiasedExponent, signBit)

    fun mkFp128DeclBiased(
        significandBits: KBitVecValue<*>,
        biasedExponent: KBitVecValue<*>,
        signBit: Boolean
    ): KFp128Decl = mkFp128Decl(
        significandBits = significandBits,
        unbiasedExponent = unbiasFp128Exponent(biasedExponent),
        signBit = signBit
    )

    fun <T : KFpSort> mkFpCustomSizeDecl(
        significandSize: UInt,
        exponentSize: UInt,
        significand: KBitVecValue<*>,
        unbiasedExponent: KBitVecValue<*>,
        signBit: Boolean
    ): KFpDecl<T> {
        val sort = mkFpSort(exponentSize, significandSize)

        if (sort is KFpCustomSizeSort) {
            return KFpCustomSizeDecl(
                this, significandSize, exponentSize, significand, unbiasedExponent, signBit
            ).cast()
        }

        val intSignBit = if (signBit) 1 else 0

        return when (sort) {
            is KFp16Sort -> {
                val fp16Number = constructFp16Number(unbiasedExponent.longValue(), significand.longValue(), intSignBit)

                mkFp16Decl(fp16Number).cast()
            }

            is KFp32Sort -> {
                val fp32Number = constructFp32Number(unbiasedExponent.longValue(), significand.longValue(), intSignBit)

                mkFp32Decl(fp32Number).cast()
            }

            is KFp64Sort -> {
                val fp64Number = constructFp64Number(unbiasedExponent.longValue(), significand.longValue(), intSignBit)

                mkFp64Decl(fp64Number).cast()
            }

            is KFp128Sort -> {
                mkFp128Decl(significand, unbiasedExponent, signBit).cast()
            }

            else -> error("Sort declaration for an unknown $sort")
        }
    }

    fun <T : KFpSort> mkFpCustomSizeDeclBiased(
        significandSize: UInt,
        exponentSize: UInt,
        significand: KBitVecValue<*>,
        biasedExponent: KBitVecValue<*>,
        signBit: Boolean
    ): KFpDecl<T> = mkFpCustomSizeDecl(
        significandSize = significandSize,
        exponentSize = exponentSize,
        significand = significand,
        unbiasedExponent = unbiasFpCustomSizeExponent(biasedExponent, exponentSize),
        signBit = signBit
    )

    fun mkFpRoundingModeDecl(value: KFpRoundingMode): KFpRoundingModeDecl =
        KFpRoundingModeDecl(this, value)

    fun <T : KFpSort> mkFpAbsDecl(valueSort: T): KFpAbsDecl<T> =
        KFpAbsDecl(this, valueSort)

    fun <T : KFpSort> mkFpNegationDecl(valueSort: T): KFpNegationDecl<T> =
        KFpNegationDecl(this, valueSort)

    fun <T : KFpSort> mkFpAddDecl(
        roundingMode: KFpRoundingModeSort,
        arg0Sort: T,
        arg1Sort: T
    ): KFpAddDecl<T> = KFpAddDecl(this, roundingMode, arg0Sort, arg1Sort)

    fun <T : KFpSort> mkFpSubDecl(
        roundingMode: KFpRoundingModeSort,
        arg0Sort: T,
        arg1Sort: T
    ): KFpSubDecl<T> = KFpSubDecl(this, roundingMode, arg0Sort, arg1Sort)

    fun <T : KFpSort> mkFpMulDecl(
        roundingMode: KFpRoundingModeSort,
        arg0Sort: T,
        arg1Sort: T
    ): KFpMulDecl<T> = KFpMulDecl(this, roundingMode, arg0Sort, arg1Sort)

    fun <T : KFpSort> mkFpDivDecl(
        roundingMode: KFpRoundingModeSort,
        arg0Sort: T,
        arg1Sort: T
    ): KFpDivDecl<T> = KFpDivDecl(this, roundingMode, arg0Sort, arg1Sort)

    fun <T : KFpSort> mkFpFusedMulAddDecl(
        roundingMode: KFpRoundingModeSort,
        arg0Sort: T,
        arg1Sort: T,
        arg2Sort: T
    ): KFpFusedMulAddDecl<T> = KFpFusedMulAddDecl(this, roundingMode, arg0Sort, arg1Sort, arg2Sort)

    fun <T : KFpSort> mkFpSqrtDecl(roundingMode: KFpRoundingModeSort, valueSort: T): KFpSqrtDecl<T> =
        KFpSqrtDecl(this, roundingMode, valueSort)

    fun <T : KFpSort> mkFpRemDecl(arg0Sort: T, arg1Sort: T): KFpRemDecl<T> =
        KFpRemDecl(this, arg0Sort, arg1Sort)

    fun <T : KFpSort> mkFpRoundToIntegralDecl(
        roundingMode: KFpRoundingModeSort,
        valueSort: T
    ): KFpRoundToIntegralDecl<T> = KFpRoundToIntegralDecl(this, roundingMode, valueSort)

    fun <T : KFpSort> mkFpMinDecl(arg0Sort: T, arg1Sort: T): KFpMinDecl<T> =
        KFpMinDecl(this, arg0Sort, arg1Sort)

    fun <T : KFpSort> mkFpMaxDecl(arg0Sort: T, arg1Sort: T): KFpMaxDecl<T> =
        KFpMaxDecl(this, arg0Sort, arg1Sort)

    fun <T : KFpSort> mkFpLessOrEqualDecl(arg0Sort: T, arg1Sort: T): KFpLessOrEqualDecl<T> =
        KFpLessOrEqualDecl(this, arg0Sort, arg1Sort)

    fun <T : KFpSort> mkFpLessDecl(arg0Sort: T, arg1Sort: T): KFpLessDecl<T> =
        KFpLessDecl(this, arg0Sort, arg1Sort)

    fun <T : KFpSort> mkFpGreaterOrEqualDecl(arg0Sort: T, arg1Sort: T): KFpGreaterOrEqualDecl<T> =
        KFpGreaterOrEqualDecl(this, arg0Sort, arg1Sort)

    fun <T : KFpSort> mkFpGreaterDecl(arg0Sort: T, arg1Sort: T): KFpGreaterDecl<T> =
        KFpGreaterDecl(this, arg0Sort, arg1Sort)

    fun <T : KFpSort> mkFpEqualDecl(arg0Sort: T, arg1Sort: T): KFpEqualDecl<T> =
        KFpEqualDecl(this, arg0Sort, arg1Sort)

    fun <T : KFpSort> mkFpIsNormalDecl(valueSort: T): KFpIsNormalDecl<T> =
        KFpIsNormalDecl(this, valueSort)

    fun <T : KFpSort> mkFpIsSubnormalDecl(valueSort: T): KFpIsSubnormalDecl<T> =
        KFpIsSubnormalDecl(this, valueSort)

    fun <T : KFpSort> mkFpIsZeroDecl(valueSort: T): KFpIsZeroDecl<T> =
        KFpIsZeroDecl(this, valueSort)

    fun <T : KFpSort> mkFpIsInfiniteDecl(valueSort: T): KFpIsInfiniteDecl<T> =
        KFpIsInfiniteDecl(this, valueSort)

    fun <T : KFpSort> mkFpIsNaNDecl(valueSort: T): KFpIsNaNDecl<T> =
        KFpIsNaNDecl(this, valueSort)

    fun <T : KFpSort> mkFpIsNegativeDecl(valueSort: T): KFpIsNegativeDecl<T> =
        KFpIsNegativeDecl(this, valueSort)

    fun <T : KFpSort> mkFpIsPositiveDecl(valueSort: T): KFpIsPositiveDecl<T> =
        KFpIsPositiveDecl(this, valueSort)

    fun <T : KFpSort> mkFpToBvDecl(
        roundingMode: KFpRoundingModeSort,
        valueSort: T,
        bvSize: Int,
        isSigned: Boolean
    ): KFpToBvDecl<T> = KFpToBvDecl(this, roundingMode, valueSort, bvSize, isSigned)

    fun <T : KFpSort> mkFpToRealDecl(valueSort: T): KFpToRealDecl<T> =
        KFpToRealDecl(this, valueSort)

    fun <T : KFpSort> mkFpToIEEEBvDecl(valueSort: T): KFpToIEEEBvDecl<T> =
        KFpToIEEEBvDecl(this, valueSort)

    fun <T : KFpSort> mkFpFromBvDecl(
        signSort: KBv1Sort,
        expSort: KBvSort,
        significandSort: KBvSort
    ): KFpFromBvDecl<T> {
        val exponentBits = expSort.sizeBits
        val significandBits = significandSort.sizeBits + 1u
        val sort = mkFpSort(exponentBits, significandBits)

        return KFpFromBvDecl(this, sort, signSort, expSort, significandSort).cast()
    }

    fun <T : KFpSort> mkFpToFpDecl(sort: T, rm: KFpRoundingModeSort, value: KFpSort): KFpToFpDecl<T> =
        KFpToFpDecl(this, sort, rm, value)

    fun <T : KFpSort> mkRealToFpDecl(sort: T, rm: KFpRoundingModeSort, value: KRealSort): KRealToFpDecl<T> =
        KRealToFpDecl(this, sort, rm, value)

    fun <T : KFpSort> mkBvToFpDecl(sort: T, rm: KFpRoundingModeSort, value: KBvSort, signed: Boolean): KBvToFpDecl<T> =
        KBvToFpDecl(this, sort, rm, value, signed)

    /*
    * KAst
    * */

    /**
     * String representations are not cached since
     * it requires a lot of memory.
     * For example, (and a b) will store a full copy
     * of a and b string representations
     * */
    val KAst.stringRepr: String
        get() = buildString { print(this) }

    // context utils
    fun ensureContextMatch(ast: KAst) {
        require(this === ast.ctx) { "Context mismatch" }
    }

    fun ensureContextMatch(ast0: KAst, ast1: KAst) {
        ensureContextMatch(ast0)
        ensureContextMatch(ast1)
    }

    fun ensureContextMatch(ast0: KAst, ast1: KAst, ast2: KAst) {
        ensureContextMatch(ast0)
        ensureContextMatch(ast1)
        ensureContextMatch(ast2)
    }

    fun ensureContextMatch(vararg args: KAst) {
        args.forEach {
            ensureContextMatch(it)
        }
    }

    fun ensureContextMatch(args: List<KAst>) {
        args.forEach {
            ensureContextMatch(it)
        }
    }

    protected inline fun <T> ensureContextActive(block: () -> T): T {
        check(isActive) { "Context is not active" }
        return block()
    }

    protected inline fun <T> AstInterner<T>.createIfContextActive(
        builder: () -> T
    ): T where T : KAst, T : KInternedObject = ensureContextActive {
        intern(builder())
    }

    protected fun <T> mkAstInterner(): AstInterner<T> where T : KAst, T : KInternedObject =
        mkAstInterner(operationMode, astManagementMode)

    private inline fun <T : KSort, A0> mkSimplified(
        a0: A0,
        simplifier: KContext.(A0) -> KExpr<T>,
        createNoSimplify: (A0) -> KExpr<T>
    ): KExpr<T> = ensureContextActive {
        when (simplificationMode) {
            SIMPLIFY -> simplifier(a0)
            NO_SIMPLIFY -> createNoSimplify(a0)
        }
    }

    private inline fun <T : KSort, A0, A1> mkSimplified(
        a0: A0, a1: A1,
        simplifier: KContext.(A0, A1) -> KExpr<T>,
        createNoSimplify: (A0, A1) -> KExpr<T>
    ): KExpr<T> = ensureContextActive {
        when (simplificationMode) {
            SIMPLIFY -> simplifier(a0, a1)
            NO_SIMPLIFY -> createNoSimplify(a0, a1)
        }
    }

    private inline fun <T : KSort, A0, A1, A2> mkSimplified(
        a0: A0, a1: A1, a2: A2,
        simplifier: KContext.(A0, A1, A2) -> KExpr<T>,
        createNoSimplify: (A0, A1, A2) -> KExpr<T>
    ): KExpr<T> = ensureContextActive {
        when (simplificationMode) {
            SIMPLIFY -> simplifier(a0, a1, a2)
            NO_SIMPLIFY -> createNoSimplify(a0, a1, a2)
        }
    }

    @Suppress("LongParameterList")
    private inline fun <T : KSort, A0, A1, A2, A3> mkSimplified(
        a0: A0, a1: A1, a2: A2, a3: A3,
        simplifier: KContext.(A0, A1, A2, A3) -> KExpr<T>,
        createNoSimplify: (A0, A1, A2, A3) -> KExpr<T>
    ): KExpr<T> = ensureContextActive {
        when (simplificationMode) {
            SIMPLIFY -> simplifier(a0, a1, a2, a3)
            NO_SIMPLIFY -> createNoSimplify(a0, a1, a2, a3)
        }
    }

    @Suppress("LongParameterList")
    private inline fun <T : KSort, A0, A1, A2, A3, A4> mkSimplified(
        a0: A0, a1: A1, a2: A2, a3: A3, a4: A4,
        simplifier: KContext.(A0, A1, A2, A3, A4) -> KExpr<T>,
        createNoSimplify: (A0, A1, A2, A3, A4) -> KExpr<T>
    ): KExpr<T> = ensureContextActive {
        when (simplificationMode) {
            SIMPLIFY -> simplifier(a0, a1, a2, a3, a4)
            NO_SIMPLIFY -> createNoSimplify(a0, a1, a2, a3, a4)
        }
    }
}
