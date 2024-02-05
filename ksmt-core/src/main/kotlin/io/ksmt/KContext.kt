package io.ksmt

import io.ksmt.KContext.AstManagementMode.GC
import io.ksmt.KContext.AstManagementMode.NO_GC
import io.ksmt.KContext.OperationMode.CONCURRENT
import io.ksmt.KContext.OperationMode.SINGLE_THREAD
import io.ksmt.KContext.SimplificationMode.NO_SIMPLIFY
import io.ksmt.KContext.SimplificationMode.SIMPLIFY
import io.ksmt.cache.AstInterner
import io.ksmt.cache.KInternedObject
import io.ksmt.cache.mkAstInterner
import io.ksmt.cache.mkCache
import io.ksmt.decl.KAndDecl
import io.ksmt.decl.KArithAddDecl
import io.ksmt.decl.KArithDivDecl
import io.ksmt.decl.KArithGeDecl
import io.ksmt.decl.KArithGtDecl
import io.ksmt.decl.KArithLeDecl
import io.ksmt.decl.KArithLtDecl
import io.ksmt.decl.KArithMulDecl
import io.ksmt.decl.KArithPowerDecl
import io.ksmt.decl.KArithSubDecl
import io.ksmt.decl.KArithUnaryMinusDecl
import io.ksmt.decl.KArray2SelectDecl
import io.ksmt.decl.KArray2StoreDecl
import io.ksmt.decl.KArray3SelectDecl
import io.ksmt.decl.KArray3StoreDecl
import io.ksmt.decl.KArrayConstDecl
import io.ksmt.decl.KArrayNSelectDecl
import io.ksmt.decl.KArrayNStoreDecl
import io.ksmt.decl.KArraySelectDecl
import io.ksmt.decl.KArrayStoreDecl
import io.ksmt.decl.KBitVec16ValueDecl
import io.ksmt.decl.KBitVec1ValueDecl
import io.ksmt.decl.KBitVec32ValueDecl
import io.ksmt.decl.KBitVec64ValueDecl
import io.ksmt.decl.KBitVec8ValueDecl
import io.ksmt.decl.KBitVecCustomSizeValueDecl
import io.ksmt.decl.KBv2IntDecl
import io.ksmt.decl.KBvAddDecl
import io.ksmt.decl.KBvAddNoOverflowDecl
import io.ksmt.decl.KBvAddNoUnderflowDecl
import io.ksmt.decl.KBvAndDecl
import io.ksmt.decl.KBvArithShiftRightDecl
import io.ksmt.decl.KBvConcatDecl
import io.ksmt.decl.KBvDivNoOverflowDecl
import io.ksmt.decl.KBvExtractDecl
import io.ksmt.decl.KBvLogicalShiftRightDecl
import io.ksmt.decl.KBvMulDecl
import io.ksmt.decl.KBvMulNoOverflowDecl
import io.ksmt.decl.KBvMulNoUnderflowDecl
import io.ksmt.decl.KBvNAndDecl
import io.ksmt.decl.KBvNegNoOverflowDecl
import io.ksmt.decl.KBvNegationDecl
import io.ksmt.decl.KBvNorDecl
import io.ksmt.decl.KBvNotDecl
import io.ksmt.decl.KBvOrDecl
import io.ksmt.decl.KBvReductionAndDecl
import io.ksmt.decl.KBvReductionOrDecl
import io.ksmt.decl.KBvRepeatDecl
import io.ksmt.decl.KBvRotateLeftDecl
import io.ksmt.decl.KBvRotateLeftIndexedDecl
import io.ksmt.decl.KBvRotateRightDecl
import io.ksmt.decl.KBvRotateRightIndexedDecl
import io.ksmt.decl.KBvShiftLeftDecl
import io.ksmt.decl.KBvSignedDivDecl
import io.ksmt.decl.KBvSignedGreaterDecl
import io.ksmt.decl.KBvSignedGreaterOrEqualDecl
import io.ksmt.decl.KBvSignedLessDecl
import io.ksmt.decl.KBvSignedLessOrEqualDecl
import io.ksmt.decl.KBvSignedModDecl
import io.ksmt.decl.KBvSignedRemDecl
import io.ksmt.decl.KBvSubDecl
import io.ksmt.decl.KBvSubNoOverflowDecl
import io.ksmt.decl.KBvSubNoUnderflowDecl
import io.ksmt.decl.KBvToFpDecl
import io.ksmt.decl.KBvUnsignedDivDecl
import io.ksmt.decl.KBvUnsignedGreaterDecl
import io.ksmt.decl.KBvUnsignedGreaterOrEqualDecl
import io.ksmt.decl.KBvUnsignedLessDecl
import io.ksmt.decl.KBvUnsignedLessOrEqualDecl
import io.ksmt.decl.KBvUnsignedRemDecl
import io.ksmt.decl.KBvXNorDecl
import io.ksmt.decl.KBvXorDecl
import io.ksmt.decl.KConstDecl
import io.ksmt.decl.KDecl
import io.ksmt.decl.KDistinctDecl
import io.ksmt.decl.KEqDecl
import io.ksmt.decl.KFalseDecl
import io.ksmt.decl.KFp128Decl
import io.ksmt.decl.KFp16Decl
import io.ksmt.decl.KFp32Decl
import io.ksmt.decl.KFp64Decl
import io.ksmt.decl.KFpAbsDecl
import io.ksmt.decl.KFpAddDecl
import io.ksmt.decl.KFpCustomSizeDecl
import io.ksmt.decl.KFpDecl
import io.ksmt.decl.KFpDivDecl
import io.ksmt.decl.KFpEqualDecl
import io.ksmt.decl.KFpFromBvDecl
import io.ksmt.decl.KFpFusedMulAddDecl
import io.ksmt.decl.KFpGreaterDecl
import io.ksmt.decl.KFpGreaterOrEqualDecl
import io.ksmt.decl.KFpIsInfiniteDecl
import io.ksmt.decl.KFpIsNaNDecl
import io.ksmt.decl.KFpIsNegativeDecl
import io.ksmt.decl.KFpIsNormalDecl
import io.ksmt.decl.KFpIsPositiveDecl
import io.ksmt.decl.KFpIsSubnormalDecl
import io.ksmt.decl.KFpIsZeroDecl
import io.ksmt.decl.KFpLessDecl
import io.ksmt.decl.KFpLessOrEqualDecl
import io.ksmt.decl.KFpMaxDecl
import io.ksmt.decl.KFpMinDecl
import io.ksmt.decl.KFpMulDecl
import io.ksmt.decl.KFpNegationDecl
import io.ksmt.decl.KFpRemDecl
import io.ksmt.decl.KFpRoundToIntegralDecl
import io.ksmt.decl.KFpRoundingModeDecl
import io.ksmt.decl.KFpSqrtDecl
import io.ksmt.decl.KFpSubDecl
import io.ksmt.decl.KFpToBvDecl
import io.ksmt.decl.KFpToFpDecl
import io.ksmt.decl.KFpToIEEEBvDecl
import io.ksmt.decl.KFpToRealDecl
import io.ksmt.decl.KFuncDecl
import io.ksmt.decl.KImpliesDecl
import io.ksmt.decl.KIntModDecl
import io.ksmt.decl.KIntNumDecl
import io.ksmt.decl.KIntRemDecl
import io.ksmt.decl.KIntToRealDecl
import io.ksmt.decl.KIteDecl
import io.ksmt.decl.KNotDecl
import io.ksmt.decl.KOrDecl
import io.ksmt.decl.KRealIsIntDecl
import io.ksmt.decl.KRealNumDecl
import io.ksmt.decl.KRealToFpDecl
import io.ksmt.decl.KRealToIntDecl
import io.ksmt.decl.KSignExtDecl
import io.ksmt.decl.KTrueDecl
import io.ksmt.decl.KUninterpretedConstDecl
import io.ksmt.decl.KUninterpretedFuncDecl
import io.ksmt.decl.KUninterpretedSortValueDecl
import io.ksmt.decl.KXorDecl
import io.ksmt.decl.KZeroExtDecl
import io.ksmt.expr.KAddArithExpr
import io.ksmt.expr.KAndBinaryExpr
import io.ksmt.expr.KAndExpr
import io.ksmt.expr.KAndNaryExpr
import io.ksmt.expr.KApp
import io.ksmt.expr.KArray2Lambda
import io.ksmt.expr.KArray2Select
import io.ksmt.expr.KArray2Store
import io.ksmt.expr.KArray3Lambda
import io.ksmt.expr.KArray3Select
import io.ksmt.expr.KArray3Store
import io.ksmt.expr.KArrayConst
import io.ksmt.expr.KArrayLambda
import io.ksmt.expr.KArrayNLambda
import io.ksmt.expr.KArrayNSelect
import io.ksmt.expr.KArrayNStore
import io.ksmt.expr.KArraySelect
import io.ksmt.expr.KArrayStore
import io.ksmt.expr.KArrayStoreBase
import io.ksmt.expr.KBitVec16Value
import io.ksmt.expr.KBitVec1Value
import io.ksmt.expr.KBitVec32Value
import io.ksmt.expr.KBitVec64Value
import io.ksmt.expr.KBitVec8Value
import io.ksmt.expr.KBitVecCustomValue
import io.ksmt.expr.KBitVecNumberValue
import io.ksmt.expr.KBitVecValue
import io.ksmt.expr.KBv2IntExpr
import io.ksmt.expr.KBvAddExpr
import io.ksmt.expr.KBvAddNoOverflowExpr
import io.ksmt.expr.KBvAddNoUnderflowExpr
import io.ksmt.expr.KBvAndExpr
import io.ksmt.expr.KBvArithShiftRightExpr
import io.ksmt.expr.KBvConcatExpr
import io.ksmt.expr.KBvDivNoOverflowExpr
import io.ksmt.expr.KBvExtractExpr
import io.ksmt.expr.KBvLogicalShiftRightExpr
import io.ksmt.expr.KBvMulExpr
import io.ksmt.expr.KBvMulNoOverflowExpr
import io.ksmt.expr.KBvMulNoUnderflowExpr
import io.ksmt.expr.KBvNAndExpr
import io.ksmt.expr.KBvNegNoOverflowExpr
import io.ksmt.expr.KBvNegationExpr
import io.ksmt.expr.KBvNorExpr
import io.ksmt.expr.KBvNotExpr
import io.ksmt.expr.KBvOrExpr
import io.ksmt.expr.KBvReductionAndExpr
import io.ksmt.expr.KBvReductionOrExpr
import io.ksmt.expr.KBvRepeatExpr
import io.ksmt.expr.KBvRotateLeftExpr
import io.ksmt.expr.KBvRotateLeftIndexedExpr
import io.ksmt.expr.KBvRotateRightExpr
import io.ksmt.expr.KBvRotateRightIndexedExpr
import io.ksmt.expr.KBvShiftLeftExpr
import io.ksmt.expr.KBvSignExtensionExpr
import io.ksmt.expr.KBvSignedDivExpr
import io.ksmt.expr.KBvSignedGreaterExpr
import io.ksmt.expr.KBvSignedGreaterOrEqualExpr
import io.ksmt.expr.KBvSignedLessExpr
import io.ksmt.expr.KBvSignedLessOrEqualExpr
import io.ksmt.expr.KBvSignedModExpr
import io.ksmt.expr.KBvSignedRemExpr
import io.ksmt.expr.KBvSubExpr
import io.ksmt.expr.KBvSubNoOverflowExpr
import io.ksmt.expr.KBvSubNoUnderflowExpr
import io.ksmt.expr.KBvToFpExpr
import io.ksmt.expr.KBvUnsignedDivExpr
import io.ksmt.expr.KBvUnsignedGreaterExpr
import io.ksmt.expr.KBvUnsignedGreaterOrEqualExpr
import io.ksmt.expr.KBvUnsignedLessExpr
import io.ksmt.expr.KBvUnsignedLessOrEqualExpr
import io.ksmt.expr.KBvUnsignedRemExpr
import io.ksmt.expr.KBvXNorExpr
import io.ksmt.expr.KBvXorExpr
import io.ksmt.expr.KBvZeroExtensionExpr
import io.ksmt.expr.KConst
import io.ksmt.expr.KDistinctExpr
import io.ksmt.expr.KDivArithExpr
import io.ksmt.expr.KEqExpr
import io.ksmt.expr.KExistentialQuantifier
import io.ksmt.expr.KExpr
import io.ksmt.expr.KFalse
import io.ksmt.expr.KFp128Value
import io.ksmt.expr.KFp16Value
import io.ksmt.expr.KFp32Value
import io.ksmt.expr.KFp64Value
import io.ksmt.expr.KFpAbsExpr
import io.ksmt.expr.KFpAddExpr
import io.ksmt.expr.KFpCustomSizeValue
import io.ksmt.expr.KFpDivExpr
import io.ksmt.expr.KFpEqualExpr
import io.ksmt.expr.KFpFromBvExpr
import io.ksmt.expr.KFpFusedMulAddExpr
import io.ksmt.expr.KFpGreaterExpr
import io.ksmt.expr.KFpGreaterOrEqualExpr
import io.ksmt.expr.KFpIsInfiniteExpr
import io.ksmt.expr.KFpIsNaNExpr
import io.ksmt.expr.KFpIsNegativeExpr
import io.ksmt.expr.KFpIsNormalExpr
import io.ksmt.expr.KFpIsPositiveExpr
import io.ksmt.expr.KFpIsSubnormalExpr
import io.ksmt.expr.KFpIsZeroExpr
import io.ksmt.expr.KFpLessExpr
import io.ksmt.expr.KFpLessOrEqualExpr
import io.ksmt.expr.KFpMaxExpr
import io.ksmt.expr.KFpMinExpr
import io.ksmt.expr.KFpMulExpr
import io.ksmt.expr.KFpNegationExpr
import io.ksmt.expr.KFpRemExpr
import io.ksmt.expr.KFpRoundToIntegralExpr
import io.ksmt.expr.KFpRoundingMode
import io.ksmt.expr.KFpRoundingModeExpr
import io.ksmt.expr.KFpSqrtExpr
import io.ksmt.expr.KFpSubExpr
import io.ksmt.expr.KFpToBvExpr
import io.ksmt.expr.KFpToFpExpr
import io.ksmt.expr.KFpToIEEEBvExpr
import io.ksmt.expr.KFpToRealExpr
import io.ksmt.expr.KFpValue
import io.ksmt.expr.KFunctionApp
import io.ksmt.expr.KFunctionAsArray
import io.ksmt.expr.KGeArithExpr
import io.ksmt.expr.KGtArithExpr
import io.ksmt.expr.KImpliesExpr
import io.ksmt.expr.KInt32NumExpr
import io.ksmt.expr.KInt64NumExpr
import io.ksmt.expr.KIntBigNumExpr
import io.ksmt.expr.KIntNumExpr
import io.ksmt.expr.KIsIntRealExpr
import io.ksmt.expr.KIteExpr
import io.ksmt.expr.KLeArithExpr
import io.ksmt.expr.KLtArithExpr
import io.ksmt.expr.KModIntExpr
import io.ksmt.expr.KMulArithExpr
import io.ksmt.expr.KNotExpr
import io.ksmt.expr.KOrBinaryExpr
import io.ksmt.expr.KOrExpr
import io.ksmt.expr.KOrNaryExpr
import io.ksmt.expr.KPowerArithExpr
import io.ksmt.expr.KRealNumExpr
import io.ksmt.expr.KRealToFpExpr
import io.ksmt.expr.KRemIntExpr
import io.ksmt.expr.KSubArithExpr
import io.ksmt.expr.KToIntRealExpr
import io.ksmt.expr.KToRealIntExpr
import io.ksmt.expr.KTrue
import io.ksmt.expr.KUnaryMinusArithExpr
import io.ksmt.expr.KUninterpretedSortValue
import io.ksmt.expr.KUniversalQuantifier
import io.ksmt.expr.KXorExpr
import io.ksmt.expr.printer.PrinterParams
import io.ksmt.expr.rewrite.simplify.simplifyAnd
import io.ksmt.expr.rewrite.simplify.simplifyArithAdd
import io.ksmt.expr.rewrite.simplify.simplifyArithDiv
import io.ksmt.expr.rewrite.simplify.simplifyArithGe
import io.ksmt.expr.rewrite.simplify.simplifyArithGt
import io.ksmt.expr.rewrite.simplify.simplifyArithLe
import io.ksmt.expr.rewrite.simplify.simplifyArithLt
import io.ksmt.expr.rewrite.simplify.simplifyArithMul
import io.ksmt.expr.rewrite.simplify.simplifyArithPower
import io.ksmt.expr.rewrite.simplify.simplifyArithSub
import io.ksmt.expr.rewrite.simplify.simplifyArithUnaryMinus
import io.ksmt.expr.rewrite.simplify.simplifyArraySelect
import io.ksmt.expr.rewrite.simplify.simplifyArrayNSelect
import io.ksmt.expr.rewrite.simplify.simplifyArrayStore
import io.ksmt.expr.rewrite.simplify.simplifyArrayNStore
import io.ksmt.expr.rewrite.simplify.simplifyBv2IntExpr
import io.ksmt.expr.rewrite.simplify.simplifyBvAddExpr
import io.ksmt.expr.rewrite.simplify.simplifyBvAddNoOverflowExpr
import io.ksmt.expr.rewrite.simplify.simplifyBvAddNoUnderflowExpr
import io.ksmt.expr.rewrite.simplify.simplifyBvAndExpr
import io.ksmt.expr.rewrite.simplify.simplifyBvArithShiftRightExpr
import io.ksmt.expr.rewrite.simplify.simplifyBvConcatExpr
import io.ksmt.expr.rewrite.simplify.simplifyBvDivNoOverflowExpr
import io.ksmt.expr.rewrite.simplify.simplifyBvExtractExpr
import io.ksmt.expr.rewrite.simplify.simplifyBvLogicalShiftRightExpr
import io.ksmt.expr.rewrite.simplify.simplifyBvMulExpr
import io.ksmt.expr.rewrite.simplify.simplifyBvMulNoOverflowExpr
import io.ksmt.expr.rewrite.simplify.simplifyBvMulNoUnderflowExpr
import io.ksmt.expr.rewrite.simplify.simplifyBvNAndExpr
import io.ksmt.expr.rewrite.simplify.simplifyBvNegationExpr
import io.ksmt.expr.rewrite.simplify.simplifyBvNegationNoOverflowExpr
import io.ksmt.expr.rewrite.simplify.simplifyBvNorExpr
import io.ksmt.expr.rewrite.simplify.simplifyBvNotExpr
import io.ksmt.expr.rewrite.simplify.simplifyBvOrExpr
import io.ksmt.expr.rewrite.simplify.simplifyBvReductionAndExpr
import io.ksmt.expr.rewrite.simplify.simplifyBvReductionOrExpr
import io.ksmt.expr.rewrite.simplify.simplifyBvRepeatExpr
import io.ksmt.expr.rewrite.simplify.simplifyBvRotateLeftExpr
import io.ksmt.expr.rewrite.simplify.simplifyBvRotateLeftIndexedExpr
import io.ksmt.expr.rewrite.simplify.simplifyBvRotateRightExpr
import io.ksmt.expr.rewrite.simplify.simplifyBvRotateRightIndexedExpr
import io.ksmt.expr.rewrite.simplify.simplifyBvShiftLeftExpr
import io.ksmt.expr.rewrite.simplify.simplifyBvSignExtensionExpr
import io.ksmt.expr.rewrite.simplify.simplifyBvSignedDivExpr
import io.ksmt.expr.rewrite.simplify.simplifyBvSignedGreaterExpr
import io.ksmt.expr.rewrite.simplify.simplifyBvSignedGreaterOrEqualExpr
import io.ksmt.expr.rewrite.simplify.simplifyBvSignedLessExpr
import io.ksmt.expr.rewrite.simplify.simplifyBvSignedLessOrEqualExpr
import io.ksmt.expr.rewrite.simplify.simplifyBvSignedModExpr
import io.ksmt.expr.rewrite.simplify.simplifyBvSignedRemExpr
import io.ksmt.expr.rewrite.simplify.simplifyBvSubExpr
import io.ksmt.expr.rewrite.simplify.simplifyBvSubNoOverflowExpr
import io.ksmt.expr.rewrite.simplify.simplifyBvSubNoUnderflowExpr
import io.ksmt.expr.rewrite.simplify.simplifyBvToFpExpr
import io.ksmt.expr.rewrite.simplify.simplifyBvUnsignedDivExpr
import io.ksmt.expr.rewrite.simplify.simplifyBvUnsignedGreaterExpr
import io.ksmt.expr.rewrite.simplify.simplifyBvUnsignedGreaterOrEqualExpr
import io.ksmt.expr.rewrite.simplify.simplifyBvUnsignedLessExpr
import io.ksmt.expr.rewrite.simplify.simplifyBvUnsignedLessOrEqualExpr
import io.ksmt.expr.rewrite.simplify.simplifyBvUnsignedRemExpr
import io.ksmt.expr.rewrite.simplify.simplifyBvXNorExpr
import io.ksmt.expr.rewrite.simplify.simplifyBvXorExpr
import io.ksmt.expr.rewrite.simplify.simplifyBvZeroExtensionExpr
import io.ksmt.expr.rewrite.simplify.simplifyDistinct
import io.ksmt.expr.rewrite.simplify.simplifyEq
import io.ksmt.expr.rewrite.simplify.simplifyFpAbsExpr
import io.ksmt.expr.rewrite.simplify.simplifyFpAddExpr
import io.ksmt.expr.rewrite.simplify.simplifyFpDivExpr
import io.ksmt.expr.rewrite.simplify.simplifyFpEqualExpr
import io.ksmt.expr.rewrite.simplify.simplifyFpFromBvExpr
import io.ksmt.expr.rewrite.simplify.simplifyFpFusedMulAddExpr
import io.ksmt.expr.rewrite.simplify.simplifyFpGreaterExpr
import io.ksmt.expr.rewrite.simplify.simplifyFpGreaterOrEqualExpr
import io.ksmt.expr.rewrite.simplify.simplifyFpIsInfiniteExpr
import io.ksmt.expr.rewrite.simplify.simplifyFpIsNaNExpr
import io.ksmt.expr.rewrite.simplify.simplifyFpIsNegativeExpr
import io.ksmt.expr.rewrite.simplify.simplifyFpIsNormalExpr
import io.ksmt.expr.rewrite.simplify.simplifyFpIsPositiveExpr
import io.ksmt.expr.rewrite.simplify.simplifyFpIsSubnormalExpr
import io.ksmt.expr.rewrite.simplify.simplifyFpIsZeroExpr
import io.ksmt.expr.rewrite.simplify.simplifyFpLessExpr
import io.ksmt.expr.rewrite.simplify.simplifyFpLessOrEqualExpr
import io.ksmt.expr.rewrite.simplify.simplifyFpMaxExpr
import io.ksmt.expr.rewrite.simplify.simplifyFpMinExpr
import io.ksmt.expr.rewrite.simplify.simplifyFpMulExpr
import io.ksmt.expr.rewrite.simplify.simplifyFpNegationExpr
import io.ksmt.expr.rewrite.simplify.simplifyFpRemExpr
import io.ksmt.expr.rewrite.simplify.simplifyFpRoundToIntegralExpr
import io.ksmt.expr.rewrite.simplify.simplifyFpSqrtExpr
import io.ksmt.expr.rewrite.simplify.simplifyFpSubExpr
import io.ksmt.expr.rewrite.simplify.simplifyFpToBvExpr
import io.ksmt.expr.rewrite.simplify.simplifyFpToFpExpr
import io.ksmt.expr.rewrite.simplify.simplifyFpToIEEEBvExpr
import io.ksmt.expr.rewrite.simplify.simplifyFpToRealExpr
import io.ksmt.expr.rewrite.simplify.simplifyImplies
import io.ksmt.expr.rewrite.simplify.simplifyIntMod
import io.ksmt.expr.rewrite.simplify.simplifyIntRem
import io.ksmt.expr.rewrite.simplify.simplifyIntToReal
import io.ksmt.expr.rewrite.simplify.simplifyIte
import io.ksmt.expr.rewrite.simplify.simplifyNot
import io.ksmt.expr.rewrite.simplify.simplifyOr
import io.ksmt.expr.rewrite.simplify.simplifyRealIsInt
import io.ksmt.expr.rewrite.simplify.simplifyRealToFpExpr
import io.ksmt.expr.rewrite.simplify.simplifyRealToInt
import io.ksmt.expr.rewrite.simplify.simplifyXor
import io.ksmt.sort.KArithSort
import io.ksmt.sort.KArray2Sort
import io.ksmt.sort.KArray3Sort
import io.ksmt.sort.KArrayNSort
import io.ksmt.sort.KArraySort
import io.ksmt.sort.KArraySortBase
import io.ksmt.sort.KBoolSort
import io.ksmt.sort.KBv16Sort
import io.ksmt.sort.KBv1Sort
import io.ksmt.sort.KBv32Sort
import io.ksmt.sort.KBv64Sort
import io.ksmt.sort.KBv8Sort
import io.ksmt.sort.KBvCustomSizeSort
import io.ksmt.sort.KBvSort
import io.ksmt.sort.KFp128Sort
import io.ksmt.sort.KFp16Sort
import io.ksmt.sort.KFp32Sort
import io.ksmt.sort.KFp64Sort
import io.ksmt.sort.KFpCustomSizeSort
import io.ksmt.sort.KFpRoundingModeSort
import io.ksmt.sort.KFpSort
import io.ksmt.sort.KIntSort
import io.ksmt.sort.KRealSort
import io.ksmt.sort.KSort
import io.ksmt.sort.KSortVisitor
import io.ksmt.sort.KUninterpretedSort
import io.ksmt.utils.DefaultValueSampler
import io.ksmt.utils.FpUtils.biasFpExponent
import io.ksmt.utils.FpUtils.fpInfExponentBiased
import io.ksmt.utils.FpUtils.fpInfSignificand
import io.ksmt.utils.FpUtils.fpNaNExponentBiased
import io.ksmt.utils.FpUtils.fpNaNSignificand
import io.ksmt.utils.FpUtils.fpZeroExponentBiased
import io.ksmt.utils.FpUtils.fpZeroSignificand
import io.ksmt.utils.FpUtils.isNaN
import io.ksmt.utils.FpUtils.unbiasFpExponent
import io.ksmt.utils.booleanSignBit
import io.ksmt.utils.cast
import io.ksmt.utils.extractExponent
import io.ksmt.utils.extractSignificand
import io.ksmt.utils.getHalfPrecisionExponent
import io.ksmt.utils.halfPrecisionSignificand
import io.ksmt.utils.normalizeValue
import io.ksmt.utils.sampleValue
import io.ksmt.utils.signBit
import io.ksmt.utils.toBigInteger
import io.ksmt.utils.toULongValue
import io.ksmt.utils.toUnsignedBigInteger
import io.ksmt.utils.uncheckedCast
import java.lang.Double.longBitsToDouble
import java.lang.Float.intBitsToFloat
import java.math.BigInteger

@Suppress("TooManyFunctions", "LargeClass", "unused")
open class KContext(
    val operationMode: OperationMode = CONCURRENT,
    val astManagementMode: AstManagementMode = GC,
    val simplificationMode: SimplificationMode = SIMPLIFY,
    val printerParams: PrinterParams = PrinterParams()
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

    /**
     * Create a Bool sort.
     * */
    fun mkBoolSort(): KBoolSort = boolSort

    private val arraySortCache = mkCache<KArraySort<*, *>, KArraySort<*, *>>(operationMode)
    private val array2SortCache = mkCache<KArray2Sort<*, *, *>, KArray2Sort<*, *, *>>(operationMode)
    private val array3SortCache = mkCache<KArray3Sort<*, *, *, *>, KArray3Sort<*, *, *, *>>(operationMode)
    private val arrayNSortCache = mkCache<KArrayNSort<*>, KArrayNSort<*>>(operationMode)

    /**
     * Create an array sort (Array [domain] [range]).
     * */
    fun <D : KSort, R : KSort> mkArraySort(domain: D, range: R): KArraySort<D, R> =
        ensureContextActive {
            ensureContextMatch(domain, range)
            val sort = KArraySort(this, domain, range)
            (arraySortCache.putIfAbsent(sort, sort) ?: sort).uncheckedCast()
        }

    /**
     * Create an array sort (Array [domain0] [domain1] [range]).
     * */
    fun <D0 : KSort, D1 : KSort, R : KSort> mkArraySort(domain0: D0, domain1: D1, range: R): KArray2Sort<D0, D1, R> =
        ensureContextActive {
            ensureContextMatch(domain0, domain1, range)
            val sort = KArray2Sort(this, domain0, domain1, range)
            (array2SortCache.putIfAbsent(sort, sort) ?: sort).uncheckedCast()
        }

    /**
     * Create an array sort (Array [domain0] [domain1] [domain2] [range]).
     * */
    fun <D0 : KSort, D1 : KSort, D2 : KSort, R : KSort> mkArraySort(
        domain0: D0, domain1: D1, domain2: D2, range: R
    ): KArray3Sort<D0, D1, D2, R> =
        ensureContextActive {
            ensureContextMatch(domain0, domain1, domain2, range)
            val sort = KArray3Sort(this, domain0, domain1, domain2, range)
            (array3SortCache.putIfAbsent(sort, sort) ?: sort).uncheckedCast()
        }


    /**
     * Create a n-ary array sort (Array [domain]_0 ... [domain]_n [range]).
     * */
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

    /**
     * Create an Int sort.
     * */
    fun mkIntSort(): KIntSort = intSortCache

    private val realSortCache by lazy {
        ensureContextActive { KRealSort(this) }
    }

    /**
     * Create a Real sort.
     * */
    fun mkRealSort(): KRealSort = realSortCache

    // bit-vec
    private val bv1SortCache: KBv1Sort by lazy { KBv1Sort(this) }
    private val bv8SortCache: KBv8Sort by lazy { KBv8Sort(this) }
    private val bv16SortCache: KBv16Sort by lazy { KBv16Sort(this) }
    private val bv32SortCache: KBv32Sort by lazy { KBv32Sort(this) }
    private val bv64SortCache: KBv64Sort by lazy { KBv64Sort(this) }
    private val bvCustomSizeSortCache = mkCache<UInt, KBvSort>(operationMode)

    /**
     * Create a BitVec sort with 1 bit length (_ BitVec 1).
     * */
    fun mkBv1Sort(): KBv1Sort = bv1SortCache

    /**
     * Create a BitVec sort with 8 bits length (_ BitVec 8).
     * */
    fun mkBv8Sort(): KBv8Sort = bv8SortCache

    /**
     * Create a BitVec sort with 16 bits length (_ BitVec 16).
     * */
    fun mkBv16Sort(): KBv16Sort = bv16SortCache

    /**
     * Create a BitVec sort with 32 bits length (_ BitVec 32).
     * */
    fun mkBv32Sort(): KBv32Sort = bv32SortCache

    /**
     * Create a BitVec sort with 64 bits length (_ BitVec 64).
     * */
    fun mkBv64Sort(): KBv64Sort = bv64SortCache

    /**
     * Create a BitVec sort with [sizeBits] bits length (_ BitVec [sizeBits]).
     * */
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

    /**
     * Create an uninterpreted sort named [name].
     * */
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

    /**
     * Create a 16-bit IEEE floating point sort (_ FloatingPoint 5 11).
     * */
    fun mkFp16Sort(): KFp16Sort = fp16SortCache

    /**
     * Create a 32-bit IEEE floating point sort (_ FloatingPoint 8 24).
     * */
    fun mkFp32Sort(): KFp32Sort = fp32SortCache

    /**
     * Create a 64-bit IEEE floating point sort (_ FloatingPoint 11 53).
     * */
    fun mkFp64Sort(): KFp64Sort = fp64SortCache

    /**
     * Create a 128-bit IEEE floating point sort (_ FloatingPoint 15 113).
     * */
    fun mkFp128Sort(): KFp128Sort = fp128SortCache

    /**
     * Create an arbitrary precision IEEE floating point sort (_ FloatingPoint [exponentBits] [significandBits]).
     * */
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

    /**
     * Create a floating point rounding mode sort.
     * */
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

    /**
     * Create boolean AND expression.
     *
     * @param flat flat nested AND expressions
     * @param order reorder arguments to ensure that (and a b) == (and b a)
     * */
    @JvmOverloads
    open fun mkAnd(
        args: List<KExpr<KBoolSort>>,
        flat: Boolean = true,
        order: Boolean = true
    ): KExpr<KBoolSort> = mkSimplified(
        args,
        simplifier = { exprArgs -> simplifyAnd(exprArgs, flat, order) },
        createNoSimplify = ::mkAndNoSimplify
    )

    /**
     * Create boolean binary AND expression.
     *
     * @param flat flat nested AND expressions
     * @param order reorder arguments to ensure that (and a b) == (and b a)
     * */
    @JvmOverloads
    open fun mkAnd(
        lhs: KExpr<KBoolSort>,
        rhs: KExpr<KBoolSort>,
        flat: Boolean = true,
        order: Boolean = true
    ): KExpr<KBoolSort> = mkSimplified(
        lhs, rhs,
        simplifier = { a, b -> simplifyAnd(a, b, flat, order) },
        createNoSimplify = ::mkAndNoSimplify
    )

    /**
     * Create boolean AND expression.
     * */
    open fun mkAndNoSimplify(args: List<KExpr<KBoolSort>>): KAndExpr =
        if (args.size == 2) {
            mkAndNoSimplify(args.first(), args.last())
        } else {
            andNaryCache.createIfContextActive {
                ensureContextMatch(args)
                KAndNaryExpr(this, args)
            }
        }

    /**
     * Create boolean binary AND expression.
     * */
    open fun mkAndNoSimplify(lhs: KExpr<KBoolSort>, rhs: KExpr<KBoolSort>): KAndBinaryExpr =
        andBinaryCache.createIfContextActive {
            ensureContextMatch(lhs, rhs)
            KAndBinaryExpr(this, lhs, rhs)
        }

    private val orNaryCache = mkAstInterner<KOrNaryExpr>()
    private val orBinaryCache = mkAstInterner<KOrBinaryExpr>()

    /**
     * Create boolean OR expression.
     *
     * @param flat flat nested OR expressions
     * @param order reorder arguments to ensure that (or a b) == (or b a)
     * */
    @JvmOverloads
    open fun mkOr(
        args: List<KExpr<KBoolSort>>,
        flat: Boolean = true,
        order: Boolean = true
    ): KExpr<KBoolSort> = mkSimplified(
        args,
        simplifier = { exprArgs -> simplifyOr(exprArgs, flat, order) },
        createNoSimplify = ::mkOrNoSimplify
    )

    /**
     * Create boolean binary OR expression.
     *
     * @param flat flat nested OR expressions
     * @param order reorder arguments to ensure that (or a b) == (or b a)
     * */
    @JvmOverloads
    open fun mkOr(
        lhs: KExpr<KBoolSort>,
        rhs: KExpr<KBoolSort>,
        flat: Boolean = true,
        order: Boolean = true
    ): KExpr<KBoolSort> = mkSimplified(
        lhs, rhs,
        simplifier = { a, b -> simplifyOr(a, b, flat, order) },
        createNoSimplify = ::mkOrNoSimplify
    )

    /**
     * Create boolean OR expression.
     * */
    open fun mkOrNoSimplify(args: List<KExpr<KBoolSort>>): KOrExpr =
        if (args.size == 2) {
            mkOrNoSimplify(args.first(), args.last())
        } else {
            orNaryCache.createIfContextActive {
                ensureContextMatch(args)
                KOrNaryExpr(this, args)
            }
        }

    /**
     * Create boolean binary OR expression.
     * */
    open fun mkOrNoSimplify(lhs: KExpr<KBoolSort>, rhs: KExpr<KBoolSort>): KOrBinaryExpr =
        orBinaryCache.createIfContextActive {
            ensureContextMatch(lhs, rhs)
            KOrBinaryExpr(this, lhs, rhs)
        }

    private val notCache = mkAstInterner<KNotExpr>()

    /**
     * Create boolean NOT expression.
     * */
    open fun mkNot(arg: KExpr<KBoolSort>): KExpr<KBoolSort> =
        mkSimplified(arg, KContext::simplifyNot, ::mkNotNoSimplify)

    /**
     * Create boolean NOT expression.
     * */
    open fun mkNotNoSimplify(arg: KExpr<KBoolSort>): KNotExpr = notCache.createIfContextActive {
        ensureContextMatch(arg)
        KNotExpr(this, arg)
    }

    private val impliesCache = mkAstInterner<KImpliesExpr>()

    /**
     * Create boolean `=>` (implication) expression.
     * */
    open fun mkImplies(p: KExpr<KBoolSort>, q: KExpr<KBoolSort>): KExpr<KBoolSort> =
        mkSimplified(p, q, KContext::simplifyImplies, ::mkImpliesNoSimplify)

    /**
     * Create boolean `=>` (implication) expression.
     * */
    open fun mkImpliesNoSimplify(
        p: KExpr<KBoolSort>,
        q: KExpr<KBoolSort>
    ): KImpliesExpr = impliesCache.createIfContextActive {
        ensureContextMatch(p, q)
        KImpliesExpr(this, p, q)
    }

    private val xorCache = mkAstInterner<KXorExpr>()

    /**
     * Create boolean XOR expression.
     * */
    open fun mkXor(a: KExpr<KBoolSort>, b: KExpr<KBoolSort>): KExpr<KBoolSort> =
        mkSimplified(a, b, KContext::simplifyXor, ::mkXorNoSimplify)

    /**
     * Create boolean XOR expression.
     * */
    open fun mkXorNoSimplify(a: KExpr<KBoolSort>, b: KExpr<KBoolSort>): KXorExpr =
        xorCache.createIfContextActive {
            ensureContextMatch(a, b)
            KXorExpr(this, a, b)
        }

    val trueExpr: KTrue = KTrue(this)
    val falseExpr: KFalse = KFalse(this)

    /**
     * Create boolean True constant.
     * */
    fun mkTrue(): KTrue = trueExpr

    /**
     * Create boolean False constant.
     * */
    fun mkFalse(): KFalse = falseExpr

    private val eqCache = mkAstInterner<KEqExpr<out KSort>>()

    /**
     * Create EQ expression.
     *
     * @param order reorder arguments to ensure that (= a b) == (= b a)
     * */
    @JvmOverloads
    open fun <T : KSort> mkEq(
        lhs: KExpr<T>,
        rhs: KExpr<T>,
        order: Boolean = true
    ): KExpr<KBoolSort> = mkSimplified(
        lhs, rhs,
        simplifier = { l, r -> simplifyEq(l, r, order) },
        createNoSimplify = ::mkEqNoSimplify
    )

    /**
     * Create EQ expression.
     * */
    open fun <T : KSort> mkEqNoSimplify(lhs: KExpr<T>, rhs: KExpr<T>): KEqExpr<T> =
        eqCache.createIfContextActive {
            ensureContextMatch(lhs, rhs)
            KEqExpr(this, lhs, rhs)
        }.cast()

    private val distinctCache = mkAstInterner<KDistinctExpr<out KSort>>()

    /**
     * Create DISTINCT expression.
     *
     * @param order reorder arguments to ensure that (distinct a b) == (distinct b a)
     * */
    @JvmOverloads
    open fun <T : KSort> mkDistinct(
        args: List<KExpr<T>>,
        order: Boolean = true
    ): KExpr<KBoolSort> = mkSimplified(
        args,
        simplifier = { exprArgs -> simplifyDistinct(exprArgs, order) },
        createNoSimplify = ::mkDistinctNoSimplify
    )

    /**
     * Create DISTINCT expression.
     * */
    open fun <T : KSort> mkDistinctNoSimplify(args: List<KExpr<T>>): KDistinctExpr<T> =
        distinctCache.createIfContextActive {
            ensureContextMatch(args)
            KDistinctExpr(this, args)
        }.cast()

    private val iteCache = mkAstInterner<KIteExpr<out KSort>>()

    /**
     * Create ITE (if-then-else) expression.
     * */
    open fun <T : KSort> mkIte(
        condition: KExpr<KBoolSort>,
        trueBranch: KExpr<T>,
        falseBranch: KExpr<T>
    ): KExpr<T> = mkSimplified(condition, trueBranch, falseBranch, KContext::simplifyIte, ::mkIteNoSimplify)

    /**
     * Create ITE (if-then-else) expression.
     * */
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

    /**
     * Create function app expression.
     *
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

    /**
     * Create constant (function without arguments) app expression.
     *
     * @see [mkApp]
     * */
    open fun <T : KSort> mkConstApp(decl: KDecl<T>): KConst<T> = constAppCache.createIfContextActive {
        ensureContextMatch(decl)
        KConst(this, decl)
    }.cast()

    /**
     * Create a constant named [name] with sort [sort].
     * */
    fun <T : KSort> mkConst(name: String, sort: T): KApp<T, *> = with(mkConstDecl(name, sort)) { apply() }

    /**
     * Create a fresh constant with name prefix [name] and sort [sort].
     *
     * It is guaranteed that a fresh constant is not equal by `==` to any other constant.
     * */
    fun <T : KSort> mkFreshConst(name: String, sort: T): KApp<T, *> = with(mkFreshConstDecl(name, sort)) { apply() }

    // array
    private val arrayStoreCache = mkAstInterner<KArrayStore<out KSort, out KSort>>()
    private val array2StoreCache = mkAstInterner<KArray2Store<out KSort, out KSort, out KSort>>()
    private val array3StoreCache = mkAstInterner<KArray3Store<out KSort, out KSort, out KSort, out KSort>>()
    private val arrayNStoreCache = mkAstInterner<KArrayNStore<out KSort>>()

    /**
     * Create an array store expression (store [array] [index] [value]).
     * */
    open fun <D : KSort, R : KSort> mkArrayStore(
        array: KExpr<KArraySort<D, R>>,
        index: KExpr<D>,
        value: KExpr<R>
    ): KExpr<KArraySort<D, R>> =
        mkSimplified(array, index, value, KContext::simplifyArrayStore, ::mkArrayStoreNoSimplify)

    /**
     * Create an array store expression (store [array] [index0] [index1] [value]).
     * */
    open fun <D0 : KSort, D1 : KSort, R : KSort> mkArrayStore(
        array: KExpr<KArray2Sort<D0, D1, R>>,
        index0: KExpr<D0>,
        index1: KExpr<D1>,
        value: KExpr<R>
    ): KExpr<KArray2Sort<D0, D1, R>> =
        mkSimplified(array, index0, index1, value, KContext::simplifyArrayStore, ::mkArrayStoreNoSimplify)

    /**
     * Create an array store expression (store [array] [index0] [index1] [index2] [value]).
     * */
    open fun <D0 : KSort, D1 : KSort, D2 : KSort, R : KSort> mkArrayStore(
        array: KExpr<KArray3Sort<D0, D1, D2, R>>,
        index0: KExpr<D0>,
        index1: KExpr<D1>,
        index2: KExpr<D2>,
        value: KExpr<R>
    ): KExpr<KArray3Sort<D0, D1, D2, R>> =
        mkSimplified(array, index0, index1, index2, value, KContext::simplifyArrayStore, ::mkArrayStoreNoSimplify)

    /**
     * Create n-ary array store expression (store [array] [indices]_0 ... [indices]_n [value]).
     * */
    open fun <R : KSort> mkArrayNStore(
        array: KExpr<KArrayNSort<R>>,
        indices: List<KExpr<*>>,
        value: KExpr<R>
    ): KExpr<KArrayNSort<R>> =
        mkSimplified(array, indices, value, KContext::simplifyArrayNStore, ::mkArrayNStoreNoSimplify)

    /**
     * Create an array store expression (store [array] [index] [value]).
     * */
    open fun <D : KSort, R : KSort> mkArrayStoreNoSimplify(
        array: KExpr<KArraySort<D, R>>,
        index: KExpr<D>,
        value: KExpr<R>
    ): KArrayStore<D, R> =
        mkArrayStoreNoSimplifyNoAnalyze(array, index, value)
            .analyzeIfSimplificationEnabled()

    /**
     * Create an array store expression (store [array] [index0] [index1] [value]).
     * */
    open fun <D0 : KSort, D1 : KSort, R : KSort> mkArrayStoreNoSimplify(
        array: KExpr<KArray2Sort<D0, D1, R>>,
        index0: KExpr<D0>,
        index1: KExpr<D1>,
        value: KExpr<R>
    ): KArray2Store<D0, D1, R> =
        mkArrayStoreNoSimplifyNoAnalyze(array, index0, index1, value)
            .analyzeIfSimplificationEnabled()

    /**
     * Create an array store expression (store [array] [index0] [index1] [index2] [value]).
     * */
    open fun <D0 : KSort, D1 : KSort, D2 : KSort, R : KSort> mkArrayStoreNoSimplify(
        array: KExpr<KArray3Sort<D0, D1, D2, R>>,
        index0: KExpr<D0>,
        index1: KExpr<D1>,
        index2: KExpr<D2>,
        value: KExpr<R>
    ): KArray3Store<D0, D1, D2, R> =
        mkArrayStoreNoSimplifyNoAnalyze(array, index0, index1, index2, value)
            .analyzeIfSimplificationEnabled()

    /**
     * Create n-ary array store expression (store [array] [indices]_0 ... [indices]_n [value]).
     * */
    open fun <R : KSort> mkArrayNStoreNoSimplify(
        array: KExpr<KArrayNSort<R>>,
        indices: List<KExpr<*>>,
        value: KExpr<R>
    ): KArrayNStore<R> =
        mkArrayNStoreNoSimplifyNoAnalyze(array, indices, value)
            .analyzeIfSimplificationEnabled()

    /**
     * Create an array store expression (store [array] [index] [value]) without cache.
     * Cache is used to speed up [mkArraySelect] operation simplification 
     * but can result in a huge memory consumption.
     *
     * @see [KArrayStoreBase] for the cache details.
     * */
    open fun <D : KSort, R : KSort> mkArrayStoreNoSimplifyNoAnalyze(
        array: KExpr<KArraySort<D, R>>,
        index: KExpr<D>,
        value: KExpr<R>
    ): KArrayStore<D, R> = arrayStoreCache.createIfContextActive {
        ensureContextMatch(array, index, value)
        KArrayStore(this, array, index, value)
    }.cast()

    /**
     * Create an array store expression (store [array] [index0] [index1] [value]) without cache.
     * Cache is used to speed up [mkArraySelect] operation simplification 
     * but can result in a huge memory consumption.
     *
     * @see [KArrayStoreBase] for the cache details.
     * */
    open fun <D0 : KSort, D1 : KSort, R : KSort> mkArrayStoreNoSimplifyNoAnalyze(
        array: KExpr<KArray2Sort<D0, D1, R>>,
        index0: KExpr<D0>,
        index1: KExpr<D1>,
        value: KExpr<R>
    ): KArray2Store<D0, D1, R> = array2StoreCache.createIfContextActive {
        ensureContextMatch(array, index0, index1, value)
        KArray2Store(this, array, index0, index1, value)
    }.cast()

    /**
     * Create an array store expression (store [array] [index0] [index1] [index2] [value]) without cache.
     * Cache is used to speed up [mkArraySelect] operation simplification 
     * but can result in a huge memory consumption.
     *
     * @see [KArrayStoreBase] for the cache details.
     * */
    open fun <D0 : KSort, D1 : KSort, D2 : KSort, R : KSort> mkArrayStoreNoSimplifyNoAnalyze(
        array: KExpr<KArray3Sort<D0, D1, D2, R>>,
        index0: KExpr<D0>,
        index1: KExpr<D1>,
        index2: KExpr<D2>,
        value: KExpr<R>
    ): KArray3Store<D0, D1, D2, R> = array3StoreCache.createIfContextActive {
        ensureContextMatch(array, index0, index1, index2, value)
        KArray3Store(this, array, index0, index1, index2, value)
    }.cast()

    /**
     * Create n-ary array store expression (store [array] [indices]_0 ... [indices]_n [value]) without cache.
     * Cache is used to speed up [mkArrayNSelect] operation simplification
     * but can result in a huge memory consumption.
     *
     * @see [KArrayStoreBase] for the cache details.
     * */
    open fun <R : KSort> mkArrayNStoreNoSimplifyNoAnalyze(
        array: KExpr<KArrayNSort<R>>,
        indices: List<KExpr<*>>,
        value: KExpr<R>
    ): KArrayNStore<R> = arrayNStoreCache.createIfContextActive {
        ensureContextMatch(indices)
        ensureContextMatch(array, value)

        KArrayNStore(this, array, indices.uncheckedCast(), value)
    }.cast()

    private fun <S : KArrayStoreBase<*, *>> S.analyzeIfSimplificationEnabled(): S {
        /**
         * Analyze store indices only when simplification is enabled since
         * we don't expect any benefit from the analyzed stores
         * if we don't use simplifications.
         * */
        if (simplificationMode == SIMPLIFY) {
            analyzeStore()
        }
        return this
    }

    private val arraySelectCache = mkAstInterner<KArraySelect<out KSort, out KSort>>()
    private val array2SelectCache = mkAstInterner<KArray2Select<out KSort, out KSort, out KSort>>()
    private val array3SelectCache = mkAstInterner<KArray3Select<out KSort, out KSort, out KSort, out KSort>>()
    private val arrayNSelectCache = mkAstInterner<KArrayNSelect<out KSort>>()

    /**
     * Create an array select expression (select [array] [index]).
     * */
    open fun <D : KSort, R : KSort> mkArraySelect(
        array: KExpr<KArraySort<D, R>>,
        index: KExpr<D>
    ): KExpr<R> = mkSimplified(array, index, KContext::simplifyArraySelect, ::mkArraySelectNoSimplify)

    /**
     * Create an array select expression (select [array] [index0] [index1]).
     * */
    open fun <D0 : KSort, D1 : KSort, R : KSort> mkArraySelect(
        array: KExpr<KArray2Sort<D0, D1, R>>,
        index0: KExpr<D0>,
        index1: KExpr<D1>
    ): KExpr<R> = mkSimplified(array, index0, index1, KContext::simplifyArraySelect, ::mkArraySelectNoSimplify)

    /**
     * Create an array select expression (select [array] [index0] [index1] [index2]).
     * */
    open fun <D0 : KSort, D1 : KSort, D2 : KSort, R : KSort> mkArraySelect(
        array: KExpr<KArray3Sort<D0, D1, D2, R>>,
        index0: KExpr<D0>,
        index1: KExpr<D1>,
        index2: KExpr<D2>
    ): KExpr<R> = mkSimplified(array, index0, index1, index2, KContext::simplifyArraySelect, ::mkArraySelectNoSimplify)

    /**
     * Create n-ary array select expression (select [array] [indices]_0 ... [indices]_n).
     * */
    open fun <R : KSort> mkArrayNSelect(
        array: KExpr<KArrayNSort<R>>,
        indices: List<KExpr<*>>
    ): KExpr<R> = mkSimplified(array, indices, KContext::simplifyArrayNSelect, ::mkArrayNSelectNoSimplify)

    /**
     * Create an array select expression (select [array] [index]).
     * */
    open fun <D : KSort, R : KSort> mkArraySelectNoSimplify(
        array: KExpr<KArraySort<D, R>>,
        index: KExpr<D>
    ): KArraySelect<D, R> = arraySelectCache.createIfContextActive {
        ensureContextMatch(array, index)
        KArraySelect(this, array, index)
    }.cast()

    /**
     * Create an array select expression (select [array] [index0] [index1]).
     * */
    open fun <D0 : KSort, D1 : KSort, R : KSort> mkArraySelectNoSimplify(
        array: KExpr<KArray2Sort<D0, D1, R>>,
        index0: KExpr<D0>,
        index1: KExpr<D1>
    ): KArray2Select<D0, D1, R> = array2SelectCache.createIfContextActive {
        ensureContextMatch(array, index0, index1)
        KArray2Select(this, array, index0, index1)
    }.cast()

    /**
     * Create an array select expression (select [array] [index0] [index1] [index2]).
     * */
    open fun <D0 : KSort, D1 : KSort, D2 : KSort, R : KSort> mkArraySelectNoSimplify(
        array: KExpr<KArray3Sort<D0, D1, D2, R>>,
        index0: KExpr<D0>,
        index1: KExpr<D1>,
        index2: KExpr<D2>
    ): KArray3Select<D0, D1, D2, R> = array3SelectCache.createIfContextActive {
        ensureContextMatch(array, index0, index1, index2)
        KArray3Select(this, array, index0, index1, index2)
    }.cast()

    /**
     * Create n-ary array select expression (select [array] [indices]_0 ... [indices]_n).
     * */
    open fun <R : KSort> mkArrayNSelectNoSimplify(
        array: KExpr<KArrayNSort<R>>,
        indices: List<KExpr<*>>
    ): KArrayNSelect<R> = arrayNSelectCache.createIfContextActive {
        ensureContextMatch(array)
        ensureContextMatch(indices)

        KArrayNSelect(this, array, indices.uncheckedCast())
    }.cast()

    private val arrayConstCache = mkAstInterner<KArrayConst<out KArraySortBase<out KSort>, out KSort>>()

    /**
     * Create a constant array expression ((as const [arraySort]) [value]).
     *
     * Maps all indices to some fixed [value].
     * If `(= C ((as const (Array D R)) value))`
     * then `(forall (i D) (= (select C i) value))`.
     * */
    open fun <A : KArraySortBase<R>, R : KSort> mkArrayConst(
        arraySort: A,
        value: KExpr<R>
    ): KArrayConst<A, R> = arrayConstCache.createIfContextActive {
        ensureContextMatch(arraySort, value)
        KArrayConst(this, arraySort, value)
    }.cast()

    private val functionAsArrayCache = mkAstInterner<KFunctionAsArray<out KArraySortBase<out KSort>, out KSort>>()

    /**
     * Create a function-as-array expression (_ as-array [function]).
     *
     * Maps all array indices to the corresponding value of [function].
     * If `(= A (_ as-array f))`
     * then `(forall (i (domain f)) (= (select A i) (f i)))`
     * */
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

    /**
     * Create an array lambda expression (lambda ([indexVar]) [body]).
     *
     * The sort of the lambda expression is an array
     * where the domain is the array are the [indexVar] sort and
     * the range is the sort of the [body] of the lambda expression.
     *
     * If `(= L (lambda (i D)) (body i))`
     * then `(forall (i D) (= (select L i) (body i)))`.
     * */
    open fun <D : KSort, R : KSort> mkArrayLambda(
        indexVar: KDecl<D>, body: KExpr<R>
    ): KArrayLambda<D, R> = arrayLambdaCache.createIfContextActive {
        ensureContextMatch(indexVar, body)
        KArrayLambda(this, indexVar, body)
    }.cast()

    /**
     * Create an array lambda expression (lambda ([indexVar0] [indexVar1]) [body]).
     *
     * @see [mkArrayLambda]
     * */
    open fun <D0 : KSort, D1 : KSort, R : KSort> mkArrayLambda(
        indexVar0: KDecl<D0>, indexVar1: KDecl<D1>, body: KExpr<R>
    ): KArray2Lambda<D0, D1, R> = array2LambdaCache.createIfContextActive {
        ensureContextMatch(indexVar0, indexVar1, body)
        KArray2Lambda(this, indexVar0, indexVar1, body)
    }.cast()

    /**
     * Create an array lambda expression (lambda ([indexVar0] [indexVar1] [indexVar2]) [body]).
     *
     * @see [mkArrayLambda]
     * */
    open fun <D0 : KSort, D1 : KSort, D2 : KSort, R : KSort> mkArrayLambda(
        indexVar0: KDecl<D0>, indexVar1: KDecl<D1>, indexVar2: KDecl<D2>, body: KExpr<R>
    ): KArray3Lambda<D0, D1, D2, R> = array3LambdaCache.createIfContextActive {
        ensureContextMatch(indexVar0, indexVar1, indexVar2, body)
        KArray3Lambda(this, indexVar0, indexVar1, indexVar2, body)
    }.cast()

    /**
     * Create n-ary array lambda expression (lambda ([indices]_0 ... [indices]_n) [body]).
     *
     * @see [mkArrayLambda]
     * */
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

    /**
     * Create an Int/Real arithmetic addition expression.
     * */
    open fun <T : KArithSort> mkArithAdd(args: List<KExpr<T>>): KExpr<T> =
        mkSimplified(args, KContext::simplifyArithAdd, ::mkArithAddNoSimplify)

    /**
     * Create an Int/Real arithmetic addition expression.
     * */
    open fun <T : KArithSort> mkArithAddNoSimplify(args: List<KExpr<T>>): KAddArithExpr<T> =
        arithAddCache.createIfContextActive {
            ensureContextMatch(args)
            KAddArithExpr(this, args)
        }.cast()

    private val arithMulCache = mkAstInterner<KMulArithExpr<out KArithSort>>()

    /**
     * Create an Int/Real arithmetic multiplication expression.
     * */
    open fun <T : KArithSort> mkArithMul(args: List<KExpr<T>>): KExpr<T> =
        mkSimplified(args, KContext::simplifyArithMul, ::mkArithMulNoSimplify)

    /**
     * Create an Int/Real arithmetic multiplication expression.
     * */
    open fun <T : KArithSort> mkArithMulNoSimplify(args: List<KExpr<T>>): KMulArithExpr<T> =
        arithMulCache.createIfContextActive {
            ensureContextMatch(args)
            KMulArithExpr(this, args)
        }.cast()

    private val arithSubCache = mkAstInterner<KSubArithExpr<out KArithSort>>()

    /**
     * Create an Int/Real arithmetic subtraction expression.
     * */
    open fun <T : KArithSort> mkArithSub(args: List<KExpr<T>>): KExpr<T> =
        mkSimplified(args, KContext::simplifyArithSub, ::mkArithSubNoSimplify)

    /**
     * Create an Int/Real arithmetic subtraction expression.
     * */
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

    /**
     * Create an Int/Real arithmetic negation expression.
     * */
    open fun <T : KArithSort> mkArithUnaryMinus(arg: KExpr<T>): KExpr<T> =
        mkSimplified(arg, KContext::simplifyArithUnaryMinus, ::mkArithUnaryMinusNoSimplify)

    /**
     * Create an Int/Real arithmetic negation expression.
     * */
    open fun <T : KArithSort> mkArithUnaryMinusNoSimplify(arg: KExpr<T>): KUnaryMinusArithExpr<T> =
        arithUnaryMinusCache.createIfContextActive {
            ensureContextMatch(arg)
            KUnaryMinusArithExpr(this, arg)
        }.cast()

    private val arithDivCache = mkAstInterner<KDivArithExpr<out KArithSort>>()

    /**
     * Create an Int/Real arithmetic division expression.
     *
     * Note for Int division:
     *
     *   when [rhs] is positive, (div [lhs] [rhs]) is the floor (round toward zero)
     *   of the rational number [lhs]/[rhs]
     *
     *   when [rhs] is negative, (div [lhs] [rhs]) is the ceiling (round toward positive infinity)
     *   of the rational number [lhs]/[rhs]
     *
     *   For example:
     *    `47 div 13 = 3`
     *    `47 div -13 = -3`
     *    `-47 div 13 = -4`
     *    `-47 div -13 = 4`
     * */
    open fun <T : KArithSort> mkArithDiv(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<T> =
        mkSimplified(lhs, rhs, KContext::simplifyArithDiv, ::mkArithDivNoSimplify)

    /**
     * Create an Int/Real arithmetic division expression.
     *
     * @see mkArithDiv for the operation details.
     * */
    open fun <T : KArithSort> mkArithDivNoSimplify(lhs: KExpr<T>, rhs: KExpr<T>): KDivArithExpr<T> =
        arithDivCache.createIfContextActive {
            ensureContextMatch(lhs, rhs)
            KDivArithExpr(this, lhs, rhs)
        }.cast()

    private val arithPowerCache = mkAstInterner<KPowerArithExpr<out KArithSort>>()

    /**
     * Create an Int/Real arithmetic power expression.
     * */
    open fun <T : KArithSort> mkArithPower(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<T> =
        mkSimplified(lhs, rhs, KContext::simplifyArithPower, ::mkArithPowerNoSimplify)

    /**
     * Create an Int/Real arithmetic power expression.
     * */
    open fun <T : KArithSort> mkArithPowerNoSimplify(lhs: KExpr<T>, rhs: KExpr<T>): KPowerArithExpr<T> =
        arithPowerCache.createIfContextActive {
            ensureContextMatch(lhs, rhs)
            KPowerArithExpr(this, lhs, rhs)
        }.cast()

    private val arithLtCache = mkAstInterner<KLtArithExpr<out KArithSort>>()

    /**
     * Create an Int/Real arithmetic `<` (less) expression.
     * */
    open fun <T : KArithSort> mkArithLt(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<KBoolSort> =
        mkSimplified(lhs, rhs, KContext::simplifyArithLt, ::mkArithLtNoSimplify)

    /**
     * Create an Int/Real arithmetic `<` (less) expression.
     * */
    open fun <T : KArithSort> mkArithLtNoSimplify(lhs: KExpr<T>, rhs: KExpr<T>): KLtArithExpr<T> =
        arithLtCache.createIfContextActive {
            ensureContextMatch(lhs, rhs)
            KLtArithExpr(this, lhs, rhs)
        }.cast()

    private val arithLeCache = mkAstInterner<KLeArithExpr<out KArithSort>>()

    /**
     * Create an Int/Real arithmetic `<=` (less-or-equal) expression.
     * */
    open fun <T : KArithSort> mkArithLe(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<KBoolSort> =
        mkSimplified(lhs, rhs, KContext::simplifyArithLe, ::mkArithLeNoSimplify)

    /**
     * Create an Int/Real arithmetic `<=` (less-or-equal) expression.
     * */
    open fun <T : KArithSort> mkArithLeNoSimplify(lhs: KExpr<T>, rhs: KExpr<T>): KLeArithExpr<T> =
        arithLeCache.createIfContextActive {
            ensureContextMatch(lhs, rhs)
            KLeArithExpr(this, lhs, rhs)
        }.cast()

    private val arithGtCache = mkAstInterner<KGtArithExpr<out KArithSort>>()

    /**
     * Create an Int/Real arithmetic `>` (greater) expression.
     * */
    open fun <T : KArithSort> mkArithGt(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<KBoolSort> =
        mkSimplified(lhs, rhs, KContext::simplifyArithGt, ::mkArithGtNoSimplify)

    /**
     * Create an Int/Real arithmetic `>` (greater) expression.
     * */
    open fun <T : KArithSort> mkArithGtNoSimplify(lhs: KExpr<T>, rhs: KExpr<T>): KGtArithExpr<T> =
        arithGtCache.createIfContextActive {
            ensureContextMatch(lhs, rhs)
            KGtArithExpr(this, lhs, rhs)
        }.cast()

    private val arithGeCache = mkAstInterner<KGeArithExpr<out KArithSort>>()

    /**
     * Create an Int/Real arithmetic `>=` (greater-or-equal) expression.
     * */
    open fun <T : KArithSort> mkArithGe(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<KBoolSort> =
        mkSimplified(lhs, rhs, KContext::simplifyArithGe, ::mkArithGeNoSimplify)

    /**
     * Create an Int/Real arithmetic `>=` (greater-or-equal) expression.
     * */
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

    /**
     * Create an Int mod expression.
     *
     * The result value is a number `r` such that
     * [lhs] = `r` + ([rhs] * (div [lhs] [rhs]))
     * where `div` is an Int division that works according to the [mkArithDiv] rules.
     * The result value is always positive or zero.
     *
     * For example:
     *   `47 mod 13 = 8`
     *   `47 mod -13 = 8`
     *   `-47 mod 13 = 5`
     *   `-47 mod -13 = 5`
     * */
    open fun mkIntMod(lhs: KExpr<KIntSort>, rhs: KExpr<KIntSort>): KExpr<KIntSort> =
        mkSimplified(lhs, rhs, KContext::simplifyIntMod, ::mkIntModNoSimplify)

    /**
     * Create an Int mod expression.
     *
     * @see mkIntMod for the operation details.
     * */
    open fun mkIntModNoSimplify(
        lhs: KExpr<KIntSort>,
        rhs: KExpr<KIntSort>
    ): KModIntExpr = intModCache.createIfContextActive {
        ensureContextMatch(lhs, rhs)
        KModIntExpr(this, lhs, rhs)
    }

    private val intRemCache = mkAstInterner<KRemIntExpr>()

    /**
     * Create an Int rem expression.
     * The result value is either (mod lhs rhs) when rhs >= 0 and (neg (mod lhs rhs)) otherwise.
     * The result sign matches the [rhs] sign.
     *
     * @see mkIntMod
     * */
    open fun mkIntRem(lhs: KExpr<KIntSort>, rhs: KExpr<KIntSort>): KExpr<KIntSort> =
        mkSimplified(lhs, rhs, KContext::simplifyIntRem, ::mkIntRemNoSimplify)

    /**
     * Create an Int rem expression.
     *
     * @see mkIntRem for the operation details.
     * */
    open fun mkIntRemNoSimplify(
        lhs: KExpr<KIntSort>,
        rhs: KExpr<KIntSort>
    ): KRemIntExpr = intRemCache.createIfContextActive {
        ensureContextMatch(lhs, rhs)
        KRemIntExpr(this, lhs, rhs)
    }

    private val intToRealCache = mkAstInterner<KToRealIntExpr>()

    /**
     * Convert an Int expression to a corresponding Real expression.
     * */
    open fun mkIntToReal(arg: KExpr<KIntSort>): KExpr<KRealSort> =
        mkSimplified(arg, KContext::simplifyIntToReal, ::mkIntToRealNoSimplify)

    /**
     * Convert an Int expression to a corresponding Real expression.
     * */
    open fun mkIntToRealNoSimplify(arg: KExpr<KIntSort>): KToRealIntExpr = intToRealCache.createIfContextActive {
        ensureContextMatch(arg)
        KToRealIntExpr(this, arg)
    }

    private val int32NumCache = mkAstInterner<KInt32NumExpr>()
    private val int64NumCache = mkAstInterner<KInt64NumExpr>()
    private val intBigNumCache = mkAstInterner<KIntBigNumExpr>()

    /**
     * Create an Int value.
     * */
    fun mkIntNum(value: Int): KIntNumExpr = int32NumCache.createIfContextActive {
        KInt32NumExpr(this, value)
    }

    /**
     * Create an Int value.
     * */
    fun mkIntNum(value: Long): KIntNumExpr = if (value.toInt().toLong() == value) {
        mkIntNum(value.toInt())
    } else {
        int64NumCache.createIfContextActive {
            KInt64NumExpr(this, value)
        }
    }

    /**
     * Create an Int value.
     * */
    fun mkIntNum(value: BigInteger): KIntNumExpr = if (value.toLong().toBigInteger() == value) {
        mkIntNum(value.toLong())
    } else {
        intBigNumCache.createIfContextActive {
            KIntBigNumExpr(this, value)
        }
    }

    /**
     * Create an Int value.
     * */
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

    /**
     * Convert Real expression to an Int expression (floor division).
     * */
    open fun mkRealToInt(arg: KExpr<KRealSort>): KExpr<KIntSort> =
        mkSimplified(arg, KContext::simplifyRealToInt, ::mkRealToIntNoSimplify)

    /**
     * Convert Real expression to an Int expression (floor division).
     * */
    open fun mkRealToIntNoSimplify(arg: KExpr<KRealSort>): KToIntRealExpr = realToIntCache.createIfContextActive {
        ensureContextMatch(arg)
        KToIntRealExpr(this, arg)
    }

    private val realIsIntCache = mkAstInterner<KIsIntRealExpr>()

    /**
     * Check whether the given Real expression has an integer value.
     * */
    open fun mkRealIsInt(arg: KExpr<KRealSort>): KExpr<KBoolSort> =
        mkSimplified(arg, KContext::simplifyRealIsInt, ::mkRealIsIntNoSimplify)

    /**
     * Check whether the given Real expression has an integer value.
     * */
    open fun mkRealIsIntNoSimplify(arg: KExpr<KRealSort>): KIsIntRealExpr = realIsIntCache.createIfContextActive {
        ensureContextMatch(arg)
        KIsIntRealExpr(this, arg)
    }

    private val realNumCache = mkAstInterner<KRealNumExpr>()

    /**
     * Create a Real value.
     * */
    fun mkRealNum(numerator: KIntNumExpr, denominator: KIntNumExpr): KRealNumExpr =
        realNumCache.createIfContextActive {
            ensureContextMatch(numerator, denominator)
            KRealNumExpr(this, numerator, denominator)
        }

    /**
     * Create a Real value.
     * */
    fun mkRealNum(numerator: KIntNumExpr) = mkRealNum(numerator, 1.expr)

    /**
     * Create a Real value.
     * */
    fun mkRealNum(numerator: Int) = mkRealNum(mkIntNum(numerator))

    /**
     * Create a Real value.
     * */
    fun mkRealNum(numerator: Int, denominator: Int) = mkRealNum(mkIntNum(numerator), mkIntNum(denominator))

    /**
     * Create a Real value.
     * */
    fun mkRealNum(numerator: Long) = mkRealNum(mkIntNum(numerator))

    /**
     * Create a Real value.
     * */
    fun mkRealNum(numerator: Long, denominator: Long) = mkRealNum(mkIntNum(numerator), mkIntNum(denominator))

    /**
     * Create a Real value from a string of the form `"123/456"`.
     * */
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


    /**
     * Create a BitVec value with 1 bit length.
     * */
    fun mkBv(value: Boolean): KBitVec1Value = bv1Cache.createIfContextActive { KBitVec1Value(this, value) }

    /**
     * Create a BitVec value with [sizeBits] bit length.
     *
     * Note: if [sizeBits] is greater than 1, the [value] bit will be repeated.
     * */
    fun mkBv(value: Boolean, sizeBits: UInt): KBitVecValue<KBvSort> {
        if (sizeBits == bv1Sort.sizeBits) return mkBv(value).cast()
        val intValue = (if (value) 1 else 0) as Number
        return mkBv(intValue, sizeBits)
    }

    /**
     * Create a BitVec value of the BitVec [sort].
     *
     * Note: if [sort] size is greater than 1, the [value] bit will be repeated.
     * */
    fun <T : KBvSort> mkBv(value: Boolean, sort: T): KBitVecValue<T> =
        if (sort == bv1Sort) mkBv(value).cast() else mkBv(value, sort.sizeBits).cast()

    fun Boolean.toBv(): KBitVec1Value = mkBv(this)
    fun Boolean.toBv(sizeBits: UInt): KBitVecValue<KBvSort> = mkBv(this, sizeBits)
    fun <T : KBvSort> Boolean.toBv(sort: T): KBitVecValue<T> = mkBv(this, sort)

    /**
     * Create a BitVec value with 8 bit length.
     * */
    fun mkBv(value: Byte): KBitVec8Value = bv8Cache.createIfContextActive { KBitVec8Value(this, value) }

    /**
     * Create a BitVec value with [sizeBits] bit length.
     *
     * Note: if [sizeBits] is less than 8,
     * the last [sizeBits] bits of the [value] will be taken.
     *
     * At the same time, if [sizeBits] is greater than 8,
     * binary representation of the [value] will be padded from the start with its sign bit.
     * */
    fun mkBv(value: Byte, sizeBits: UInt): KBitVecValue<KBvSort> =
        if (sizeBits == bv8Sort.sizeBits) mkBv(value).cast() else mkBv(value as Number, sizeBits)

    /**
     * Create a BitVec value of the BitVec [sort].
     *
     * Note: if [sort] size is less than 8,
     * the last [sort] size bits of the [value] will be taken.
     *
     * At the same time, if [sort] size is greater than 8,
     * binary representation of the [value] will be padded from the start with its sign bit.
     * */
    fun <T : KBvSort> mkBv(value: Byte, sort: T): KBitVecValue<T> =
        if (sort == bv8Sort) mkBv(value).cast() else mkBv(value as Number, sort)

    fun Byte.toBv(): KBitVec8Value = mkBv(this)
    fun Byte.toBv(sizeBits: UInt): KBitVecValue<KBvSort> = mkBv(this, sizeBits)
    fun <T : KBvSort> Byte.toBv(sort: T): KBitVecValue<T> = mkBv(this, sort)
    fun UByte.toBv(): KBitVec8Value = mkBv(toByte())

    /**
     * Create a BitVec value with 16 bit length.
     * */
    fun mkBv(value: Short): KBitVec16Value = bv16Cache.createIfContextActive { KBitVec16Value(this, value) }

    /**
     * Create a BitVec value with [sizeBits] bit length.
     *
     * Note: if [sizeBits] is less than 16,
     * the last [sizeBits] bits of the [value] will be taken.
     *
     * At the same time, if [sizeBits] is greater than 16,
     * binary representation of the [value] will be padded from the start with its sign bit.
     * */
    fun mkBv(value: Short, sizeBits: UInt): KBitVecValue<KBvSort> =
        if (sizeBits == bv16Sort.sizeBits) mkBv(value).cast() else mkBv(value as Number, sizeBits)

    /**
     * Create a BitVec value of the BitVec [sort].
     *
     * Note: if [sort] size is less than 16,
     * the last [sort] size bits of the [value] will be taken.
     *
     * At the same time, if [sort] size is greater than 16,
     * binary representation of the [value] will be padded from the start with its sign bit.
     * */
    fun <T : KBvSort> mkBv(value: Short, sort: T): KBitVecValue<T>  =
        if (sort == bv16Sort) mkBv(value).cast() else mkBv(value as Number, sort)

    fun Short.toBv(): KBitVec16Value = mkBv(this)
    fun Short.toBv(sizeBits: UInt): KBitVecValue<KBvSort> = mkBv(this, sizeBits)
    fun <T : KBvSort> Short.toBv(sort: T): KBitVecValue<T> = mkBv(this, sort)
    fun UShort.toBv(): KBitVec16Value = mkBv(toShort())

    /**
     * Create a BitVec value with 32 bit length.
     * */
    fun mkBv(value: Int): KBitVec32Value = bv32Cache.createIfContextActive { KBitVec32Value(this, value) }

    /**
     * Create a BitVec value with [sizeBits] bit length.
     *
     * Note: if [sizeBits] is less than 32,
     * the last [sizeBits] bits of the [value] will be taken.
     *
     * At the same time, if [sizeBits] is greater than 32,
     * binary representation of the [value] will be padded from the start with its sign bit.
     * */
    fun mkBv(value: Int, sizeBits: UInt): KBitVecValue<KBvSort> =
        if (sizeBits == bv32Sort.sizeBits) mkBv(value).cast() else mkBv(value as Number, sizeBits)

    /**
     * Create a BitVec value of the BitVec [sort].
     *
     * Note: if [sort] size is less than 32,
     * the last [sort] size bits of the [value] will be taken.
     *
     * At the same time, if [sort] size is greater than 32,
     * binary representation of the [value] will be padded from the start with its sign bit.
     * */
    fun <T : KBvSort> mkBv(value: Int, sort: T): KBitVecValue<T> =
        if (sort == bv32Sort) mkBv(value).cast() else mkBv(value as Number, sort)

    fun Int.toBv(): KBitVec32Value = mkBv(this)
    fun Int.toBv(sizeBits: UInt): KBitVecValue<KBvSort> = mkBv(this, sizeBits)
    fun <T : KBvSort> Int.toBv(sort: T): KBitVecValue<T> = mkBv(this, sort)
    fun UInt.toBv(): KBitVec32Value = mkBv(toInt())

    /**
     * Create a BitVec value with 64 bit length.
     * */
    fun mkBv(value: Long): KBitVec64Value = bv64Cache.createIfContextActive { KBitVec64Value(this, value) }

    /**
     * Create a BitVec value with [sizeBits] bit length.
     *
     * Note: if [sizeBits] is less than 64,
     * the last [sizeBits] bits of the [value] will be taken.
     *
     * At the same time, if [sizeBits] is greater than 64,
     * binary representation of the [value] will be padded from the start with its sign bit.
     * */
    fun mkBv(value: Long, sizeBits: UInt): KBitVecValue<KBvSort> =
        if (sizeBits == bv64Sort.sizeBits) mkBv(value).cast() else mkBv(value as Number, sizeBits)

    /**
     * Create a BitVec value of the BitVec [sort].
     *
     * Note: if [sort] size is less than 64,
     * the last [sort] size bits of the [value] will be taken.
     *
     * At the same time, if [sort] size is greater than 64,
     * binary representation of the [value] will be padded from the start with its sign bit.
     * */
    fun <T : KBvSort> mkBv(value: Long, sort: T): KBitVecValue<T> =
        if (sort == bv64Sort) mkBv(value).cast() else mkBv(value as Number, sort)

    fun Long.toBv(): KBitVec64Value = mkBv(this)
    fun Long.toBv(sizeBits: UInt): KBitVecValue<KBvSort> = mkBv(this, sizeBits)
    fun <T : KBvSort> Long.toBv(sort: T): KBitVecValue<T> = mkBv(this, sort)
    fun ULong.toBv(): KBitVec64Value = mkBv(toLong())

    /**
     * Create a BitVec value with [sizeBits] bit length.
     * */
    fun mkBv(value: BigInteger, sizeBits: UInt): KBitVecValue<KBvSort> = mkBv(value as Number, sizeBits)

    /**
     * Create a BitVec value of the BitVec [sort].
     * */
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

    /**
     * Create a BitVec value with [sizeBits] bit length from the given binary string [value].
     * */
    fun mkBv(value: String, sizeBits: UInt): KBitVecValue<KBvSort> =
        mkBv(value.toBigInteger(radix = 2), sizeBits)

    /**
     * Create a BitVec value with [sizeBits] bit length from the given hex string [value].
     * */
    fun mkBvHex(value: String, sizeBits: UInt): KBitVecValue<KBvSort> =
        mkBv(value.toBigInteger(radix = 16), sizeBits)

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

    /**
     * Create bitwise NOT (`bvnot`) expression.
     * */
    open fun <T : KBvSort> mkBvNotExpr(value: KExpr<T>): KExpr<T> =
        mkSimplified(value, KContext::simplifyBvNotExpr, ::mkBvNotExprNoSimplify)

    /**
     * Create bitwise NOT (`bvnot`) expression.
     * */
    open fun <T : KBvSort> mkBvNotExprNoSimplify(value: KExpr<T>): KBvNotExpr<T> =
        bvNotExprCache.createIfContextActive {
            ensureContextMatch(value)
            KBvNotExpr(this, value)
        }.cast()

    private val bvRedAndExprCache = mkAstInterner<KBvReductionAndExpr<out KBvSort>>()

    /**
     * Create bitwise AND reduction (`bvredand`) expression.
     * Reduce all bits to a single bit with AND operation.
     * */
    open fun <T : KBvSort> mkBvReductionAndExpr(value: KExpr<T>): KExpr<KBv1Sort> =
        mkSimplified(value, KContext::simplifyBvReductionAndExpr, ::mkBvReductionAndExprNoSimplify)

    /**
     * Create bitwise AND reduction (`bvredand`) expression.
     * Reduce all bits to a single bit with AND operation.
     * */
    open fun <T : KBvSort> mkBvReductionAndExprNoSimplify(value: KExpr<T>): KBvReductionAndExpr<T> =
        bvRedAndExprCache.createIfContextActive {
            ensureContextMatch(value)
            KBvReductionAndExpr(this, value)
        }.cast()

    fun <T : KBvSort> KExpr<T>.reductionAnd() = mkBvReductionAndExpr(this)

    private val bvRedOrExprCache = mkAstInterner<KBvReductionOrExpr<out KBvSort>>()

    /**
     * Create bitwise OR reduction (`bvredor`) expression.
     * Reduce all bits to a single bit with OR operation.
     * */
    open fun <T : KBvSort> mkBvReductionOrExpr(value: KExpr<T>): KExpr<KBv1Sort> =
        mkSimplified(value, KContext::simplifyBvReductionOrExpr, ::mkBvReductionOrExprNoSimplify)

    /**
     * Create bitwise OR reduction (`bvredor`) expression.
     * Reduce all bits to a single bit with OR operation.
     * */
    open fun <T : KBvSort> mkBvReductionOrExprNoSimplify(value: KExpr<T>): KBvReductionOrExpr<T> =
        bvRedOrExprCache.createIfContextActive {
            ensureContextMatch(value)
            KBvReductionOrExpr(this, value)
        }.cast()

    fun <T : KBvSort> KExpr<T>.reductionOr() = mkBvReductionOrExpr(this)

    private val bvAndExprCache = mkAstInterner<KBvAndExpr<out KBvSort>>()

    /**
     * Create bitwise AND (`bvand`) expression.
     * */
    open fun <T : KBvSort> mkBvAndExpr(arg0: KExpr<T>, arg1: KExpr<T>): KExpr<T> =
        mkSimplified(arg0, arg1, KContext::simplifyBvAndExpr, ::mkBvAndExprNoSimplify)

    /**
     * Create bitwise AND (`bvand`) expression.
     * */
    open fun <T : KBvSort> mkBvAndExprNoSimplify(arg0: KExpr<T>, arg1: KExpr<T>): KBvAndExpr<T> =
        bvAndExprCache.createIfContextActive {
            ensureContextMatch(arg0, arg1)
            KBvAndExpr(this, arg0, arg1)
        }.cast()

    private val bvOrExprCache = mkAstInterner<KBvOrExpr<out KBvSort>>()


    /**
     * Create bitwise OR (`bvor`) expression.
     * */
    open fun <T : KBvSort> mkBvOrExpr(arg0: KExpr<T>, arg1: KExpr<T>): KExpr<T> =
        mkSimplified(arg0, arg1, KContext::simplifyBvOrExpr, ::mkBvOrExprNoSimplify)

    /**
     * Create bitwise OR (`bvor`) expression.
     * */
    open fun <T : KBvSort> mkBvOrExprNoSimplify(arg0: KExpr<T>, arg1: KExpr<T>): KBvOrExpr<T> =
        bvOrExprCache.createIfContextActive {
            ensureContextMatch(arg0, arg1)
            KBvOrExpr(this, arg0, arg1)
        }.cast()

    private val bvXorExprCache = mkAstInterner<KBvXorExpr<out KBvSort>>()

    /**
     * Create bitwise XOR (`bvxor`) expression.
     * */
    open fun <T : KBvSort> mkBvXorExpr(arg0: KExpr<T>, arg1: KExpr<T>): KExpr<T> =
        mkSimplified(arg0, arg1, KContext::simplifyBvXorExpr, ::mkBvXorExprNoSimplify)

    /**
     * Create bitwise XOR (`bvxor`) expression.
     * */
    open fun <T : KBvSort> mkBvXorExprNoSimplify(arg0: KExpr<T>, arg1: KExpr<T>): KBvXorExpr<T> =
        bvXorExprCache.createIfContextActive {
            ensureContextMatch(arg0, arg1)
            KBvXorExpr(this, arg0, arg1)
        }.cast()

    private val bvNAndExprCache = mkAstInterner<KBvNAndExpr<out KBvSort>>()

    /**
     * Create bitwise NAND (`bvnand`) expression.
     * */
    open fun <T : KBvSort> mkBvNAndExpr(arg0: KExpr<T>, arg1: KExpr<T>): KExpr<T> =
        mkSimplified(arg0, arg1, KContext::simplifyBvNAndExpr, ::mkBvNAndExprNoSimplify)

    /**
     * Create bitwise NAND (`bvnand`) expression.
     * */
    open fun <T : KBvSort> mkBvNAndExprNoSimplify(arg0: KExpr<T>, arg1: KExpr<T>): KBvNAndExpr<T> =
        bvNAndExprCache.createIfContextActive {
            ensureContextMatch(arg0, arg1)
            KBvNAndExpr(this, arg0, arg1)
        }.cast()

    private val bvNorExprCache = mkAstInterner<KBvNorExpr<out KBvSort>>()

    /**
     * Create bitwise NOR (`bvnor`) expression.
     * */
    open fun <T : KBvSort> mkBvNorExpr(arg0: KExpr<T>, arg1: KExpr<T>): KExpr<T> =
        mkSimplified(arg0, arg1, KContext::simplifyBvNorExpr, ::mkBvNorExprNoSimplify)

    /**
     * Create bitwise NOR (`bvnor`) expression.
     * */
    open fun <T : KBvSort> mkBvNorExprNoSimplify(arg0: KExpr<T>, arg1: KExpr<T>): KBvNorExpr<T> =
        bvNorExprCache.createIfContextActive {
            ensureContextMatch(arg0, arg1)
            KBvNorExpr(this, arg0, arg1)
        }.cast()

    private val bvXNorExprCache = mkAstInterner<KBvXNorExpr<out KBvSort>>()

    /**
     * Create bitwise XNOR (`bvxnor`) expression.
     * */
    open fun <T : KBvSort> mkBvXNorExpr(arg0: KExpr<T>, arg1: KExpr<T>): KExpr<T> =
        mkSimplified(arg0, arg1, KContext::simplifyBvXNorExpr, ::mkBvXNorExprNoSimplify)

    /**
     * Create bitwise XNOR (`bvxnor`) expression.
     * */
    open fun <T : KBvSort> mkBvXNorExprNoSimplify(arg0: KExpr<T>, arg1: KExpr<T>): KBvXNorExpr<T> =
        bvXNorExprCache.createIfContextActive {
            ensureContextMatch(arg0, arg1)
            KBvXNorExpr(this, arg0, arg1)
        }.cast()

    private val bvNegationExprCache = mkAstInterner<KBvNegationExpr<out KBvSort>>()

    /**
     * Create BitVec arithmetic negation (`bvneg`) expression.
     *
     * @see mkBvNotExpr for bitwise not.
     * */
    open fun <T : KBvSort> mkBvNegationExpr(value: KExpr<T>): KExpr<T> =
        mkSimplified(value, KContext::simplifyBvNegationExpr, ::mkBvNegationExprNoSimplify)

    /**
     * Create BitVec arithmetic negation (`bvneg`) expression.
     *
     * @see mkBvNotExpr for bitwise not.
     * */
    open fun <T : KBvSort> mkBvNegationExprNoSimplify(value: KExpr<T>): KBvNegationExpr<T> =
        bvNegationExprCache.createIfContextActive {
            ensureContextMatch(value)
            KBvNegationExpr(this, value)
        }.cast()

    private val bvAddExprCache = mkAstInterner<KBvAddExpr<out KBvSort>>()

    /**
     * Create BitVec arithmetic addition (`bvadd`) expression.
     * */
    open fun <T : KBvSort> mkBvAddExpr(arg0: KExpr<T>, arg1: KExpr<T>): KExpr<T> =
        mkSimplified(arg0, arg1, KContext::simplifyBvAddExpr, ::mkBvAddExprNoSimplify)

    /**
     * Create BitVec arithmetic addition (`bvadd`) expression.
     * */
    open fun <T : KBvSort> mkBvAddExprNoSimplify(arg0: KExpr<T>, arg1: KExpr<T>): KBvAddExpr<T> =
        bvAddExprCache.createIfContextActive {
            ensureContextMatch(arg0, arg1)
            KBvAddExpr(this, arg0, arg1)
        }.cast()

    private val bvSubExprCache = mkAstInterner<KBvSubExpr<out KBvSort>>()

    /**
     * Create BitVec arithmetic subtraction (`bvsub`) expression.
     * */
    open fun <T : KBvSort> mkBvSubExpr(arg0: KExpr<T>, arg1: KExpr<T>): KExpr<T> =
        mkSimplified(arg0, arg1, KContext::simplifyBvSubExpr, ::mkBvSubExprNoSimplify)

    /**
     * Create BitVec arithmetic subtraction (`bvsub`) expression.
     * */
    open fun <T : KBvSort> mkBvSubExprNoSimplify(arg0: KExpr<T>, arg1: KExpr<T>): KBvSubExpr<T> =
        bvSubExprCache.createIfContextActive {
            ensureContextMatch(arg0, arg1)
            KBvSubExpr(this, arg0, arg1)
        }.cast()

    private val bvMulExprCache = mkAstInterner<KBvMulExpr<out KBvSort>>()

    /**
     * Create BitVec arithmetic multiplication (`bvmul`) expression.
     * */
    open fun <T : KBvSort> mkBvMulExpr(arg0: KExpr<T>, arg1: KExpr<T>): KExpr<T> =
        mkSimplified(arg0, arg1, KContext::simplifyBvMulExpr, ::mkBvMulExprNoSimplify)

    /**
     * Create BitVec arithmetic multiplication (`bvmul`) expression.
     * */
    open fun <T : KBvSort> mkBvMulExprNoSimplify(arg0: KExpr<T>, arg1: KExpr<T>): KBvMulExpr<T> =
        bvMulExprCache.createIfContextActive {
            ensureContextMatch(arg0, arg1)
            KBvMulExpr(this, arg0, arg1)
        }.cast()

    private val bvUnsignedDivExprCache = mkAstInterner<KBvUnsignedDivExpr<out KBvSort>>()

    /**
     * Create BitVec arithmetic unsigned division (`bvudiv`) expression.
     *
     * @see mkBvSignedDivExpr for the signed division.
     * */
    open fun <T : KBvSort> mkBvUnsignedDivExpr(arg0: KExpr<T>, arg1: KExpr<T>): KExpr<T> =
        mkSimplified(arg0, arg1, KContext::simplifyBvUnsignedDivExpr, ::mkBvUnsignedDivExprNoSimplify)

    /**
     * Create BitVec arithmetic unsigned division (`bvudiv`) expression.
     *
     * @see mkBvSignedDivExpr for the signed division.
     * */
    open fun <T : KBvSort> mkBvUnsignedDivExprNoSimplify(arg0: KExpr<T>, arg1: KExpr<T>): KBvUnsignedDivExpr<T> =
        bvUnsignedDivExprCache.createIfContextActive {
            ensureContextMatch(arg0, arg1)
            KBvUnsignedDivExpr(this, arg0, arg1)
        }.cast()

    private val bvSignedDivExprCache = mkAstInterner<KBvSignedDivExpr<out KBvSort>>()

    /**
     * Create BitVec arithmetic signed division (`bvsdiv`) expression.
     * The division result sign depends on the arguments signs.
     *
     * @see mkBvUnsignedDivExpr for the unsigned division.
     * */
    open fun <T : KBvSort> mkBvSignedDivExpr(arg0: KExpr<T>, arg1: KExpr<T>): KExpr<T> =
        mkSimplified(arg0, arg1, KContext::simplifyBvSignedDivExpr, ::mkBvSignedDivExprNoSimplify)

    /**
     * Create BitVec arithmetic signed division (`bvsdiv`) expression.
     * The division result sign depends on the arguments signs.
     *
     * @see mkBvUnsignedDivExpr for the unsigned division.
     * */
    open fun <T : KBvSort> mkBvSignedDivExprNoSimplify(arg0: KExpr<T>, arg1: KExpr<T>): KBvSignedDivExpr<T> =
        bvSignedDivExprCache.createIfContextActive {
            ensureContextMatch(arg0, arg1)
            KBvSignedDivExpr(this, arg0, arg1)
        }.cast()

    private val bvUnsignedRemExprCache = mkAstInterner<KBvUnsignedRemExpr<out KBvSort>>()

    /**
     * Create BitVec arithmetic unsigned reminder (`bvurem`) expression.
     *
     * @see mkBvSignedRemExpr for the signed remainder.
     * */
    open fun <T : KBvSort> mkBvUnsignedRemExpr(arg0: KExpr<T>, arg1: KExpr<T>): KExpr<T> =
        mkSimplified(arg0, arg1, KContext::simplifyBvUnsignedRemExpr, ::mkBvUnsignedRemExprNoSimplify)

    /**
     * Create BitVec arithmetic unsigned reminder (`bvurem`) expression.
     *
     * @see mkBvSignedRemExpr for the signed remainder.
     * */
    open fun <T : KBvSort> mkBvUnsignedRemExprNoSimplify(arg0: KExpr<T>, arg1: KExpr<T>): KBvUnsignedRemExpr<T> =
        bvUnsignedRemExprCache.createIfContextActive {
            ensureContextMatch(arg0, arg1)
            KBvUnsignedRemExpr(this, arg0, arg1)
        }.cast()

    private val bvSignedRemExprCache = mkAstInterner<KBvSignedRemExpr<out KBvSort>>()

    /**
     * Create BitVec arithmetic signed reminder (`bvsrem`) expression.
     *
     * Computes remainder of a `truncate` (round toward zero) division.
     * The result sign matches the [arg0] sign.
     *
     * For example:
     *  `47 bvsrem 13 = 8`
     *  `47 bvsrem -13 = 8`
     *  `-47 bvsrem 13 = -8`
     *  `-47 bvsrem -13 = -8`
     *
     * @see mkBvUnsignedRemExpr for the unsigned remainder.
     * */
    open fun <T : KBvSort> mkBvSignedRemExpr(arg0: KExpr<T>, arg1: KExpr<T>): KExpr<T> =
        mkSimplified(arg0, arg1, KContext::simplifyBvSignedRemExpr, ::mkBvSignedRemExprNoSimplify)

    /**
     * Create BitVec arithmetic signed reminder (`bvsrem`) expression.
     * @see [mkBvSignedRemExpr] for the operation details.
     * @see mkBvUnsignedRemExpr for the unsigned remainder.
     * */
    open fun <T : KBvSort> mkBvSignedRemExprNoSimplify(arg0: KExpr<T>, arg1: KExpr<T>): KBvSignedRemExpr<T> =
        bvSignedRemExprCache.createIfContextActive {
            ensureContextMatch(arg0, arg1)
            KBvSignedRemExpr(this, arg0, arg1)
        }.cast()

    private val bvSignedModExprCache = mkAstInterner<KBvSignedModExpr<out KBvSort>>()

    /**
     * Create BitVec arithmetic signed mod (`bvsmod`) expression.
     *
     * Computes remainder of a `floor` (round toward negative infinity) division.
     * The result sign matches the [arg1] sign.
     *
     * For example:
     *  `47 bvsmod 13 = 8`
     *  `47 bvsmod -13 = -5`
     *  `-47 bvsmod 13 = 5`
     *  `-47 bvsmod -13 = -8`
     * */
    open fun <T : KBvSort> mkBvSignedModExpr(arg0: KExpr<T>, arg1: KExpr<T>): KExpr<T> =
        mkSimplified(arg0, arg1, KContext::simplifyBvSignedModExpr, ::mkBvSignedModExprNoSimplify)

    /**
     * Create BitVec arithmetic signed mod (`bvsmod`) expression.
     * @see [mkBvSignedModExpr] for the operation details.
     * */
    open fun <T : KBvSort> mkBvSignedModExprNoSimplify(arg0: KExpr<T>, arg1: KExpr<T>): KBvSignedModExpr<T> =
        bvSignedModExprCache.createIfContextActive {
            ensureContextMatch(arg0, arg1)
            KBvSignedModExpr(this, arg0, arg1)
        }.cast()

    private val bvUnsignedLessExprCache = mkAstInterner<KBvUnsignedLessExpr<out KBvSort>>()

    /**
     * Create BitVec unsigned less (`bvult`) expression.
     * */
    open fun <T : KBvSort> mkBvUnsignedLessExpr(arg0: KExpr<T>, arg1: KExpr<T>): KExpr<KBoolSort> =
        mkSimplified(arg0, arg1, KContext::simplifyBvUnsignedLessExpr, ::mkBvUnsignedLessExprNoSimplify)

    /**
     * Create BitVec unsigned less (`bvult`) expression.
     * */
    open fun <T : KBvSort> mkBvUnsignedLessExprNoSimplify(arg0: KExpr<T>, arg1: KExpr<T>): KBvUnsignedLessExpr<T> =
        bvUnsignedLessExprCache.createIfContextActive {
            ensureContextMatch(arg0, arg1)
            KBvUnsignedLessExpr(this, arg0, arg1)
        }.cast()

    private val bvSignedLessExprCache = mkAstInterner<KBvSignedLessExpr<out KBvSort>>()

    /**
     * Create BitVec signed less (`bvslt`) expression.
     * */
    open fun <T : KBvSort> mkBvSignedLessExpr(arg0: KExpr<T>, arg1: KExpr<T>): KExpr<KBoolSort> =
        mkSimplified(arg0, arg1, KContext::simplifyBvSignedLessExpr, ::mkBvSignedLessExprNoSimplify)

    /**
     * Create BitVec signed less (`bvslt`) expression.
     * */
    open fun <T : KBvSort> mkBvSignedLessExprNoSimplify(arg0: KExpr<T>, arg1: KExpr<T>): KBvSignedLessExpr<T> =
        bvSignedLessExprCache.createIfContextActive {
            ensureContextMatch(arg0, arg1)
            KBvSignedLessExpr(this, arg0, arg1)
        }.cast()

    private val bvSignedLessOrEqualExprCache = mkAstInterner<KBvSignedLessOrEqualExpr<out KBvSort>>()

    /**
     * Create BitVec signed less-or-equal (`bvsle`) expression.
     * */
    open fun <T : KBvSort> mkBvSignedLessOrEqualExpr(arg0: KExpr<T>, arg1: KExpr<T>): KExpr<KBoolSort> =
        mkSimplified(arg0, arg1, KContext::simplifyBvSignedLessOrEqualExpr, ::mkBvSignedLessOrEqualExprNoSimplify)

    /**
     * Create BitVec signed less-or-equal (`bvsle`) expression.
     * */
    open fun <T : KBvSort> mkBvSignedLessOrEqualExprNoSimplify(
        arg0: KExpr<T>, arg1: KExpr<T>
    ): KBvSignedLessOrEqualExpr<T> =
        bvSignedLessOrEqualExprCache.createIfContextActive {
            ensureContextMatch(arg0, arg1)
            KBvSignedLessOrEqualExpr(this, arg0, arg1)
        }.cast()

    private val bvUnsignedLessOrEqualExprCache = mkAstInterner<KBvUnsignedLessOrEqualExpr<out KBvSort>>()

    /**
     * Create BitVec unsigned less-or-equal (`bvule`) expression.
     * */
    open fun <T : KBvSort> mkBvUnsignedLessOrEqualExpr(arg0: KExpr<T>, arg1: KExpr<T>): KExpr<KBoolSort> =
        mkSimplified(arg0, arg1, KContext::simplifyBvUnsignedLessOrEqualExpr, ::mkBvUnsignedLessOrEqualExprNoSimplify)

    /**
     * Create BitVec unsigned less-or-equal (`bvule`) expression.
     * */
    open fun <T : KBvSort> mkBvUnsignedLessOrEqualExprNoSimplify(
        arg0: KExpr<T>,
        arg1: KExpr<T>
    ): KBvUnsignedLessOrEqualExpr<T> = bvUnsignedLessOrEqualExprCache.createIfContextActive {
        ensureContextMatch(arg0, arg1)
        KBvUnsignedLessOrEqualExpr(this, arg0, arg1)
    }.cast()

    private val bvUnsignedGreaterOrEqualExprCache = mkAstInterner<KBvUnsignedGreaterOrEqualExpr<out KBvSort>>()

    /**
     * Create BitVec unsigned greater-or-equal (`bvuge`) expression.
     * */
    open fun <T : KBvSort> mkBvUnsignedGreaterOrEqualExpr(arg0: KExpr<T>, arg1: KExpr<T>): KExpr<KBoolSort> =
        mkSimplified(
            arg0,
            arg1,
            KContext::simplifyBvUnsignedGreaterOrEqualExpr,
            ::mkBvUnsignedGreaterOrEqualExprNoSimplify
        )

    /**
     * Create BitVec unsigned greater-or-equal (`bvuge`) expression.
     * */
    open fun <T : KBvSort> mkBvUnsignedGreaterOrEqualExprNoSimplify(
        arg0: KExpr<T>,
        arg1: KExpr<T>
    ): KBvUnsignedGreaterOrEqualExpr<T> = bvUnsignedGreaterOrEqualExprCache.createIfContextActive {
        ensureContextMatch(arg0, arg1)
        KBvUnsignedGreaterOrEqualExpr(this, arg0, arg1)
    }.cast()

    private val bvSignedGreaterOrEqualExprCache = mkAstInterner<KBvSignedGreaterOrEqualExpr<out KBvSort>>()

    /**
     * Create BitVec signed greater-or-equal (`bvsge`) expression.
     * */
    open fun <T : KBvSort> mkBvSignedGreaterOrEqualExpr(arg0: KExpr<T>, arg1: KExpr<T>): KExpr<KBoolSort> =
        mkSimplified(arg0, arg1, KContext::simplifyBvSignedGreaterOrEqualExpr, ::mkBvSignedGreaterOrEqualExprNoSimplify)

    /**
     * Create BitVec signed greater-or-equal (`bvsge`) expression.
     * */
    open fun <T : KBvSort> mkBvSignedGreaterOrEqualExprNoSimplify(
        arg0: KExpr<T>,
        arg1: KExpr<T>
    ): KBvSignedGreaterOrEqualExpr<T> = bvSignedGreaterOrEqualExprCache.createIfContextActive {
        ensureContextMatch(arg0, arg1)
        KBvSignedGreaterOrEqualExpr(this, arg0, arg1)
    }.cast()

    private val bvUnsignedGreaterExprCache = mkAstInterner<KBvUnsignedGreaterExpr<out KBvSort>>()

    /**
     * Create BitVec unsigned greater (`bvugt`) expression.
     * */
    open fun <T : KBvSort> mkBvUnsignedGreaterExpr(arg0: KExpr<T>, arg1: KExpr<T>): KExpr<KBoolSort> =
        mkSimplified(arg0, arg1, KContext::simplifyBvUnsignedGreaterExpr, ::mkBvUnsignedGreaterExprNoSimplify)

    /**
     * Create BitVec unsigned greater (`bvugt`) expression.
     * */
    open fun <T : KBvSort> mkBvUnsignedGreaterExprNoSimplify(
        arg0: KExpr<T>, arg1: KExpr<T>
    ): KBvUnsignedGreaterExpr<T> =
        bvUnsignedGreaterExprCache.createIfContextActive {
            ensureContextMatch(arg0, arg1)
            KBvUnsignedGreaterExpr(this, arg0, arg1)
        }.cast()

    private val bvSignedGreaterExprCache = mkAstInterner<KBvSignedGreaterExpr<out KBvSort>>()

    /**
     * Create BitVec signed greater (`bvsgt`) expression.
     * */
    open fun <T : KBvSort> mkBvSignedGreaterExpr(arg0: KExpr<T>, arg1: KExpr<T>): KExpr<KBoolSort> =
        mkSimplified(arg0, arg1, KContext::simplifyBvSignedGreaterExpr, ::mkBvSignedGreaterExprNoSimplify)

    /**
     * Create BitVec signed greater (`bvsgt`) expression.
     * */
    open fun <T : KBvSort> mkBvSignedGreaterExprNoSimplify(arg0: KExpr<T>, arg1: KExpr<T>): KBvSignedGreaterExpr<T> =
        bvSignedGreaterExprCache.createIfContextActive {
            ensureContextMatch(arg0, arg1)
            KBvSignedGreaterExpr(this, arg0, arg1)
        }.cast()

    private val concatExprCache = mkAstInterner<KBvConcatExpr>()

    /**
     * Create BitVec concatenation (`concat`) expression.
     * */
    open fun <T : KBvSort, S : KBvSort> mkBvConcatExpr(arg0: KExpr<T>, arg1: KExpr<S>): KExpr<KBvSort> =
        mkSimplified(arg0, arg1, KContext::simplifyBvConcatExpr, ::mkBvConcatExprNoSimplify)

    /**
     * Create BitVec concatenation (`concat`) expression.
     * */
    open fun <T : KBvSort, S : KBvSort> mkBvConcatExprNoSimplify(arg0: KExpr<T>, arg1: KExpr<S>): KBvConcatExpr =
        concatExprCache.createIfContextActive {
            ensureContextMatch(arg0, arg1)
            KBvConcatExpr(this, arg0.cast(), arg1.cast())
        }

    private val extractExprCache = mkAstInterner<KBvExtractExpr>()

    /**
     * Create BitVec extract (`extract`) expression.
     * Extract bits from [low] (including) to [high] (including) as a new BitVec.
     * */
    open fun <T : KBvSort> mkBvExtractExpr(high: Int, low: Int, value: KExpr<T>): KExpr<KBvSort> =
        mkSimplified(high, low, value, KContext::simplifyBvExtractExpr, ::mkBvExtractExprNoSimplify)

    /**
     * Create BitVec extract (`extract`) expression.
     * Extract bits from [low] (including) to [high] (including) as a new BitVec.
     * */
    open fun <T : KBvSort> mkBvExtractExprNoSimplify(high: Int, low: Int, value: KExpr<T>): KBvExtractExpr =
        extractExprCache.createIfContextActive {
            ensureContextMatch(value)
            KBvExtractExpr(this, high, low, value.cast())
        }

    private val signExtensionExprCache = mkAstInterner<KBvSignExtensionExpr>()

    /**
     * Create BitVec signed extension (`signext`) expression.
     * Returns a BitVec expression with [extensionSize] extra sign (leftmost, highest) bits.
     * The extra bits are prepended to the provided [value].
     * */
    open fun <T : KBvSort> mkBvSignExtensionExpr(extensionSize: Int, value: KExpr<T>): KExpr<KBvSort> =
        mkSimplified(extensionSize, value, KContext::simplifyBvSignExtensionExpr, ::mkBvSignExtensionExprNoSimplify)

    /**
     * Create BitVec signed extension (`signext`) expression.
     * Returns a BitVec expression with [extensionSize] extra sign (leftmost, highest) bits.
     * The extra bits are prepended to the provided [value].
     * */
    open fun <T : KBvSort> mkBvSignExtensionExprNoSimplify(
        extensionSize: Int,
        value: KExpr<T>
    ): KBvSignExtensionExpr = signExtensionExprCache.createIfContextActive {
        ensureContextMatch(value)
        KBvSignExtensionExpr(this, extensionSize, value.cast())
    }

    private val zeroExtensionExprCache = mkAstInterner<KBvZeroExtensionExpr>()

    /**
     * Create BitVec signed extension (`signext`) expression.
     * Returns a BitVec expression with [extensionSize] extra sign (leftmost, highest) bits.
     * The extra bits are prepended to the provided [value].
     * */
    open fun <T : KBvSort> mkBvZeroExtensionExpr(extensionSize: Int, value: KExpr<T>): KExpr<KBvSort> =
        mkSimplified(extensionSize, value, KContext::simplifyBvZeroExtensionExpr, ::mkBvZeroExtensionExprNoSimplify)

    /**
     * Create BitVec signed extension (`signext`) expression.
     * Returns a BitVec expression with [extensionSize] extra sign (leftmost, highest) bits.
     * The extra bits are prepended to the provided [value].
     * */
    open fun <T : KBvSort> mkBvZeroExtensionExprNoSimplify(
        extensionSize: Int,
        value: KExpr<T>
    ): KBvZeroExtensionExpr = zeroExtensionExprCache.createIfContextActive {
        ensureContextMatch(value)
        KBvZeroExtensionExpr(this, extensionSize, value.cast())
    }

    private val repeatExprCache = mkAstInterner<KBvRepeatExpr>()

    /*
     * Create BitVec repeat (`repeat`) expression.
     * Returns a BitVec expression with [repeatNumber] concatenated copies of [value].
     * */
    open fun <T : KBvSort> mkBvRepeatExpr(repeatNumber: Int, value: KExpr<T>): KExpr<KBvSort> =
        mkSimplified(repeatNumber, value, KContext::simplifyBvRepeatExpr, ::mkBvRepeatExprNoSimplify)

    /**
     * Create BitVec repeat (`repeat`) expression.
     * Returns a BitVec expression with [repeatNumber] concatenated copies of [value].
     * */
    open fun <T : KBvSort> mkBvRepeatExprNoSimplify(
        repeatNumber: Int,
        value: KExpr<T>
    ): KBvRepeatExpr = repeatExprCache.createIfContextActive {
        ensureContextMatch(value)
        KBvRepeatExpr(this, repeatNumber, value.cast())
    }

    private val bvShiftLeftExprCache = mkAstInterner<KBvShiftLeftExpr<out KBvSort>>()

    /**
     * Create BitVec shift left (`bvshl`) expression.
     * */
    open fun <T : KBvSort> mkBvShiftLeftExpr(arg: KExpr<T>, shift: KExpr<T>): KExpr<T> =
        mkSimplified(arg, shift, KContext::simplifyBvShiftLeftExpr, ::mkBvShiftLeftExprNoSimplify)

    /**
     * Create BitVec shift left (`bvshl`) expression.
     * */
    open fun <T : KBvSort> mkBvShiftLeftExprNoSimplify(arg: KExpr<T>, shift: KExpr<T>): KBvShiftLeftExpr<T> =
        bvShiftLeftExprCache.createIfContextActive {
            ensureContextMatch(arg, shift)
            KBvShiftLeftExpr(this, arg, shift)
        }.cast()

    private val bvLogicalShiftRightExprCache = mkAstInterner<KBvLogicalShiftRightExpr<out KBvSort>>()

    /**
     * Create BitVec logical shift right (`bvlshr`) expression.
     * The shifted expressions is padded with zero bits.
     * */
    open fun <T : KBvSort> mkBvLogicalShiftRightExpr(arg: KExpr<T>, shift: KExpr<T>): KExpr<T> =
        mkSimplified(arg, shift, KContext::simplifyBvLogicalShiftRightExpr, ::mkBvLogicalShiftRightExprNoSimplify)

    /**
     * Create BitVec logical shift right (`bvlshr`) expression.
     * The shifted expressions is padded with zero bits.
     * */
    open fun <T : KBvSort> mkBvLogicalShiftRightExprNoSimplify(
        arg: KExpr<T>, shift: KExpr<T>
    ): KBvLogicalShiftRightExpr<T> =
        bvLogicalShiftRightExprCache.createIfContextActive {
            ensureContextMatch(arg, shift)
            KBvLogicalShiftRightExpr(this, arg, shift)
        }.cast()

    private val bvArithShiftRightExprCache = mkAstInterner<KBvArithShiftRightExpr<out KBvSort>>()

    /**
     * Create BitVec arithmetic shift right (`bvashr`) expression.
     * The shifted expressions is padded with sign (leftmost, highest) bits.
     * */
    open fun <T : KBvSort> mkBvArithShiftRightExpr(arg: KExpr<T>, shift: KExpr<T>): KExpr<T> =
        mkSimplified(arg, shift, KContext::simplifyBvArithShiftRightExpr, ::mkBvArithShiftRightExprNoSimplify)

    /**
     * Create BitVec arithmetic shift right (`bvashr`) expression.
     * The shifted expressions is padded with sign (leftmost, highest) bits.
     * */
    open fun <T : KBvSort> mkBvArithShiftRightExprNoSimplify(
        arg: KExpr<T>, shift: KExpr<T>
    ): KBvArithShiftRightExpr<T> =
        bvArithShiftRightExprCache.createIfContextActive {
            ensureContextMatch(arg, shift)
            KBvArithShiftRightExpr(this, arg, shift)
        }.cast()

    private val bvRotateLeftExprCache = mkAstInterner<KBvRotateLeftExpr<out KBvSort>>()

    /**
     * Create BitVec rotate left (`rotateleft`) expression.
     * The result expression is rotated left [rotation] times.
     *
     * @see [mkBvRotateLeftIndexedExpr] for the rotation by a known number of bits.
     * */
    open fun <T : KBvSort> mkBvRotateLeftExpr(arg: KExpr<T>, rotation: KExpr<T>): KExpr<T> =
        mkSimplified(arg, rotation, KContext::simplifyBvRotateLeftExpr, ::mkBvRotateLeftExprNoSimplify)

    /**
     * Create BitVec rotate left (`rotateleft`) expression.
     * The result expression is rotated left [rotation] times.
     *
     * @see [mkBvRotateLeftIndexedExpr] for the rotation by a known number of bits.
     * */
    open fun <T : KBvSort> mkBvRotateLeftExprNoSimplify(arg: KExpr<T>, rotation: KExpr<T>): KBvRotateLeftExpr<T> =
        bvRotateLeftExprCache.createIfContextActive {
            ensureContextMatch(arg, rotation)
            KBvRotateLeftExpr(this, arg, rotation)
        }.cast()

    private val bvRotateLeftIndexedExprCache = mkAstInterner<KBvRotateLeftIndexedExpr<out KBvSort>>()

    /**
     * Create BitVec rotate left (`rotateleft`) expression.
     * The result expression is rotated left [rotation] times.
     *
     * @see [mkBvRotateLeftExpr] for the rotation by a symbolic (any BitVec expression) number of bits.
     * */
    open fun <T : KBvSort> mkBvRotateLeftIndexedExpr(rotation: Int, value: KExpr<T>): KExpr<T> =
        mkSimplified(rotation, value, KContext::simplifyBvRotateLeftIndexedExpr, ::mkBvRotateLeftIndexedExprNoSimplify)

    /**
     * Create BitVec rotate left (`rotateleft`) expression.
     * The result expression is rotated left [rotation] times.
     *
     * @see [mkBvRotateLeftExpr] for the rotation by a symbolic (any BitVec expression) number of bits.
     * */
    open fun <T : KBvSort> mkBvRotateLeftIndexedExprNoSimplify(
        rotation: Int, value: KExpr<T>
    ): KBvRotateLeftIndexedExpr<T> =
        bvRotateLeftIndexedExprCache.createIfContextActive {
            ensureContextMatch(value)
            KBvRotateLeftIndexedExpr(this, rotation, value)
        }.cast()

    private val bvRotateRightIndexedExprCache = mkAstInterner<KBvRotateRightIndexedExpr<out KBvSort>>()

    /**
     * Create BitVec rotate right (`rotateright`) expression.
     * The result expression is rotated right [rotation] times.
     *
     * @see [mkBvRotateRightExpr] for the rotation by a symbolic (any BitVec expression) number of bits.
     * */
    open fun <T : KBvSort> mkBvRotateRightIndexedExpr(rotation: Int, value: KExpr<T>): KExpr<T> =
        mkSimplified(
            rotation,
            value,
            KContext::simplifyBvRotateRightIndexedExpr,
            ::mkBvRotateRightIndexedExprNoSimplify
        )

    /**
     * Create BitVec rotate right (`rotateright`) expression.
     * The result expression is rotated right [rotation] times.
     *
     * @see [mkBvRotateRightExpr] for the rotation by a symbolic (any BitVec expression) number of bits.
     * */
    open fun <T : KBvSort> mkBvRotateRightIndexedExprNoSimplify(
        rotation: Int,
        value: KExpr<T>
    ): KBvRotateRightIndexedExpr<T> =
        bvRotateRightIndexedExprCache.createIfContextActive {
            ensureContextMatch(value)
            KBvRotateRightIndexedExpr(this, rotation, value)
        }.cast()

    private val bvRotateRightExprCache = mkAstInterner<KBvRotateRightExpr<out KBvSort>>()

    /**
     * Create BitVec rotate right (`rotateright`) expression.
     * The result expression is rotated right [rotation] times.
     *
     * @see [mkBvRotateRightIndexedExpr] for the rotation by a known number of bits.
     * */
    open fun <T : KBvSort> mkBvRotateRightExpr(arg: KExpr<T>, rotation: KExpr<T>): KExpr<T> =
        mkSimplified(arg, rotation, KContext::simplifyBvRotateRightExpr, ::mkBvRotateRightExprNoSimplify)

    /**
     * Create BitVec rotate right (`rotateright`) expression.
     * The result expression is rotated right [rotation] times.
     *
     * @see [mkBvRotateRightIndexedExpr] for the rotation by a known number of bits.
     * */
    open fun <T : KBvSort> mkBvRotateRightExprNoSimplify(arg: KExpr<T>, rotation: KExpr<T>): KBvRotateRightExpr<T> =
        bvRotateRightExprCache.createIfContextActive {
            ensureContextMatch(arg, rotation)
            KBvRotateRightExpr(this, arg, rotation)
        }.cast()

    private val bv2IntExprCache = mkAstInterner<KBv2IntExpr>()

    /**
     * Convert BitVec expressions [value] to the arithmetic Int expressions.
     * When [isSigned] is set, the sign (highest, leftmost) bit is treated as the [value] sign.
     * */
    open fun <T : KBvSort> mkBv2IntExpr(value: KExpr<T>, isSigned: Boolean): KExpr<KIntSort> =
        mkSimplified(value, isSigned, KContext::simplifyBv2IntExpr, ::mkBv2IntExprNoSimplify)

    /**
     * Convert BitVec expressions [value] to the arithmetic Int expressions.
     * When [isSigned] is set, the sign (highest, leftmost) bit is treated as the [value] sign.
     * */
    open fun <T : KBvSort> mkBv2IntExprNoSimplify(value: KExpr<T>, isSigned: Boolean): KBv2IntExpr =
        bv2IntExprCache.createIfContextActive {
            ensureContextMatch(value)
            KBv2IntExpr(this, value.cast(), isSigned)
        }

    private val bvAddNoOverflowExprCache = mkAstInterner<KBvAddNoOverflowExpr<out KBvSort>>()

    /**
     * Create BitVec add-no-overflow check expression.
     * Determines that BitVec arithmetic addition does not overflow.
     * An overflow occurs when the addition result value is greater than max supported value.
     *
     * @see mkBvAddNoUnderflowExpr for the underflow check.
     * */
    open fun <T : KBvSort> mkBvAddNoOverflowExpr(arg0: KExpr<T>, arg1: KExpr<T>, isSigned: Boolean): KExpr<KBoolSort> =
        mkSimplified(arg0, arg1, isSigned, KContext::simplifyBvAddNoOverflowExpr, ::mkBvAddNoOverflowExprNoSimplify)

    /**
     * Create BitVec add-no-overflow check expression.
     * Determines that BitVec arithmetic addition does not overflow.
     * An overflow occurs when the addition result value is greater than max supported value.
     *
     * @see mkBvAddNoUnderflowExpr for the underflow check.
     * */
    open fun <T : KBvSort> mkBvAddNoOverflowExprNoSimplify(
        arg0: KExpr<T>,
        arg1: KExpr<T>,
        isSigned: Boolean
    ): KBvAddNoOverflowExpr<T> = bvAddNoOverflowExprCache.createIfContextActive {
        ensureContextMatch(arg0, arg1)
        KBvAddNoOverflowExpr(this, arg0, arg1, isSigned)
    }.cast()

    private val bvAddNoUnderflowExprCache = mkAstInterner<KBvAddNoUnderflowExpr<out KBvSort>>()

    /**
     * Create BitVec add-no-underflow check expression.
     * Determines that BitVec arithmetic addition does not underflow.
     * An underflow occurs when the addition result value is lower than min supported value.
     *
     * @see mkBvAddNoOverflowExpr for the overflow check.
     * */
    open fun <T : KBvSort> mkBvAddNoUnderflowExpr(arg0: KExpr<T>, arg1: KExpr<T>): KExpr<KBoolSort> =
        mkSimplified(arg0, arg1, KContext::simplifyBvAddNoUnderflowExpr, ::mkBvAddNoUnderflowExprNoSimplify)

    /**
     * Create BitVec add-no-underflow check expression.
     * Determines that BitVec arithmetic addition does not underflow.
     * An underflow occurs when the addition result value is lower than min supported value.
     *
     * @see mkBvAddNoOverflowExpr for the overflow check.
     * */
    open fun <T : KBvSort> mkBvAddNoUnderflowExprNoSimplify(arg0: KExpr<T>, arg1: KExpr<T>): KBvAddNoUnderflowExpr<T> =
        bvAddNoUnderflowExprCache.createIfContextActive {
            ensureContextMatch(arg0, arg1)
            KBvAddNoUnderflowExpr(this, arg0, arg1)
        }.cast()

    private val bvSubNoOverflowExprCache = mkAstInterner<KBvSubNoOverflowExpr<out KBvSort>>()

    /**
     * Create BitVec sub-no-overflow check expression.
     * Determines that BitVec arithmetic subtraction does not overflow.
     * An overflow occurs when the subtraction result value is greater than max supported value.
     *
     * @see mkBvSubNoUnderflowExpr for the underflow check.
     * */
    open fun <T : KBvSort> mkBvSubNoOverflowExpr(arg0: KExpr<T>, arg1: KExpr<T>): KExpr<KBoolSort> =
        mkSimplified(arg0, arg1, KContext::simplifyBvSubNoOverflowExpr, ::mkBvSubNoOverflowExprNoSimplify)

    /**
     * Create BitVec sub-no-overflow check expression.
     * Determines that BitVec arithmetic subtraction does not overflow.
     * An overflow occurs when the subtraction result value is greater than max supported value.
     *
     * @see mkBvSubNoUnderflowExpr for the underflow check.
     * */
    open fun <T : KBvSort> mkBvSubNoOverflowExprNoSimplify(arg0: KExpr<T>, arg1: KExpr<T>): KBvSubNoOverflowExpr<T> =
        bvSubNoOverflowExprCache.createIfContextActive {
            ensureContextMatch(arg0, arg1)
            KBvSubNoOverflowExpr(this, arg0, arg1)
        }.cast()

    private val bvSubNoUnderflowExprCache = mkAstInterner<KBvSubNoUnderflowExpr<out KBvSort>>()

    /**
     * Create BitVec sub-no-underflow check expression.
     * Determines that BitVec arithmetic subtraction does not underflow.
     * An underflow occurs when the subtraction result value is lower than min supported value.
     *
     * @see mkBvSubNoOverflowExpr for the overflow check.
     * */
    open fun <T : KBvSort> mkBvSubNoUnderflowExpr(arg0: KExpr<T>, arg1: KExpr<T>, isSigned: Boolean): KExpr<KBoolSort> =
        mkSimplified(arg0, arg1, isSigned, KContext::simplifyBvSubNoUnderflowExpr, ::mkBvSubNoUnderflowExprNoSimplify)

    /**
     * Create BitVec sub-no-underflow check expression.
     * Determines that BitVec arithmetic subtraction does not underflow.
     * An underflow occurs when the subtraction result value is lower than min supported value.
     *
     * @see mkBvSubNoOverflowExpr for the overflow check.
     * */
    open fun <T : KBvSort> mkBvSubNoUnderflowExprNoSimplify(
        arg0: KExpr<T>,
        arg1: KExpr<T>,
        isSigned: Boolean
    ): KBvSubNoUnderflowExpr<T> = bvSubNoUnderflowExprCache.createIfContextActive {
        ensureContextMatch(arg0, arg1)
        KBvSubNoUnderflowExpr(this, arg0, arg1, isSigned)
    }.cast()

    private val bvDivNoOverflowExprCache = mkAstInterner<KBvDivNoOverflowExpr<out KBvSort>>()

    /**
     * Create BitVec div-no-overflow check expression.
     * Determines that BitVec arithmetic signed division does not overflow.
     * */
    open fun <T : KBvSort> mkBvDivNoOverflowExpr(arg0: KExpr<T>, arg1: KExpr<T>): KExpr<KBoolSort> =
        mkSimplified(arg0, arg1, KContext::simplifyBvDivNoOverflowExpr, ::mkBvDivNoOverflowExprNoSimplify)

    /**
     * Create BitVec div-no-overflow check expression.
     * Determines that BitVec arithmetic signed division does not overflow.
     * */
    open fun <T : KBvSort> mkBvDivNoOverflowExprNoSimplify(arg0: KExpr<T>, arg1: KExpr<T>): KBvDivNoOverflowExpr<T> =
        bvDivNoOverflowExprCache.createIfContextActive {
            ensureContextMatch(arg0, arg1)
            KBvDivNoOverflowExpr(this, arg0, arg1)
        }.cast()

    private val bvNegNoOverflowExprCache = mkAstInterner<KBvNegNoOverflowExpr<out KBvSort>>()

    /**
     * Create BitVec neg-no-overflow check expression.
     * Determines that BitVec arithmetic negation does not overflow.
     * */
    open fun <T : KBvSort> mkBvNegationNoOverflowExpr(value: KExpr<T>): KExpr<KBoolSort> =
        mkSimplified(value, KContext::simplifyBvNegationNoOverflowExpr, ::mkBvNegationNoOverflowExprNoSimplify)

    /**
     * Create BitVec neg-no-overflow check expression.
     * Determines that BitVec arithmetic negation does not overflow.
     * */
    open fun <T : KBvSort> mkBvNegationNoOverflowExprNoSimplify(value: KExpr<T>): KBvNegNoOverflowExpr<T> =
        bvNegNoOverflowExprCache.createIfContextActive {
            ensureContextMatch(value)
            KBvNegNoOverflowExpr(this, value)
        }.cast()

    private val bvMulNoOverflowExprCache = mkAstInterner<KBvMulNoOverflowExpr<out KBvSort>>()

    /**
     * Create BitVec mul-no-overflow check expression.
     * Determines that BitVec arithmetic multiplication does not overflow.
     * An overflow occurs when the multiplication result value is greater than max supported value.
     *
     * @see mkBvMulNoUnderflowExpr for the underflow check.
     * */
    open fun <T : KBvSort> mkBvMulNoOverflowExpr(arg0: KExpr<T>, arg1: KExpr<T>, isSigned: Boolean): KExpr<KBoolSort> =
        mkSimplified(arg0, arg1, isSigned, KContext::simplifyBvMulNoOverflowExpr, ::mkBvMulNoOverflowExprNoSimplify)

    /**
     * Create BitVec mul-no-overflow check expression.
     * Determines that BitVec arithmetic multiplication does not overflow.
     * An overflow occurs when the multiplication result value is greater than max supported value.
     *
     * @see mkBvMulNoUnderflowExpr for the underflow check.
     * */
    open fun <T : KBvSort> mkBvMulNoOverflowExprNoSimplify(
        arg0: KExpr<T>,
        arg1: KExpr<T>,
        isSigned: Boolean
    ): KBvMulNoOverflowExpr<T> = bvMulNoOverflowExprCache.createIfContextActive {
        ensureContextMatch(arg0, arg1)
        KBvMulNoOverflowExpr(this, arg0, arg1, isSigned)
    }.cast()

    private val bvMulNoUnderflowExprCache = mkAstInterner<KBvMulNoUnderflowExpr<out KBvSort>>()

    /**
     * Create BitVec mul-no-underflow check expression.
     * Determines that BitVec arithmetic multiplication does not underflow.
     * An underflow occurs when the multiplication result value is lower than min supported value.
     *
     * @see mkBvMulNoOverflowExpr for the overflow check.
     * */
    open fun <T : KBvSort> mkBvMulNoUnderflowExpr(arg0: KExpr<T>, arg1: KExpr<T>): KExpr<KBoolSort> =
        mkSimplified(arg0, arg1, KContext::simplifyBvMulNoUnderflowExpr, ::mkBvMulNoUnderflowExprNoSimplify)

    /**
     * Create BitVec mul-no-underflow check expression.
     * Determines that BitVec arithmetic multiplication does not underflow.
     * An underflow occurs when the multiplication result value is lower than min supported value.
     *
     * @see mkBvMulNoOverflowExpr for the overflow check.
     * */
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
     * Create FP16 from the [value].
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

    /**
     * Create FP16 NaN value.
     * */
    fun mkFp16NaN(): KFp16Value = mkFp16WithoutNaNCheck(Float.NaN)
    private fun mkFp16WithoutNaNCheck(value: Float): KFp16Value =
        fp16Cache.createIfContextActive { KFp16Value(this, value) }

    /**
     * Create FP32 from the [value].
     * */
    fun mkFp32(value: Float): KFp32Value = if (value.isNaN()) mkFp32NaN() else mkFp32WithoutNaNCheck(value)

    /**
     * Create FP32 NaN value.
     * */
    fun mkFp32NaN(): KFp32Value = mkFp32WithoutNaNCheck(Float.NaN)
    private fun mkFp32WithoutNaNCheck(value: Float): KFp32Value =
        fp32Cache.createIfContextActive { KFp32Value(this, value) }

    /**
     * Create FP64 from the [value].
     * */
    fun mkFp64(value: Double): KFp64Value = if (value.isNaN()) mkFp64NaN() else mkFp64WithoutNaNCheck(value)

    /**
     * Create FP64 NaN value.
     * */
    fun mkFp64NaN(): KFp64Value = mkFp64WithoutNaNCheck(Double.NaN)
    private fun mkFp64WithoutNaNCheck(value: Double): KFp64Value =
        fp64Cache.createIfContextActive { KFp64Value(this, value) }

    /**
     * Create FP128 from the IEEE binary representation.
     * Note: [biasedExponent] here is biased, i.e. unsigned.
     * */
    fun mkFp128Biased(significand: KBitVecValue<*>, biasedExponent: KBitVecValue<*>, signBit: Boolean): KFp128Value =
        if (KFp128Value(this, significand, biasedExponent, signBit).isNaN()) {
            mkFp128NaN()
        } else {
            mkFp128BiasedWithoutNaNCheck(significand, biasedExponent, signBit)
        }

    /**
     * Create FP128 NaN value.
     * */
    fun mkFp128NaN(): KFp128Value = mkFpNaN(mkFp128Sort()).cast()
    private fun mkFp128BiasedWithoutNaNCheck(
        significand: KBitVecValue<*>,
        biasedExponent: KBitVecValue<*>,
        signBit: Boolean
    ): KFp128Value = fp128Cache.createIfContextActive {
        ensureContextMatch(significand, biasedExponent)
        KFp128Value(this, significand, biasedExponent, signBit)
    }

    /**
     * Create FP128 from the IEEE binary representation.
     * Note: [unbiasedExponent] here is unbiased, i.e. signed.
     * */
    fun mkFp128(significand: KBitVecValue<*>, unbiasedExponent: KBitVecValue<*>, signBit: Boolean): KFp128Value =
        mkFp128Biased(
            significand = significand,
            biasedExponent = biasFp128Exponent(unbiasedExponent),
            signBit = signBit
        )

    /**
     * Create FP128 from the IEEE binary representation.
     * Note: [unbiasedExponent] here is unbiased, i.e. signed.
     * */
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
     * Create FP with a custom size from the IEEE binary representation.
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
     * Create FP with a custom size from the IEEE binary representation.
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

    /**
     * Create FP with a custom size from the IEEE binary representation.
     * Important: [unbiasedExponent] here is an **unbiased** value.
     */
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
     * Create Fp zero value with [signBit] sign.
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

    /**
     * Create Fp Inf value with [signBit] sign.
     * */
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

    /**
     * Create Fp NaN value.
     * */
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

    /**
     * Create Fp rounding mode.
     *
     * @see [KFpRoundingMode]
     * */
    fun mkFpRoundingModeExpr(
        value: KFpRoundingMode
    ): KFpRoundingModeExpr = roundingModeCache.createIfContextActive {
        KFpRoundingModeExpr(this, value)
    }

    private val fpAbsExprCache = mkAstInterner<KFpAbsExpr<out KFpSort>>()

    /**
     * Create Fp absolute value (`fp.abs`) expression.
     * */
    open fun <T : KFpSort> mkFpAbsExpr(value: KExpr<T>): KExpr<T> =
        mkSimplified(value, KContext::simplifyFpAbsExpr, ::mkFpAbsExprNoSimplify)

    /**
     * Create Fp absolute value (`fp.abs`) expression.
     * */
    open fun <T : KFpSort> mkFpAbsExprNoSimplify(
        value: KExpr<T>
    ): KFpAbsExpr<T> = fpAbsExprCache.createIfContextActive {
        ensureContextMatch(value)
        KFpAbsExpr(this, value)
    }.cast()

    private val fpNegationExprCache = mkAstInterner<KFpNegationExpr<out KFpSort>>()

    /**
     * Create Fp negation (`fp.neg`) expression.
     * */
    open fun <T : KFpSort> mkFpNegationExpr(value: KExpr<T>): KExpr<T> =
        mkSimplified(value, KContext::simplifyFpNegationExpr, ::mkFpNegationExprNoSimplify)

    /**
     * Create Fp negation (`fp.neg`) expression.
     * */
    open fun <T : KFpSort> mkFpNegationExprNoSimplify(
        value: KExpr<T>
    ): KFpNegationExpr<T> = fpNegationExprCache.createIfContextActive {
        ensureContextMatch(value)
        KFpNegationExpr(this, value)
    }.cast()

    private val fpAddExprCache = mkAstInterner<KFpAddExpr<out KFpSort>>()

    /**
     * Create Fp addition (`fp.add`) expression.
     * The result is rounded according to the provided [roundingMode].
     * */
    open fun <T : KFpSort> mkFpAddExpr(
        roundingMode: KExpr<KFpRoundingModeSort>, arg0: KExpr<T>, arg1: KExpr<T>
    ): KExpr<T> =
        mkSimplified(roundingMode, arg0, arg1, KContext::simplifyFpAddExpr, ::mkFpAddExprNoSimplify)

    /**
     * Create Fp addition (`fp.add`) expression.
     * The result is rounded according to the provided [roundingMode].
     * */
    open fun <T : KFpSort> mkFpAddExprNoSimplify(
        roundingMode: KExpr<KFpRoundingModeSort>,
        arg0: KExpr<T>,
        arg1: KExpr<T>
    ): KFpAddExpr<T> = fpAddExprCache.createIfContextActive {
        ensureContextMatch(roundingMode, arg0, arg1)
        KFpAddExpr(this, roundingMode, arg0, arg1)
    }.cast()

    private val fpSubExprCache = mkAstInterner<KFpSubExpr<out KFpSort>>()

    /**
     * Create Fp subtraction (`fp.sub`) expression.
     * The result is rounded according to the provided [roundingMode].
     * */
    open fun <T : KFpSort> mkFpSubExpr(
        roundingMode: KExpr<KFpRoundingModeSort>, arg0: KExpr<T>, arg1: KExpr<T>
    ): KExpr<T> =
        mkSimplified(roundingMode, arg0, arg1, KContext::simplifyFpSubExpr, ::mkFpSubExprNoSimplify)

    /**
     * Create Fp subtraction (`fp.sub`) expression.
     * The result is rounded according to the provided [roundingMode].
     * */
    open fun <T : KFpSort> mkFpSubExprNoSimplify(
        roundingMode: KExpr<KFpRoundingModeSort>,
        arg0: KExpr<T>,
        arg1: KExpr<T>
    ): KFpSubExpr<T> = fpSubExprCache.createIfContextActive {
        ensureContextMatch(roundingMode, arg0, arg1)
        KFpSubExpr(this, roundingMode, arg0, arg1)
    }.cast()

    private val fpMulExprCache = mkAstInterner<KFpMulExpr<out KFpSort>>()

    /**
     * Create Fp multiplication (`fp.mul`) expression.
     * The result is rounded according to the provided [roundingMode].
     * */
    open fun <T : KFpSort> mkFpMulExpr(
        roundingMode: KExpr<KFpRoundingModeSort>, arg0: KExpr<T>, arg1: KExpr<T>
    ): KExpr<T> =
        mkSimplified(roundingMode, arg0, arg1, KContext::simplifyFpMulExpr, ::mkFpMulExprNoSimplify)

    /**
     * Create Fp multiplication (`fp.mul`) expression.
     * The result is rounded according to the provided [roundingMode].
     * */
    open fun <T : KFpSort> mkFpMulExprNoSimplify(
        roundingMode: KExpr<KFpRoundingModeSort>,
        arg0: KExpr<T>,
        arg1: KExpr<T>
    ): KFpMulExpr<T> = fpMulExprCache.createIfContextActive {
        ensureContextMatch(roundingMode, arg0, arg1)
        KFpMulExpr(this, roundingMode, arg0, arg1)
    }.cast()

    private val fpDivExprCache = mkAstInterner<KFpDivExpr<out KFpSort>>()

    /**
     * Create Fp division (`fp.div`) expression.
     * The result is rounded according to the provided [roundingMode].
     * */
    open fun <T : KFpSort> mkFpDivExpr(
        roundingMode: KExpr<KFpRoundingModeSort>, arg0: KExpr<T>, arg1: KExpr<T>
    ): KExpr<T> =
        mkSimplified(roundingMode, arg0, arg1, KContext::simplifyFpDivExpr, ::mkFpDivExprNoSimplify)

    /**
     * Create Fp division (`fp.div`) expression.
     * The result is rounded according to the provided [roundingMode].
     * */
    open fun <T : KFpSort> mkFpDivExprNoSimplify(
        roundingMode: KExpr<KFpRoundingModeSort>,
        arg0: KExpr<T>,
        arg1: KExpr<T>
    ): KFpDivExpr<T> = fpDivExprCache.createIfContextActive {
        ensureContextMatch(roundingMode, arg0, arg1)
        KFpDivExpr(this, roundingMode, arg0, arg1)
    }.cast()

    private val fpFusedMulAddExprCache = mkAstInterner<KFpFusedMulAddExpr<out KFpSort>>()

    /**
     * Create Fp fused multiplication and addition (`fp.fma`) expression.
     * Computes ([arg0] * [arg1]) + [arg2].
     * The result is rounded according to the provided [roundingMode].
     * */
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


    /**
     * Create Fp fused multiplication and addition (`fp.fma`) expression.
     * Computes ([arg0] * [arg1]) + [arg2].
     * The result is rounded according to the provided [roundingMode].
     * */
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

    /**
     * Create Fp square root (`fp.sqrt`) expression.
     * The result is rounded according to the provided [roundingMode].
     * */
    open fun <T : KFpSort> mkFpSqrtExpr(roundingMode: KExpr<KFpRoundingModeSort>, value: KExpr<T>): KExpr<T> =
        mkSimplified(roundingMode, value, KContext::simplifyFpSqrtExpr, ::mkFpSqrtExprNoSimplify)

    /**
     * Create Fp square root (`fp.sqrt`) expression.
     * The result is rounded according to the provided [roundingMode].
     * */
    open fun <T : KFpSort> mkFpSqrtExprNoSimplify(
        roundingMode: KExpr<KFpRoundingModeSort>,
        value: KExpr<T>
    ): KFpSqrtExpr<T> = fpSqrtExprCache.createIfContextActive {
        ensureContextMatch(roundingMode, value)
        KFpSqrtExpr(this, roundingMode, value)
    }.cast()

    private val fpRemExprCache = mkAstInterner<KFpRemExpr<out KFpSort>>()

    /**
     * Create Fp IEEE remainder (`fp.IEEERem`) expression.
     * */
    open fun <T : KFpSort> mkFpRemExpr(arg0: KExpr<T>, arg1: KExpr<T>): KExpr<T> =
        mkSimplified(arg0, arg1, KContext::simplifyFpRemExpr, ::mkFpRemExprNoSimplify)

    /**
     * Create Fp IEEE remainder (`fp.IEEERem`) expression.
     * */
    open fun <T : KFpSort> mkFpRemExprNoSimplify(arg0: KExpr<T>, arg1: KExpr<T>): KFpRemExpr<T> =
        fpRemExprCache.createIfContextActive {
            ensureContextMatch(arg0, arg1)
            KFpRemExpr(this, arg0, arg1)
        }.cast()

    private val fpRoundToIntegralExprCache = mkAstInterner<KFpRoundToIntegralExpr<out KFpSort>>()

    /**
     * Create Fp round to integral (`fp.roundToIntegral`) expression.
     * The rounding is performed according to the provided [roundingMode].
     * */
    open fun <T : KFpSort> mkFpRoundToIntegralExpr(
        roundingMode: KExpr<KFpRoundingModeSort>, value: KExpr<T>
    ): KExpr<T> =
        mkSimplified(roundingMode, value, KContext::simplifyFpRoundToIntegralExpr, ::mkFpRoundToIntegralExprNoSimplify)

    /**
     * Create Fp round to integral (`fp.roundToIntegral`) expression.
     * The rounding is performed according to the provided [roundingMode].
     * */
    open fun <T : KFpSort> mkFpRoundToIntegralExprNoSimplify(
        roundingMode: KExpr<KFpRoundingModeSort>,
        value: KExpr<T>
    ): KFpRoundToIntegralExpr<T> = fpRoundToIntegralExprCache.createIfContextActive {
        ensureContextMatch(roundingMode, value)
        KFpRoundToIntegralExpr(this, roundingMode, value)
    }.cast()

    private val fpMinExprCache = mkAstInterner<KFpMinExpr<out KFpSort>>()

    /**
     * Create Fp minimum (`fp.min`) expression.
     * */
    open fun <T : KFpSort> mkFpMinExpr(arg0: KExpr<T>, arg1: KExpr<T>): KExpr<T> =
        mkSimplified(arg0, arg1, KContext::simplifyFpMinExpr, ::mkFpMinExprNoSimplify)

    /**
     * Create Fp minimum (`fp.min`) expression.
     * */
    open fun <T : KFpSort> mkFpMinExprNoSimplify(arg0: KExpr<T>, arg1: KExpr<T>): KFpMinExpr<T> =
        fpMinExprCache.createIfContextActive {
            ensureContextMatch(arg0, arg1)
            KFpMinExpr(this, arg0, arg1)
        }.cast()

    private val fpMaxExprCache = mkAstInterner<KFpMaxExpr<out KFpSort>>()

    /**
     * Create Fp maximum (`fp.max`) expression.
     * */
    open fun <T : KFpSort> mkFpMaxExpr(arg0: KExpr<T>, arg1: KExpr<T>): KExpr<T> =
        mkSimplified(arg0, arg1, KContext::simplifyFpMaxExpr, ::mkFpMaxExprNoSimplify)

    /**
     * Create Fp maximum (`fp.max`) expression.
     * */
    open fun <T : KFpSort> mkFpMaxExprNoSimplify(arg0: KExpr<T>, arg1: KExpr<T>): KFpMaxExpr<T> =
        fpMaxExprCache.createIfContextActive {
            ensureContextMatch(arg0, arg1)
            KFpMaxExpr(this, arg0, arg1)
        }.cast()

    private val fpLessOrEqualExprCache = mkAstInterner<KFpLessOrEqualExpr<out KFpSort>>()

    /**
     * Create Fp less-or-equal comparison (`fp.leq`) expression.
     * */
    open fun <T : KFpSort> mkFpLessOrEqualExpr(arg0: KExpr<T>, arg1: KExpr<T>): KExpr<KBoolSort> =
        mkSimplified(arg0, arg1, KContext::simplifyFpLessOrEqualExpr, ::mkFpLessOrEqualExprNoSimplify)

    /**
     * Create Fp less-or-equal comparison (`fp.leq`) expression.
     * */
    open fun <T : KFpSort> mkFpLessOrEqualExprNoSimplify(arg0: KExpr<T>, arg1: KExpr<T>): KFpLessOrEqualExpr<T> =
        fpLessOrEqualExprCache.createIfContextActive {
            ensureContextMatch(arg0, arg1)
            KFpLessOrEqualExpr(this, arg0, arg1)
        }.cast()

    private val fpLessExprCache = mkAstInterner<KFpLessExpr<out KFpSort>>()

    /**
     * Create Fp less comparison (`fp.lt`) expression.
     * */
    open fun <T : KFpSort> mkFpLessExpr(arg0: KExpr<T>, arg1: KExpr<T>): KExpr<KBoolSort> =
        mkSimplified(arg0, arg1, KContext::simplifyFpLessExpr, ::mkFpLessExprNoSimplify)

    /**
     * Create Fp less comparison (`fp.lt`) expression.
     * */
    open fun <T : KFpSort> mkFpLessExprNoSimplify(arg0: KExpr<T>, arg1: KExpr<T>): KFpLessExpr<T> =
        fpLessExprCache.createIfContextActive {
            ensureContextMatch(arg0, arg1)
            KFpLessExpr(this, arg0, arg1)
        }.cast()

    private val fpGreaterOrEqualExprCache = mkAstInterner<KFpGreaterOrEqualExpr<out KFpSort>>()

    /**
     * Create Fp greater-or-equal comparison (`fp.geq`) expression.
     * */
    open fun <T : KFpSort> mkFpGreaterOrEqualExpr(arg0: KExpr<T>, arg1: KExpr<T>): KExpr<KBoolSort> =
        mkSimplified(arg0, arg1, KContext::simplifyFpGreaterOrEqualExpr, ::mkFpGreaterOrEqualExprNoSimplify)

    /**
     * Create Fp greater-or-equal comparison (`fp.geq`) expression.
     * */
    open fun <T : KFpSort> mkFpGreaterOrEqualExprNoSimplify(arg0: KExpr<T>, arg1: KExpr<T>): KFpGreaterOrEqualExpr<T> =
        fpGreaterOrEqualExprCache.createIfContextActive {
            ensureContextMatch(arg0, arg1)
            KFpGreaterOrEqualExpr(this, arg0, arg1)
        }.cast()

    private val fpGreaterExprCache = mkAstInterner<KFpGreaterExpr<out KFpSort>>()

    /**
     * Create Fp greater comparison (`fp.gt`) expression.
     * */
    open fun <T : KFpSort> mkFpGreaterExpr(arg0: KExpr<T>, arg1: KExpr<T>): KExpr<KBoolSort> =
        mkSimplified(arg0, arg1, KContext::simplifyFpGreaterExpr, ::mkFpGreaterExprNoSimplify)

    /**
     * Create Fp greater comparison (`fp.gt`) expression.
     * */
    open fun <T : KFpSort> mkFpGreaterExprNoSimplify(arg0: KExpr<T>, arg1: KExpr<T>): KFpGreaterExpr<T> =
        fpGreaterExprCache.createIfContextActive {
            ensureContextMatch(arg0, arg1)
            KFpGreaterExpr(this, arg0, arg1)
        }.cast()

    private val fpEqualExprCache = mkAstInterner<KFpEqualExpr<out KFpSort>>()

    /**
     * Create Fp equality comparison (`fp.eq`) expression.
     * Checks arguments equality using IEEE 754-2008 rules.
     *
     * @see [mkEq] for smt equality.
     * */
    open fun <T : KFpSort> mkFpEqualExpr(arg0: KExpr<T>, arg1: KExpr<T>): KExpr<KBoolSort> =
        mkSimplified(arg0, arg1, KContext::simplifyFpEqualExpr, ::mkFpEqualExprNoSimplify)

    /**
     * Create Fp equality comparison (`fp.eq`) expression.
     * Checks arguments equality using IEEE 754-2008 rules.
     *
     * @see [mkEq] for smt equality.
     * */
    open fun <T : KFpSort> mkFpEqualExprNoSimplify(arg0: KExpr<T>, arg1: KExpr<T>): KFpEqualExpr<T> =
        fpEqualExprCache.createIfContextActive {
            ensureContextMatch(arg0, arg1)
            KFpEqualExpr(this, arg0, arg1)
        }.cast()

    private val fpIsNormalExprCache = mkAstInterner<KFpIsNormalExpr<out KFpSort>>()

    /**
     * Create Fp is-normal check (`fp.isNormal`) expression.
     * */
    open fun <T : KFpSort> mkFpIsNormalExpr(value: KExpr<T>): KExpr<KBoolSort> =
        mkSimplified(value, KContext::simplifyFpIsNormalExpr, ::mkFpIsNormalExprNoSimplify)

    /**
     * Create Fp is-normal check (`fp.isNormal`) expression.
     * */
    open fun <T : KFpSort> mkFpIsNormalExprNoSimplify(value: KExpr<T>): KFpIsNormalExpr<T> =
        fpIsNormalExprCache.createIfContextActive {
            ensureContextMatch(value)
            KFpIsNormalExpr(this, value)
        }.cast()

    private val fpIsSubnormalExprCache = mkAstInterner<KFpIsSubnormalExpr<out KFpSort>>()

    /**
     * Create Fp is-subnormal check (`fp.isSubnormal`) expression.
     * */
    open fun <T : KFpSort> mkFpIsSubnormalExpr(value: KExpr<T>): KExpr<KBoolSort> =
        mkSimplified(value, KContext::simplifyFpIsSubnormalExpr, ::mkFpIsSubnormalExprNoSimplify)

    /**
     * Create Fp is-subnormal check (`fp.isSubnormal`) expression.
     * */
    open fun <T : KFpSort> mkFpIsSubnormalExprNoSimplify(value: KExpr<T>): KFpIsSubnormalExpr<T> =
        fpIsSubnormalExprCache.createIfContextActive {
            ensureContextMatch(value)
            KFpIsSubnormalExpr(this, value)
        }.cast()

    private val fpIsZeroExprCache = mkAstInterner<KFpIsZeroExpr<out KFpSort>>()

    /**
     * Create Fp is-zero check (`fp.isZero`) expression.
     * */
    open fun <T : KFpSort> mkFpIsZeroExpr(value: KExpr<T>): KExpr<KBoolSort> =
        mkSimplified(value, KContext::simplifyFpIsZeroExpr, ::mkFpIsZeroExprNoSimplify)

    /**
     * Create Fp is-zero check (`fp.isZero`) expression.
     * */
    open fun <T : KFpSort> mkFpIsZeroExprNoSimplify(value: KExpr<T>): KFpIsZeroExpr<T> =
        fpIsZeroExprCache.createIfContextActive {
            ensureContextMatch(value)
            KFpIsZeroExpr(this, value)
        }.cast()

    private val fpIsInfiniteExprCache = mkAstInterner<KFpIsInfiniteExpr<out KFpSort>>()

    /**
     * Create Fp is-inf check (`fp.isInfinite`) expression.
     * */
    open fun <T : KFpSort> mkFpIsInfiniteExpr(value: KExpr<T>): KExpr<KBoolSort> =
        mkSimplified(value, KContext::simplifyFpIsInfiniteExpr, ::mkFpIsInfiniteExprNoSimplify)

    /**
     * Create Fp is-inf check (`fp.isInfinite`) expression.
     * */
    open fun <T : KFpSort> mkFpIsInfiniteExprNoSimplify(value: KExpr<T>): KFpIsInfiniteExpr<T> =
        fpIsInfiniteExprCache.createIfContextActive {
            ensureContextMatch(value)
            KFpIsInfiniteExpr(this, value)
        }.cast()

    private val fpIsNaNExprCache = mkAstInterner<KFpIsNaNExpr<out KFpSort>>()

    /**
     * Create Fp is-nan check (`fp.isNaN`) expression.
     * */
    open fun <T : KFpSort> mkFpIsNaNExpr(value: KExpr<T>): KExpr<KBoolSort> =
        mkSimplified(value, KContext::simplifyFpIsNaNExpr, ::mkFpIsNaNExprNoSimplify)

    /**
     * Create Fp is-nan check (`fp.isNaN`) expression.
     * */
    open fun <T : KFpSort> mkFpIsNaNExprNoSimplify(value: KExpr<T>): KFpIsNaNExpr<T> =
        fpIsNaNExprCache.createIfContextActive {
            ensureContextMatch(value)
            KFpIsNaNExpr(this, value)
        }.cast()

    private val fpIsNegativeExprCache = mkAstInterner<KFpIsNegativeExpr<out KFpSort>>()

    /**
     * Create Fp is-negative check (`fp.isNegative`) expression.
     * */
    open fun <T : KFpSort> mkFpIsNegativeExpr(value: KExpr<T>): KExpr<KBoolSort> =
        mkSimplified(value, KContext::simplifyFpIsNegativeExpr, ::mkFpIsNegativeExprNoSimplify)

    /**
     * Create Fp is-negative check (`fp.isNegative`) expression.
     * */
    open fun <T : KFpSort> mkFpIsNegativeExprNoSimplify(value: KExpr<T>): KFpIsNegativeExpr<T> =
        fpIsNegativeExprCache.createIfContextActive {
            ensureContextMatch(value)
            KFpIsNegativeExpr(this, value)
        }.cast()

    private val fpIsPositiveExprCache = mkAstInterner<KFpIsPositiveExpr<out KFpSort>>()

    /**
     * Create Fp is-positive check (`fp.isPositive`) expression.
     * */
    open fun <T : KFpSort> mkFpIsPositiveExpr(value: KExpr<T>): KExpr<KBoolSort> =
        mkSimplified(value, KContext::simplifyFpIsPositiveExpr, ::mkFpIsPositiveExprNoSimplify)

    /**
     * Create Fp is-positive check (`fp.isPositive`) expression.
     * */
    open fun <T : KFpSort> mkFpIsPositiveExprNoSimplify(value: KExpr<T>): KFpIsPositiveExpr<T> =
        fpIsPositiveExprCache.createIfContextActive {
            ensureContextMatch(value)
            KFpIsPositiveExpr(this, value)
        }.cast()

    private val fpToBvExprCache = mkAstInterner<KFpToBvExpr<out KFpSort>>()

    /**
     * Create Fp to BitVec conversion (`fp.to_sbv`, `fp.to_ubv`) expression.
     * Provided Fp [value] is rounded to the nearest integral according to the [roundingMode].
     *
     * @see [mkFpToIEEEBvExpr] for binary conversion.
     * */
    open fun <T : KFpSort> mkFpToBvExpr(
        roundingMode: KExpr<KFpRoundingModeSort>,
        value: KExpr<T>,
        bvSize: Int,
        isSigned: Boolean
    ): KExpr<KBvSort> =
        mkSimplified(roundingMode, value, bvSize, isSigned, KContext::simplifyFpToBvExpr, ::mkFpToBvExprNoSimplify)

    /**
     * Create Fp to BitVec conversion (`fp.to_sbv`, `fp.to_ubv`) expression.
     * Provided Fp [value] is rounded to the nearest integral according to the [roundingMode].
     *
     * @see [mkFpToIEEEBvExpr] for binary conversion.
     * */
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

    /**
     * Create Fp to Real conversion (`fp.to_real`) expression.
     * */
    open fun <T : KFpSort> mkFpToRealExpr(value: KExpr<T>): KExpr<KRealSort> =
        mkSimplified(value, KContext::simplifyFpToRealExpr, ::mkFpToRealExprNoSimplify)

    /**
     * Create Fp to Real conversion (`fp.to_real`) expression.
     * */
    open fun <T : KFpSort> mkFpToRealExprNoSimplify(value: KExpr<T>): KFpToRealExpr<T> =
        fpToRealExprCache.createIfContextActive {
            ensureContextMatch(value)
            KFpToRealExpr(this, value)
        }.cast()

    private val fpToIEEEBvExprCache = mkAstInterner<KFpToIEEEBvExpr<out KFpSort>>()

    /**
     * Create Fp to IEEE BitVec conversion expression.
     * Converts Fp value to the corresponding IEEE 754-2008 binary format.
     *
     * Note: conversion is unspecified for NaN values.
     * */
    open fun <T : KFpSort> mkFpToIEEEBvExpr(value: KExpr<T>): KExpr<KBvSort> =
        mkSimplified(value, KContext::simplifyFpToIEEEBvExpr, ::mkFpToIEEEBvExprNoSimplify)

    /**
     * Create Fp to IEEE BitVec conversion expression.
     * Converts Fp value to the corresponding IEEE 754-2008 binary format.
     *
     * Note: conversion is unspecified for NaN values.
     * */
    open fun <T : KFpSort> mkFpToIEEEBvExprNoSimplify(value: KExpr<T>): KFpToIEEEBvExpr<T> =
        fpToIEEEBvExprCache.createIfContextActive {
            ensureContextMatch(value)
            KFpToIEEEBvExpr(this, value)
        }.cast()

    private val fpFromBvExprCache = mkAstInterner<KFpFromBvExpr<out KFpSort>>()

    /**
     * Create Fp from IEEE BitVec expressions.
     * */
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

    /**
     * Create Fp from IEEE BitVec expressions.
     * */
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

    /**
     * Create Fp to another Fp.
     * Rounding is performed according to the [roundingMode].
     * */
    open fun <T : KFpSort> mkFpToFpExpr(
        sort: T,
        roundingMode: KExpr<KFpRoundingModeSort>,
        value: KExpr<out KFpSort>
    ): KExpr<T> = mkSimplified(sort, roundingMode, value, KContext::simplifyFpToFpExpr, ::mkFpToFpExprNoSimplify)

    /**
     * Create Fp to another Fp.
     * Rounding is performed according to the [roundingMode].
     * */
    open fun <T : KFpSort> mkFpToFpExprNoSimplify(
        sort: T,
        roundingMode: KExpr<KFpRoundingModeSort>,
        value: KExpr<out KFpSort>
    ): KFpToFpExpr<T> = fpToFpExprCache.createIfContextActive {
        ensureContextMatch(sort, roundingMode, value)
        KFpToFpExpr(this, sort, roundingMode, value)
    }.cast()

    /**
     * Create Fp from Real expression.
     * Rounding is performed according to the [roundingMode].
     * */
    open fun <T : KFpSort> mkRealToFpExpr(
        sort: T,
        roundingMode: KExpr<KFpRoundingModeSort>,
        value: KExpr<KRealSort>
    ): KExpr<T> = mkSimplified(sort, roundingMode, value, KContext::simplifyRealToFpExpr, ::mkRealToFpExprNoSimplify)

    /**
     * Create Fp from Real expression.
     * Rounding is performed according to the [roundingMode].
     * */
    open fun <T : KFpSort> mkRealToFpExprNoSimplify(
        sort: T,
        roundingMode: KExpr<KFpRoundingModeSort>,
        value: KExpr<KRealSort>
    ): KRealToFpExpr<T> = realToFpExprCache.createIfContextActive {
        ensureContextMatch(sort, roundingMode, value)
        KRealToFpExpr(this, sort, roundingMode, value)
    }.cast()

    /**
     * Create Fp from BitVec value.
     * Rounding is performed according to the [roundingMode].
     *
     * @see [mkFpFromBvExpr] for IEEE binary conversion.
     * */
    open fun <T : KFpSort> mkBvToFpExpr(
        sort: T,
        roundingMode: KExpr<KFpRoundingModeSort>,
        value: KExpr<KBvSort>,
        signed: Boolean
    ): KExpr<T> =
        mkSimplified(sort, roundingMode, value, signed, KContext::simplifyBvToFpExpr, ::mkBvToFpExprNoSimplify)

    /**
     * Create Fp from BitVec value.
     * Rounding is performed according to the [roundingMode].
     *
     * @see [mkFpFromBvExpr] for IEEE binary conversion.
     * */
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

    /**
     * Create existential quantifier (`exists`).
     * */
    open fun mkExistentialQuantifier(body: KExpr<KBoolSort>, bounds: List<KDecl<*>>) =
        existentialQuantifierCache.createIfContextActive {
            ensureContextMatch(body)
            ensureContextMatch(bounds)
            KExistentialQuantifier(this, body, bounds)
        }

    private val universalQuantifierCache = mkAstInterner<KUniversalQuantifier>()

    /**
     * Create universal quantifier (`forall`).
     * */
    open fun mkUniversalQuantifier(body: KExpr<KBoolSort>, bounds: List<KDecl<*>>) =
        universalQuantifierCache.createIfContextActive {
            ensureContextMatch(body)
            ensureContextMatch(bounds)
            KUniversalQuantifier(this, body, bounds)
        }

    private val uninterpretedSortValueCache = mkAstInterner<KUninterpretedSortValue>()

    /**
     * Create uninterpreted sort value.
     *
     * Note: uninterpreted sort values with different [valueIdx] are distinct.
     * */
    open fun mkUninterpretedSortValue(sort: KUninterpretedSort, valueIdx: Int): KUninterpretedSortValue =
        uninterpretedSortValueCache.createIfContextActive {
            ensureContextMatch(sort)
            KUninterpretedSortValue(this, sort, valueIdx)
        }

    // default values
    val defaultValueSampler: KSortVisitor<KExpr<*>> by lazy {
        mkDefaultValueSampler()
    }

    open fun mkDefaultValueSampler(): KSortVisitor<KExpr<*>> =
        DefaultValueSampler(this)

    open fun boolSortDefaultValue(): KExpr<KBoolSort> = trueExpr

    open fun intSortDefaultValue(): KExpr<KIntSort> = mkIntNum(0)

    open fun realSortDefaultValue(): KExpr<KRealSort> = mkRealNum(0)

    open fun <S : KBvSort> bvSortDefaultValue(sort: S): KExpr<S> = mkBv(0, sort)

    open fun <S : KFpSort> fpSortDefaultValue(sort: S): KExpr<S> = mkFpZero(signBit = false, sort)

    open fun fpRoundingModeSortDefaultValue(): KExpr<KFpRoundingModeSort> =
        mkFpRoundingModeExpr(KFpRoundingMode.RoundNearestTiesToEven)

    open fun <A : KArraySortBase<R>, R : KSort> arraySortDefaultValue(sort: A): KExpr<A> =
        mkArrayConst(sort, sort.range.sampleValue())

    open fun uninterpretedSortDefaultValue(sort: KUninterpretedSort): KUninterpretedSortValue =
        mkUninterpretedSortValue(sort, valueIdx = 0)

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

    fun mkBvHexDecl(value: String, sizeBits: UInt): KDecl<KBvSort> =
        mkBvDecl(value.toBigInteger(radix = 16), sizeBits)

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

    fun mkUninterpretedSortValueDecl(sort: KUninterpretedSort, valueIdx: Int): KUninterpretedSortValueDecl =
        KUninterpretedSortValueDecl(this, sort, valueIdx)

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
