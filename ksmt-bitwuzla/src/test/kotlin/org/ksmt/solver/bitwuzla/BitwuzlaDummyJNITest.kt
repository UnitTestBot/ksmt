package org.ksmt.solver.bitwuzla;


import org.junit.jupiter.api.Assertions.assertFalse
import org.junit.jupiter.api.Assertions.assertTrue
import org.ksmt.KContext
import org.ksmt.solver.bitwuzla.bindings.BitwuzlaSort
import org.ksmt.solver.bitwuzla.bindings.BitwuzlaTerm
import org.ksmt.solver.bitwuzla.bindings.BitwuzlaOption
import org.ksmt.solver.bitwuzla.bindings.BitwuzlaResult
import kotlin.test.Test;
import org.ksmt.solver.bitwuzla.bindings.Native
import kotlin.test.assertEquals

class BitwuzlaDummyJNITest {

    private val ctx = KContext()
    private val solver = KBitwuzlaSolver(ctx)

    @Test
    fun test1(): Unit = with(ctx) {
        val bitwuzla = Native.bitwuzlaNew();
        Native.bitwuzlaReset(bitwuzla);
        val copyRight = Native.bitwuzlaCopyright(bitwuzla);
        val true_val = Native.bitwuzlaMkTrue(bitwuzla);
        val version = Native.bitwuzlaVersion(bitwuzla);
        val gitid = Native.bitwuzlaGitId(bitwuzla);
        println(version);
        println(gitid);
        val termination = Native.bitwuzlaTerminate(bitwuzla);
        println(termination);
        val bitwuzla2 = Native.bitwuzlaNew();
        Native.bitwuzlaSetOption(bitwuzla, BitwuzlaOption.BITWUZLA_OPT_PRODUCE_MODELS.value, 0);
        Native.bitwuzlaSetOptionStr(bitwuzla, BitwuzlaOption.BITWUZLA_OPT_ENGINE.value, "aigprop");
        val optionValue = Native.bitwuzlaGetOptionStr(bitwuzla, BitwuzlaOption.BITWUZLA_OPT_ENGINE.value);
        println(optionValue);
        val optionIntValue = Native.bitwuzlaGetOption(bitwuzla, BitwuzlaOption.BITWUZLA_OPT_PRODUCE_MODELS.value);
        println(optionIntValue);
        val sort4 = Native.bitwuzlaMkRmSort(bitwuzla);
        val sort = Native.bitwuzlaMkBoolSort(bitwuzla);
        val sort2 = Native.bitwuzlaMkBvSort(bitwuzla, 1);
        val sort3 = Native.bitwuzlaMkFpSort(bitwuzla, 4,2);
        val t = Native.bitwuzlaMkTrue(bitwuzla);
        val f = Native.bitwuzlaMkFalse(bitwuzla);
        val zero = Native.bitwuzlaMkBvZero(bitwuzla, sort);
        val one = Native.bitwuzlaMkBvOne(bitwuzla, sort);
        val ones = Native.bitwuzlaMkBvOnes(bitwuzla, sort);
        val minsigned = Native.bitwuzlaMkBvMinSigned(bitwuzla, sort2);
        val maxsigned = Native.bitwuzlaMkBvMinSigned(bitwuzla, sort2);
        //BitwuzlaJNIException - SymFPU not configured
        //val mkFPPosZero = Native.bitwuzlaMkFpPosZero(bitwuzla, sort3);
        //val mkFPNegZero = Native.bitwuzlaMkFpNegZero(bitwuzla, sort3);
        //val mkFPPosInf = Native.bitwuzlaMkFpPosInf(bitwuzla, sort3);
        //val mkFPNegInf = Native.bitwuzlaMkFpNegInf(bitwuzla, sort3);
        //val mkFPNegInf = Native.bitwuzlaMkFpNan(bitwuzla, sort3);
        val mkFPNegInf = Native.bitwuzlaMkBvValueUint64(bitwuzla, sort2, 5);
        val bitwuzlaMkVar = Native.bitwuzlaMkVar(bitwuzla, sort, "H");
        val bitwuzlaMkConst = Native.bitwuzlaMkConst(bitwuzla, sort, "H");
        val a : Array<BitwuzlaSort> = Array(1) { sort }
        //assertThrowsExactly(BitwuzlaJNIException , ()->Native.bitwuzlaMkFunSort(bitwuzla, 1, a, sort))
        //Native.bitwuzlaMkBvValue(bitwuzla, sort, "A", BitwuzlaBVBase.BITWUZLA_BV_BASE_HEX.value)
        //Native.bitwuzlaMkTerm1(bitwuzla, BitwuzlaKind.BITWUZLA_KIND_CONST.value, t)
        //Native.bitwuzlaMkTerm2(bitwuzla, BitwuzlaKind.BITWUZLA_KIND_CONST.value, t, t)
        //Native.bitwuzlaMkTerm3(bitwuzla, BitwuzlaKind.BITWUZLA_KIND_CONST.value, t, t, t)
        val ar : Array<BitwuzlaTerm> = Array(1) { t }
//        Native.bitwuzlaMkTerm1Indexed1(bitwuzla, BitwuzlaKind.BITWUZLA_KIND_CONST.value, t, 1);
//        Native.bitwuzlaMkTerm1Indexed2(bitwuzla, BitwuzlaKind.BITWUZLA_KIND_CONST.value, t, 1, 1);
//        Native.bitwuzlaMkTerm2Indexed1(bitwuzla, BitwuzlaKind.BITWUZLA_KIND_CONST.value, t, t,1);
//        Native.bitwuzlaMkTerm2Indexed2(bitwuzla, BitwuzlaKind.BITWUZLA_KIND_CONST.value, t, t,1, 1);
        //Native.bitwuzlaMkTerm(bitwuzla, BitwuzlaKind.BITWUZLA_KIND_CONST.value, ar);
        //Native.bitwuzlaMkConstArray(bitwuzla, sort, t);
//        Native.bitwuzlaGetUnsatAssumptions(bitwuzla);
        //val y = Native.bitwuzlaGetUnsatCore(bitwuzla);
        val result = Native.bitwuzlaCheckSat(bitwuzla);
        assertEquals(BitwuzlaResult.BITWUZLA_SAT, BitwuzlaResult.fromValue(result));
        //Native.bitwuzlaGetValue(bitwuzla, t);
        //incremental usage not enabled
        //Native.bitwuzlaPush(bitwuzla, 1);
        //Native.bitwuzlaPop(bitwuzla, 1);
        Native.bitwuzlaAssert(bitwuzla, t);
        //incremental usage not enabled
        //Native.bitwuzlaAssume(bitwuzla, t);
        //Native.bitwuzlaFixateAssumptions(bitwuzla);
        //Native.bitwuzlaResetAssumptions(bitwuzla);
        val hash = Native.bitwuzlaSortHash(sort);
        val bvsize = Native.bitwuzlaSortBvGetSize(sort);
        val fpExpSize = Native.bitwuzlaSortFpGetExpSize(sort3);
        val fpSigSize = Native.bitwuzlaSortFpGetSigSize(sort3);

        val arSort = Native.bitwuzlaMkArraySort(bitwuzla, sort2, sort2);
        val arSortI = Native.bitwuzlaSortArrayGetIndex(arSort);
        val arSortE = Native.bitwuzlaSortArrayGetElement(arSort);

        val sort_true = Native.bitwuzlaSortIsEqual(arSort, sort2);
        println(sort_true);
        println(Native.bitwuzlaSortIsArray(arSort));
        println(Native.bitwuzlaSortIsArray(sort2));
        println(Native.bitwuzlaSortIsBv(arSort));
        println(Native.bitwuzlaSortIsBv(sort2));
        println(Native.bitwuzlaSortIsFp(arSort));
        println(Native.bitwuzlaSortIsFp(sort3));
        println(Native.bitwuzlaSortIsFun(arSort));
        println(Native.bitwuzlaSortIsRm(sort4));
        println(Native.bitwuzlaSortIsRm(arSort));
        var hash_t = Native.bitwuzlaTermHash(t);
        println(Native.bitwuzlaTermIsIndexed(t));
        var bitwuzla_N = Native.bitwuzlaTermGetBitwuzla(t);
        println("============================")
        println(bitwuzla == bitwuzla_N)
        var s1 = Native.bitwuzlaTermGetSort(t);
        //expected array term
        //var s2 = Native.bitwuzlaTermArrayGetIndexSort(t);
        //var s2 = Native.bitwuzlaTermArrayGetElementSort(t);
        //Native.bitwuzlaTermFunGetCodomainSort(t);
        Native.bitwuzlaTermBvGetSize(ones);
        //Native.bitwuzlaTermFpGetExpSize(ones);
        //Native.bitwuzlaTermFpGetSigSize(ones);
        //Native.bitwuzlaTermFunGetArity(ones);
        Native.bitwuzlaTermSetSymbol(ones, "F");
        println(Native.bitwuzlaTermGetSymbol(ones));
        assertTrue(Native.bitwuzlaTermIsEqualSort(ones, ones));
        assertFalse(Native.bitwuzlaTermIsArray(ones));
        assertFalse(Native.bitwuzlaTermIsConst(t));
        assertFalse(Native.bitwuzlaTermIsFun(t));
        assertFalse(Native.bitwuzlaTermIsVar(t));
        assertFalse(Native.bitwuzlaTermIsBoundVar(bitwuzlaMkVar));
        assertTrue(Native.bitwuzlaTermIsValue(one));
        assertTrue(Native.bitwuzlaTermIsBvValue(one));
        assertFalse(Native.bitwuzlaTermIsFpValue(one));
        assertFalse(Native.bitwuzlaTermIsRmValue(one));
        assertTrue(Native.bitwuzlaTermIsBv(ones));
        assertFalse(Native.bitwuzlaTermIsFp(ones));
        assertFalse(Native.bitwuzlaTermIsRm(ones));
        assertTrue(Native.bitwuzlaTermIsBvValueZero(zero));
        assertTrue(Native.bitwuzlaTermIsBvValueOne(one));
        assertTrue(Native.bitwuzlaTermIsBvValueOnes(ones));
        assertFalse(Native.bitwuzlaTermIsBvValueMinSigned(zero));
        assertTrue(Native.bitwuzlaTermIsBvValueMaxSigned(zero));
        assertFalse(Native.bitwuzlaTermIsFpValuePosZero(zero));
        assertFalse(Native.bitwuzlaTermIsFpValueNegZero(zero));
        assertFalse(Native.bitwuzlaTermIsFpValuePosInf(zero));
        assertFalse(Native.bitwuzlaTermIsFpValueNegInf(zero));
        assertFalse(Native.bitwuzlaTermIsFpValueNan(zero));
        assertFalse(Native.bitwuzlaTermIsRmValueRna(zero));
        assertFalse(Native.bitwuzlaTermIsRmValueRne(zero));
        assertFalse(Native.bitwuzlaTermIsRmValueRtn(zero));
        assertFalse(Native.bitwuzlaTermIsRmValueRtp(zero));
        assertFalse(Native.bitwuzlaTermIsRmValueRtz(zero));
        assertFalse(Native.bitwuzlaTermIsConstArray(zero));

        val r = Native.bitwuzlaBvConstNodeGetBits(t);
        println(r);
        val w = Native.bitwuzlaBvBitsGetWidth(r);
        println(w);
        val w2 = Native.bitwuzlaBvBitsToUInt64(r);
        println(w2);
        val w3 = Native.bitwuzlaBvBitsGetBit(r, 2);
        println(w3);

//        val p = Native.bitwuzlaFpConstNodeGetBits(bitwuzla, t);
//        println(p);

        //Native.bitwuzlaGetBvValue(bitwuzla, f);
        //val ara = Native.bitwuzlaGetArrayValue(bitwuzla, t);
        //val fun_v = Native.bitwuzlaGetFunValue(bitwuzla, f);
    }
}
