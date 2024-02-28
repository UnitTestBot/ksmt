#include <jni.h>

#ifndef _Included_bitwuzla_jni
#define _Included_bitwuzla_jni
#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT void JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaInit
        (JNIEnv* env, jobject native_class);

JNIEXPORT jlong JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermManagerNew(JNIEnv* env, jobject native_class);

JNIEXPORT void JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermManagerRelease
        (JNIEnv* env, jobject native_class, jlong bitwuzla_term_manager);

JNIEXPORT void JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermManagerDelete
        (JNIEnv* env, jobject native_class, jlong bitwuzla_term_manager);

JNIEXPORT void JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaSortRelease
        (JNIEnv* env, jobject native_class, jlong sort);

JNIEXPORT void JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermRelease
        (JNIEnv* env, jobject native_class, jlong term);

JNIEXPORT jlong JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaNew
        (JNIEnv* env, jobject native_class, jlong bitwuzla_term_manager, jlong bitwuzla_options);

JNIEXPORT void JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaDelete
        (JNIEnv* env, jobject native_class, jlong bitwuzla);

JNIEXPORT jlong JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaOptionsNew
(JNIEnv* env, jobject native_class);

JNIEXPORT void JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaOptionsDelete
(JNIEnv* env, jobject native_class, jlong options);

JNIEXPORT void JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaSetOption
        (JNIEnv* env, jobject native_class, jlong options, jint bitwuzla_option, jlong value);

JNIEXPORT void JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaSetOptionMode
        (JNIEnv* env, jobject native_class, jlong options, jint bitwuzla_option, jstring value);

JNIEXPORT jlong JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaGetOption
        (JNIEnv* env, jobject native_class, jlong options, jint bitwuzla_option);

JNIEXPORT jstring JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaGetOptionMode
        (JNIEnv* env, jobject native_class, jlong options, jint bitwuzla_option);

JNIEXPORT jlong JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkArraySort
        (JNIEnv* env, jobject native_class, jlong bitwuzla_term_manager, jlong index, jlong element);

JNIEXPORT jlong JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkBoolSort
        (JNIEnv* env, jobject native_class, jlong bitwuzla_term_manager);

JNIEXPORT jlong JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkBvSort
        (JNIEnv* env, jobject native_class, jlong bitwuzla_term_manager, jlong size);

JNIEXPORT jlong JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkFpSort
        (JNIEnv* env, jobject native_class, jlong bitwuzla_term_manager, jlong exp_size, jlong sig_size);

JNIEXPORT jlong JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkFunSort
        (JNIEnv* env, jobject native_class, jlong bitwuzla_term_manager, jlong arity, jlongArray domain, jlong codomain);

JNIEXPORT jlong JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkRmSort
        (JNIEnv* env, jobject native_class, jlong bitwuzla_term_manager);

JNIEXPORT jlong JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkUninterpretedSort
(JNIEnv* env, jobject native_class, jlong bitwuzla_term_manager, jstring value);

JNIEXPORT jlong JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkTrue
        (JNIEnv* env, jobject native_class, jlong bitwuzla_term_manager);

JNIEXPORT jlong JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkFalse
        (JNIEnv* env, jobject native_class, jlong bitwuzla_term_manager);

JNIEXPORT jlong JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkBvZero
        (JNIEnv* env, jobject native_class, jlong bitwuzla_term_manager, jlong bitwuzla_sort);

JNIEXPORT jlong JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkBvOne
        (JNIEnv* env, jobject native_class, jlong bitwuzla_term_manager, jlong bitwuzla_sort);

JNIEXPORT jlong JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkBvOnes
        (JNIEnv* env, jobject native_class, jlong bitwuzla_term_manager, jlong bitwuzla_sort);

JNIEXPORT jlong JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkBvMinSigned
        (JNIEnv* env, jobject native_class, jlong bitwuzla_term_manager, jlong bitwuzla_sort);

JNIEXPORT jlong JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkBvMaxSigned
        (JNIEnv* env, jobject native_class, jlong bitwuzla_term_manager, jlong bitwuzla_sort);

JNIEXPORT jlong JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkFpPosZero
        (JNIEnv* env, jobject native_class, jlong bitwuzla_term_manager, jlong bitwuzla_sort);

JNIEXPORT jlong JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkFpNegZero
        (JNIEnv* env, jobject native_class, jlong bitwuzla_term_manager, jlong bitwuzla_sort);

JNIEXPORT jlong JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkFpPosInf
        (JNIEnv* env, jobject native_class, jlong bitwuzla_term_manager, jlong bitwuzla_sort);

JNIEXPORT jlong JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkFpNegInf
        (JNIEnv* env, jobject native_class, jlong bitwuzla_term_manager, jlong bitwuzla_sort);

JNIEXPORT jlong JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkFpNan
        (JNIEnv* env, jobject native_class, jlong bitwuzla_term_manager, jlong bitwuzla_sort);

JNIEXPORT jlong JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkBvValue
        (JNIEnv* env, jobject native_class, jlong bitwuzla_term_manager, jlong bitwuzla_sort, jstring value, jbyte base);

JNIEXPORT jlong JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkBvValueUint64
        (JNIEnv* env, jobject native_class, jlong bitwuzla_term_manager, jlong bitwuzla_sort, jlong value);

JNIEXPORT jlong JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkBvValueUint64Array
        (JNIEnv* env, jobject native_class, jlong bitwuzla_term_manager, jlong bv_width, jlongArray value);

JNIEXPORT jlong JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkFpValue
        (JNIEnv* env, jobject native_class, jlong bitwuzla_term_manager, jlong bitwuzla_bvSign, jlong bitwuzla_bvExponent,
         jlong bitwuzla_bvSignificand);

JNIEXPORT jlong JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkFpValueFromReal
        (JNIEnv* env, jobject native_class, jlong bitwuzla_term_manager, jlong sort, jlong rm, jstring real);

JNIEXPORT jlong JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkFpValueFromRational
        (JNIEnv* env, jobject native_class, jlong bitwuzla_term_manager, jlong sort, jlong rm, jstring num, jstring den);

JNIEXPORT jlong JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkRmValue
        (JNIEnv* env, jobject native_class, jlong bitwuzla_term_manager, jint rm);

JNIEXPORT jlong JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkTerm1
        (JNIEnv* env, jobject native_class, jlong bitwuzla_term_manager, jint kind, jlong arg);

JNIEXPORT jlong JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkTerm2
        (JNIEnv* env, jobject native_class, jlong bitwuzla_term_manager, jint kind, jlong arg0, jlong arg1);

JNIEXPORT jlong JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkTerm3
        (JNIEnv* env, jobject native_class, jlong bitwuzla_term_manager, jint kind, jlong arg0, jlong arg1, jlong arg2);

JNIEXPORT jlong JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkTerm
        (JNIEnv* env, jobject native_class, jlong bitwuzla_term_manager, jint kind, jlongArray args);

JNIEXPORT jlong JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkTerm1Indexed1
        (JNIEnv* env, jobject native_class, jlong bitwuzla_term_manager, jint kind, jlong term, jlong idx);

JNIEXPORT jlong JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkTerm1Indexed2
        (JNIEnv* env, jobject native_class, jlong bitwuzla_term_manager, jint kind, jlong term, jlong idx0, jlong idx1);

JNIEXPORT jlong JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkTerm2Indexed1
        (JNIEnv* env, jobject native_class, jlong bitwuzla_term_manager, jint kind, jlong term0, jlong term1, jlong idx0);

JNIEXPORT jlong JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkTerm2Indexed2
        (JNIEnv* env, jobject native_class, jlong bitwuzla_term_manager, jint kind, jlong term0, jlong term1, jlong idx0, jlong idx1);

JNIEXPORT jlong JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkTermIndexed
        (JNIEnv* env, jobject native_class, jlong bitwuzla_term_manager, jint kind, jlongArray args, jlongArray idxs);

JNIEXPORT jlong JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkConst
        (JNIEnv* env, jobject native_class, jlong bitwuzla_term_manager, jlong bitwuzla_sort, jstring symbol);

JNIEXPORT jlong JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkConstArray
        (JNIEnv* env, jobject native_class, jlong bitwuzla_term_manager, jlong bitwuzla_sort, jlong value);

JNIEXPORT jlong JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkVar
        (JNIEnv* env, jobject native_class, jlong bitwuzla_term_manager, jlong bitwuzla_sort, jstring symbol);

JNIEXPORT void JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaPush
        (JNIEnv* env, jobject native_class, jlong bitwuzla, jlong n_levels);

JNIEXPORT void JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaPop
        (JNIEnv* env, jobject native_class, jlong bitwuzla, jlong n_levels);

JNIEXPORT void JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaAssert
        (JNIEnv* env, jobject native_class, jlong bitwuzla, jlong term);

JNIEXPORT jboolean JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaIsUnsatAssumption
        (JNIEnv* env, jobject native_class, jlong bitwuzla, jlong term);

JNIEXPORT jlongArray JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaGetUnsatAssumptions
        (JNIEnv* env, jobject native_class, jlong bitwuzla);

JNIEXPORT jlongArray JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaGetUnsatCore
        (JNIEnv* env, jobject native_class, jlong bitwuzla);

JNIEXPORT void JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaSimplify
        (JNIEnv* env, jobject native_class, jlong bitwuzla);

JNIEXPORT jint JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaCheckSat
        (JNIEnv* env, jobject native_class, jlong bitwuzla);

JNIEXPORT jint JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaCheckSatAssuming
        (JNIEnv* env, jobject native_class, jlong bitwuzla, jlongArray assumptions);

JNIEXPORT jint JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaCheckSatTimeout
        (JNIEnv* env, jobject native_class, jlong bitwuzla, jlong timeout);

JNIEXPORT jint JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaCheckSatAssumingTimeout
        (JNIEnv* env, jobject native_class, jlong bitwuzla, jlongArray assumptions, jlong timeout);

JNIEXPORT void JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaForceTerminate
        (JNIEnv* env, jobject native_class, jlong bitwuzla);

JNIEXPORT jlong JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaGetValue
        (JNIEnv* env, jobject native_class, jlong bitwuzla, jlong bitwuzla_term);

JNIEXPORT jboolean JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermValueGetBool
(JNIEnv* env, jobject native_class, jlong bitwuzla_term);

JNIEXPORT jint JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermValueGetRm
(JNIEnv* env, jobject native_class, jlong bitwuzla_term);

JNIEXPORT jlong JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermValueGetBvUint64
(JNIEnv* env, jobject native_class, jlong bitwuzla_term);

JNIEXPORT jlong JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermValueGetFpUint64
(JNIEnv* env, jobject native_class, jlong bitwuzla_term);

JNIEXPORT jlongArray JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermValueGetBvUint64Array
(JNIEnv* env, jobject native_class, jlong bitwuzla_term);

JNIEXPORT jlongArray JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermValueGetFpUint64Array
(JNIEnv* env, jobject native_class, jlong bitwuzla_term);

JNIEXPORT jlong JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaSubstituteTerm
        (JNIEnv* env, jobject native_class, jlong bitwuzla_term,
         jlongArray map_keys, jlongArray map_values);

JNIEXPORT jlongArray JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaSubstituteTerms
        (JNIEnv* env, jobject native_class, jlongArray terms,
         jlongArray map_keys, jlongArray map_values);

JNIEXPORT jlong JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaSortHash
        (JNIEnv* env, jobject native_class, jlong bitwuzla_sort);

JNIEXPORT jint JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermGetKind
        (JNIEnv* env, jobject native_class, jlong bitwuzla_term);

JNIEXPORT jlongArray JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermGetChildren
        (JNIEnv* env, jobject native_class, jlong bitwuzla_term);

JNIEXPORT jlongArray JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermGetIndices
        (JNIEnv* env, jobject native_class, jlong bitwuzla_term);

JNIEXPORT jlong JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaSortBvGetSize
        (JNIEnv* env, jobject native_class, jlong bitwuzla_sort);

JNIEXPORT jlong JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaSortFpGetExpSize
        (JNIEnv* env, jobject native_class, jlong bitwuzla_sort);

JNIEXPORT jlong JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaSortFpGetSigSize
        (JNIEnv* env, jobject native_class, jlong bitwuzla_sort);

JNIEXPORT jlong JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaSortArrayGetIndex
        (JNIEnv* env, jobject native_class, jlong bitwuzla_sort);

JNIEXPORT jlong JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaSortArrayGetElement
        (JNIEnv* env, jobject native_class, jlong bitwuzla_sort);

JNIEXPORT jlongArray JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaSortFunGetDomainSorts
        (JNIEnv* env, jobject native_class, jlong bitwuzla_sort);

JNIEXPORT jlong JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaSortFunGetCodomain
        (JNIEnv* env, jobject native_class, jlong bitwuzla_sort);

JNIEXPORT jlong JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaSortFunGetArity
        (JNIEnv* env, jobject native_class, jlong bitwuzla_sort);

JNIEXPORT jstring JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaSortGetUninterpretedSymbol
(JNIEnv* env, jobject native_class, jlong bitwuzla_sort);

JNIEXPORT jboolean JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaSortIsArray
        (JNIEnv* env, jobject native_class, jlong bitwuzla_sort);

JNIEXPORT jboolean JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaSortIsBool
        (JNIEnv* env, jobject native_class, jlong bitwuzla_sort);

JNIEXPORT jboolean JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaSortIsBv
        (JNIEnv* env, jobject native_class, jlong bitwuzla_sort);

JNIEXPORT jboolean JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaSortIsFp
        (JNIEnv* env, jobject native_class, jlong bitwuzla_sort);

JNIEXPORT jboolean JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaSortIsFun
        (JNIEnv* env, jobject native_class, jlong bitwuzla_sort);

JNIEXPORT jboolean JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaSortIsRm
        (JNIEnv* env, jobject native_class, jlong bitwuzla_sort);

JNIEXPORT jboolean JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaSortIsUninterpreted
        (JNIEnv* env, jobject native_class, jlong bitwuzla_sort);

JNIEXPORT jlong JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermHash
        (JNIEnv* env, jobject native_class, jlong bitwuzla_term);

JNIEXPORT jboolean JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsIndexed
        (JNIEnv* env, jobject native_class, jlong bitwuzla_term);

JNIEXPORT jlong JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermGetSort
        (JNIEnv* env, jobject native_class, jlong bitwuzla_term);

JNIEXPORT jlong JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermArrayGetIndexSort
        (JNIEnv* env, jobject native_class, jlong bitwuzla_term);

JNIEXPORT jlong JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermArrayGetElementSort
        (JNIEnv* env, jobject native_class, jlong bitwuzla_term);

JNIEXPORT jlongArray JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermFunGetDomainSorts
        (JNIEnv* env, jobject native_class, jlong bitwuzla_term);

JNIEXPORT jlong JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermFunGetCodomainSort
        (JNIEnv* env, jobject native_class, jlong bitwuzla_term);

JNIEXPORT jlong JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermBvGetSize
        (JNIEnv* env, jobject native_class, jlong bitwuzla_term);

JNIEXPORT jlong JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermFpGetExpSize
        (JNIEnv* env, jobject native_class, jlong bitwuzla_term);

JNIEXPORT jlong JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermFpGetSigSize
        (JNIEnv* env, jobject native_class, jlong bitwuzla_term);

JNIEXPORT jlong JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermFunGetArity
        (JNIEnv* env, jobject native_class, jlong bitwuzla_term);

JNIEXPORT jstring JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermGetSymbol
        (JNIEnv* env, jobject native_class, jlong bitwuzla_term);

JNIEXPORT jboolean JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsEqualSort
        (JNIEnv* env, jobject native_class, jlong bitwuzla_term_1, jlong bitwuzla_term_2);

JNIEXPORT jboolean JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsArray
        (JNIEnv* env, jobject native_class, jlong bitwuzla_term);

JNIEXPORT jboolean JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsConst
        (JNIEnv* env, jobject native_class, jlong bitwuzla_term);

JNIEXPORT jboolean JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsFun
        (JNIEnv* env, jobject native_class, jlong bitwuzla_term);

JNIEXPORT jboolean JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsVar
        (JNIEnv* env, jobject native_class, jlong bitwuzla_term);

JNIEXPORT jboolean JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsValue
        (JNIEnv* env, jobject native_class, jlong bitwuzla_term);

JNIEXPORT jboolean JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsBvValue
        (JNIEnv* env, jobject native_class, jlong bitwuzla_term);

JNIEXPORT jboolean JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsFpValue
        (JNIEnv* env, jobject native_class, jlong bitwuzla_term);

JNIEXPORT jboolean JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsRmValue
        (JNIEnv* env, jobject native_class, jlong bitwuzla_term);

JNIEXPORT jboolean JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsBool
        (JNIEnv* env, jobject native_class, jlong bitwuzla_term);

JNIEXPORT jboolean JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsBv
        (JNIEnv* env, jobject native_class, jlong bitwuzla_term);

JNIEXPORT jboolean JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsFp
        (JNIEnv* env, jobject native_class, jlong bitwuzla_term);

JNIEXPORT jboolean JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsRm
        (JNIEnv* env, jobject native_class, jlong bitwuzla_term);

JNIEXPORT jboolean JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsUninterpreted
        (JNIEnv* env, jobject native_class, jlong bitwuzla_term);

JNIEXPORT jboolean JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsTrue
        (JNIEnv* env, jobject native_class, jlong bitwuzla_term);

JNIEXPORT jboolean JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsFalse
        (JNIEnv* env, jobject native_class, jlong bitwuzla_term);

JNIEXPORT jboolean JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsBvValueZero
        (JNIEnv* env, jobject native_class, jlong bitwuzla_term);

JNIEXPORT jboolean JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsBvValueOne
        (JNIEnv* env, jobject native_class, jlong bitwuzla_term);

JNIEXPORT jboolean JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsBvValueOnes
        (JNIEnv* env, jobject native_class, jlong bitwuzla_term);

JNIEXPORT jboolean JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsBvValueMinSigned
        (JNIEnv* env, jobject native_class, jlong bitwuzla_term);

JNIEXPORT jboolean JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsBvValueMaxSigned
        (JNIEnv* env, jobject native_class, jlong bitwuzla_term);

JNIEXPORT jboolean JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsFpValuePosZero
        (JNIEnv* env, jobject native_class, jlong bitwuzla_term);

JNIEXPORT jboolean JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsFpValueNegZero
        (JNIEnv* env, jobject native_class, jlong bitwuzla_term);

JNIEXPORT jboolean JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsFpValuePosInf
        (JNIEnv* env, jobject native_class, jlong bitwuzla_term);

JNIEXPORT jboolean JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsFpValueNegInf
        (JNIEnv* env, jobject native_class, jlong bitwuzla_term);

JNIEXPORT jboolean JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsFpValueNan
        (JNIEnv* env, jobject native_class, jlong bitwuzla_term);

JNIEXPORT jboolean JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsRmValueRna
        (JNIEnv* env, jobject native_class, jlong bitwuzla_term);

JNIEXPORT jboolean JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsRmValueRne
        (JNIEnv* env, jobject native_class, jlong bitwuzla_term);

JNIEXPORT jboolean JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsRmValueRtn
        (JNIEnv* env, jobject native_class, jlong bitwuzla_term);

JNIEXPORT jboolean JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsRmValueRtp
        (JNIEnv* env, jobject native_class, jlong bitwuzla_term);

JNIEXPORT jboolean JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsRmValueRtz
        (JNIEnv* env, jobject native_class, jlong bitwuzla_term);

JNIEXPORT jstring JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaSortToString
        (JNIEnv* env, jobject native_class, jlong sort);

JNIEXPORT jstring JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermToString
        (JNIEnv* env, jobject native_class, jlong term);

JNIEXPORT void JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaDumpFormula
        (JNIEnv* env, jobject native_class, jlong bitwuzla, jstring format, jstring file_path);

#ifdef __cplusplus
}
#endif
#endif
