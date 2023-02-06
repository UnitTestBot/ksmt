#include <iostream>
#include <atomic>
#include <bitwuzla.h>
#include <memory>
#include <unistd.h>
#include <vector>
#include "bitwuzla_jni.hpp"
#include "bitwuzla_extension.h"

#define BITWUZLA_JNI_EXCEPTION_CLS "org/ksmt/solver/bitwuzla/bindings/BitwuzlaNativeException"
#define BITWUZLA_CATCH_STATEMENT catch (const std::exception& e)
#define BITWUZLA_JNI_EXCEPTION jclass exception = env->FindClass(BITWUZLA_JNI_EXCEPTION_CLS);
#define BITWUZLA_JNI_THROW env->ThrowNew(exception, e.what());
#define BZLA_TRY(CODE, ERROR_VAL) try { CODE } BITWUZLA_CATCH_STATEMENT { BITWUZLA_JNI_EXCEPTION BITWUZLA_JNI_THROW return ERROR_VAL;}
#define BZLA_TRY_OR_ZERO(CODE) BZLA_TRY(CODE, 0)
#define BZLA_TRY_VOID(CODE) BZLA_TRY(CODE, )
#define BZLA_TRY_OR_NULL(CODE) BZLA_TRY(CODE, nullptr)

template<typename T>
struct JniPoinerArray {
    JNIEnv* env;
    jlongArray array;
    T* ptr_array;

    JniPoinerArray(JNIEnv* env, jlongArray array) : env(env), array(array) {
#if defined(__LP64__) || defined(_WIN64)
        // pointers are 64 bits, we can simply cast an array
        ptr_array = (T*) env->GetLongArrayElements(array, nullptr);
#else
        jlong* tmp = env->GetLongArrayElements(array, nullptr);
        unsigned int size = env->GetArrayLength(array);
        ptr_array = (T*) new int[size];
        for (unsigned i = 0; i < size; i++) {
            ptr_array[i] = reinterpret_cast<T>(tmp[i]);
        }
        env->ReleaseLongArrayElements(array, tmp, JNI_ABORT);
#endif
    }

    ~JniPoinerArray() {
#if defined(__LP64__) || defined(_WIN64)
        env->ReleaseLongArrayElements(array, (jlong*) ptr_array, JNI_ABORT);
#else
        delete[] ptr_array;
#endif
    }
};

struct JniString {
    JNIEnv* env;
    jstring str;
    const char* chars;

    JniString(JNIEnv* env, jstring str) : env(env), str(str) {
        chars = env->GetStringUTFChars(str, nullptr);
    }

    ~JniString() {
        env->ReleaseStringUTFChars(str, chars);
    }
};

#define PPCAT_NX(A, B) A ## B
#define PPCAT(A, B) PPCAT_NX(A, B)

#define GET_STRING(var_name, str) \
    JniString PPCAT(str_wrap_, var_name)(env, str); \
    const char* var_name = PPCAT(str_wrap_, var_name).chars;

#define GET_PTR_ARRAY(type, var_name, array) \
    JniPoinerArray<type> PPCAT(array_wrap_, var_name)(env, array);\
    type* var_name = PPCAT(array_wrap_, var_name).ptr_array;

jlongArray create_ptr_array(JNIEnv* env, size_t size) {
    return env->NewLongArray((jsize) size);
}

template<typename T>
void set_ptr_array(JNIEnv* env, jlongArray array, T* ptr_array, size_t size) {
#if defined(__LP64__) || defined(_WIN64)
    env->SetLongArrayRegion(array, 0, (jsize) size, (jlong*) ptr_array);
#else
    jlong* temp = new jlong[size];
    for (int i = 0; i < size; i++) {
        temp[i] = reinterpret_cast<jlong>(ptr_array[i]);
    }
    env->SetLongArrayRegion(array, 0, (jsize) size, temp);
    delete[] temp;
#endif
}

#define BZLA (Bitwuzla*) bitwuzla
#define TERM(t) (BitwuzlaTerm*) t
#define SORT(s) (BitwuzlaSort *) s

void abort_callback(const char* msg) {
    throw std::runtime_error(msg);
}

int32_t termination_callback(void* state) {
    return 0;
}

void Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaInit(JNIEnv* env, jobject native_class) {
    bitwuzla_set_abort_callback(abort_callback);
}

jlong Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaNew(JNIEnv* env, jobject native_class) {
    BZLA_TRY_OR_ZERO({
                         return (jlong) bitwuzla_new();
                     })
}

void Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaDelete(JNIEnv* env, jobject native_class, jlong bitwuzla) {
    BZLA_TRY_VOID({
                      bitwuzla_delete(BZLA);
                  })
}

void Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaReset(JNIEnv* env, jobject native_class, jlong bitwuzla) {
    BZLA_TRY_VOID({
                      bitwuzla_reset(BZLA);
                  })
}

jstring
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaCopyright(JNIEnv* env, jobject native_class, jlong bitwuzla) {
    BZLA_TRY_OR_NULL({
                         const char* c = bitwuzla_copyright(BZLA);
                         jstring result = env->NewStringUTF(c);
                         return result;
                     })
}

jstring
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaVersion(JNIEnv* env, jobject native_class, jlong bitwuzla) {
    BZLA_TRY_OR_NULL({
                         const char* c = bitwuzla_version(BZLA);
                         jstring result = env->NewStringUTF(c);
                         return result;
                     })
}

jstring Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaGitId(JNIEnv* env, jobject native_class, jlong bitwuzla) {
    BZLA_TRY_OR_NULL({
                         const char* c = bitwuzla_git_id(BZLA);
                         jstring result = env->NewStringUTF(c);
                         return result;
                     })
}

jboolean
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTerminate(JNIEnv* env, jobject native_class, jlong bitwuzla) {
    BZLA_TRY_OR_ZERO({
                         return static_cast<jboolean>(bitwuzla_terminate(BZLA));
                     })
}

void Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaSetTerminationCallback(JNIEnv* env, jobject native_class,
                                                                                  jlong bitwuzla) {
    BZLA_TRY_VOID({
                      bitwuzla_set_termination_callback(BZLA, termination_callback, nullptr);
                  })
}

void Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaResetTerminationCallback(JNIEnv* env, jobject native_class,
                                                                                    jlong bitwuzla) {
    BZLA_TRY_VOID({
                      bitwuzla_set_termination_callback(BZLA, termination_callback, nullptr);
                  })
}

void Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaSetOption(JNIEnv* env, jobject native_class, jlong bitwuzla,
                                                                     jint bitwuzla_option, jint value) {
    BZLA_TRY_VOID({
                      bitwuzla_set_option(BZLA, BitwuzlaOption(bitwuzla_option), value);
                  })
}

void
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaSetOptionStr(JNIEnv* env, jobject native_class, jlong bitwuzla,
                                                                   jint bitwuzla_option, jstring value) {
    BZLA_TRY_VOID({
                      GET_STRING(nativeString, value);
                      bitwuzla_set_option_str(BZLA, BitwuzlaOption(bitwuzla_option), nativeString);
                  })
}

jint Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaGetOption(JNIEnv* env, jobject native_class, jlong bitwuzla,
                                                                     jint bitwuzla_option) {
    BZLA_TRY_OR_ZERO({
                         return (jint) bitwuzla_get_option(BZLA, BitwuzlaOption(bitwuzla_option));
                     })
}

jstring
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaGetOptionStr(JNIEnv* env, jobject native_class, jlong bitwuzla,
                                                                   jint bitwuzla_option) {
    BZLA_TRY_OR_NULL({
                         const char* value = bitwuzla_get_option_str(BZLA, BitwuzlaOption(bitwuzla_option));
                         return env->NewStringUTF(value);
                     })
}

jlong
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkArraySort(JNIEnv* env, jobject native_class, jlong bitwuzla,
                                                                  jlong index, jlong element) {
    BZLA_TRY_OR_ZERO({
                         return (jlong) bitwuzla_mk_array_sort(BZLA, SORT(index), SORT(element));
                     })
}

jlong
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkBoolSort(JNIEnv* env, jobject native_class, jlong bitwuzla) {
    BZLA_TRY_OR_ZERO({
                         return (jlong) bitwuzla_mk_bool_sort(BZLA);
                     })
}

jlong Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkBvSort(JNIEnv* env, jobject native_class, jlong bitwuzla,
                                                                     jint size) {
    BZLA_TRY_OR_ZERO({
                         return (jlong) bitwuzla_mk_bv_sort(BZLA, size);
                     })
}

jlong Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkFpSort(JNIEnv* env, jobject native_class, jlong bitwuzla,
                                                                     jint exp_size, jint sig_size) {
    BZLA_TRY_OR_ZERO({
                         const BitwuzlaSort* value = bitwuzla_mk_fp_sort(BZLA, exp_size, sig_size);
                         return (jlong) value;
                     })
}

jlong Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkFunSort(JNIEnv* env, jobject native_class, jlong bitwuzla,
                                                                      jint arity, jlongArray domain, jlong codomain) {
    BZLA_TRY_OR_ZERO({
                         GET_PTR_ARRAY(BitwuzlaSort const*, domain_ptr, domain);
                         jsize len = env->GetArrayLength(domain);
                         const BitwuzlaSort* value = bitwuzla_mk_fun_sort(BZLA, arity, domain_ptr, SORT(codomain));
                         return (jlong) value;
                     })
}

jlong
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkRmSort(JNIEnv* env, jobject native_class, jlong bitwuzla) {
    BZLA_TRY_OR_ZERO({
                         const BitwuzlaSort* value = bitwuzla_mk_rm_sort(BZLA);
                         return (jlong) value;
                     })
}

jlong Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkTrue(JNIEnv* env, jobject native_class, jlong bitwuzla) {
    BZLA_TRY_OR_ZERO({
                         return (jlong) bitwuzla_mk_true(BZLA);
                     })
}

jlong Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkFalse(JNIEnv* env, jobject native_class, jlong bitwuzla) {
    BZLA_TRY_OR_ZERO({
                         return (jlong) bitwuzla_mk_false(BZLA);
                     })
}

jlong Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkBvZero(JNIEnv* env, jobject native_class, jlong bitwuzla,
                                                                     jlong bitwuzla_sort) {
    BZLA_TRY_OR_ZERO({
                         return (jlong) bitwuzla_mk_bv_zero(BZLA, SORT(bitwuzla_sort));
                     })
}

jlong Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkBvOne(JNIEnv* env, jobject native_class, jlong bitwuzla,
                                                                    jlong bitwuzla_sort) {
    BZLA_TRY_OR_ZERO({
                         return (jlong) bitwuzla_mk_bv_one(BZLA, SORT(bitwuzla_sort));
                     })
}

jlong Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkBvOnes(JNIEnv* env, jobject native_class, jlong bitwuzla,
                                                                     jlong bitwuzla_sort) {
    BZLA_TRY_OR_ZERO({
                         return (jlong) bitwuzla_mk_bv_ones(BZLA, SORT(bitwuzla_sort));
                     })
}

jlong
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkBvMinSigned(JNIEnv* env, jobject native_class, jlong bitwuzla,
                                                                    jlong bitwuzla_sort) {
    BZLA_TRY_OR_ZERO({
                         return (jlong) bitwuzla_mk_bv_min_signed(BZLA, SORT(bitwuzla_sort));
                     })
}

jlong
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkBvMaxSigned(JNIEnv* env, jobject native_class, jlong bitwuzla,
                                                                    jlong bitwuzla_sort) {
    BZLA_TRY_OR_ZERO({
                         return (jlong) bitwuzla_mk_bv_max_signed(BZLA, SORT(bitwuzla_sort));
                     })
}

jlong
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkFpPosZero(JNIEnv* env, jobject native_class, jlong bitwuzla,
                                                                  jlong bitwuzla_sort) {
    BZLA_TRY_OR_ZERO({
                         const BitwuzlaTerm* result = bitwuzla_mk_fp_pos_zero(BZLA, SORT(bitwuzla_sort));
                         return (jlong) result;
                     })
}

jlong
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkFpNegZero(JNIEnv* env, jobject native_class, jlong bitwuzla,
                                                                  jlong bitwuzla_sort) {
    BZLA_TRY_OR_ZERO({
                         const BitwuzlaTerm* result = bitwuzla_mk_fp_neg_zero(BZLA, SORT(bitwuzla_sort));
                         return (jlong) result;
                     })
}

jlong
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkFpPosInf(JNIEnv* env, jobject native_class, jlong bitwuzla,
                                                                 jlong bitwuzla_sort) {
    BZLA_TRY_OR_ZERO({
                         const BitwuzlaTerm* result = bitwuzla_mk_fp_pos_inf(BZLA, SORT(bitwuzla_sort));
                         return (jlong) result;
                     })
}

jlong
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkFpNegInf(JNIEnv* env, jobject native_class, jlong bitwuzla,
                                                                 jlong bitwuzla_sort) {
    BZLA_TRY_OR_ZERO({
                         const BitwuzlaTerm* result = bitwuzla_mk_fp_neg_inf(BZLA, SORT(bitwuzla_sort));
                         return (jlong) result;
                     })
}

jlong Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkFpNan(JNIEnv* env, jobject native_class, jlong bitwuzla,
                                                                    jlong bitwuzla_sort) {
    BZLA_TRY_OR_ZERO({
                         const BitwuzlaTerm* result = bitwuzla_mk_fp_nan(BZLA, SORT(bitwuzla_sort));
                         return (jlong) result;
                     })
}

jlong Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkBvValue(JNIEnv* env, jobject native_class, jlong bitwuzla,
                                                                      jlong bitwuzla_sort, jstring value, jint base) {
    BZLA_TRY_OR_ZERO({
                         GET_STRING(native_value, value);
                         const BitwuzlaTerm* result = bitwuzla_mk_bv_value(
                                 BZLA, SORT(bitwuzla_sort), native_value, BitwuzlaBVBase(base)
                         );
                         return (jlong) result;
                     })
}

jlong
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkBvValueUint64(JNIEnv* env, jobject native_class, jlong bitwuzla,
                                                                      jlong bitwuzla_sort, jlong value) {
    BZLA_TRY_OR_ZERO({
                         const BitwuzlaTerm* result = bitwuzla_mk_bv_value_uint64(BZLA, SORT(bitwuzla_sort), value);
                         return (jlong) result;
                     })
}

jlong Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkTerm1(JNIEnv* env, jobject native_class, jlong bitwuzla,
                                                                    jint kind, jlong arg) {
    BZLA_TRY_OR_ZERO({
                         const BitwuzlaTerm* result = bitwuzla_mk_term1(BZLA, BitwuzlaKind(kind), TERM(arg));
                         return (jlong) result;
                     })
}

jlong Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkTerm2(JNIEnv* env, jobject native_class, jlong bitwuzla,
                                                                    jint kind, jlong arg0, jlong arg1) {
    BZLA_TRY_OR_ZERO({
                         const BitwuzlaTerm* result = bitwuzla_mk_term2(
                                 BZLA, BitwuzlaKind(kind), TERM(arg0), TERM(arg1)
                         );
                         return (jlong) result;
                     })
}

jlong Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkTerm3(JNIEnv* env, jobject native_class, jlong bitwuzla,
                                                                    jint kind, jlong arg0, jlong arg1, jlong arg2) {
    BZLA_TRY_OR_ZERO({
                         const BitwuzlaTerm* result = bitwuzla_mk_term3(
                                 BZLA, BitwuzlaKind(kind), TERM(arg0), TERM(arg1), TERM(arg2)
                         );
                         return (jlong) result;
                     })
}

jlong Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkTerm(JNIEnv* env, jobject native_class, jlong bitwuzla,
                                                                   jint kind, jlongArray args) {
    BZLA_TRY_OR_ZERO({
                         GET_PTR_ARRAY(BitwuzlaTerm const*, args_ptr, args);
                         jsize len = env->GetArrayLength(args);
                         const BitwuzlaTerm* result = bitwuzla_mk_term(BZLA, BitwuzlaKind(kind), len, args_ptr);
                         return (jlong) result;
                     })
}

jlong
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkTerm1Indexed1(JNIEnv* env, jobject native_class,
                                                                      jlong bitwuzla, jint kind, jlong term, jint idx) {
    BZLA_TRY_OR_ZERO({
                         const BitwuzlaTerm* result = bitwuzla_mk_term1_indexed1(
                                 BZLA, BitwuzlaKind(kind), TERM(term), idx
                         );
                         return (jlong) result;
                     })
}

JNIEXPORT jlong JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkTerm1Indexed2(JNIEnv* env, jobject native_class, jlong bitwuzla,
                                                                      jint kind, jlong term, jint idx0, jint idx1) {
    BZLA_TRY_OR_ZERO({
                         const BitwuzlaTerm* result = bitwuzla_mk_term1_indexed2(
                                 BZLA, BitwuzlaKind(kind), TERM(term), idx0, idx1
                         );
                         return (jlong) result;
                     })
}

JNIEXPORT jlong JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkTerm2Indexed1(JNIEnv* env, jobject native_class, jlong bitwuzla,
                                                                      jint kind, jlong term0, jlong term1, jint idx0) {
    BZLA_TRY_OR_ZERO({
                         const BitwuzlaTerm* result = bitwuzla_mk_term2_indexed1(
                                 BZLA, BitwuzlaKind(kind), TERM(term0), TERM(term1), idx0
                         );
                         return (jlong) result;
                     })
}

JNIEXPORT jlong JNICALL
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkTerm2Indexed2(JNIEnv* env, jobject native_class, jlong bitwuzla,
                                                                      jint kind, jlong term0, jlong term1, jint idx0,
                                                                      jint idx1) {
    BZLA_TRY_OR_ZERO({
                         const BitwuzlaTerm* result = bitwuzla_mk_term2_indexed2(
                                 BZLA, BitwuzlaKind(kind), TERM(term0), TERM(term1), idx0, idx1
                         );
                         return (jlong) result;
                     })
}

jlong Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkConst(JNIEnv* env, jobject native_class, jlong bitwuzla,
                                                                    jlong bitwuzla_sort, jstring symbol) {
    BZLA_TRY_OR_ZERO({
                         GET_STRING(native_symbol, symbol);
                         const BitwuzlaTerm* result = bitwuzla_mk_const(BZLA, SORT(bitwuzla_sort), native_symbol);
                         return (jlong) result;
                     })
}

jlong
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkConstArray(JNIEnv* env, jobject native_class, jlong bitwuzla,
                                                                   jlong bitwuzla_sort, jlong value) {
    BZLA_TRY_OR_ZERO({
                         const BitwuzlaTerm* result = bitwuzla_mk_const_array(BZLA, SORT(bitwuzla_sort), TERM(value));
                         return (jlong) result;
                     })
}

jlong Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkVar(JNIEnv* env, jobject native_class, jlong bitwuzla,
                                                                  jlong bitwuzla_sort, jstring symbol) {
    BZLA_TRY_OR_ZERO({
                         GET_STRING(native_symbol, symbol);
                         const BitwuzlaTerm* result = bitwuzla_mk_var(BZLA, SORT(bitwuzla_sort), native_symbol);
                         return (jlong) result;
                     })
}

void Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaPush(JNIEnv* env, jobject native_class, jlong bitwuzla,
                                                                jint n_levels) {
    BZLA_TRY_VOID({
                      bitwuzla_push(BZLA, n_levels);
                  })
}

void Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaPop(JNIEnv* env, jobject native_class, jlong bitwuzla,
                                                               jint n_levels) {
    BZLA_TRY_VOID({
                      bitwuzla_pop(BZLA, n_levels);
                  })
}

void Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaAssert(JNIEnv* env, jobject native_class, jlong bitwuzla,
                                                                  jlong term) {
    BZLA_TRY_VOID({
                      bitwuzla_assert(BZLA, TERM(term));
                  })
}

void Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaAssume(JNIEnv* env, jobject native_class, jlong bitwuzla,
                                                                  jlong term) {
    BZLA_TRY_VOID({
                      bitwuzla_assume(BZLA, TERM(term));
                  })
}

jlongArray Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaGetUnsatAssumptions(JNIEnv* env, jobject native_class,
                                                                                     jlong bitwuzla) {
    BZLA_TRY_OR_NULL({
                         size_t len = 0;
                         const BitwuzlaTerm** array = bitwuzla_get_unsat_assumptions(BZLA, &len);
                         jlongArray result = create_ptr_array(env, len);
                         set_ptr_array(env, result, array, len);
                         return result;
                     })
}

jlongArray
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaGetUnsatCore(JNIEnv* env, jobject native_class, jlong bitwuzla) {
    BZLA_TRY_OR_NULL({
                         size_t len = 0;
                         const BitwuzlaTerm** array = bitwuzla_get_unsat_core(BZLA, &len);
                         jlongArray result = create_ptr_array(env, len);
                         set_ptr_array(env, result, array, len);
                         return result;
                     })
}

void Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaFixateAssumptions(JNIEnv* env, jobject native_class,
                                                                             jlong bitwuzla) {
    BZLA_TRY_VOID({
                      bitwuzla_fixate_assumptions(BZLA);
                  })
}

void Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaResetAssumptions(JNIEnv* env, jobject native_class,
                                                                            jlong bitwuzla) {
    BZLA_TRY_VOID({
                      bitwuzla_reset_assumptions(BZLA);
                  })
}

jint Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaCheckSat(JNIEnv* env, jobject native_class, jlong bitwuzla) {
    BZLA_TRY_OR_ZERO({
                         BitwuzlaResult result = bitwuzla_check_sat(BZLA);
                         return result;
                     })
}

jlong Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaGetValue(JNIEnv* env, jobject native_class, jlong bitwuzla,
                                                                     jlong bitwuzla_term) {
    BZLA_TRY_OR_ZERO({
                         return (jlong) bitwuzla_get_value(BZLA, TERM(bitwuzla_term));
                     })
}

jstring
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaGetBvValue(JNIEnv* env, jobject native_class, jlong bitwuzla,
                                                                 jlong bitwuzla_term) {
    BZLA_TRY_OR_NULL({
                         const char* c = bitwuzla_get_bv_value(BZLA, TERM(bitwuzla_term));
                         jstring result = env->NewStringUTF(c);
                         return result;
                     })
}

jobject
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaGetFpValue(JNIEnv* env, jobject native_class, jlong bitwuzla,
                                                                 jlong bitwuzla_term) {
    BZLA_TRY_OR_NULL({
                         const char* sign;
                         const char* exponent;
                         const char* significand;
                         bitwuzla_get_fp_value(BZLA, TERM(bitwuzla_term), &sign, &exponent, &significand);

                         jclass clazz = env->FindClass("org/ksmt/solver/bitwuzla/bindings/FpValue");
                         jmethodID constructor = env->GetMethodID(clazz, "<init>", "()V");
                         jfieldID sign_id = env->GetFieldID(clazz, "sign", "Ljava/lang/String;");
                         jfieldID exponent_id = env->GetFieldID(clazz, "exponent", "Ljava/lang/String;");
                         jfieldID significand_id = env->GetFieldID(clazz, "significand", "Ljava/lang/String;");

                         jstring sign_string = env->NewStringUTF(sign);
                         jstring exponent_string = env->NewStringUTF(exponent);
                         jstring significandn_string = env->NewStringUTF(significand);

                         jobject result_object = env->NewObject(clazz, constructor);
                         env->SetObjectField(result_object, sign_id, sign_string);
                         env->SetObjectField(result_object, exponent_id, exponent_string);
                         env->SetObjectField(result_object, significand_id, significandn_string);

                         return result_object;
                     })
}

jobject
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaGetArrayValue(JNIEnv* env, jobject native_class, jlong bitwuzla,
                                                                    jlong bitwuzla_term) {
    BZLA_TRY_OR_NULL({
                         size_t len = 0;
                         BitwuzlaTerm const** indices_ptr;
                         BitwuzlaTerm const** values_ptr;
                         BitwuzlaTerm const* default_value_ptr;
                         bitwuzla_get_array_value(
                                 BZLA, TERM(bitwuzla_term), &indices_ptr, &values_ptr, &len, &default_value_ptr
                         );

                         jclass clazz = env->FindClass("org/ksmt/solver/bitwuzla/bindings/ArrayValue");
                         jmethodID constructor = env->GetMethodID(clazz, "<init>", "()V");
                         jfieldID size_id = env->GetFieldID(clazz, "size", "I");
                         jfieldID indices_id = env->GetFieldID(clazz, "indices", "[J");
                         jfieldID values_id = env->GetFieldID(clazz, "values", "[J");
                         jfieldID defaultValue_id = env->GetFieldID(clazz, "defaultValue", "J");

                         jlongArray indices_array = create_ptr_array(env, len);
                         set_ptr_array(env, indices_array, indices_ptr, len);
                         jlongArray values_array = create_ptr_array(env, len);
                         set_ptr_array(env, values_array, values_ptr, len);

                         jobject result_object = env->NewObject(clazz, constructor);
                         env->SetIntField(result_object, size_id, (jint) len);
                         env->SetObjectField(result_object, indices_id, indices_array);
                         env->SetObjectField(result_object, values_id, values_array);
                         env->SetLongField(result_object, defaultValue_id, (jlong) default_value_ptr);

                         return result_object;
                     })
}

jobject
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaGetFunValue(JNIEnv* env, jobject native_class, jlong bitwuzla,
                                                                  jlong bitwuzla_term) {
    BZLA_TRY_OR_NULL({
                         size_t size = 0;
                         size_t arity = 0;
                         BitwuzlaTerm const*** args_ptr;
                         BitwuzlaTerm const** values_ptr;
                         bitwuzla_get_fun_value(BZLA, TERM(bitwuzla_term), &args_ptr, &arity, &values_ptr, &size);

                         jclass clazz = env->FindClass("org/ksmt/solver/bitwuzla/bindings/FunValue");
                         jmethodID constructor = env->GetMethodID(clazz, "<init>", "()V");
                         jfieldID size_id = env->GetFieldID(clazz, "size", "I");
                         jfieldID arity_id = env->GetFieldID(clazz, "arity", "I");
                         jfieldID args_id = env->GetFieldID(clazz, "args", "[[J");
                         jfieldID values_id = env->GetFieldID(clazz, "values", "[J");

                         jclass clazz_long_array = env->FindClass("[J");
                         jobjectArray argsArray = env->NewObjectArray((jsize) size, clazz_long_array, nullptr);
                         for (unsigned int i = 0; i < size; i++) {
                             jlongArray args_i = create_ptr_array(env, arity);
                             set_ptr_array(env, args_i, args_ptr[i], arity);
                             env->SetObjectArrayElement(argsArray, (jsize) i, args_i);
                         }
                         jlongArray valuesArray = create_ptr_array(env, size);
                         set_ptr_array(env, valuesArray, values_ptr, size);

                         jobject result_object = env->NewObject(clazz, constructor);
                         env->SetIntField(result_object, size_id, (jint) size);
                         env->SetIntField(result_object, arity_id, (jint) arity);
                         env->SetObjectField(result_object, args_id, argsArray);
                         env->SetObjectField(result_object, values_id, valuesArray);
                         return result_object;
                     })
}

jlong
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaSortHash(JNIEnv* env, jobject native_class, jlong bitwuzla_sort) {
    BZLA_TRY_OR_ZERO({
                         return bitwuzla_sort_hash(SORT(bitwuzla_sort));
                     })
}

jint Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermGetKind(JNIEnv* env, jobject native_class,
                                                                       jlong bitwuzla_term) {
    BZLA_TRY_OR_ZERO({
                         return bitwuzla_term_get_kind(TERM(bitwuzla_term));
                     })
}

jlongArray Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermGetChildren(JNIEnv* env, jobject native_class,
                                                                                 jlong bitwuzla_term) {
    BZLA_TRY_OR_NULL({
                         size_t len = 0;
                         const BitwuzlaTerm** array = bitwuzla_term_get_children(TERM(bitwuzla_term), &len);
                         jlongArray result = create_ptr_array(env, len);
                         set_ptr_array(env, result, array, len);
                         return result;
                     })
}

jintArray Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermGetIndices(JNIEnv* env, jobject native_class,
                                                                               jlong bitwuzla_term) {
    BZLA_TRY_OR_NULL({
                         size_t len = 0;
                         uint32_t* array = bitwuzla_term_get_indices(TERM(bitwuzla_term), &len);
                         jintArray result = env->NewIntArray((jsize) len);
                         env->SetIntArrayRegion(result, 0, len, (jint*) array);
                         return result;
                     })
}

jint Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaSortBvGetSize(JNIEnv* env, jobject native_class,
                                                                         jlong bitwuzla_sort) {
    BZLA_TRY_OR_ZERO({
                         return bitwuzla_sort_bv_get_size(SORT(bitwuzla_sort));
                     })
}

jint Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaSortFpGetExpSize(JNIEnv* env, jobject native_class,
                                                                            jlong bitwuzla_sort) {
    BZLA_TRY_OR_ZERO({
                         return bitwuzla_sort_fp_get_exp_size(SORT(bitwuzla_sort));
                     })
}

jint Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaSortFpGetSigSize(JNIEnv* env, jobject native_class,
                                                                            jlong bitwuzla_sort) {
    BZLA_TRY_OR_ZERO({
                         return bitwuzla_sort_fp_get_sig_size(SORT(bitwuzla_sort));
                     })
}

jlong Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaSortArrayGetIndex(JNIEnv* env, jobject native_class,
                                                                              jlong bitwuzla_sort) {
    BZLA_TRY_OR_ZERO({
                         const BitwuzlaSort* bw_s = bitwuzla_sort_array_get_index(SORT(bitwuzla_sort));
                         return (jlong) bw_s;
                     })
}

jlong Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaSortArrayGetElement(JNIEnv* env, jobject native_class,
                                                                                jlong bitwuzla_sort) {
    BZLA_TRY_OR_ZERO({
                         const BitwuzlaSort* bw_s = bitwuzla_sort_array_get_element(SORT(bitwuzla_sort));
                         return (jlong) bw_s;
                     })
}

jboolean Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaSortIsEqual(JNIEnv* env, jobject native_class,
                                                                           jlong bitwuzla_sort_1,
                                                                           jlong bitwuzla_sort_2) {
    BZLA_TRY_OR_ZERO({
                         bool result = bitwuzla_sort_is_equal(SORT(bitwuzla_sort_1), SORT(bitwuzla_sort_2));
                         return (jboolean) result;
                     })
}

jboolean Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaSortIsArray(JNIEnv* env, jobject native_class,
                                                                           jlong bitwuzla_sort) {
    BZLA_TRY_OR_ZERO({
                         bool result = bitwuzla_sort_is_array(SORT(bitwuzla_sort));
                         return (jboolean) result;
                     })
}

jboolean
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaSortIsBv(JNIEnv* env, jobject native_class, jlong bitwuzla_sort) {
    BZLA_TRY_OR_ZERO({
                         bool result = bitwuzla_sort_is_bv(SORT(bitwuzla_sort));
                         return (jboolean) result;
                     })
}

jboolean
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaSortIsFp(JNIEnv* env, jobject native_class, jlong bitwuzla_sort) {
    BZLA_TRY_OR_ZERO({
                         bool result = bitwuzla_sort_is_fp(SORT(bitwuzla_sort));
                         return (jboolean) result;
                     })
}

jboolean Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaSortIsFun(JNIEnv* env, jobject native_class,
                                                                         jlong bitwuzla_sort) {
    BZLA_TRY_OR_ZERO({
                         bool result = bitwuzla_sort_is_fun(SORT(bitwuzla_sort));
                         return (jboolean) result;
                     })
}

jboolean
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaSortIsRm(JNIEnv* env, jobject native_class, jlong bitwuzla_sort) {
    BZLA_TRY_OR_ZERO({
                         bool result = bitwuzla_sort_is_rm(SORT(bitwuzla_sort));
                         return (jboolean) result;
                     })
}

jlong
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermHash(JNIEnv* env, jobject native_class, jlong bitwuzla_term) {
    BZLA_TRY_OR_ZERO({
                         return bitwuzla_term_hash(TERM(bitwuzla_term));
                     })
}

jboolean Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsIndexed(JNIEnv* env, jobject native_class,
                                                                             jlong bitwuzla_term) {
    BZLA_TRY_OR_ZERO({
                         bool result = bitwuzla_term_is_indexed(TERM(bitwuzla_term));
                         return (jboolean) result;
                     })
}

jlong Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermGetBitwuzla(JNIEnv* env, jobject native_class,
                                                                            jlong bitwuzla_term) {
    BZLA_TRY_OR_ZERO({
                         Bitwuzla* bw = bitwuzla_term_get_bitwuzla(TERM(bitwuzla_term));
                         return (jlong) bw;
                     })
}

jlong Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermGetSort(JNIEnv* env, jobject native_class,
                                                                        jlong bitwuzla_term) {
    BZLA_TRY_OR_ZERO({
                         const BitwuzlaSort* bw_sort = bitwuzla_term_get_sort(TERM(bitwuzla_term));
                         return (jlong) bw_sort;
                     })
}

jlong Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermArrayGetIndexSort(JNIEnv* env, jobject native_class,
                                                                                  jlong bitwuzla_term) {
    BZLA_TRY_OR_ZERO({
                         const BitwuzlaSort* bw_sort = bitwuzla_term_array_get_index_sort(TERM(bitwuzla_term));
                         return (jlong) bw_sort;
                     })
}

jlong Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermArrayGetElementSort(JNIEnv* env, jobject native_class,
                                                                                    jlong bitwuzla_term) {
    BZLA_TRY_OR_ZERO({
                         const BitwuzlaSort* bw_sort = bitwuzla_term_array_get_element_sort(TERM(bitwuzla_term));
                         return (jlong) bw_sort;
                     })
}

jlong Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermFunGetCodomainSort(JNIEnv* env, jobject native_class,
                                                                                   jlong bitwuzla_term) {
    BZLA_TRY_OR_ZERO({
                         const BitwuzlaSort* bw_sort = bitwuzla_term_fun_get_codomain_sort(TERM(bitwuzla_term));
                         return (jlong) bw_sort;
                     })
}

jint Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermBvGetSize(JNIEnv* env, jobject native_class,
                                                                         jlong bitwuzla_term) {
    BZLA_TRY_OR_ZERO({
                         uint32_t result = bitwuzla_term_bv_get_size(TERM(bitwuzla_term));
                         return (jint) result;
                     })
}

jint Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermFpGetExpSize(JNIEnv* env, jobject native_class,
                                                                            jlong bitwuzla_term) {
    BZLA_TRY_OR_ZERO({
                         uint32_t result = bitwuzla_term_fp_get_exp_size(TERM(bitwuzla_term));
                         return (jint) result;
                     })
}

jint Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermFpGetSigSize(JNIEnv* env, jobject native_class,
                                                                            jlong bitwuzla_term) {
    BZLA_TRY_OR_ZERO({
                         uint32_t result = bitwuzla_term_fp_get_sig_size(TERM(bitwuzla_term));
                         return (jint) result;
                     })
}

jint Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermFunGetArity(JNIEnv* env, jobject native_class,
                                                                           jlong bitwuzla_term) {
    BZLA_TRY_OR_ZERO({
                         uint32_t result = bitwuzla_term_fun_get_arity(TERM(bitwuzla_term));
                         return (jint) result;
                     })
}

jstring Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermGetSymbol(JNIEnv* env, jobject native_class,
                                                                            jlong bitwuzla_term) {
    BZLA_TRY_OR_NULL({
                         const char* c = bitwuzla_term_get_symbol(TERM(bitwuzla_term));
                         jstring result = env->NewStringUTF(c);
                         return result;
                     })
}

void Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermSetSymbol(JNIEnv* env, jobject native_class,
                                                                         jlong bitwuzla_term, jstring symbol) {
    BZLA_TRY_VOID({
                      GET_STRING(native_symbol, symbol);
                      bitwuzla_term_set_symbol(TERM(bitwuzla_term), native_symbol);
                  })
}

jboolean Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsEqualSort(JNIEnv* env, jobject native_class,
                                                                               jlong bitwuzla_term_1,
                                                                               jlong bitwuzla_term_2) {
    BZLA_TRY_OR_ZERO({
                         bool result = bitwuzla_term_is_equal_sort(TERM(bitwuzla_term_1), TERM(bitwuzla_term_2));
                         return (jboolean) result;
                     })
}

jboolean Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsArray(JNIEnv* env, jobject native_class,
                                                                           jlong bitwuzla_term) {
    BZLA_TRY_OR_ZERO({
                         bool result = bitwuzla_term_is_array(TERM(bitwuzla_term));
                         return (jboolean) result;
                     })
}

jboolean Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsConst(JNIEnv* env, jobject native_class,
                                                                           jlong bitwuzla_term) {
    BZLA_TRY_OR_ZERO({
                         bool result = bitwuzla_term_is_const(TERM(bitwuzla_term));
                         return (jboolean) result;
                     })
}

jboolean Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsFun(JNIEnv* env, jobject native_class,
                                                                         jlong bitwuzla_term) {
    BZLA_TRY_OR_ZERO({
                         bool result = bitwuzla_term_is_fun(TERM(bitwuzla_term));
                         return (jboolean) result;
                     })
}

jboolean Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsVar(JNIEnv* env, jobject native_class,
                                                                         jlong bitwuzla_term) {
    BZLA_TRY_OR_ZERO({
                         bool result = bitwuzla_term_is_var(TERM(bitwuzla_term));
                         return (jboolean) result;
                     })
}

jboolean Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsBoundVar(JNIEnv* env, jobject native_class,
                                                                              jlong bitwuzla_term) {
    BZLA_TRY_OR_ZERO({
                         bool result = bitwuzla_term_is_bound_var(TERM(bitwuzla_term));
                         return (jboolean) result;
                     })
}

jboolean Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsValue(JNIEnv* env, jobject native_class,
                                                                           jlong bitwuzla_term) {
    BZLA_TRY_OR_ZERO({
                         bool result = bitwuzla_term_is_value(TERM(bitwuzla_term));
                         return (jboolean) result;
                     })
}

jboolean Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsBvValue(JNIEnv* env, jobject native_class,
                                                                             jlong bitwuzla_term) {
    BZLA_TRY_OR_ZERO({
                         bool result = bitwuzla_term_is_bv_value(TERM(bitwuzla_term));
                         return (jboolean) result;
                     })
}

jboolean Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsFpValue(JNIEnv* env, jobject native_class,
                                                                             jlong bitwuzla_term) {
    BZLA_TRY_OR_ZERO({
                         bool result = bitwuzla_term_is_fp_value(TERM(bitwuzla_term));
                         return (jboolean) result;
                     })
}

jboolean Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsRmValue(JNIEnv* env, jobject native_class,
                                                                             jlong bitwuzla_term) {
    BZLA_TRY_OR_ZERO({
                         bool result = bitwuzla_term_is_rm_value(TERM(bitwuzla_term));
                         return (jboolean) result;
                     })
}

jboolean
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsBv(JNIEnv* env, jobject native_class, jlong bitwuzla_term) {
    BZLA_TRY_OR_ZERO({
                         bool result = bitwuzla_term_is_bv(TERM(bitwuzla_term));
                         return (jboolean) result;
                     })
}

jboolean
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsFp(JNIEnv* env, jobject native_class, jlong bitwuzla_term) {
    BZLA_TRY_OR_ZERO({
                         bool result = bitwuzla_term_is_fp(TERM(bitwuzla_term));
                         return (jboolean) result;
                     })
}

jboolean
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsRm(JNIEnv* env, jobject native_class, jlong bitwuzla_term) {
    BZLA_TRY_OR_ZERO({
                         bool result = bitwuzla_term_is_rm(TERM(bitwuzla_term));
                         return (jboolean) result;
                     })
}

jboolean Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsBvValueZero(JNIEnv* env, jobject native_class,
                                                                                 jlong bitwuzla_term) {
    BZLA_TRY_OR_ZERO({
                         bool result = bitwuzla_term_is_bv_value_zero(TERM(bitwuzla_term));
                         return (jboolean) result;
                     })
}

jboolean Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsBvValueOne(JNIEnv* env, jobject native_class,
                                                                                jlong bitwuzla_term) {
    BZLA_TRY_OR_ZERO({
                         bool result = bitwuzla_term_is_bv_value_one(TERM(bitwuzla_term));
                         return (jboolean) result;
                     })
}

jboolean Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsBvValueOnes(JNIEnv* env, jobject native_class,
                                                                                 jlong bitwuzla_term) {
    BZLA_TRY_OR_ZERO({
                         bool result = bitwuzla_term_is_bv_value_ones(TERM(bitwuzla_term));
                         return (jboolean) result;
                     })
}

jboolean Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsBvValueMinSigned(JNIEnv* env, jobject native_class,
                                                                                      jlong bitwuzla_term) {
    BZLA_TRY_OR_ZERO({
                         bool result = bitwuzla_term_is_bv_value_min_signed(TERM(bitwuzla_term));
                         return (jboolean) result;
                     })
}

jboolean Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsBvValueMaxSigned(JNIEnv* env, jobject native_class,
                                                                                      jlong bitwuzla_term) {
    BZLA_TRY_OR_ZERO({
                         bool result = bitwuzla_term_is_bv_value_max_signed(TERM(bitwuzla_term));
                         return (jboolean) result;
                     })
}

jboolean Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsFpValuePosZero(JNIEnv* env, jobject native_class,
                                                                                    jlong bitwuzla_term) {
    BZLA_TRY_OR_ZERO({
                         bool result = bitwuzla_term_is_fp_value_pos_zero(TERM(bitwuzla_term));
                         return (jboolean) result;
                     })
}

jboolean Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsFpValueNegZero(JNIEnv* env, jobject native_class,
                                                                                    jlong bitwuzla_term) {
    BZLA_TRY_OR_ZERO({
                         bool result = bitwuzla_term_is_fp_value_neg_zero(TERM(bitwuzla_term));
                         return (jboolean) result;
                     })
}

jboolean Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsFpValuePosInf(JNIEnv* env, jobject native_class,
                                                                                   jlong bitwuzla_term) {
    BZLA_TRY_OR_ZERO({
                         bool result = bitwuzla_term_is_fp_value_pos_inf(TERM(bitwuzla_term));
                         return (jboolean) result;
                     })
}

jboolean Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsFpValueNegInf(JNIEnv* env, jobject native_class,
                                                                                   jlong bitwuzla_term) {
    BZLA_TRY_OR_ZERO({
                         bool result = bitwuzla_term_is_fp_value_neg_inf(TERM(bitwuzla_term));
                         return (jboolean) result;
                     })
}

jboolean Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsFpValueNan(JNIEnv* env, jobject native_class,
                                                                                jlong bitwuzla_term) {
    BZLA_TRY_OR_ZERO({
                         bool result = bitwuzla_term_is_fp_value_nan(TERM(bitwuzla_term));
                         return (jboolean) result;
                     })
}

jboolean Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsRmValueRna(JNIEnv* env, jobject native_class,
                                                                                jlong bitwuzla_term) {
    BZLA_TRY_OR_ZERO({
                         bool result = bitwuzla_term_is_rm_value_rna(TERM(bitwuzla_term));
                         return (jboolean) result;
                     })
}

jboolean Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsRmValueRne(JNIEnv* env, jobject native_class,
                                                                                jlong bitwuzla_term) {
    BZLA_TRY_OR_ZERO({
                         bool result = bitwuzla_term_is_rm_value_rne(TERM(bitwuzla_term));
                         return (jboolean) result;
                     })
}

jboolean Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsRmValueRtn(JNIEnv* env, jobject native_class,
                                                                                jlong bitwuzla_term) {
    BZLA_TRY_OR_ZERO({
                         bool result = bitwuzla_term_is_rm_value_rtn(TERM(bitwuzla_term));
                         return (jboolean) result;
                     })
}

jboolean Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsRmValueRtp(JNIEnv* env, jobject native_class,
                                                                                jlong bitwuzla_term) {
    BZLA_TRY_OR_ZERO({
                         bool result = bitwuzla_term_is_rm_value_rtp(TERM(bitwuzla_term));
                         return (jboolean) result;
                     })
}

jboolean Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsRmValueRtz(JNIEnv* env, jobject native_class,
                                                                                jlong bitwuzla_term) {
    BZLA_TRY_OR_ZERO({
                         bool result = bitwuzla_term_is_rm_value_rtz(TERM(bitwuzla_term));
                         return (jboolean) result;
                     })
}

jboolean Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsConstArray(JNIEnv* env, jobject native_class,
                                                                                jlong bitwuzla_term) {
    BZLA_TRY_OR_ZERO({
                         bool result = bitwuzla_term_is_const_array(TERM(bitwuzla_term));
                         return (jboolean) result;
                     })
}

jlong Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaBvConstNodeGetBits(JNIEnv* env, jobject native_class,
                                                                               jlong bitwuzla_term) {
    BZLA_TRY_OR_ZERO({
                         const BzlaBitVector* bw_vector = bitwuzla_extension_node_bv_const_get_bits(
                                 TERM(bitwuzla_term));
                         return (jlong) bw_vector;
                     })
}

jint Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaBvBitsGetWidth(JNIEnv* env, jobject native_class,
                                                                          jlong bitwuzla_bit_vector) {
    BZLA_TRY_OR_ZERO({
                         const BzlaBitVector* bw_vector = (BzlaBitVector*) bitwuzla_bit_vector;
                         uint32_t i = bitwuzla_extension_bv_get_width(bw_vector);
                         return i;
                     })
}

jlong Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaBvBitsToUInt64(JNIEnv* env, jobject native_class,
                                                                           jlong bitwuzla_bit_vector) {
    BZLA_TRY_OR_ZERO({
                         const BzlaBitVector* bw_vector = (BzlaBitVector*) bitwuzla_bit_vector;
                         uint64_t i = bitwuzla_extension_bv_to_uint64(bw_vector);
                         return i;
                     })
}

jlong Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkFpValue(JNIEnv* env, jobject native_class, jlong bitwuzla,
                                                                      jlong bitwuzla_bvSign, jlong bitwuzla_bvExponent,
                                                                      jlong bitwuzla_bvSignificand) {
    BZLA_TRY_OR_ZERO({
                         const BitwuzlaTerm* bv_sign = (BitwuzlaTerm*) bitwuzla_bvSign;
                         const BitwuzlaTerm* bv_exponent = (BitwuzlaTerm*) bitwuzla_bvExponent;
                         const BitwuzlaTerm* bv_significand = (BitwuzlaTerm*) bitwuzla_bvSignificand;
                         return (jlong) bitwuzla_mk_fp_value(BZLA, bv_sign, bv_exponent, bv_significand);
                     })
}

jlong Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkRmValue(JNIEnv* env, jobject native_class, jlong bitwuzla,
                                                                      jint rm) {
    BZLA_TRY_OR_ZERO({
                         return (jlong) bitwuzla_mk_rm_value(BZLA, (BitwuzlaRoundingMode) rm);
                     })
}

jint Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaBvBitsGetBit(JNIEnv* env, jobject native_class,
                                                                        jlong bitwuzla_bit_vector, jint pos) {
    BZLA_TRY_OR_ZERO({
                         const BzlaBitVector* bw_vector = (BzlaBitVector*) bitwuzla_bit_vector;
                         uint32_t i = bitwuzla_extension_bv_get_bit(bw_vector, pos);
                         return i;
                     })
}

jlong Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaFpConstNodeGetBits(JNIEnv* env, jobject native_class,
                                                                               jlong bitwuzla, jlong bitwuzla_term) {
    BZLA_TRY_OR_ZERO({
                         auto bzla = bitwuzla_extension_get_bzla(BZLA);
                         auto fp = bitwuzla_extension_node_fp_const_get_fp(TERM(bitwuzla_term));
                         const BzlaBitVector* bw_vector = bitwuzla_extension_fp_as_bv(bzla, fp);
                         return (jlong) bw_vector;
                     })
}

void Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaPrintModel(JNIEnv* env, jobject native_class, jlong bitwuzla,
                                                                      jstring format, jstring file_path) {
    BZLA_TRY_VOID({
                      GET_STRING(print_format, format);
                      GET_STRING(path, file_path);

                      auto f = fopen(path, "w");
                      bitwuzla_print_model(BZLA, print_format, f);
                      fclose(f);
                  })
}

void
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaDumpFormula(JNIEnv* env, jobject native_class, jlong bitwuzla,
                                                                  jstring format, jstring file_path) {
    BZLA_TRY_VOID({
                      GET_STRING(print_format, format);
                      GET_STRING(path, file_path);

                      auto f = fopen(path, "w");
                      bitwuzla_dump_formula(BZLA, print_format, f);
                      fclose(f);
                  })
}

jstring read_file_to_java_str(JNIEnv* env, FILE* file_ptr) {
    std::vector<char> result;

    int c;
    while ((c = std::fgetc(file_ptr)) != EOF) {
        result.push_back((char) c);
    }

    std::string str(result.data(), result.size());
    return env->NewStringUTF(str.c_str());
}

jstring Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaSortDump(JNIEnv* env, jobject native_class, jlong sort,
                                                                       jstring format) {
    BZLA_TRY_OR_NULL({
                         GET_STRING(print_format, format);
                         auto&& f = std::shared_ptr<FILE>(tmpfile(), fclose);

                         bitwuzla_sort_dump(SORT(sort), print_format, f.get());
                         rewind(f.get());
                         return read_file_to_java_str(env, f.get());
                     })
}

jstring Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermDump(JNIEnv* env, jobject native_class, jlong term,
                                                                       jstring format) {
    BZLA_TRY_OR_NULL({
                         GET_STRING(print_format, format);
                         auto&& f = std::shared_ptr<FILE>(tmpfile(), fclose);

                         bitwuzla_term_dump(TERM(term), print_format, f.get());
                         rewind(f.get());
                         return read_file_to_java_str(env, f.get());
                     })
}
