#include <iostream>
#include <atomic>
#include <bitwuzla.h>
#include <memory>
#include <unistd.h>
#include <vector>
#include <chrono>
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

struct JniIntArray {
    JNIEnv* env;
    jintArray jni_array;
    jint* elements;

    JniIntArray(JNIEnv* env, jintArray jni_array) : env(env), jni_array(jni_array) {
        elements = env->GetIntArrayElements(jni_array, nullptr);
    }

    ~JniIntArray() {
        env->ReleaseIntArrayElements(jni_array, elements, JNI_ABORT);
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
#define SORT(s) (BitwuzlaSort*) s

void abort_callback(const char* msg) {
    throw std::runtime_error(msg);
}

enum TerminationState {
    NOT_ACTIVE, ACTIVE, TERMINATED
};

struct BitwuzlaTerminationCallbackState {
    std::chrono::milliseconds time_mark;
    std::atomic_int termination_state;

    void reset() {
        termination_state = TerminationState::NOT_ACTIVE;
    }

    void terminate() {
        int expected_state = TerminationState::ACTIVE;
        int new_state = TerminationState::TERMINATED;
        termination_state.compare_exchange_strong(expected_state, new_state);
    }

    void setup_timeout(uint64_t timeout) {
        auto current_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()
        );
        time_mark = current_time + std::chrono::milliseconds(timeout);
        termination_state = TerminationState::ACTIVE;
    }

    bool terminated() {
        int current_state = termination_state;
        if (current_state == TerminationState::TERMINATED) {
            return true;
        }
        if (current_state != TerminationState::ACTIVE) {
            return false;
        }

        std::chrono::milliseconds current_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()
        );
        if (current_time > time_mark) {
            int expected_state = TerminationState::ACTIVE;
            int new_state = TerminationState::TERMINATED;
            termination_state.compare_exchange_strong(expected_state, new_state);
        }

        int final_state = termination_state;
        return final_state == TerminationState::TERMINATED;
    }
};

int32_t termination_callback(void* state) {
    auto termination_state = reinterpret_cast<BitwuzlaTerminationCallbackState*>(state);
    if (termination_state != nullptr && termination_state->terminated()) {
        return 1;
    }
    return 0;
}

BitwuzlaTerminationCallbackState* get_termination_state(Bitwuzla* bitwuzla) {
    auto state = bitwuzla_get_termination_callback_state(bitwuzla);
    return reinterpret_cast<BitwuzlaTerminationCallbackState*>(state);
}

struct ScopedTimeout {
    BitwuzlaTerminationCallbackState* state;

    ScopedTimeout(BitwuzlaTerminationCallbackState* state, uint64_t timeout) : state(state) {
        state->setup_timeout(timeout);
    }

    ~ScopedTimeout() {
        state->reset();
    }
};

void Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaInit(JNIEnv* env, jobject native_class) {
    bitwuzla_set_abort_callback(abort_callback);
}

jlong Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaNew(JNIEnv* env, jobject native_class) {
    BZLA_TRY_OR_ZERO({
                         Bitwuzla* bzla = bitwuzla_new();

                         auto termination_state = new BitwuzlaTerminationCallbackState();
                         termination_state->reset();
                         bitwuzla_set_termination_callback(bzla, termination_callback, termination_state);

                         return (jlong) bzla;
                     })
}

void Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaDelete(JNIEnv* env, jobject native_class, jlong bitwuzla) {
    BZLA_TRY_VOID({

                      auto termination_state = get_termination_state(BZLA);
                      delete termination_state;

                      bitwuzla_delete(BZLA);
                  })
}

void Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaReset(JNIEnv* env, jobject native_class, jlong bitwuzla) {
    BZLA_TRY_VOID({
                      auto termination_state = get_termination_state(BZLA);
                      delete termination_state;

                      bitwuzla_reset(BZLA);

                      auto new_termination_state = new BitwuzlaTerminationCallbackState();
                      new_termination_state->reset();
                      bitwuzla_set_termination_callback(BZLA, termination_callback, new_termination_state);
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
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkBvValueUint32(JNIEnv* env, jobject native_class, jlong bitwuzla,
                                                                      jlong bitwuzla_sort, jint value) {
    BZLA_TRY_OR_ZERO({
                         // We can't use all 64 bits because of some problems on Windows
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

jlong
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkTerm1Indexed2(JNIEnv* env, jobject native_class, jlong bitwuzla,
                                                                      jint kind, jlong term, jint idx0, jint idx1) {
    BZLA_TRY_OR_ZERO({
                         const BitwuzlaTerm* result = bitwuzla_mk_term1_indexed2(
                                 BZLA, BitwuzlaKind(kind), TERM(term), idx0, idx1
                         );
                         return (jlong) result;
                     })
}

jlong
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkTerm2Indexed1(JNIEnv* env, jobject native_class, jlong bitwuzla,
                                                                      jint kind, jlong term0, jlong term1, jint idx0) {
    BZLA_TRY_OR_ZERO({
                         const BitwuzlaTerm* result = bitwuzla_mk_term2_indexed1(
                                 BZLA, BitwuzlaKind(kind), TERM(term0), TERM(term1), idx0
                         );
                         return (jlong) result;
                     })
}

jlong
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

jlong
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkTermIndexed(JNIEnv* env, jobject native_class, jlong bitwuzla,
                                                                    jint kind, jlongArray args, jintArray idxs) {
    BZLA_TRY_OR_ZERO({
                         GET_PTR_ARRAY(BitwuzlaTerm const*, args_ptr, args);
                         jsize argc = env->GetArrayLength(args);

                         JniIntArray indices_array(env, idxs);
                         auto indices = (uint32_t*) indices_array.elements;
                         jsize idxc = env->GetArrayLength(idxs);

                         const BitwuzlaTerm* result = bitwuzla_mk_term_indexed(
                                 BZLA, BitwuzlaKind(kind), argc, args_ptr, idxc, indices
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

jboolean
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaIsUnsatAssumption(JNIEnv* env, jobject native_class,
                                                                        jlong bitwuzla, jlong term) {
    BZLA_TRY_OR_ZERO({
                         bool result = bitwuzla_is_unsat_assumption(BZLA, TERM(term));
                         if (result) {
                             return JNI_TRUE;
                         } else {
                             return JNI_FALSE;
                         }
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

jint
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaSimplify(JNIEnv* env, jobject native_class, jlong bitwuzla) {
    BZLA_TRY_OR_ZERO({
                         BitwuzlaResult result = bitwuzla_simplify(BZLA);
                         return result;
                     })
}

jint Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaCheckSat(JNIEnv* env, jobject native_class, jlong bitwuzla) {
    BZLA_TRY_OR_ZERO({
                         BitwuzlaResult result = bitwuzla_check_sat(BZLA);
                         return result;
                     })
}

jint
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaCheckSatTimeout(JNIEnv* env, jobject native_class, jlong bitwuzla,
                                                                      jlong timeout) {
    BZLA_TRY_OR_ZERO({
                         auto termination_state = get_termination_state(BZLA);
                         ScopedTimeout _timeout(termination_state, timeout);

                         BitwuzlaResult result = bitwuzla_check_sat(BZLA);
                         return result;
                     })
}

void
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaForceTerminate(JNIEnv* env, jobject native_class,
                                                                     jlong bitwuzla) {
    BZLA_TRY_VOID({
                      auto termination_state = get_termination_state(BZLA);
                      termination_state->terminate();
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

jstring
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaGetRmValue(JNIEnv* env, jobject native_class, jlong bitwuzla,
                                                                 jlong bitwuzla_term) {
    BZLA_TRY_OR_NULL({
                         const char* result = bitwuzla_get_rm_value(BZLA, TERM(bitwuzla_term));
                         jstring result_str = env->NewStringUTF(result);

                         return result_str;
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
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaSubstituteTerm(JNIEnv* env, jobject native_class, jlong bitwuzla,
                                                                     jlong bitwuzla_term, jlongArray map_keys,
                                                                     jlongArray map_values) {
    BZLA_TRY_OR_ZERO({
                         GET_PTR_ARRAY(BitwuzlaTerm const*, keys_ptr, map_keys);
                         GET_PTR_ARRAY(BitwuzlaTerm const*, values_ptr, map_values);
                         jsize num_keys = env->GetArrayLength(map_keys);

                         const BitwuzlaTerm* result = bitwuzla_substitute_term(
                                 BZLA, TERM(bitwuzla_term), num_keys, keys_ptr, values_ptr
                         );

                         return (jlong) result;
                     })
}

jlongArray
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaSubstituteTerms(JNIEnv* env, jobject native_class, jlong bitwuzla,
                                                                      jlongArray terms, jlongArray map_keys,
                                                                      jlongArray map_values) {
    BZLA_TRY_OR_NULL({
                         GET_PTR_ARRAY(BitwuzlaTerm const*, terms_ptr, terms);
                         jsize num_terms = env->GetArrayLength(terms);

                         GET_PTR_ARRAY(BitwuzlaTerm const*, keys_ptr, map_keys);
                         GET_PTR_ARRAY(BitwuzlaTerm const*, values_ptr, map_values);
                         jsize num_keys = env->GetArrayLength(map_keys);

                         bitwuzla_substitute_terms(
                                 BZLA, num_terms, terms_ptr, num_keys, keys_ptr, values_ptr
                         );

                         jlongArray result_terms = create_ptr_array(env, num_terms);
                         set_ptr_array(env, result_terms, terms_ptr, num_terms);

                         return result_terms;
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

jlongArray
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaSortFunGetDomainSorts(JNIEnv* env, jobject native_class,
                                                                            jlong bitwuzla_sort) {
    BZLA_TRY_OR_NULL({
                         size_t result_size = 0;
                         const BitwuzlaSort** result = bitwuzla_sort_fun_get_domain_sorts(
                                 SORT(bitwuzla_sort), &result_size
                         );

                         jlongArray result_array = create_ptr_array(env, result_size);
                         set_ptr_array(env, result_array, result, result_size);

                         return result_array;
                     })
}

jlong
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaSortFunGetCodomain(JNIEnv* env, jobject native_class,
                                                                         jlong bitwuzla_sort) {
    BZLA_TRY_OR_ZERO({
                         const BitwuzlaSort* result = bitwuzla_sort_fun_get_codomain(SORT(bitwuzla_sort));
                         return (jlong) result;
                     })
}

jint
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaSortFunGetArity(JNIEnv* env, jobject native_class,
                                                                      jlong bitwuzla_sort) {
    BZLA_TRY_OR_ZERO({
                         uint32_t result = bitwuzla_sort_fun_get_arity(SORT(bitwuzla_sort));
                         return (jint) result;
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

jlongArray
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermFunGetDomainSorts(JNIEnv* env, jobject native_class,
                                                                            jlong bitwuzla_term) {
    BZLA_TRY_OR_NULL({
                         size_t result_size = 0;
                         const BitwuzlaSort** result = bitwuzla_term_fun_get_domain_sorts(
                                 TERM(bitwuzla_term), &result_size
                         );

                         jlongArray result_array = create_ptr_array(env, result_size);
                         set_ptr_array(env, result_array, result, result_size);

                         return result_array;
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

jlong
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkFpValueFromReal(JNIEnv* env, jobject native_class,
                                                                        jlong bitwuzla, jlong sort, jlong rm,
                                                                        jstring real) {
    BZLA_TRY_OR_ZERO({
                         GET_STRING(real_str, real);
                         const BitwuzlaTerm* result = bitwuzla_mk_fp_value_from_real(
                                 BZLA, SORT(sort), TERM(rm), real_str
                         );

                         return (jlong) result;
                     })
}

jlong
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkFpValueFromRational(JNIEnv* env, jobject native_class,
                                                                            jlong bitwuzla, jlong sort, jlong rm,
                                                                            jstring num, jstring den) {
    BZLA_TRY_OR_ZERO({
                         GET_STRING(num_str, num);
                         GET_STRING(den_str, den);

                         const BitwuzlaTerm* result = bitwuzla_mk_fp_value_from_rational(
                                 BZLA, SORT(sort), TERM(rm), num_str, den_str
                         );

                         return (jlong) result;
                     })
}

jlong Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkRmValue(JNIEnv* env, jobject native_class, jlong bitwuzla,
                                                                      jint rm) {
    BZLA_TRY_OR_ZERO({
                         return (jlong) bitwuzla_mk_rm_value(BZLA, (BitwuzlaRoundingMode) rm);
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

// Bv bits utils

jlong Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkBvValueUint32Array(JNIEnv* env, jobject native_class,
                                                                                 jlong bitwuzla, jint bv_width,
                                                                                 jintArray value) {
    BZLA_TRY_OR_ZERO({
                         JniIntArray value_array(env, value);
                         unsigned int value_length = env->GetArrayLength(value);

                         auto bzla_core = bitwuzla_extension_get_bzla(BZLA);
                         auto bzla_memory = bitwuzla_extension_get_bzla_memory(bzla_core);

                         auto remaining_width = (int32_t) bv_width;
                         BzlaBitVector* bv = nullptr;

                         unsigned int idx = 0;
                         while (remaining_width > 0) {
                             idx++;
                             uint32_t chunk_value = value_array.elements[value_length - idx];
                             uint32_t chunk_width = remaining_width > 32 ? 32 : remaining_width;
                             remaining_width -= 32;

                             auto chunk_bv = bitwuzla_extension_bzla_bv_uint64_to_bv(
                                     bzla_memory, chunk_value, chunk_width
                             );

                             if (bv != nullptr) {
                                 auto previous_chunk = bv;
                                 bv = bitwuzla_extension_bzla_bv_concat(bzla_memory, chunk_bv, previous_chunk);

                                 bitwuzla_extension_bzla_bv_free(bzla_memory, previous_chunk);
                                 bitwuzla_extension_bzla_bv_free(bzla_memory, chunk_bv);
                             } else {
                                 bv = chunk_bv;
                             }
                         }

                         BzlaNode* res = bitwuzla_extension_bzla_exp_bv_const(bzla_core, bv);
                         bitwuzla_extension_bzla_bv_free(bzla_memory, bv);

                         bitwuzla_extension_bzla_node_inc_ext_ref_counter(bzla_core, res);

                         return (jlong) res;
                     })
}

jint Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaBvConstNodeGetBitsUInt32(JNIEnv* env, jobject native_class,
                                                                                    jlong bitwuzla, jlong bitwuzla_term) {
    BZLA_TRY_OR_ZERO({
                         const BzlaBitVector* bw_vector = bitwuzla_extension_node_bv_const_get_bits(
                                 TERM(bitwuzla_term));
                         uint64_t bits = bitwuzla_extension_bv_to_uint64(bw_vector);

                         return (jint) bits;
                     })
}

jintArray get_bv_bits_as_jint_array(JNIEnv* env, jlong bitwuzla, const BzlaBitVector* bw_vector) {
    auto bzla_core = bitwuzla_extension_get_bzla(BZLA);
    auto bzla_memory = bitwuzla_extension_get_bzla_memory(bzla_core);

    uint32_t width = bitwuzla_extension_bv_get_width(bw_vector);

    uint32_t idx = 0;
    int32_t remaining_width = width;
    std::vector<uint32_t> chunks;

    while (remaining_width > 0) {
        uint32_t lower = idx * 32;
        uint32_t upper = std::min(lower + 32, width);
        idx++;
        remaining_width -= 32;

        auto chunk_bv = bitwuzla_extension_bzla_bv_slice(bzla_memory, bw_vector, upper, lower);
        uint64_t bits = bitwuzla_extension_bv_to_uint64(chunk_bv);
        chunks.push_back(bits);

        bitwuzla_extension_bzla_bv_free(bzla_memory, chunk_bv);
    }

    jintArray result = env->NewIntArray(chunks.size());
    env->SetIntArrayRegion(result, 0, chunks.size(), (jint*) chunks.data());

    return result;
}

jintArray
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaBvConstNodeGetBitsUIntArray(JNIEnv* env, jobject native_class,
                                                                                  jlong bitwuzla, jlong bitwuzla_term) {
    BZLA_TRY_OR_NULL({
                         auto bzla_core = bitwuzla_extension_get_bzla(BZLA);
                         auto bzla_memory = bitwuzla_extension_get_bzla_memory(bzla_core);

                         const BzlaBitVector* bw_vector = bitwuzla_extension_node_bv_const_get_bits(
                                 TERM(bitwuzla_term));

                         return get_bv_bits_as_jint_array(env, bitwuzla, bw_vector);
                     })
}

jint Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaFpConstNodeGetBitsUInt32(JNIEnv* env, jobject native_class,
                                                                                    jlong bitwuzla, jlong bitwuzla_term) {
    BZLA_TRY_OR_ZERO({
                         auto bzla = bitwuzla_extension_get_bzla(BZLA);
                         auto fp = bitwuzla_extension_node_fp_const_get_fp(TERM(bitwuzla_term));
                         const BzlaBitVector* bw_vector = bitwuzla_extension_fp_bits_as_bv_bits(bzla, fp);

                         uint64_t bits = bitwuzla_extension_bv_to_uint64(bw_vector);
                         return (jint) bits;
                     })
}

jintArray
Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaFpConstNodeGetBitsUIntArray(JNIEnv* env, jobject native_class,
                                                                                  jlong bitwuzla, jlong bitwuzla_term) {
    BZLA_TRY_OR_NULL({
                         auto bzla = bitwuzla_extension_get_bzla(BZLA);
                         auto fp = bitwuzla_extension_node_fp_const_get_fp(TERM(bitwuzla_term));
                         const BzlaBitVector* bw_vector = bitwuzla_extension_fp_bits_as_bv_bits(bzla, fp);

                         return get_bv_bits_as_jint_array(env, bitwuzla, bw_vector);
                     })
}

