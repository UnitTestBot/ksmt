#include <iostream>
#include <atomic>
#include <bitwuzla.h>
#include <memory>
#include <unistd.h>
#include <vector>
#include <chrono>
#include "bitwuzla_jni.hpp"

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
#define BZLA_OPTIONS(opts) (BitwuzlaOptions*) opts
#define TERM(t) (BitwuzlaTerm) t
#define SORT(s) (BitwuzlaSort) s

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

void Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaInit(JNIEnv *env, jclass native_class) {
    bitwuzla_set_abort_callback(abort_callback);
}

void Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermDecRef(JNIEnv *env, jclass native_class, jlong term) {
    BZLA_TRY_VOID({
        bitwuzla_term_dec_ref(TERM(term));
    })
}

void Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaSortDecRef(JNIEnv *env, jclass native_class, jlong sort) {
    BZLA_TRY_VOID({
        bitwuzla_sort_dec_ref(SORT(sort));
    })
}

void Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaForceTerminate(JNIEnv* env, jclass native_class, jlong bitwuzla) {
    BZLA_TRY_VOID({
        auto termination_state = get_termination_state(BZLA);
        termination_state->terminate();
    })
}

jstring Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaCopyright(JNIEnv *env, jclass native_class) {
    BZLA_TRY_OR_NULL({
         const char *c = bitwuzla_copyright();
         jstring result = env->NewStringUTF(c);
         return result;
    })
}

jstring Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaVersion(JNIEnv *env, jclass native_class) {
    BZLA_TRY_OR_NULL({
         const char* c = bitwuzla_version();
         jstring result = env->NewStringUTF(c);
         return result;
    })
}

jstring Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaGitId(JNIEnv *env, jclass native_class) {
    BZLA_TRY_OR_NULL({
         const char* c = bitwuzla_git_id();
         jstring result = env->NewStringUTF(c);
         return result;
    })
}

jlong Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaOptionsNew(JNIEnv *env, jclass native_class) {
    BZLA_TRY_OR_ZERO({
        BitwuzlaOptions *bitwuzla_options = bitwuzla_options_new();
        return (jlong) bitwuzla_options;
    })
}

void Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaOptionsDelete(JNIEnv *env, jclass native_class, jlong options) {
    BZLA_TRY_VOID({
         BitwuzlaOptions * bitwuzla_options_delete(BZLA_OPTIONS(options));
     })
}

jboolean Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaOptionIsNumeric(JNIEnv *env, jclass native_class, jlong options, jint option) {
    BZLA_TRY_OR_ZERO({
        return (jboolean) bitwuzla_option_is_numeric(BZLA_OPTIONS(options), BitwuzlaOption(option));
    })
}

jboolean Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaOptionIsMode(JNIEnv *env, jclass native_class, jlong options, jint option) {
    BZLA_TRY_OR_ZERO({
         return (jboolean) bitwuzla_option_is_mode(BZLA_OPTIONS(options), BitwuzlaOption(option));
    })
}

void Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaSetOption(JNIEnv *env, jclass native_class, jlong options, jint option, jlong value) {
    BZLA_TRY_VOID({
          bitwuzla_set_option(BZLA_OPTIONS(options), BitwuzlaOption(option), (uint64_t) value);
    })
}

void Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaSetOptionMode(JNIEnv *env, jclass native_class, jlong options, jint option, jstring value) {
    BZLA_TRY_VOID({
        GET_STRING(nativeString, value);
        bitwuzla_set_option_mode(BZLA_OPTIONS(options), BitwuzlaOption(option), nativeString);
    })
}

jlong Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaGetOption(JNIEnv *env, jclass native_class, jlong options, jint option) {
    BZLA_TRY_OR_ZERO({
        return (jlong) bitwuzla_get_option(BZLA_OPTIONS(options), BitwuzlaOption(option));
    })
}

jstring Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaGetOptionMode(JNIEnv *env, jclass native_class, jlong options, jint option) {
    BZLA_TRY_OR_ZERO({
        const char* value = bitwuzla_get_option_mode(BZLA_OPTIONS(options), BitwuzlaOption(option));
        return env->NewStringUTF(value);
    })
}

jlong Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaSortHash(JNIEnv *env, jclass native_class, jlong sort) {
    BZLA_TRY_OR_ZERO({
        return (jlong) bitwuzla_sort_hash(SORT(sort));
    })
}

jlong Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaSortBvGetSize(JNIEnv *env, jclass native_class, jlong sort) {
    BZLA_TRY_OR_ZERO({
        return (jlong) bitwuzla_sort_bv_get_size(SORT(sort));
    })
}

jlong Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaSortFpGetExpSize(JNIEnv *env, jclass native_class, jlong sort) {
    BZLA_TRY_OR_ZERO({
        return (jlong) bitwuzla_sort_fp_get_exp_size(SORT(sort));
    })
}

jlong Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaSortFpGetSigSize(JNIEnv *env, jclass native_class, jlong sort) {
    BZLA_TRY_OR_ZERO({
        return (jlong) bitwuzla_sort_fp_get_sig_size(SORT(sort));
    })
}

jlong Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaSortArrayGetIndex(JNIEnv *env, jclass native_class, jlong sort) {
    BZLA_TRY_OR_ZERO({
        return (jlong) bitwuzla_sort_array_get_index(SORT(sort));
    })
}

jlong Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaSortArrayGetElement(JNIEnv *env, jclass native_class, jlong sort) {
    BZLA_TRY_OR_ZERO({
        return (jlong) bitwuzla_sort_array_get_element(SORT(sort));
    })
}

jlongArray Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaSortFunGetDomainSorts(JNIEnv *env, jclass native_class, jlong sort) {
    BZLA_TRY_OR_NULL({
        size_t len = 0;
        const BitwuzlaSort* array = bitwuzla_sort_fun_get_domain_sorts(SORT(sort), &len);
        jlongArray result = create_ptr_array(env, len);
        set_ptr_array(env, result, array, len);
        return result;
    })
}

jlong Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaSortFunGetCodomain(JNIEnv *env, jclass native_class, jlong sort) {
    BZLA_TRY_OR_ZERO({
        return (jlong) bitwuzla_sort_fun_get_codomain(SORT(sort));
    })
}

jlong Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaSortFunGetArity(JNIEnv *env, jclass native_class, jlong sort) {
    BZLA_TRY_OR_ZERO({
        return (jlong) bitwuzla_sort_fun_get_arity(SORT(sort));
    })
}

jstring Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaSortGetUninterpretedSymbol(JNIEnv *env, jclass native_class, jlong sort) {
    BZLA_TRY_OR_ZERO({
        const char *symbol = bitwuzla_sort_get_uninterpreted_symbol(SORT(sort));
        return env->NewStringUTF(symbol);
    })
}

jboolean Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaSortIsEqual(JNIEnv *env, jclass native_class, jlong sort0, jlong sort1) {
    BZLA_TRY_OR_ZERO({
        return (jboolean) bitwuzla_sort_is_equal(SORT(sort0), SORT(sort1));
    })
}

jboolean Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaSortIsArray(JNIEnv *env, jclass native_class, jlong sort) {
    BZLA_TRY_OR_ZERO({
        return (jboolean) bitwuzla_sort_is_array(SORT(sort));
    })
}

jboolean Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaSortIsBool(JNIEnv *env, jclass native_class, jlong sort) {
    BZLA_TRY_OR_ZERO({
        return (jboolean) bitwuzla_sort_is_bool(SORT(sort));
    })
}

jboolean Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaSortIsBv(JNIEnv *env, jclass native_class, jlong sort) {
    BZLA_TRY_OR_ZERO({
        return (jboolean) bitwuzla_sort_is_bv(SORT(sort));
    })
}

jboolean Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaSortIsFp(JNIEnv *env, jclass native_class, jlong sort) {
    BZLA_TRY_OR_ZERO({
        return (jboolean) bitwuzla_sort_is_fp(SORT(sort));
    })
}

jboolean Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaSortIsFun(JNIEnv *env, jclass native_class, jlong sort) {
    BZLA_TRY_OR_ZERO({
        return (jboolean) bitwuzla_sort_is_fun(SORT(sort));
    })
}

jboolean Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaSortIsRm(JNIEnv *env, jclass native_class, jlong sort) {
    BZLA_TRY_OR_ZERO({
        return (jboolean) bitwuzla_sort_is_rm(SORT(sort));
    })
}

jboolean Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaSortIsUninterpreted(JNIEnv *env, jclass native_class, jlong sort) {
    BZLA_TRY_OR_ZERO({
        return (jboolean) bitwuzla_sort_is_uninterpreted(SORT(sort));
    })
}

jstring Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaSortToString(JNIEnv *env, jclass native_class, jlong sort) {
    BZLA_TRY_OR_NULL({
        const char* result = bitwuzla_sort_to_string(SORT(sort));
        return env->NewStringUTF(result);
    })
}

jlong Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermHash(JNIEnv *env, jclass native_class, jlong term) {
    BZLA_TRY_OR_ZERO({
        return (jlong) bitwuzla_term_hash(TERM(term));
    })
}

jlong Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermGetKindNative(JNIEnv *env, jclass native_class, jlong term) {
    BZLA_TRY_OR_ZERO({
        return (jlong) bitwuzla_term_get_kind(TERM(term));
    })
}

jlongArray Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermGetChildren(JNIEnv *env, jclass native_class, jlong term) {
    BZLA_TRY_OR_NULL({
        size_t len = 0;
        const BitwuzlaTerm *array = bitwuzla_term_get_children(TERM(term), &len);
        jlongArray result = create_ptr_array(env, len);
        set_ptr_array(env, result, array, len);
        return result;
    })
}

jlongArray Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermGetIndices(JNIEnv *env, jclass native_class, jlong term) {
    BZLA_TRY_OR_NULL({
        size_t len = 0;
        const uint64_t *array = bitwuzla_term_get_indices(TERM(term), &len);
        jlongArray result = create_ptr_array(env, len);
        set_ptr_array(env, result, array, len);
        return result;
    })
}

jboolean Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsIndexed(JNIEnv *env, jclass native_class, jlong term) {
    BZLA_TRY_OR_ZERO({
        return (jboolean) bitwuzla_term_is_indexed(TERM(term));
    })
}

jlong Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermGetSort(JNIEnv *env, jclass native_class, jlong term) {
    BZLA_TRY_OR_ZERO({
        return (jlong) bitwuzla_term_get_sort(TERM(term));
    })
}

jlong Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermArrayGetIndexSort(JNIEnv *env, jclass native_class, jlong term) {
    BZLA_TRY_OR_ZERO({
        return (jlong) bitwuzla_term_array_get_index_sort(TERM(term));
    })
}

jlong Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermArrayGetElementSort(JNIEnv *env, jclass native_class, jlong term) {
    BZLA_TRY_OR_ZERO({
        return (jlong) bitwuzla_term_array_get_element_sort(TERM(term));
    })
}

jlongArray Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermFunGetDomainSorts(JNIEnv *env, jclass native_class, jlong term) {
    BZLA_TRY_OR_NULL({
        size_t len = 0;
        const uint64_t *array = bitwuzla_term_fun_get_domain_sorts(TERM(term), &len);
        jlongArray result = create_ptr_array(env, len);
        set_ptr_array(env, result, array, len);
        return result;
    })
}

jlong Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermFunGetCodomainSort(JNIEnv *env, jclass native_class, jlong term) {
    BZLA_TRY_OR_ZERO({
        return (jlong) bitwuzla_term_fun_get_codomain_sort(TERM(term));
    })
}

jlong Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermBvGetSize(JNIEnv *env, jclass native_class, jlong term) {
    BZLA_TRY_OR_ZERO({
        return (jlong) bitwuzla_term_bv_get_size(TERM(term));
    })
}

jlong Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermFpGetExpSize(JNIEnv *env, jclass native_class, jlong term) {
    BZLA_TRY_OR_ZERO({
        return (jlong) bitwuzla_term_fp_get_exp_size(TERM(term));
    })
}

jlong Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermFpGetSigSize(JNIEnv *env, jclass native_class, jlong term) {
    BZLA_TRY_OR_ZERO({
        return (jlong) bitwuzla_term_fp_get_sig_size(TERM(term));
    })
}

jlong Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermFunGetArity(JNIEnv *env, jclass native_class, jlong term) {
    BZLA_TRY_OR_ZERO({
        return (jlong) bitwuzla_term_fun_get_arity(TERM(term));
    })
}

jstring Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermGetSymbol(JNIEnv *env, jclass native_class, jlong term) {
    BZLA_TRY_OR_NULL({
        const char *result = bitwuzla_term_get_symbol(TERM(term));
        return env->NewStringUTF(result);
    })
}

jboolean Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsEqualSort(JNIEnv *env, jclass native_class, jlong term0, jlong term1) {
    BZLA_TRY_OR_ZERO({
        return (jboolean) bitwuzla_term_is_equal_sort(TERM(term0), TERM(term1));
    })
}

jboolean Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsArray(JNIEnv *env, jclass native_class, jlong term) {
    BZLA_TRY_OR_ZERO({
        return (jboolean) bitwuzla_term_is_array(TERM(term));
    })
}

jboolean Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsConst(JNIEnv *env, jclass native_class, jlong term) {
    BZLA_TRY_OR_ZERO({
        return (jboolean) bitwuzla_term_is_const(TERM(term));
    })
}

jboolean Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsFun(JNIEnv *env, jclass native_class, jlong term) {
    BZLA_TRY_OR_ZERO({
        return (jboolean) bitwuzla_term_is_fun(TERM(term));
    })
}

jboolean Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsVar(JNIEnv *env, jclass native_class, jlong term) {
    BZLA_TRY_OR_ZERO({
        return (jboolean) bitwuzla_term_is_var(TERM(term));
    })
}

jboolean Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsValue(JNIEnv *env, jclass native_class, jlong term) {
    BZLA_TRY_OR_ZERO({
        return (jboolean) bitwuzla_term_is_value(TERM(term));
    })
}

jboolean Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsBvValue(JNIEnv *env, jclass native_class, jlong term) {
    BZLA_TRY_OR_ZERO({
        return (jboolean) bitwuzla_term_is_bv_value(TERM(term));
    })
}

jboolean Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsFpValue(JNIEnv *env, jclass native_class, jlong term) {
    BZLA_TRY_OR_ZERO({
        return (jboolean) bitwuzla_term_is_fp_value(TERM(term));
    })
}

jboolean Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsRmValue(JNIEnv *env, jclass native_class, jlong term) {
    BZLA_TRY_OR_ZERO({
        return (jboolean) bitwuzla_term_is_rm_value(TERM(term));
    })
}

jboolean Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsBool(JNIEnv *env, jclass native_class, jlong term) {
    BZLA_TRY_OR_ZERO({
        return (jboolean) bitwuzla_term_is_bool(TERM(term));
    })
}

jboolean Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsBv(JNIEnv *env, jclass native_class, jlong term) {
    BZLA_TRY_OR_ZERO({
        return (jboolean) bitwuzla_term_is_bv(TERM(term));
    })
}

jboolean Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsFp(JNIEnv *env, jclass native_class, jlong term) {
    BZLA_TRY_OR_ZERO({
        return (jboolean) bitwuzla_term_is_fp(TERM(term));
    })
}

jboolean Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsRm(JNIEnv *env, jclass native_class, jlong term) {
    BZLA_TRY_OR_ZERO({
        return (jboolean) bitwuzla_term_is_rm(TERM(term));
    })
}

jboolean Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsUninterpreted(JNIEnv *env, jclass native_class, jlong term) {
    BZLA_TRY_OR_ZERO({
        return (jboolean) bitwuzla_term_is_uninterpreted(TERM(term));
    })
}

jboolean Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsTrue(JNIEnv *env, jclass native_class, jlong term) {
    BZLA_TRY_OR_ZERO({
        return (jboolean) bitwuzla_term_is_true(TERM(term));
    })
}

jboolean Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsFalse(JNIEnv *env, jclass native_class, jlong term) {
    BZLA_TRY_OR_ZERO({
        return (jboolean) bitwuzla_term_is_false(TERM(term));
    })
}

jboolean Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsBvValueZero(JNIEnv *env, jclass native_class, jlong term) {
    BZLA_TRY_OR_ZERO({
        return (jboolean) bitwuzla_term_is_bv_value_zero(TERM(term));
    })
}

jboolean Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsBvValueOne(JNIEnv *env, jclass native_class, jlong term) {
    BZLA_TRY_OR_ZERO({
        return (jboolean) bitwuzla_term_is_bv_value_one(TERM(term));
    })
}

jboolean Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsBvValueOnes(JNIEnv *env, jclass native_class, jlong term) {
    BZLA_TRY_OR_ZERO({
        return (jboolean) bitwuzla_term_is_bv_value_ones(TERM(term));
    })
}

jboolean Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsBvValueMinSigned(JNIEnv *env, jclass native_class, jlong term) {
    BZLA_TRY_OR_ZERO({
        return (jboolean) bitwuzla_term_is_bv_value_min_signed(TERM(term));
    })
}

jboolean Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsBvValueMaxSigned(JNIEnv *env, jclass native_class, jlong term) {
    BZLA_TRY_OR_ZERO({
        return (jboolean) bitwuzla_term_is_bv_value_max_signed(TERM(term));
    })
}

jboolean Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsFpValuePosZero(JNIEnv *env, jclass native_class, jlong term) {
    BZLA_TRY_OR_ZERO({
        return (jboolean) bitwuzla_term_is_fp_value_pos_zero(TERM(term));
    })
}

jboolean Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsFpValueNegZero(JNIEnv *env, jclass native_class, jlong term) {
    BZLA_TRY_OR_ZERO({
        return (jboolean) bitwuzla_term_is_fp_value_neg_zero(TERM(term));
    })
}

jboolean Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsFpValuePosInf(JNIEnv *env, jclass native_class, jlong term) {
    BZLA_TRY_OR_ZERO({
        return (jboolean) bitwuzla_term_is_fp_value_pos_inf(TERM(term));
    })
}

jboolean Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsFpValueNegInf(JNIEnv *env, jclass native_class, jlong term) {
    BZLA_TRY_OR_ZERO({
        return (jboolean) bitwuzla_term_is_fp_value_neg_inf(TERM(term));
    })
}

jboolean Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsFpValueNan(JNIEnv *env, jclass native_class, jlong term) {
    BZLA_TRY_OR_ZERO({
        return (jboolean) bitwuzla_term_is_fp_value_nan(TERM(term));
    })
}

jboolean Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsRmValueRna(JNIEnv *env, jclass native_class, jlong term) {
    BZLA_TRY_OR_ZERO({
        return (jboolean) bitwuzla_term_is_rm_value_rna(TERM(term));
    })
}

jboolean Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsRmValueRne(JNIEnv *env, jclass native_class, jlong term) {
    BZLA_TRY_OR_ZERO({
        return (jboolean) bitwuzla_term_is_rm_value_rne(TERM(term));
    })
}

jboolean Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsRmValueRtn(JNIEnv *env, jclass native_class, jlong term) {
    BZLA_TRY_OR_ZERO({
        return (jboolean) bitwuzla_term_is_rm_value_rtn(TERM(term));
    })
}

jboolean Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsRmValueRtp(JNIEnv *env, jclass native_class, jlong term) {
    BZLA_TRY_OR_ZERO({
        return (jboolean) bitwuzla_term_is_rm_value_rtp(TERM(term));
    })
}

jboolean Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsRmValueRtz(JNIEnv *env, jclass native_class, jlong term) {
    BZLA_TRY_OR_ZERO({
        return (jboolean) bitwuzla_term_is_rm_value_rtz(TERM(term));
    })
}

jboolean Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermValueGetBool(JNIEnv *env, jclass native_class, jlong term) {
    BZLA_TRY_OR_ZERO({
        return (jboolean) bitwuzla_term_value_get_bool(TERM(term));
    })
}

jstring Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermValueGetStr(JNIEnv *env, jclass native_class, jlong term) {
    BZLA_TRY_OR_NULL({
        const char *result = bitwuzla_term_value_get_str(TERM(term));
        return env->NewStringUTF(result);
    })
}

jstring Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermValueGetStrFmt(JNIEnv *env, jclass native_class, jlong term, jbyte base) {
    BZLA_TRY_OR_NULL({
        const char *result = bitwuzla_term_value_get_str_fmt(TERM(term), (uint8_t) base);
        return env->NewStringUTF(result);
    })
}

jint Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermValueGetRm(JNIEnv *env, jclass native_class, jlong term) {
    BZLA_TRY_OR_ZERO({
        return BitwuzlaRoundingMode(bitwuzla_term_value_get_rm(TERM(term)));
    })
}

jstring Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermToString(JNIEnv *env, jclass native_class, jlong term) {
    BZLA_TRY_OR_NULL({
        const char *result = bitwuzla_term_to_string(TERM(term));
        return env->NewStringUTF(result);
    })
}

jstring Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermToStringFmt(JNIEnv *env, jclass native_class, jlong term, jbyte base) {
    BZLA_TRY_OR_NULL({
        const char *result = bitwuzla_term_to_string_fmt(TERM(term), (uint8_t) base);
        return env->NewStringUTF(result);
    })
}

jlong Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaNew(JNIEnv *env, jclass native_class, jlong options) {
    BZLA_TRY_OR_ZERO({
        Bitwuzla *bzla = bitwuzla_new(BZLA_OPTIONS(options));

        auto termination_state = new BitwuzlaTerminationCallbackState();
        termination_state->reset();
        bitwuzla_set_termination_callback(bzla, termination_callback, termination_state);

        return (jlong) bzla;
    })
}

void Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaDelete(JNIEnv *env, jclass native_class, jlong bitwuzla) {
    BZLA_TRY_VOID({
        auto termination_state = get_termination_state(BZLA);
        delete termination_state;

        bitwuzla_delete(BZLA);
    })
}

void Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaPush(JNIEnv *env, jclass native_class, jlong bitwuzla, jlong nlevels) {
    BZLA_TRY_VOID({
        bitwuzla_push(BZLA, (uint64_t) nlevels);
    })
}

void Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaPop(JNIEnv *env, jclass native_class, jlong bitwuzla, jlong nlevels) {
    BZLA_TRY_VOID({
        bitwuzla_pop(BZLA, (uint64_t) nlevels);
    })
}

void Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaAssert(JNIEnv *env, jclass native_class, jlong bitwuzla, jlong term) {
    BZLA_TRY_VOID({
        bitwuzla_assert(BZLA, TERM(term));
    })
}

jlongArray Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaGetAssertions(JNIEnv *env, jclass native_class, jlong bitwuzla) {
    BZLA_TRY_OR_NULL({
        size_t len = 0;
        const BitwuzlaTerm *array = bitwuzla_get_assertions(BZLA, &len);
        jlongArray result = create_ptr_array(env, len);
        set_ptr_array(env, result, array, len);
        return result;
    })
}

jboolean Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaIsUnsatAssumption(JNIEnv *env, jclass native_class, jlong bitwuzla, jlong term) {
    BZLA_TRY_OR_ZERO({
        return bitwuzla_is_unsat_assumption(BZLA, TERM(term));
    })
}

jlongArray Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaGetUnsatAssumptions(JNIEnv *env, jclass native_class, jlong bitwuzla) {
    BZLA_TRY_OR_NULL({
        size_t len = 0;
        const BitwuzlaTerm *array = bitwuzla_get_unsat_assumptions(BZLA, &len);
        jlongArray result = create_ptr_array(env, len);
        set_ptr_array(env, result, array, len);
        return result;
    })
}

jlongArray Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaGetUnsatCore(JNIEnv *env, jclass native_class, jlong bitwuzla) {
    BZLA_TRY_OR_NULL({
        size_t len = 0;
        const BitwuzlaTerm *array = bitwuzla_get_unsat_core(BZLA, &len);
        jlongArray result = create_ptr_array(env, len);
        set_ptr_array(env, result, array, len);
        return result;
    })
}

void Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaSimplify(JNIEnv *env, jclass native_class, jlong bitwuzla) {
    BZLA_TRY_VOID({
        bitwuzla_simplify(BZLA);
    })
}

jint Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaCheckSatNative(JNIEnv *env, jclass native_class, jlong bitwuzla) {
    BZLA_TRY_OR_ZERO({
        return (jint) bitwuzla_check_sat(BZLA);
    })
}

jint Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaCheckSatAssumingNative(JNIEnv *env, jclass native_class, jlong bitwuzla, jlongArray args) {
    BZLA_TRY_OR_ZERO({
        GET_PTR_ARRAY(BitwuzlaTerm, args_ptr, args);
        jsize len = env->GetArrayLength(args);
        return (jint) bitwuzla_check_sat_assuming(BZLA, (uint32_t) len, args_ptr);
    })
}

jlong Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaGetValue(JNIEnv *env, jclass native_class, jlong bitwuzla, jlong term) {
    BZLA_TRY_OR_ZERO({
        return (jlong) bitwuzla_get_value(BZLA, TERM(term));
    })
}

jlong Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaGetBvValueUInt64(JNIEnv *env, jclass native_class, jlong term) {
    BZLA_TRY_OR_ZERO({
        return (jlong) bitwuzla_term_value_get_bv_uint64(TERM(term));
    })
}

jstring Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaGetBvValueString(JNIEnv *env, jclass native_class, jlong term) {
    BZLA_TRY_OR_ZERO({
        const char * str = bitwuzla_term_value_get_bv_str(TERM(term), 2);
        return env->NewStringUTF(str);
    })
}

jobject Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaGetFpValue(JNIEnv *env, jclass native_class, jlong term) {
    BZLA_TRY_OR_NULL({
        const char* sign;
        const char* exponent;
        const char* significand;

        bitwuzla_term_value_get_fp_ieee(term, &sign, &exponent, &significand, 2);
        jclass clazz = env->FindClass("org/ksmt/solver/bitwuzla/bindings/FpValue");
        jmethodID constructor = env->GetMethodID(clazz, "<init>", "()V");
        jfieldID sign_id = env->GetFieldID(clazz, "sign", "Ljava/lang/String;");
        jfieldID exponent_id = env->GetFieldID(clazz, "exponent", "Ljava/lang/String;");
        jfieldID significand_id = env->GetFieldID(clazz, "significand", "Ljava/lang/String;");

        jstring sign_str = env->NewStringUTF(sign);
        jstring exponent_str = env->NewStringUTF(exponent);
        jstring significand_str = env->NewStringUTF(significand);

        jobject result = env->NewObject(clazz, constructor);
        env->SetObjectField(result, sign_id, sign_str);
        env->SetObjectField(result, exponent_id, exponent_str);
        env->SetObjectField(result, significand_id, significand_str);

        return result;
    })
}


jlong Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkArraySort(JNIEnv *env, jclass native_class, jlong index, jlong element) {
    BZLA_TRY_OR_ZERO({
        return (jlong) bitwuzla_mk_array_sort(SORT(index), SORT(element));
    })
}

jlong Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkBoolSort(JNIEnv *env, jclass native_class) {
    BZLA_TRY_OR_ZERO({
        return (jlong) bitwuzla_mk_bool_sort();
    })
}

jlong Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkBvSort(JNIEnv *env, jclass native_class, jlong size) {
    BZLA_TRY_OR_ZERO({
        return (jlong) bitwuzla_mk_bv_sort((uint64_t) size);
    })
}

jlong Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkFpSort(JNIEnv *env, jclass native_class, jlong expSize, jlong sigSize) {
    BZLA_TRY_OR_ZERO({
        return (jlong) bitwuzla_mk_fp_sort((uint64_t) expSize, (uint64_t) sigSize);
    })
}

jlong Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkFunSort(JNIEnv *env, jclass native_class, jlongArray domain, jlong codomain) {
    BZLA_TRY_OR_ZERO({
        GET_PTR_ARRAY(BitwuzlaSort, domain_ptr, domain);
        jsize arity = env->GetArrayLength(domain);
        return (jlong) bitwuzla_mk_fun_sort((uint64_t) arity, domain_ptr, SORT(codomain));
    })
}

jlong Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkRmSort(JNIEnv *env, jclass native_class) {
    BZLA_TRY_OR_ZERO({
        return (jlong) bitwuzla_mk_rm_sort();
    })
}

jlong Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkUninterpretedSort(JNIEnv *env, jclass native_class, jstring symbol) {
    BZLA_TRY_OR_ZERO({
        GET_STRING(native_symbol, symbol);
        return (jlong) bitwuzla_mk_uninterpreted_sort(native_symbol);
    })
}

jlong Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkTrue(JNIEnv *env, jclass native_class) {
    BZLA_TRY_OR_ZERO({
        return (jlong) bitwuzla_mk_true();
    })
}

jlong Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkFalse(JNIEnv *env, jclass native_class) {
    BZLA_TRY_OR_ZERO({
        return (jlong) bitwuzla_mk_false();
    })
}

jlong Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkBvZero(JNIEnv *env, jclass native_class, jlong sort) {
    BZLA_TRY_OR_ZERO({
        return (jlong) bitwuzla_mk_bv_zero(SORT(sort));
    })
}

jlong Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkBvOne(JNIEnv *env, jclass native_class, jlong sort) {
    BZLA_TRY_OR_ZERO({
        return (jlong) bitwuzla_mk_bv_one(SORT(sort));
    })
}

jlong Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkBvOnes(JNIEnv *env, jclass native_class, jlong sort) {
    BZLA_TRY_OR_ZERO({
        return (jlong) bitwuzla_mk_bv_ones(SORT(sort));
    })
}

jlong Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkBvMinSigned(JNIEnv *env, jclass native_class, jlong sort) {
    BZLA_TRY_OR_ZERO({
        return (jlong) bitwuzla_mk_bv_min_signed(SORT(sort));
    })
}

jlong Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkBvMaxSigned(JNIEnv *env, jclass native_class, jlong sort) {
    BZLA_TRY_OR_ZERO({
        return (jlong) bitwuzla_mk_bv_max_signed(SORT(sort));
    })
}

jlong Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkFpPosZero(JNIEnv *env, jclass native_class, jlong sort) {
    BZLA_TRY_OR_ZERO({
        return (jlong) bitwuzla_mk_fp_pos_zero(SORT(sort));
    })
}

jlong Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkFpNegZero(JNIEnv *env, jclass native_class, jlong sort) {
    BZLA_TRY_OR_ZERO({
        return (jlong) bitwuzla_mk_fp_neg_zero(SORT(sort));
    })
}

jlong Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkFpPosInf(JNIEnv *env, jclass native_class, jlong sort) {
    BZLA_TRY_OR_ZERO({
        return (jlong) bitwuzla_mk_fp_pos_inf(SORT(sort));
    })
}

jlong Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkFpNegInf(JNIEnv *env, jclass native_class, jlong sort) {
    BZLA_TRY_OR_ZERO({
        return (jlong) bitwuzla_mk_fp_neg_inf(SORT(sort));
    })
}

jlong Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkFpNan(JNIEnv *env, jclass native_class, jlong sort) {
    BZLA_TRY_OR_ZERO({
        return (jlong) bitwuzla_mk_fp_nan(SORT(sort));
    })
}

jlong Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkBvValue(JNIEnv *env, jclass native_class, jlong sort, jstring value, jbyte base) {
    BZLA_TRY_OR_ZERO({
        GET_STRING(native_value, value);
        return (jlong) bitwuzla_mk_bv_value(SORT(sort), native_value, (uint8_t) base);
    })
}

jlong Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkBvValueUint64(JNIEnv *env, jclass native_class, jlong sort, jlong value) {
    BZLA_TRY_OR_ZERO({
        return (jlong) bitwuzla_mk_bv_value_uint64(SORT(sort), (uint64_t) value);
    })
}

jlong Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkBvValueInt64(JNIEnv *env, jclass native_class, jlong sort, jlong value) {
    BZLA_TRY_OR_ZERO({
        return (jlong) bitwuzla_mk_bv_value_int64(SORT(sort), (int64_t) value);
    })
}

jlong Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkFpValue(JNIEnv *env, jclass native_class, jlong bvSign, jlong bvExponent, jlong bvSignificand) {
    BZLA_TRY_OR_ZERO({
        return (jlong) bitwuzla_mk_fp_value(TERM(bvSign), TERM(bvExponent), TERM(bvSignificand));
    })
}

jlong Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkFpFromReal(JNIEnv *env, jclass native_class, jlong sort, jlong rm, jstring real) {
    BZLA_TRY_OR_ZERO({
        GET_STRING(native_real, real);
        return (jlong) bitwuzla_mk_fp_from_real(SORT(sort), TERM(rm), native_real);
    })
}

jlong Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkFpFromRational(JNIEnv *env, jclass native_class, jlong sort, jlong rm, jstring num, jstring den) {
    BZLA_TRY_OR_ZERO({
        GET_STRING(native_num, num);
        GET_STRING(native_den, den);
        return (jlong) bitwuzla_mk_fp_from_rational(SORT(sort), TERM(rm), native_num, native_den);
    })
}

jlong Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkRmValue(JNIEnv *env, jclass native_class, jint rm) {
    BZLA_TRY_OR_ZERO({
        return (jlong) bitwuzla_mk_rm_value(BitwuzlaRoundingMode(rm));
    })
}

jlong Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkTerm1(JNIEnv *env, jclass native_class, jint kind, jlong arg) {
    BZLA_TRY_OR_ZERO({
        return (jlong) bitwuzla_mk_term1(BitwuzlaKind(kind), TERM(arg));
    })
}

jlong Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkTerm2(JNIEnv *env, jclass native_class, jint kind, jlong arg0, jlong arg1) {
    BZLA_TRY_OR_ZERO({
        return (jlong) bitwuzla_mk_term2(BitwuzlaKind(kind), TERM(arg0), TERM(arg1));
    })
}

jlong Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkTerm3(JNIEnv *env, jclass native_class, jint kind, jlong arg0, jlong arg1, jlong arg2) {
    BZLA_TRY_OR_ZERO({
        return (jlong) bitwuzla_mk_term3(BitwuzlaKind(kind), TERM(arg0), TERM(arg1), TERM(arg2));
    })
}

jlong Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkTerm(JNIEnv *env, jclass native_class, jint kind, jlongArray args) {
    BZLA_TRY_OR_ZERO({
        GET_PTR_ARRAY(BitwuzlaTerm, args_ptr, args);
        jsize argc = env->GetArrayLength(args);
        return (jlong) bitwuzla_mk_term(BitwuzlaKind(kind), (uint32_t) argc, args_ptr);
    })
}

jlong Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkTerm1Indexed1(JNIEnv *env, jclass native_class, jint kind, jlong arg, jlong idx) {
    BZLA_TRY_OR_ZERO({
        return (jlong) bitwuzla_mk_term1_indexed1(BitwuzlaKind(kind), TERM(arg), (uint64_t) idx);
    })
}

jlong Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkTerm1Indexed2(JNIEnv *env, jclass native_class, jint kind, jlong arg, jlong idx0, jlong idx1) {
    BZLA_TRY_OR_ZERO({
        return (jlong) bitwuzla_mk_term1_indexed2(BitwuzlaKind(kind), TERM(arg), (uint64_t) idx0, (uint64_t) idx1);
    })
}

jlong Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkTerm2Indexed1(JNIEnv *env, jclass native_class, jint kind, jlong arg0, jlong arg1, jlong idx) {
    BZLA_TRY_OR_ZERO({
        return (jlong) bitwuzla_mk_term2_indexed1(BitwuzlaKind(kind), TERM(arg0), TERM(arg1), (uint64_t) idx);
    })
}

jlong Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkTerm2Indexed2(JNIEnv *env, jclass native_class, jint kind, jlong arg0, jlong arg1, jlong idx0, jlong idx1) {
    BZLA_TRY_OR_ZERO({
        return (jlong) bitwuzla_mk_term2_indexed2(BitwuzlaKind(kind), TERM(arg0), TERM(arg1), (uint64_t) idx0, (uint64_t) idx1);
    })
}

jlong Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkTermIndexed(JNIEnv *env, jclass native_class, jint kind, jlongArray args, jlongArray idxs) {
    BZLA_TRY_OR_ZERO({
        GET_PTR_ARRAY(BitwuzlaTerm, args_ptr, args);
        jsize argc = env->GetArrayLength(args);

        GET_PTR_ARRAY(BitwuzlaTerm, idxs_ptr, idxs);
        jsize idxc = env->GetArrayLength(idxs);

        return (jlong) bitwuzla_mk_term_indexed(BitwuzlaKind(kind), (uint32_t) argc, args_ptr, (uint32_t) idxc, idxs_ptr);
    })
}

jlong Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkConst(JNIEnv *env, jclass native_class, jlong sort, jstring symbol) {
    BZLA_TRY_OR_ZERO({
        GET_STRING(native_symbol, symbol);
        return (jlong) bitwuzla_mk_const(SORT(sort), native_symbol);
    })
}

jlong Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkConstArray(JNIEnv *env, jclass native_class, jlong sort, jlong value) {
    BZLA_TRY_OR_ZERO({
        return (jlong) bitwuzla_mk_const_array(SORT(sort), TERM(value));
    })
}

jlong Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkVar(JNIEnv *env, jclass native_class, jlong sort, jstring symbol) {
    BZLA_TRY_OR_ZERO({
        GET_STRING(native_symbol, symbol);
        return (jlong) bitwuzla_mk_var(SORT(sort), native_symbol);
    })
}

jlong Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaSubstituteTerm(JNIEnv *env, jclass native_class, jlong bitwuzla, jlong term, jlongArray mapKeys, jlongArray mapValues) {
    BZLA_TRY_OR_ZERO({
        GET_PTR_ARRAY(BitwuzlaTerm, mapKeys_ptr, mapKeys);
        GET_PTR_ARRAY(BitwuzlaTerm, mapValues_ptr, mapValues);
        jsize map_size = env->GetArrayLength(mapKeys);
        return (jlong) bitwuzla_substitute_term(BZLA, TERM(term), map_size, mapKeys_ptr, mapValues_ptr);
    })
}

void Java_org_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaSubstituteTerms(JNIEnv *env, jclass native_class, jlong bitwuzla, jlongArray terms, jlongArray mapKeys, jlongArray mapValues) {
    BZLA_TRY_VOID({
        GET_PTR_ARRAY(BitwuzlaTerm, terms_ptr, terms);
        jsize terms_size = env->GetArrayLength(terms);

        GET_PTR_ARRAY(BitwuzlaTerm, mapKeys_ptr, mapKeys);
        GET_PTR_ARRAY(BitwuzlaTerm, mapValues_ptr, mapValues);
        jsize map_size = env->GetArrayLength(mapKeys);

        bitwuzla_substitute_terms(BZLA, (size_t) terms_size, terms_ptr, map_size, mapKeys_ptr, mapValues_ptr);
    })
}
