#include <iostream>
#include <atomic>
#include <memory>
#include <unistd.h>
#include <vector>
#include <chrono>
#include <bitwuzla/c/bitwuzla.h>

#include "bv/bitvector.h"
#include "api/c/bitwuzla_structs.h"
#include "solver/fp/floating_point.h"
#include "node/node.h"

#include "io_ksmt_solver_bitwuzla_bindings_Native.h"
#include "access_private.hpp"

template<typename T>
struct JniPoinerArray {
    JNIEnv* env;
    jlongArray array;
    T* ptr_array;

    JniPoinerArray(JNIEnv* env, jlongArray array) : env(env), array(array) {
#if defined(__LP64__) || defined(_WIN64)
        // pointers are 64 bits, we can simply cast an array
        ptr_array = reinterpret_cast<T*>(env->GetLongArrayElements(array, nullptr));
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
        env->ReleaseLongArrayElements(array, reinterpret_cast<jlong*>(ptr_array), JNI_ABORT);
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

// struct JniIntArray {
//     JNIEnv* env;
//     jintArray jni_array;
//     jint* elements;
//
//     JniIntArray(JNIEnv* env, jintArray jni_array) : env(env), jni_array(jni_array) {
//         elements = env->GetIntArrayElements(jni_array, nullptr);
//     }
//
//     ~JniIntArray() {
//         env->ReleaseIntArrayElements(jni_array, elements, JNI_ABORT);
//     }
// };

struct JniLongArray {
    JNIEnv* env;
    jlongArray jni_array;
    jlong* elements;

    JniLongArray(JNIEnv* env, jlongArray jni_array) : env(env), jni_array(jni_array) {
        elements = env->GetLongArrayElements(jni_array, nullptr);
    }

    ~JniLongArray() {
        env->ReleaseLongArrayElements(jni_array, elements, JNI_ABORT);
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
    return env->NewLongArray(static_cast<jsize>(size));
}

template<typename T>
void set_ptr_array(JNIEnv* env, jlongArray array, T* ptr_array, size_t size) {
#if defined(__LP64__) || defined(_WIN64)
    env->SetLongArrayRegion(array, 0, static_cast<jsize>(size), reinterpret_cast<jlong*>(ptr_array));
#else
    jlong* temp = new jlong[size];
    for (int i = 0; i < size; i++) {
        temp[i] = reinterpret_cast<jlong>(ptr_array[i]);
    }
    env->SetLongArrayRegion(array, 0, (jsize) size, temp);
    delete[] temp;
#endif
}

#define TERM_MANAGER (reinterpret_cast<BitwuzlaTermManager *>(bitwuzla_term_manager))
#define TERM(t) (reinterpret_cast<BitwuzlaTerm>(t))
#define SORT(s) (reinterpret_cast<BitwuzlaSort>(s))

#define EXCEPTION_STATE_UNKNOWN (-1)
#define EXCEPTION_STATE_NO_EXCEPTION (0)

static thread_local int exception_state = EXCEPTION_STATE_UNKNOWN;
static thread_local std::string exception_message = "";

void abort_callback(const char* msg)
{
    exception_state++;
    exception_message = std::string(msg);
}

#define BITWUZLA_JNI_EXCEPTION_CLS "io/ksmt/solver/bitwuzla/bindings/BitwuzlaNativeException"

# define BZLA_API_TRY(EXPR, ON_SUCCESS, ERROR_VAL) {                      \
    exception_state = EXCEPTION_STATE_NO_EXCEPTION;                       \
    bitwuzla_set_abort_callback(abort_callback);                          \
    EXPR;                                                                 \
    if (exception_state > EXCEPTION_STATE_NO_EXCEPTION) {                 \
        jclass exception = env->FindClass(BITWUZLA_JNI_EXCEPTION_CLS);    \
        env->ThrowNew(exception, exception_message.c_str());              \
        exception_state = EXCEPTION_STATE_UNKNOWN;                        \
        return ERROR_VAL;                                                 \
    }                                                                     \
    exception_state = EXCEPTION_STATE_UNKNOWN;                            \
    ON_SUCCESS;                                                           \
}                                                                         \

#define BZLA_TRY_PTR_EXPR(EXPR) BZLA_API_TRY(auto api_call_result = EXPR, return reinterpret_cast<jlong>(api_call_result), 0)
#define BZLA_TRY_UINT64_EXPR(EXPR) BZLA_API_TRY(auto api_call_result = EXPR, return static_cast<jlong>(api_call_result), 0)
#define BZLA_TRY_INT32_EXPR(EXPR) BZLA_API_TRY(auto api_call_result = EXPR, return static_cast<jint>(api_call_result), 0)
#define BZLA_TRY_BOOL_EXPR(EXPR) BZLA_API_TRY(bool api_call_result = EXPR, return api_call_result ? JNI_TRUE : JNI_FALSE, JNI_FALSE)
#define BZLA_TRY_STRING_EXPR(EXPR) BZLA_API_TRY(auto api_call_result = EXPR, return env->NewStringUTF(api_call_result), nullptr)
#define BZLA_TRY_VOID_EXPR(EXPR) BZLA_API_TRY(EXPR, , )
#define BZLA_TRY_OR_RETURN_ZERO(EXPR) BZLA_API_TRY(EXPR, , 0)
#define BZLA_TRY_OR_RETURN_NULL(EXPR) BZLA_API_TRY(EXPR, , nullptr)

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
    auto termination_state = static_cast<BitwuzlaTerminationCallbackState*>(state);
    if (termination_state != nullptr && termination_state->terminated()) {
        return 1;
    }
    return 0;
}

BitwuzlaTerminationCallbackState* get_termination_state(Bitwuzla* bitwuzla) {
    auto state = bitwuzla_get_termination_callback_state(bitwuzla);
    return static_cast<BitwuzlaTerminationCallbackState*>(state);
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

jlong Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermManagerNew(JNIEnv* env, jclass native_class) {
    BZLA_TRY_PTR_EXPR(bitwuzla_term_manager_new())
}

void Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermManagerRelease(
    JNIEnv* env, jclass native_class, jlong bitwuzla_term_manager)
{
    BZLA_TRY_VOID_EXPR(bitwuzla_term_manager_release(TERM_MANAGER))
}

void Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermManagerDelete(
    JNIEnv* env, jclass native_class, jlong bitwuzla_term_manager)
{
    BZLA_TRY_VOID_EXPR(bitwuzla_term_manager_delete(TERM_MANAGER))
}

void Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaSortRelease(
    JNIEnv* env, jclass native_class, jlong sort)
{
    BZLA_TRY_VOID_EXPR(bitwuzla_sort_release(SORT(sort)))
}

void Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermRelease(
    JNIEnv* env, jclass native_class, jlong term)
{
    BZLA_TRY_VOID_EXPR(bitwuzla_term_release(TERM(term)))
}

jlong Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaNew(JNIEnv* env, jclass native_class,
                                                                jlong bitwuzla_term_manager, jlong bitwuzla_options)
{
    Bitwuzla* bzla;
    BZLA_TRY_OR_RETURN_ZERO(bzla = bitwuzla_new(TERM_MANAGER, reinterpret_cast<BitwuzlaOptions*>(bitwuzla_options)))

    auto termination_state = new BitwuzlaTerminationCallbackState();
    termination_state->reset();
    bitwuzla_set_termination_callback(bzla, termination_callback, termination_state);
    return reinterpret_cast<jlong>(bzla);
}

void Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaDelete(JNIEnv* env, jclass native_class, jlong bitwuzla)
{
    auto bzla = reinterpret_cast<Bitwuzla*>(bitwuzla);

    auto termination_state = get_termination_state(bzla);
    delete termination_state;

    BZLA_TRY_VOID_EXPR(bitwuzla_delete(bzla))
}

jlong Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaOptionsNew(JNIEnv* env, jclass native_class)
{
    BZLA_TRY_PTR_EXPR(bitwuzla_options_new())
}

void Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaOptionsDelete(
    JNIEnv* env, jclass native_class, jlong options)
{
    BZLA_TRY_VOID_EXPR(bitwuzla_options_delete(reinterpret_cast<BitwuzlaOptions*>(options)))
}

void Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaSetOption(JNIEnv* env, jclass native_class, jlong options,
                                                                     jint bitwuzla_option, jlong value)
{
    BZLA_TRY_VOID_EXPR(
        bitwuzla_set_option(
            reinterpret_cast<BitwuzlaOptions*>(options),
            static_cast<BitwuzlaOption>(bitwuzla_option),
            static_cast<uint64_t>(value)
        )
    )
}

void
Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaSetOptionMode(JNIEnv* env, jclass native_class, jlong options,
                                                                    jint bitwuzla_option, jstring value)
{
    GET_STRING(nativeString, value);

    BZLA_TRY_VOID_EXPR(
        bitwuzla_set_option_mode(
            reinterpret_cast<BitwuzlaOptions*>(options),
            static_cast<BitwuzlaOption>(bitwuzla_option),
            nativeString
        )
    )
}

jlong Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaGetOption(JNIEnv* env, jclass native_class, jlong options,
                                                                      jint bitwuzla_option)
{
    BZLA_TRY_UINT64_EXPR(
        bitwuzla_get_option(
            reinterpret_cast<BitwuzlaOptions*>(options),
            static_cast<BitwuzlaOption>(bitwuzla_option)
        )
    )
}

jstring
Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaGetOptionMode(JNIEnv* env, jclass native_class, jlong options,
                                                                    jint bitwuzla_option)
{
    BZLA_TRY_STRING_EXPR(
        bitwuzla_get_option_mode(
            reinterpret_cast<BitwuzlaOptions*>(options),
            static_cast<BitwuzlaOption>(bitwuzla_option)
        )
    )
}

jlong
Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkArraySort(JNIEnv* env, jclass native_class,
                                                                  jlong bitwuzla_term_manager,
                                                                  jlong index, jlong element)
{
    BZLA_TRY_PTR_EXPR(bitwuzla_mk_array_sort(TERM_MANAGER, SORT(index), SORT(element)))
}

jlong
Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkBoolSort(JNIEnv* env, jclass native_class,
                                                                 jlong bitwuzla_term_manager)
{
    BZLA_TRY_PTR_EXPR(bitwuzla_mk_bool_sort(TERM_MANAGER))
}

jlong Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkBvSort(JNIEnv* env, jclass native_class,
                                                                     jlong bitwuzla_term_manager,
                                                                     jlong size)
{
    BZLA_TRY_PTR_EXPR(bitwuzla_mk_bv_sort(TERM_MANAGER, static_cast<uint64_t>(size)))
}

jlong Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkFpSort(JNIEnv* env, jclass native_class,
                                                                     jlong bitwuzla_term_manager,
                                                                     jlong exp_size, jlong sig_size)
{
    BZLA_TRY_PTR_EXPR(bitwuzla_mk_fp_sort(TERM_MANAGER, static_cast<uint64_t>(exp_size), static_cast<uint64_t>(sig_size)))
}

jlong Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkFunSort(JNIEnv* env, jclass native_class,
                                                                      jlong bitwuzla_term_manager,
                                                                      jlong arity, jlongArray domain, jlong codomain)
{
    GET_PTR_ARRAY(BitwuzlaSort, domain_ptr, domain);
    BZLA_TRY_PTR_EXPR(bitwuzla_mk_fun_sort(TERM_MANAGER, static_cast<uint64_t>(arity), domain_ptr, SORT(codomain)))
}

jlong
Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkRmSort(JNIEnv* env, jclass native_class,
                                                               jlong bitwuzla_term_manager)
{
    BZLA_TRY_PTR_EXPR(bitwuzla_mk_rm_sort(TERM_MANAGER))
}

jlong
Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkUninterpretedSort(JNIEnv* env, jclass native_class,
                                                                          jlong bitwuzla_term_manager, jstring value)
{
    GET_STRING(nativeString, value);
    BZLA_TRY_PTR_EXPR(bitwuzla_mk_uninterpreted_sort(TERM_MANAGER, nativeString))
}

jlong Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkTrue(JNIEnv* env, jclass native_class, jlong bitwuzla_term_manager) {
    BZLA_TRY_PTR_EXPR(bitwuzla_mk_true(TERM_MANAGER))
}

jlong Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkFalse(JNIEnv* env, jclass native_class, jlong bitwuzla_term_manager) {
    BZLA_TRY_PTR_EXPR(bitwuzla_mk_false(TERM_MANAGER))
}

jlong Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkBvZero(JNIEnv* env, jclass native_class, jlong bitwuzla_term_manager,
                                                                     jlong bitwuzla_sort) {
    BZLA_TRY_PTR_EXPR(bitwuzla_mk_bv_zero(TERM_MANAGER, SORT(bitwuzla_sort)))
}

jlong Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkBvOne(JNIEnv* env, jclass native_class, jlong bitwuzla_term_manager,
                                                                    jlong bitwuzla_sort) {
    BZLA_TRY_PTR_EXPR(bitwuzla_mk_bv_one(TERM_MANAGER, SORT(bitwuzla_sort)))
}

jlong Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkBvOnes(JNIEnv* env, jclass native_class, jlong bitwuzla_term_manager,
                                                                     jlong bitwuzla_sort) {
    BZLA_TRY_PTR_EXPR(bitwuzla_mk_bv_ones(TERM_MANAGER, SORT(bitwuzla_sort)))
}

jlong
Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkBvMinSigned(JNIEnv* env, jclass native_class, jlong bitwuzla_term_manager,
                                                                    jlong bitwuzla_sort) {
    BZLA_TRY_PTR_EXPR(bitwuzla_mk_bv_min_signed(TERM_MANAGER, SORT(bitwuzla_sort)))
}

jlong
Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkBvMaxSigned(JNIEnv* env, jclass native_class, jlong bitwuzla_term_manager,
                                                                    jlong bitwuzla_sort) {
    BZLA_TRY_PTR_EXPR(bitwuzla_mk_bv_max_signed(TERM_MANAGER, SORT(bitwuzla_sort)))
}

jlong
Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkFpPosZero(JNIEnv* env, jclass native_class, jlong bitwuzla_term_manager,
                                                                  jlong bitwuzla_sort) {
    BZLA_TRY_PTR_EXPR(bitwuzla_mk_fp_pos_zero(TERM_MANAGER, SORT(bitwuzla_sort)))
}

jlong
Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkFpNegZero(JNIEnv* env, jclass native_class, jlong bitwuzla_term_manager,
                                                                  jlong bitwuzla_sort) {
    BZLA_TRY_PTR_EXPR(bitwuzla_mk_fp_neg_zero(TERM_MANAGER, SORT(bitwuzla_sort)))
}

jlong
Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkFpPosInf(JNIEnv* env, jclass native_class, jlong bitwuzla_term_manager,
                                                                 jlong bitwuzla_sort) {
    BZLA_TRY_PTR_EXPR(bitwuzla_mk_fp_pos_inf(TERM_MANAGER, SORT(bitwuzla_sort)))
}

jlong
Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkFpNegInf(JNIEnv* env, jclass native_class, jlong bitwuzla_term_manager,
                                                                 jlong bitwuzla_sort) {
    BZLA_TRY_PTR_EXPR(bitwuzla_mk_fp_neg_inf(TERM_MANAGER, SORT(bitwuzla_sort)))
}

jlong Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkFpNan(JNIEnv* env, jclass native_class, jlong bitwuzla_term_manager,
                                                                    jlong bitwuzla_sort) {
    BZLA_TRY_PTR_EXPR(bitwuzla_mk_fp_nan(TERM_MANAGER, SORT(bitwuzla_sort)))
}

jlong Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkBvValue(JNIEnv* env, jclass native_class,
                                                                      jlong bitwuzla_term_manager,
                                                                      jlong bitwuzla_sort, jstring value, jbyte base)
{
    GET_STRING(native_value, value);
    BZLA_TRY_PTR_EXPR(
        bitwuzla_mk_bv_value(TERM_MANAGER, SORT(bitwuzla_sort), native_value, static_cast<uint8_t>(base))
    )
}

jlong
Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkBvValueUint64(JNIEnv* env, jclass native_class,
                                                                      jlong bitwuzla_term_manager,
                                                                      jlong bitwuzla_sort, jlong value)
{
    BZLA_TRY_PTR_EXPR(bitwuzla_mk_bv_value_uint64(TERM_MANAGER, SORT(bitwuzla_sort), static_cast<uint64_t>(value)))
}

jlong
Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkBvValueUint64Array(JNIEnv* env, jclass native_class,
                                                                                 jlong bitwuzla_term_manager,
                                                                                 jlong bv_width, jlongArray value) {
    JniLongArray value_array(env, value);
    uint64_t value_length = env->GetArrayLength(value);

    std::vector<bitwuzla::Term> bv_chunks;

    auto remaining_width = static_cast<uint64_t>(bv_width);
    uint64_t chunk_size = 64;
    uint64_t idx = 0;

    while (remaining_width > 0)
    {
        idx++;
        uint64_t chunk_value = value_array.elements[value_length - idx];
        uint64_t chunk_width = remaining_width > chunk_size ? chunk_size : remaining_width;
        remaining_width -= chunk_size;

        BZLA_TRY_OR_RETURN_ZERO(
            bv_chunks.emplace_back(
                TERM_MANAGER->d_tm.mk_bv_value_uint64(
                    TERM_MANAGER->d_tm.mk_bv_sort(chunk_width),
                    chunk_value
                )
            )
        )
    }

    BZLA_TRY_PTR_EXPR(
        TERM_MANAGER->export_term(
            TERM_MANAGER->d_tm.mk_term(bitwuzla::Kind::BV_CONCAT, bv_chunks)
        )
    )
}



jlong Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkFpValue(JNIEnv* env, jclass native_class, jlong bitwuzla_term_manager,
                                                                      jlong bitwuzla_bvSign, jlong bitwuzla_bvExponent,
                                                                      jlong bitwuzla_bvSignificand) {
    BZLA_TRY_PTR_EXPR(bitwuzla_mk_fp_value(TERM_MANAGER, TERM(bitwuzla_bvSign), TERM(bitwuzla_bvExponent), TERM(bitwuzla_bvSignificand)))
}

jlong
Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkFpValueFromReal(JNIEnv* env, jclass native_class,
                                                                        jlong bitwuzla_term_manager, jlong sort, jlong rm,
                                                                        jstring real) {
    GET_STRING(real_str, real);

    BZLA_TRY_PTR_EXPR(bitwuzla_mk_fp_from_real(TERM_MANAGER, SORT(sort), TERM(rm), real_str))
}

jlong
Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkFpValueFromRational(JNIEnv* env, jclass native_class,
                                                                            jlong bitwuzla_term_manager, jlong sort, jlong rm,
                                                                            jstring num, jstring den) {
    GET_STRING(num_str, num);
    GET_STRING(den_str, den);

    BZLA_TRY_PTR_EXPR(bitwuzla_mk_fp_from_rational(TERM_MANAGER, SORT(sort), TERM(rm), num_str, den_str))
}

jlong Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkRmValue(JNIEnv* env, jclass native_class, jlong bitwuzla_term_manager,
                                                                      jint rm) {
    BZLA_TRY_PTR_EXPR(bitwuzla_mk_rm_value(TERM_MANAGER, static_cast<BitwuzlaRoundingMode>(rm)))
}

jlong Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkTerm1(JNIEnv* env, jclass native_class, jlong bitwuzla_term_manager,
                                                                    jint kind, jlong arg) {
    BZLA_TRY_PTR_EXPR(bitwuzla_mk_term1(TERM_MANAGER, static_cast<BitwuzlaKind>(kind), TERM(arg)))
}

jlong Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkTerm2(JNIEnv* env, jclass native_class,
                                                                    jlong bitwuzla_term_manager,
                                                                    jint kind, jlong arg0, jlong arg1)
{
    BZLA_TRY_PTR_EXPR(bitwuzla_mk_term2(TERM_MANAGER, static_cast<BitwuzlaKind>(kind), TERM(arg0), TERM(arg1)))
}

jlong Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkTerm3(JNIEnv* env, jclass native_class,
                                                                    jlong bitwuzla_term_manager,
                                                                    jint kind, jlong arg0, jlong arg1, jlong arg2)
{
    BZLA_TRY_PTR_EXPR(bitwuzla_mk_term3(TERM_MANAGER, static_cast<BitwuzlaKind>(kind), TERM(arg0), TERM(arg1), TERM(arg2)))
}

jlong Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkTerm(JNIEnv* env, jclass native_class,
                                                                   jlong bitwuzla_term_manager,
                                                                   jint kind, jlongArray args)
{
    GET_PTR_ARRAY(BitwuzlaTerm, args_ptr, args);
    jsize len = env->GetArrayLength(args);

    BZLA_TRY_PTR_EXPR(
        bitwuzla_mk_term(TERM_MANAGER, static_cast<BitwuzlaKind>(kind), static_cast<uint32_t>(len), args_ptr)
    )
}

jlong
Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkTerm1Indexed1(JNIEnv* env, jclass native_class,
                                                                      jlong bitwuzla_term_manager, jint kind,
                                                                      jlong term, jlong idx)
{
    BZLA_TRY_PTR_EXPR(
        bitwuzla_mk_term1_indexed1(
            TERM_MANAGER, static_cast<BitwuzlaKind>(kind), TERM(term), static_cast<uint64_t>(idx)
        )
    )
}

jlong
Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkTerm1Indexed2(JNIEnv* env, jclass native_class,
                                                                      jlong bitwuzla_term_manager,
                                                                      jint kind, jlong term, jlong idx0, jlong idx1)
{
    BZLA_TRY_PTR_EXPR(
        bitwuzla_mk_term1_indexed2(
            TERM_MANAGER, static_cast<BitwuzlaKind>(kind), TERM(term),
            static_cast<uint64_t> (idx0), static_cast<uint64_t>(idx1)
        )
    )
}

jlong
Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkTerm2Indexed1(JNIEnv* env, jclass native_class, jlong bitwuzla_term_manager,
                                                                      jint kind, jlong term0, jlong term1, jlong idx0)
{
    BZLA_TRY_PTR_EXPR(
        bitwuzla_mk_term2_indexed1(
            TERM_MANAGER, static_cast<BitwuzlaKind>(kind), TERM(term0), TERM(term1), static_cast<uint64_t>(idx0)
        )
    )
}

jlong
Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkTerm2Indexed2(JNIEnv* env, jclass native_class, jlong bitwuzla_term_manager,
                                                                      jint kind, jlong term0, jlong term1, jlong idx0,
                                                                      jlong idx1) {
    BZLA_TRY_PTR_EXPR(
        bitwuzla_mk_term2_indexed2(
            TERM_MANAGER, static_cast<BitwuzlaKind>(kind), TERM(term0), TERM(term1),
            static_cast<uint64_t>(idx0), static_cast<uint64_t>(idx1)
        )
    )
}

jlong
Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkTermIndexed(JNIEnv* env, jclass native_class, jlong bitwuzla_term_manager,
                                                                    jint kind, jlongArray args, jlongArray idxs) {
    GET_PTR_ARRAY(BitwuzlaTerm, args_ptr, args);
    jsize argc = env->GetArrayLength(args);

    JniLongArray indices_array(env, idxs);
    jsize idxc = env->GetArrayLength(idxs);

    BZLA_TRY_PTR_EXPR(
        bitwuzla_mk_term_indexed(
            TERM_MANAGER, static_cast<BitwuzlaKind>(kind),
            static_cast<uint32_t>(argc), args_ptr,
            static_cast<uint32_t>(idxc), reinterpret_cast<uint64_t*>(indices_array.elements)
        )
    )
}

jlong Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkConst(JNIEnv* env, jclass native_class, jlong bitwuzla_term_manager,
                                                                    jlong bitwuzla_sort, jstring symbol) {
    GET_STRING(native_symbol, symbol);
    BZLA_TRY_PTR_EXPR(bitwuzla_mk_const(TERM_MANAGER, SORT(bitwuzla_sort), native_symbol))
}

jlong
Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkConstArray(JNIEnv* env, jclass native_class, jlong bitwuzla_term_manager,
                                                                   jlong bitwuzla_sort, jlong value) {
    BZLA_TRY_PTR_EXPR(bitwuzla_mk_const_array(TERM_MANAGER, SORT(bitwuzla_sort), TERM(value)))
}

jlong Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkVar(JNIEnv* env, jclass native_class, jlong bitwuzla_term_manager,
                                                                  jlong bitwuzla_sort, jstring symbol) {
    GET_STRING(native_symbol, symbol);
    BZLA_TRY_PTR_EXPR(bitwuzla_mk_var(TERM_MANAGER, SORT(bitwuzla_sort), native_symbol))
}

void Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaPush(JNIEnv* env, jclass native_class, jlong bitwuzla,
                                                                jlong n_levels) {
    BZLA_TRY_VOID_EXPR(bitwuzla_push(reinterpret_cast<Bitwuzla*>(bitwuzla), static_cast<uint64_t>(n_levels)))
}

void Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaPop(JNIEnv* env, jclass native_class, jlong bitwuzla,
                                                               jlong n_levels) {
    BZLA_TRY_VOID_EXPR(bitwuzla_pop(reinterpret_cast<Bitwuzla*>(bitwuzla), static_cast<uint64_t>(n_levels)))
}

void Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaAssert(JNIEnv* env, jclass native_class, jlong bitwuzla,
                                                                  jlong term) {
    BZLA_TRY_VOID_EXPR(bitwuzla_assert(reinterpret_cast<Bitwuzla*>(bitwuzla), TERM(term)))
}

jboolean
Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaIsUnsatAssumption(JNIEnv* env, jclass native_class,
                                                                        jlong bitwuzla, jlong term) {
    BZLA_TRY_BOOL_EXPR(bitwuzla_is_unsat_assumption(reinterpret_cast<Bitwuzla*>(bitwuzla), TERM(term)))
}

jlongArray Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaGetUnsatAssumptions(JNIEnv* env, jclass native_class,
    jlong bitwuzla)
{
    size_t len = 0;
    BitwuzlaTerm* array;
    BZLA_TRY_OR_RETURN_NULL(array = bitwuzla_get_unsat_assumptions(reinterpret_cast<Bitwuzla*>(bitwuzla), &len))

    jlongArray result = create_ptr_array(env, len);
    set_ptr_array(env, result, array, len);
    return result;
}

jlongArray
Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaGetUnsatCore(JNIEnv* env, jclass native_class, jlong bitwuzla) {
    size_t len = 0;
    BitwuzlaTerm* array;
    BZLA_TRY_OR_RETURN_NULL(array = bitwuzla_get_unsat_core(reinterpret_cast<Bitwuzla*>(bitwuzla), &len))

    jlongArray result = create_ptr_array(env, len);
    set_ptr_array(env, result, array, len);
    return result;
}

void
Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaSimplify(JNIEnv* env, jclass native_class, jlong bitwuzla) {
    BZLA_TRY_VOID_EXPR(bitwuzla_simplify(reinterpret_cast<Bitwuzla*>(bitwuzla)))
}

jint Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaCheckSat(JNIEnv* env, jclass native_class, jlong bitwuzla) {
    BZLA_TRY_INT32_EXPR(bitwuzla_check_sat(reinterpret_cast<Bitwuzla*>(bitwuzla)))
}

jint Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaCheckSatAssuming(
    JNIEnv* env, jclass native_class, jlong bitwuzla, jlongArray assumptions)
{
    GET_PTR_ARRAY(BitwuzlaTerm, assuptions_ptr, assumptions);
    jsize len = env->GetArrayLength(assumptions);

    BZLA_TRY_INT32_EXPR(
        bitwuzla_check_sat_assuming(
            reinterpret_cast<Bitwuzla*>(bitwuzla),
            static_cast<uint32_t>(len),
            assuptions_ptr
        )
    )
}

jint
Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaCheckSatTimeout(JNIEnv* env, jclass native_class, jlong bitwuzla,
                                                                      jlong timeout)
{
    auto bzla = reinterpret_cast<Bitwuzla*>(bitwuzla);

    auto termination_state = get_termination_state(bzla);
    ScopedTimeout _timeout(termination_state, timeout);

    BZLA_TRY_INT32_EXPR(bitwuzla_check_sat(bzla))
}

jint Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaCheckSatAssumingTimeout(
    JNIEnv* env, jclass native_class, jlong bitwuzla, jlongArray assumptions, jlong timeout)
{
    auto bzla = reinterpret_cast<Bitwuzla*>(bitwuzla);

    auto termination_state = get_termination_state(bzla);
    ScopedTimeout _timeout(termination_state, timeout);

    GET_PTR_ARRAY(BitwuzlaTerm, assuptions_ptr, assumptions);
    jsize len = env->GetArrayLength(assumptions);

    BZLA_TRY_INT32_EXPR(
        bitwuzla_check_sat_assuming(bzla,static_cast<uint32_t>(len), assuptions_ptr)
    )
}

void
Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaForceTerminate(JNIEnv* env, jclass native_class,
                                                                     jlong bitwuzla)
{
    auto termination_state = get_termination_state(reinterpret_cast<Bitwuzla*>(bitwuzla));
    termination_state->terminate();
}

jlong Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaGetValue(JNIEnv* env, jclass native_class, jlong bitwuzla,
                                                                     jlong bitwuzla_term) {
    BZLA_TRY_PTR_EXPR(bitwuzla_get_value(reinterpret_cast<Bitwuzla*>(bitwuzla), TERM(bitwuzla_term)))
}

jboolean
Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermValueGetBool(JNIEnv* env, jclass native_class,
                                                                     jlong bitwuzla_term) {
    BZLA_TRY_BOOL_EXPR(bitwuzla_term_value_get_bool(TERM(bitwuzla_term)))
}

jint
Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermValueGetRm(JNIEnv* env, jclass native_class,
                                                                     jlong bitwuzla_term) {
    BZLA_TRY_INT32_EXPR(bitwuzla_term_value_get_rm(TERM(bitwuzla_term)))
}

ACCESS_PRIVATE_FIELD(bitwuzla::Term, std::shared_ptr<bzla::Node>, d_node)

jlong
Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermValueGetBvUint64(JNIEnv* env, jclass native_class,
                                                                           jlong bitwuzla_term)
{
    auto&& term = BitwuzlaTermManager::import_term(TERM(bitwuzla_term));
    const bzla::BitVector& bv_value = access_private::d_node(term)->value<bzla::BitVector>();
    return static_cast<jlong>(bv_value.to_uint64());
}

jlong
Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermValueGetFpUint64(JNIEnv* env, jclass native_class,
                                                                     jlong bitwuzla_term) {
    auto&& term = BitwuzlaTermManager::import_term(TERM(bitwuzla_term));

    const bzla::FloatingPoint &fp_value = access_private::d_node(term)->value<bzla::FloatingPoint>();
    const bzla::BitVector &bv_value = fp_value.as_bv(); // IEEE 754 bv
    return static_cast<jlong>(bv_value.to_uint64());
}

jlongArray get_bv_bits_as_jlong_array(JNIEnv* env, const bzla::BitVector& bv_value) {
    auto width = static_cast<int64_t>( bv_value.size());

    int64_t idx = 0;
    int64_t chunk_size = 64;
    int64_t remaining_width = width;
    std::vector<uint64_t> chunks;

    while (remaining_width > 0) {
        int64_t lower = idx * chunk_size;
        int64_t upper = std::min(lower + chunk_size, width);
        idx++;
        remaining_width -= chunk_size;

        auto&& chunk_bv = bv_value.bvextract(upper, lower);
        chunks.push_back(chunk_bv.to_uint64());
    }

    auto length = static_cast<jsize>(chunks.size());
    jlongArray result = env->NewLongArray(length);
    env->SetLongArrayRegion(result, 0, length, reinterpret_cast<jlong*>(chunks.data()));

    return result;
}

jlongArray
Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermValueGetBvUint64Array(JNIEnv* env, jclass native_class,
                                                                           jlong bitwuzla_term)
{
    auto&& term = BitwuzlaTermManager::import_term(TERM(bitwuzla_term));
    const bzla::BitVector& bv_value = access_private::d_node(term)->value<bzla::BitVector>();

    return get_bv_bits_as_jlong_array(env, bv_value);
}

jlongArray
Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermValueGetFpUint64Array(JNIEnv* env, jclass native_class,
                                                                           jlong bitwuzla_term)
{
    auto&& term = BitwuzlaTermManager::import_term(TERM(bitwuzla_term));
    const bzla::FloatingPoint &fp_value = access_private::d_node(term)->value<bzla::FloatingPoint>();
    const bzla::BitVector &bv_value = fp_value.as_bv(); // IEEE 754 bv

    return get_bv_bits_as_jlong_array(env, bv_value);
}

jlong
Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaSubstituteTerm(JNIEnv* env, jclass native_class,
                                                                     jlong bitwuzla_term, jlongArray map_keys,
                                                                     jlongArray map_values) {

    GET_PTR_ARRAY(BitwuzlaTerm, keys_ptr, map_keys);
    GET_PTR_ARRAY(BitwuzlaTerm, values_ptr, map_values);
    jsize num_keys = env->GetArrayLength(map_keys);

    BZLA_TRY_PTR_EXPR(bitwuzla_substitute_term(TERM(bitwuzla_term), num_keys, keys_ptr, values_ptr))
}

jlongArray
Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaSubstituteTerms(JNIEnv* env, jclass native_class,
                                                                     jlongArray terms, jlongArray map_keys,
                                                                     jlongArray map_values)
{
    GET_PTR_ARRAY(BitwuzlaTerm, terms_ptr, terms);
    jsize num_terms = env->GetArrayLength(terms);

    GET_PTR_ARRAY(BitwuzlaTerm, keys_ptr, map_keys);
    GET_PTR_ARRAY(BitwuzlaTerm, values_ptr, map_values);
    jsize num_keys = env->GetArrayLength(map_keys);

    BZLA_TRY_OR_RETURN_NULL(bitwuzla_substitute_terms(num_terms, terms_ptr, num_keys, keys_ptr, values_ptr))

    jlongArray result_terms = create_ptr_array(env, num_terms);
    set_ptr_array(env, result_terms, terms_ptr, num_terms);

    return result_terms;
}

jlong
Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaSortHash(JNIEnv* env, jclass native_class, jlong bitwuzla_sort) {
    BZLA_TRY_UINT64_EXPR(bitwuzla_sort_hash(SORT(bitwuzla_sort)))
}

jint Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermGetKind(JNIEnv* env, jclass native_class,
                                                                       jlong bitwuzla_term) {
    BZLA_TRY_INT32_EXPR(bitwuzla_term_get_kind(TERM(bitwuzla_term)))
}

jlongArray Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermGetChildren(JNIEnv* env, jclass native_class,
    jlong bitwuzla_term)
{
    size_t len = 0;
    BitwuzlaTerm* array;

    BZLA_TRY_OR_RETURN_NULL(array = bitwuzla_term_get_children(TERM(bitwuzla_term), &len))

    jlongArray result = create_ptr_array(env, len);
    set_ptr_array(env, result, array, len);
    return result;
}

jlongArray Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermGetIndices(JNIEnv* env, jclass native_class,
                                                                               jlong bitwuzla_term)
{
    size_t len = 0;
    uint64_t* array;

    BZLA_TRY_OR_RETURN_NULL(array = bitwuzla_term_get_indices(TERM(bitwuzla_term), &len))

    jlongArray result = env->NewLongArray(static_cast<jsize>(len));
    env->SetLongArrayRegion(result, 0, static_cast<jsize>(len), reinterpret_cast<jlong*>(array));
    return result;
}

jlong Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaSortBvGetSize(JNIEnv* env, jclass native_class,
                                                                          jlong bitwuzla_sort)
{
    BZLA_TRY_UINT64_EXPR(bitwuzla_sort_bv_get_size(SORT(bitwuzla_sort)))
}

jlong Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaSortFpGetExpSize(JNIEnv* env, jclass native_class,
                                                                            jlong bitwuzla_sort) {
    BZLA_TRY_UINT64_EXPR(bitwuzla_sort_fp_get_exp_size(SORT(bitwuzla_sort)))
}

jlong Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaSortFpGetSigSize(JNIEnv* env, jclass native_class,
                                                                            jlong bitwuzla_sort) {
    BZLA_TRY_UINT64_EXPR( bitwuzla_sort_fp_get_sig_size(SORT(bitwuzla_sort)))
}

jlong Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaSortArrayGetIndex(JNIEnv* env, jclass native_class,
                                                                              jlong bitwuzla_sort) {
    BZLA_TRY_PTR_EXPR(bitwuzla_sort_array_get_index(SORT(bitwuzla_sort)))
}

jlong Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaSortArrayGetElement(JNIEnv* env, jclass native_class,
                                                                                jlong bitwuzla_sort) {
    BZLA_TRY_PTR_EXPR(bitwuzla_sort_array_get_element(SORT(bitwuzla_sort)))
}

jlongArray
Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaSortFunGetDomainSorts(JNIEnv* env, jclass native_class,
                                                                            jlong bitwuzla_sort) {
    size_t result_size = 0;
    BitwuzlaSort* result;

    BZLA_TRY_OR_RETURN_NULL(result = bitwuzla_sort_fun_get_domain_sorts(SORT(bitwuzla_sort), &result_size))

    jlongArray result_array = create_ptr_array(env, result_size);
    set_ptr_array(env, result_array, result, result_size);

    return result_array;
}

jlong
Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaSortFunGetCodomain(JNIEnv* env, jclass native_class,
                                                                         jlong bitwuzla_sort) {
    BZLA_TRY_PTR_EXPR(bitwuzla_sort_fun_get_codomain(SORT(bitwuzla_sort)))
}

jlong
Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaSortFunGetArity(JNIEnv* env, jclass native_class,
                                                                      jlong bitwuzla_sort) {
    BZLA_TRY_UINT64_EXPR(bitwuzla_sort_fun_get_arity(SORT(bitwuzla_sort)))
}

jstring
Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaSortGetUninterpretedSymbol(
    JNIEnv* env, jclass native_class, jlong bitwuzla_sort)
{
    BZLA_TRY_STRING_EXPR(bitwuzla_sort_get_uninterpreted_symbol(SORT(bitwuzla_sort)))
}

jboolean Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaSortIsArray(JNIEnv* env, jclass native_class,
                                                                           jlong bitwuzla_sort) {
    BZLA_TRY_BOOL_EXPR(bitwuzla_sort_is_array(SORT(bitwuzla_sort)))
}

jboolean
Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaSortIsBool(JNIEnv* env, jclass native_class, jlong bitwuzla_sort) {
    BZLA_TRY_BOOL_EXPR(bitwuzla_sort_is_bool(SORT(bitwuzla_sort)))
}

jboolean
Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaSortIsBv(JNIEnv* env, jclass native_class, jlong bitwuzla_sort) {
    BZLA_TRY_BOOL_EXPR(bitwuzla_sort_is_bv(SORT(bitwuzla_sort)))
}

jboolean
Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaSortIsFp(JNIEnv* env, jclass native_class, jlong bitwuzla_sort) {
    BZLA_TRY_BOOL_EXPR(bitwuzla_sort_is_fp(SORT(bitwuzla_sort)))
}

jboolean Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaSortIsFun(JNIEnv* env, jclass native_class,
                                                                         jlong bitwuzla_sort) {
    BZLA_TRY_BOOL_EXPR(bitwuzla_sort_is_fun(SORT(bitwuzla_sort)))
}

jboolean
Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaSortIsRm(JNIEnv* env, jclass native_class, jlong bitwuzla_sort) {
    BZLA_TRY_BOOL_EXPR(bitwuzla_sort_is_rm(SORT(bitwuzla_sort)))
}

jboolean
Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaSortIsUninterpreted(JNIEnv* env, jclass native_class, jlong bitwuzla_sort) {
    BZLA_TRY_BOOL_EXPR(bitwuzla_sort_is_uninterpreted(SORT(bitwuzla_sort)))
}

jlong
Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermHash(JNIEnv* env, jclass native_class, jlong bitwuzla_term) {
    BZLA_TRY_UINT64_EXPR(bitwuzla_term_hash(TERM(bitwuzla_term)))
}

jboolean Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsIndexed(JNIEnv* env, jclass native_class,
                                                                             jlong bitwuzla_term) {
    BZLA_TRY_BOOL_EXPR(bitwuzla_term_is_indexed(TERM(bitwuzla_term)))
}

jlong Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermGetSort(JNIEnv* env, jclass native_class,
                                                                        jlong bitwuzla_term) {
    BZLA_TRY_PTR_EXPR(bitwuzla_term_get_sort(TERM(bitwuzla_term)))
}

jlong Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermArrayGetIndexSort(JNIEnv* env, jclass native_class,
                                                                                  jlong bitwuzla_term) {
    BZLA_TRY_PTR_EXPR(bitwuzla_term_array_get_index_sort(TERM(bitwuzla_term)))
}

jlong Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermArrayGetElementSort(JNIEnv* env, jclass native_class,
                                                                                    jlong bitwuzla_term) {
    BZLA_TRY_PTR_EXPR(bitwuzla_term_array_get_element_sort(TERM(bitwuzla_term)))
}

jlongArray
Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermFunGetDomainSorts(JNIEnv* env, jclass native_class,
                                                                            jlong bitwuzla_term) {
    size_t result_size = 0;
    BitwuzlaSort* result;

    BZLA_TRY_OR_RETURN_NULL(result = bitwuzla_term_fun_get_domain_sorts(TERM(bitwuzla_term), &result_size))

    jlongArray result_array = create_ptr_array(env, result_size);
    set_ptr_array(env, result_array, result, result_size);

    return result_array;
}

jlong Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermFunGetCodomainSort(JNIEnv* env, jclass native_class,
                                                                                   jlong bitwuzla_term) {
    BZLA_TRY_PTR_EXPR(bitwuzla_term_fun_get_codomain_sort(TERM(bitwuzla_term)))
}

jlong Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermBvGetSize(JNIEnv* env, jclass native_class,
                                                                         jlong bitwuzla_term) {
    BZLA_TRY_UINT64_EXPR(bitwuzla_term_bv_get_size(TERM(bitwuzla_term)))
}

jlong Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermFpGetExpSize(JNIEnv* env, jclass native_class,
                                                                            jlong bitwuzla_term) {
    BZLA_TRY_UINT64_EXPR(bitwuzla_term_fp_get_exp_size(TERM(bitwuzla_term)))
}

jlong Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermFpGetSigSize(JNIEnv* env, jclass native_class,
                                                                            jlong bitwuzla_term) {
    BZLA_TRY_UINT64_EXPR(bitwuzla_term_fp_get_sig_size(TERM(bitwuzla_term)))
}

jlong Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermFunGetArity(JNIEnv* env, jclass native_class,
                                                                           jlong bitwuzla_term) {
    BZLA_TRY_UINT64_EXPR(bitwuzla_term_fun_get_arity(TERM(bitwuzla_term)))
}

jstring Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermGetSymbol(JNIEnv* env, jclass native_class,
                                                                            jlong bitwuzla_term)
{
    BZLA_TRY_STRING_EXPR(bitwuzla_term_get_symbol(TERM(bitwuzla_term)))
}

jboolean Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsEqualSort(JNIEnv* env, jclass native_class,
                                                                               jlong bitwuzla_term_1,
                                                                               jlong bitwuzla_term_2) {
    BZLA_TRY_BOOL_EXPR(bitwuzla_term_is_equal_sort(TERM(bitwuzla_term_1), TERM(bitwuzla_term_2)))
}

jboolean Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsArray(JNIEnv* env, jclass native_class,
                                                                           jlong bitwuzla_term) {
    BZLA_TRY_BOOL_EXPR(bitwuzla_term_is_array(TERM(bitwuzla_term)))
}

jboolean Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsConst(JNIEnv* env, jclass native_class,
                                                                           jlong bitwuzla_term) {
    BZLA_TRY_BOOL_EXPR(bitwuzla_term_is_const(TERM(bitwuzla_term)))
}

jboolean Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsFun(JNIEnv* env, jclass native_class,
                                                                         jlong bitwuzla_term) {
    BZLA_TRY_BOOL_EXPR(bitwuzla_term_is_fun(TERM(bitwuzla_term)))
}

jboolean Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsVar(JNIEnv* env, jclass native_class,
                                                                         jlong bitwuzla_term) {
    BZLA_TRY_BOOL_EXPR(bitwuzla_term_is_var(TERM(bitwuzla_term)))
}

jboolean Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsValue(JNIEnv* env, jclass native_class,
                                                                           jlong bitwuzla_term) {
    BZLA_TRY_BOOL_EXPR(bitwuzla_term_is_value(TERM(bitwuzla_term)))
}

jboolean Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsBvValue(JNIEnv* env, jclass native_class,
                                                                             jlong bitwuzla_term) {
    BZLA_TRY_BOOL_EXPR(bitwuzla_term_is_bv_value(TERM(bitwuzla_term)))
}

jboolean Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsFpValue(JNIEnv* env, jclass native_class,
                                                                             jlong bitwuzla_term) {
    BZLA_TRY_BOOL_EXPR(bitwuzla_term_is_fp_value(TERM(bitwuzla_term)))
}

jboolean Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsRmValue(JNIEnv* env, jclass native_class,
                                                                             jlong bitwuzla_term) {
    BZLA_TRY_BOOL_EXPR(bitwuzla_term_is_rm_value(TERM(bitwuzla_term)))
}

jboolean
Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsBool(JNIEnv* env, jclass native_class, jlong bitwuzla_term) {
    BZLA_TRY_BOOL_EXPR(bitwuzla_term_is_bool(TERM(bitwuzla_term)))
}

jboolean
Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsBv(JNIEnv* env, jclass native_class, jlong bitwuzla_term) {
    BZLA_TRY_BOOL_EXPR(bitwuzla_term_is_bv(TERM(bitwuzla_term)))
}

jboolean
Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsFp(JNIEnv* env, jclass native_class, jlong bitwuzla_term) {
    BZLA_TRY_BOOL_EXPR(bitwuzla_term_is_fp(TERM(bitwuzla_term)))
}

jboolean
Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsRm(JNIEnv* env, jclass native_class, jlong bitwuzla_term) {
    BZLA_TRY_BOOL_EXPR(bitwuzla_term_is_rm(TERM(bitwuzla_term)))
}

jboolean
Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsUninterpreted(JNIEnv* env, jclass native_class, jlong bitwuzla_term) {
    BZLA_TRY_BOOL_EXPR(bitwuzla_term_is_uninterpreted(TERM(bitwuzla_term)))
}

jboolean
Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsTrue(JNIEnv* env, jclass native_class, jlong bitwuzla_term) {
    BZLA_TRY_BOOL_EXPR(bitwuzla_term_is_true(TERM(bitwuzla_term)))
}

jboolean
Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsFalse(JNIEnv* env, jclass native_class, jlong bitwuzla_term) {
    BZLA_TRY_BOOL_EXPR(bitwuzla_term_is_false(TERM(bitwuzla_term)))
}

jboolean Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsBvValueZero(JNIEnv* env, jclass native_class,
                                                                                 jlong bitwuzla_term) {
    BZLA_TRY_BOOL_EXPR(bitwuzla_term_is_bv_value_zero(TERM(bitwuzla_term)))
}

jboolean Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsBvValueOne(JNIEnv* env, jclass native_class,
                                                                                jlong bitwuzla_term) {
    BZLA_TRY_BOOL_EXPR(bitwuzla_term_is_bv_value_one(TERM(bitwuzla_term)))
}

jboolean Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsBvValueOnes(JNIEnv* env, jclass native_class,
                                                                                 jlong bitwuzla_term) {
    BZLA_TRY_BOOL_EXPR(bitwuzla_term_is_bv_value_ones(TERM(bitwuzla_term)))
}

jboolean Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsBvValueMinSigned(JNIEnv* env, jclass native_class,
                                                                                      jlong bitwuzla_term) {
    BZLA_TRY_BOOL_EXPR(bitwuzla_term_is_bv_value_min_signed(TERM(bitwuzla_term)))
}

jboolean Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsBvValueMaxSigned(JNIEnv* env, jclass native_class,
                                                                                      jlong bitwuzla_term) {
    BZLA_TRY_BOOL_EXPR(bitwuzla_term_is_bv_value_max_signed(TERM(bitwuzla_term)))
}

jboolean Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsFpValuePosZero(JNIEnv* env, jclass native_class,
                                                                                    jlong bitwuzla_term) {
    BZLA_TRY_BOOL_EXPR(bitwuzla_term_is_fp_value_pos_zero(TERM(bitwuzla_term)))
}

jboolean Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsFpValueNegZero(JNIEnv* env, jclass native_class,
                                                                                    jlong bitwuzla_term) {
    BZLA_TRY_BOOL_EXPR(bitwuzla_term_is_fp_value_neg_zero(TERM(bitwuzla_term)))
}

jboolean Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsFpValuePosInf(JNIEnv* env, jclass native_class,
                                                                                   jlong bitwuzla_term) {
    BZLA_TRY_BOOL_EXPR(bitwuzla_term_is_fp_value_pos_inf(TERM(bitwuzla_term)))
}

jboolean Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsFpValueNegInf(JNIEnv* env, jclass native_class,
                                                                                   jlong bitwuzla_term) {
    BZLA_TRY_BOOL_EXPR(bitwuzla_term_is_fp_value_neg_inf(TERM(bitwuzla_term)))
}

jboolean Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsFpValueNan(JNIEnv* env, jclass native_class,
                                                                                jlong bitwuzla_term) {
    BZLA_TRY_BOOL_EXPR(bitwuzla_term_is_fp_value_nan(TERM(bitwuzla_term)))
}

jboolean Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsRmValueRna(JNIEnv* env, jclass native_class,
                                                                                jlong bitwuzla_term) {
    BZLA_TRY_BOOL_EXPR(bitwuzla_term_is_rm_value_rna(TERM(bitwuzla_term)))
}

jboolean Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsRmValueRne(JNIEnv* env, jclass native_class,
                                                                                jlong bitwuzla_term) {
    BZLA_TRY_BOOL_EXPR(bitwuzla_term_is_rm_value_rne(TERM(bitwuzla_term)))
}

jboolean Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsRmValueRtn(JNIEnv* env, jclass native_class,
                                                                                jlong bitwuzla_term) {
    BZLA_TRY_BOOL_EXPR(bitwuzla_term_is_rm_value_rtn(TERM(bitwuzla_term)))
}

jboolean Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsRmValueRtp(JNIEnv* env, jclass native_class,
                                                                                jlong bitwuzla_term) {
    BZLA_TRY_BOOL_EXPR(bitwuzla_term_is_rm_value_rtp(TERM(bitwuzla_term)))
}

jboolean Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsRmValueRtz(JNIEnv* env, jclass native_class,
                                                                                jlong bitwuzla_term) {
    BZLA_TRY_BOOL_EXPR(bitwuzla_term_is_rm_value_rtz(TERM(bitwuzla_term)))
}

void
Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaDumpFormula(JNIEnv* env, jclass native_class, jlong bitwuzla,
                                                                  jstring format, jstring file_path) {
    GET_STRING(print_format, format);
    GET_STRING(path, file_path);

    const auto f = fopen(path, "w");
    BZLA_API_TRY(
        bitwuzla_print_formula(reinterpret_cast<Bitwuzla*>(bitwuzla), print_format, f, 16),
        fclose(f),
        (void) fclose(f)
    )
}

jstring Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaSortToString(JNIEnv* env, jclass native_class, jlong sort) {
    BZLA_TRY_STRING_EXPR(bitwuzla_sort_to_string(SORT(sort)))
}

jstring Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermToString(JNIEnv* env, jclass native_class, jlong term) {
    BZLA_TRY_STRING_EXPR(bitwuzla_term_to_string(TERM(term)))
}

jstring Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaResultToString(JNIEnv* env, jclass native_class, jint result) {
    BZLA_TRY_STRING_EXPR(bitwuzla_result_to_string(static_cast<BitwuzlaResult>(result)))
}

jstring Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaRmToString(JNIEnv* env, jclass native_class, jint rm) {
    BZLA_TRY_STRING_EXPR(bitwuzla_rm_to_string(static_cast<BitwuzlaRoundingMode>(rm)))
}

jstring Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaKindToString(JNIEnv* env, jclass native_class, jint kind) {
    BZLA_TRY_STRING_EXPR(bitwuzla_kind_to_string(static_cast<BitwuzlaKind>(kind)))
}

jstring Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaOptionToString(
    JNIEnv* env, jclass native_class, jint option)
{
    BitwuzlaOptions options{};
    BitwuzlaOptionInfo info{};
    BZLA_TRY_OR_RETURN_NULL(bitwuzla_get_option_info(&options, static_cast<BitwuzlaOption>(option), &info))
    BZLA_TRY_STRING_EXPR(info.lng)
}

jstring Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaCopyright(JNIEnv* env, jclass native_class) {
    BZLA_TRY_STRING_EXPR(bitwuzla_copyright())
}

jstring Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaVersion(JNIEnv* env, jclass native_class) {
    BZLA_TRY_STRING_EXPR(bitwuzla_version())
}

jstring Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaGitId(JNIEnv* env, jclass native_class) {
    BZLA_TRY_STRING_EXPR(bitwuzla_git_id())
}
