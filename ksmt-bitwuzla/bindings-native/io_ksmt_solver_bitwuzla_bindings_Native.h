/* DO NOT EDIT THIS FILE - it is machine generated */
#include <jni.h>
/* Header for class io_ksmt_solver_bitwuzla_bindings_Native */

#ifndef _Included_io_ksmt_solver_bitwuzla_bindings_Native
#define _Included_io_ksmt_solver_bitwuzla_bindings_Native
#ifdef __cplusplus
extern "C" {
#endif
/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaInit
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaInit
  (JNIEnv *, jclass);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaCopyright
 * Signature: ()Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaCopyright
  (JNIEnv *, jclass);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaVersion
 * Signature: ()Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaVersion
  (JNIEnv *, jclass);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaGitId
 * Signature: ()Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaGitId
  (JNIEnv *, jclass);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaTermManagerNew
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermManagerNew
  (JNIEnv *, jclass);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaTermManagerRelease
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermManagerRelease
  (JNIEnv *, jclass, jlong);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaTermManagerDelete
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermManagerDelete
  (JNIEnv *, jclass, jlong);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaSortRelease
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaSortRelease
  (JNIEnv *, jclass, jlong);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaTermRelease
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermRelease
  (JNIEnv *, jclass, jlong);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaNew
 * Signature: (JJ)J
 */
JNIEXPORT jlong JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaNew
  (JNIEnv *, jclass, jlong, jlong);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaDelete
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaDelete
  (JNIEnv *, jclass, jlong);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaOptionsNew
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaOptionsNew
  (JNIEnv *, jclass);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaOptionsDelete
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaOptionsDelete
  (JNIEnv *, jclass, jlong);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaSetOption
 * Signature: (JIJ)V
 */
JNIEXPORT void JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaSetOption
  (JNIEnv *, jclass, jlong, jint, jlong);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaSetOptionMode
 * Signature: (JILjava/lang/String;)V
 */
JNIEXPORT void JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaSetOptionMode
  (JNIEnv *, jclass, jlong, jint, jstring);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaGetOption
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaGetOption
  (JNIEnv *, jclass, jlong, jint);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaGetOptionMode
 * Signature: (JI)Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaGetOptionMode
  (JNIEnv *, jclass, jlong, jint);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaMkArraySort
 * Signature: (JJJ)J
 */
JNIEXPORT jlong JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkArraySort
  (JNIEnv *, jclass, jlong, jlong, jlong);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaMkBoolSort
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkBoolSort
  (JNIEnv *, jclass, jlong);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaMkBvSort
 * Signature: (JJ)J
 */
JNIEXPORT jlong JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkBvSort
  (JNIEnv *, jclass, jlong, jlong);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaMkFpSort
 * Signature: (JJJ)J
 */
JNIEXPORT jlong JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkFpSort
  (JNIEnv *, jclass, jlong, jlong, jlong);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaMkFunSort
 * Signature: (JJ[JJ)J
 */
JNIEXPORT jlong JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkFunSort
  (JNIEnv *, jclass, jlong, jlong, jlongArray, jlong);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaMkRmSort
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkRmSort
  (JNIEnv *, jclass, jlong);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaMkUninterpretedSort
 * Signature: (JLjava/lang/String;)J
 */
JNIEXPORT jlong JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkUninterpretedSort
  (JNIEnv *, jclass, jlong, jstring);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaMkTrue
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkTrue
  (JNIEnv *, jclass, jlong);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaMkFalse
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkFalse
  (JNIEnv *, jclass, jlong);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaMkBvZero
 * Signature: (JJ)J
 */
JNIEXPORT jlong JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkBvZero
  (JNIEnv *, jclass, jlong, jlong);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaMkBvOne
 * Signature: (JJ)J
 */
JNIEXPORT jlong JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkBvOne
  (JNIEnv *, jclass, jlong, jlong);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaMkBvOnes
 * Signature: (JJ)J
 */
JNIEXPORT jlong JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkBvOnes
  (JNIEnv *, jclass, jlong, jlong);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaMkBvMinSigned
 * Signature: (JJ)J
 */
JNIEXPORT jlong JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkBvMinSigned
  (JNIEnv *, jclass, jlong, jlong);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaMkBvMaxSigned
 * Signature: (JJ)J
 */
JNIEXPORT jlong JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkBvMaxSigned
  (JNIEnv *, jclass, jlong, jlong);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaMkFpPosZero
 * Signature: (JJ)J
 */
JNIEXPORT jlong JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkFpPosZero
  (JNIEnv *, jclass, jlong, jlong);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaMkFpNegZero
 * Signature: (JJ)J
 */
JNIEXPORT jlong JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkFpNegZero
  (JNIEnv *, jclass, jlong, jlong);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaMkFpPosInf
 * Signature: (JJ)J
 */
JNIEXPORT jlong JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkFpPosInf
  (JNIEnv *, jclass, jlong, jlong);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaMkFpNegInf
 * Signature: (JJ)J
 */
JNIEXPORT jlong JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkFpNegInf
  (JNIEnv *, jclass, jlong, jlong);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaMkFpNan
 * Signature: (JJ)J
 */
JNIEXPORT jlong JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkFpNan
  (JNIEnv *, jclass, jlong, jlong);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaMkBvValue
 * Signature: (JJLjava/lang/String;B)J
 */
JNIEXPORT jlong JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkBvValue
  (JNIEnv *, jclass, jlong, jlong, jstring, jbyte);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaMkBvValueUint64
 * Signature: (JJJ)J
 */
JNIEXPORT jlong JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkBvValueUint64
  (JNIEnv *, jclass, jlong, jlong, jlong);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaMkBvValueUint64Array
 * Signature: (JJ[J)J
 */
JNIEXPORT jlong JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkBvValueUint64Array
  (JNIEnv *, jclass, jlong, jlong, jlongArray);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaMkFpValue
 * Signature: (JJJJ)J
 */
JNIEXPORT jlong JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkFpValue
  (JNIEnv *, jclass, jlong, jlong, jlong, jlong);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaMkFpValueFromReal
 * Signature: (JJJLjava/lang/String;)J
 */
JNIEXPORT jlong JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkFpValueFromReal
  (JNIEnv *, jclass, jlong, jlong, jlong, jstring);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaMkFpValueFromRational
 * Signature: (JJJLjava/lang/String;Ljava/lang/String;)J
 */
JNIEXPORT jlong JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkFpValueFromRational
  (JNIEnv *, jclass, jlong, jlong, jlong, jstring, jstring);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaMkRmValue
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkRmValue
  (JNIEnv *, jclass, jlong, jint);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaMkTerm1
 * Signature: (JIJ)J
 */
JNIEXPORT jlong JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkTerm1
  (JNIEnv *, jclass, jlong, jint, jlong);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaMkTerm2
 * Signature: (JIJJ)J
 */
JNIEXPORT jlong JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkTerm2
  (JNIEnv *, jclass, jlong, jint, jlong, jlong);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaMkTerm3
 * Signature: (JIJJJ)J
 */
JNIEXPORT jlong JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkTerm3
  (JNIEnv *, jclass, jlong, jint, jlong, jlong, jlong);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaMkTerm
 * Signature: (JI[J)J
 */
JNIEXPORT jlong JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkTerm
  (JNIEnv *, jclass, jlong, jint, jlongArray);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaMkTerm1Indexed1
 * Signature: (JIJJ)J
 */
JNIEXPORT jlong JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkTerm1Indexed1
  (JNIEnv *, jclass, jlong, jint, jlong, jlong);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaMkTerm1Indexed2
 * Signature: (JIJJJ)J
 */
JNIEXPORT jlong JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkTerm1Indexed2
  (JNIEnv *, jclass, jlong, jint, jlong, jlong, jlong);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaMkTerm2Indexed1
 * Signature: (JIJJJ)J
 */
JNIEXPORT jlong JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkTerm2Indexed1
  (JNIEnv *, jclass, jlong, jint, jlong, jlong, jlong);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaMkTerm2Indexed2
 * Signature: (JIJJJJ)J
 */
JNIEXPORT jlong JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkTerm2Indexed2
  (JNIEnv *, jclass, jlong, jint, jlong, jlong, jlong, jlong);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaMkTermIndexed
 * Signature: (JI[J[J)J
 */
JNIEXPORT jlong JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkTermIndexed
  (JNIEnv *, jclass, jlong, jint, jlongArray, jlongArray);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaMkConst
 * Signature: (JJLjava/lang/String;)J
 */
JNIEXPORT jlong JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkConst
  (JNIEnv *, jclass, jlong, jlong, jstring);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaMkConstArray
 * Signature: (JJJ)J
 */
JNIEXPORT jlong JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkConstArray
  (JNIEnv *, jclass, jlong, jlong, jlong);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaMkVar
 * Signature: (JJLjava/lang/String;)J
 */
JNIEXPORT jlong JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaMkVar
  (JNIEnv *, jclass, jlong, jlong, jstring);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaPush
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaPush
  (JNIEnv *, jclass, jlong, jlong);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaPop
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaPop
  (JNIEnv *, jclass, jlong, jlong);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaAssert
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaAssert
  (JNIEnv *, jclass, jlong, jlong);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaIsUnsatAssumption
 * Signature: (JJ)Z
 */
JNIEXPORT jboolean JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaIsUnsatAssumption
  (JNIEnv *, jclass, jlong, jlong);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaGetUnsatAssumptions
 * Signature: (J)[J
 */
JNIEXPORT jlongArray JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaGetUnsatAssumptions
  (JNIEnv *, jclass, jlong);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaGetUnsatCore
 * Signature: (J)[J
 */
JNIEXPORT jlongArray JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaGetUnsatCore
  (JNIEnv *, jclass, jlong);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaSimplify
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaSimplify
  (JNIEnv *, jclass, jlong);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaCheckSat
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaCheckSat
  (JNIEnv *, jclass, jlong);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaCheckSatAssuming
 * Signature: (J[J)I
 */
JNIEXPORT jint JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaCheckSatAssuming
  (JNIEnv *, jclass, jlong, jlongArray);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaCheckSatTimeout
 * Signature: (JJ)I
 */
JNIEXPORT jint JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaCheckSatTimeout
  (JNIEnv *, jclass, jlong, jlong);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaCheckSatAssumingTimeout
 * Signature: (J[JJ)I
 */
JNIEXPORT jint JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaCheckSatAssumingTimeout
  (JNIEnv *, jclass, jlong, jlongArray, jlong);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaForceTerminate
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaForceTerminate
  (JNIEnv *, jclass, jlong);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaGetValue
 * Signature: (JJ)J
 */
JNIEXPORT jlong JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaGetValue
  (JNIEnv *, jclass, jlong, jlong);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaTermValueGetBool
 * Signature: (J)Z
 */
JNIEXPORT jboolean JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermValueGetBool
  (JNIEnv *, jclass, jlong);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaTermValueGetBvUint64
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermValueGetBvUint64
  (JNIEnv *, jclass, jlong);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaTermValueGetBvUint64Array
 * Signature: (J)[J
 */
JNIEXPORT jlongArray JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermValueGetBvUint64Array
  (JNIEnv *, jclass, jlong);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaTermValueGetFpUint64
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermValueGetFpUint64
  (JNIEnv *, jclass, jlong);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaTermValueGetFpUint64Array
 * Signature: (J)[J
 */
JNIEXPORT jlongArray JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermValueGetFpUint64Array
  (JNIEnv *, jclass, jlong);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaTermValueGetRm
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermValueGetRm
  (JNIEnv *, jclass, jlong);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaSubstituteTerm
 * Signature: (J[J[J)J
 */
JNIEXPORT jlong JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaSubstituteTerm
  (JNIEnv *, jclass, jlong, jlongArray, jlongArray);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaSubstituteTerms
 * Signature: ([J[J[J)[J
 */
JNIEXPORT jlongArray JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaSubstituteTerms
  (JNIEnv *, jclass, jlongArray, jlongArray, jlongArray);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaSortHash
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaSortHash
  (JNIEnv *, jclass, jlong);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaTermGetKind
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermGetKind
  (JNIEnv *, jclass, jlong);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaTermGetChildren
 * Signature: (J)[J
 */
JNIEXPORT jlongArray JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermGetChildren
  (JNIEnv *, jclass, jlong);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaTermGetIndices
 * Signature: (J)[J
 */
JNIEXPORT jlongArray JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermGetIndices
  (JNIEnv *, jclass, jlong);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaSortBvGetSize
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaSortBvGetSize
  (JNIEnv *, jclass, jlong);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaSortFpGetExpSize
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaSortFpGetExpSize
  (JNIEnv *, jclass, jlong);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaSortFpGetSigSize
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaSortFpGetSigSize
  (JNIEnv *, jclass, jlong);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaSortArrayGetIndex
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaSortArrayGetIndex
  (JNIEnv *, jclass, jlong);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaSortArrayGetElement
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaSortArrayGetElement
  (JNIEnv *, jclass, jlong);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaSortFunGetDomainSorts
 * Signature: (J)[J
 */
JNIEXPORT jlongArray JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaSortFunGetDomainSorts
  (JNIEnv *, jclass, jlong);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaSortFunGetCodomain
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaSortFunGetCodomain
  (JNIEnv *, jclass, jlong);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaSortFunGetArity
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaSortFunGetArity
  (JNIEnv *, jclass, jlong);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaSortGetUninterpretedSymbol
 * Signature: (J)Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaSortGetUninterpretedSymbol
  (JNIEnv *, jclass, jlong);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaSortIsArray
 * Signature: (J)Z
 */
JNIEXPORT jboolean JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaSortIsArray
  (JNIEnv *, jclass, jlong);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaSortIsBool
 * Signature: (J)Z
 */
JNIEXPORT jboolean JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaSortIsBool
  (JNIEnv *, jclass, jlong);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaSortIsBv
 * Signature: (J)Z
 */
JNIEXPORT jboolean JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaSortIsBv
  (JNIEnv *, jclass, jlong);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaSortIsFp
 * Signature: (J)Z
 */
JNIEXPORT jboolean JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaSortIsFp
  (JNIEnv *, jclass, jlong);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaSortIsFun
 * Signature: (J)Z
 */
JNIEXPORT jboolean JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaSortIsFun
  (JNIEnv *, jclass, jlong);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaSortIsRm
 * Signature: (J)Z
 */
JNIEXPORT jboolean JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaSortIsRm
  (JNIEnv *, jclass, jlong);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaSortIsUninterpreted
 * Signature: (J)Z
 */
JNIEXPORT jboolean JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaSortIsUninterpreted
  (JNIEnv *, jclass, jlong);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaTermHash
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermHash
  (JNIEnv *, jclass, jlong);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaTermIsIndexed
 * Signature: (J)Z
 */
JNIEXPORT jboolean JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsIndexed
  (JNIEnv *, jclass, jlong);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaTermGetSort
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermGetSort
  (JNIEnv *, jclass, jlong);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaTermArrayGetIndexSort
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermArrayGetIndexSort
  (JNIEnv *, jclass, jlong);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaTermArrayGetElementSort
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermArrayGetElementSort
  (JNIEnv *, jclass, jlong);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaTermFunGetDomainSorts
 * Signature: (J)[J
 */
JNIEXPORT jlongArray JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermFunGetDomainSorts
  (JNIEnv *, jclass, jlong);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaTermFunGetCodomainSort
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermFunGetCodomainSort
  (JNIEnv *, jclass, jlong);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaTermBvGetSize
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermBvGetSize
  (JNIEnv *, jclass, jlong);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaTermFpGetExpSize
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermFpGetExpSize
  (JNIEnv *, jclass, jlong);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaTermFpGetSigSize
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermFpGetSigSize
  (JNIEnv *, jclass, jlong);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaTermFunGetArity
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermFunGetArity
  (JNIEnv *, jclass, jlong);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaTermGetSymbol
 * Signature: (J)Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermGetSymbol
  (JNIEnv *, jclass, jlong);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaTermIsEqualSort
 * Signature: (JJ)Z
 */
JNIEXPORT jboolean JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsEqualSort
  (JNIEnv *, jclass, jlong, jlong);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaTermIsArray
 * Signature: (J)Z
 */
JNIEXPORT jboolean JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsArray
  (JNIEnv *, jclass, jlong);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaTermIsConst
 * Signature: (J)Z
 */
JNIEXPORT jboolean JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsConst
  (JNIEnv *, jclass, jlong);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaTermIsFun
 * Signature: (J)Z
 */
JNIEXPORT jboolean JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsFun
  (JNIEnv *, jclass, jlong);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaTermIsVar
 * Signature: (J)Z
 */
JNIEXPORT jboolean JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsVar
  (JNIEnv *, jclass, jlong);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaTermIsValue
 * Signature: (J)Z
 */
JNIEXPORT jboolean JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsValue
  (JNIEnv *, jclass, jlong);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaTermIsBvValue
 * Signature: (J)Z
 */
JNIEXPORT jboolean JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsBvValue
  (JNIEnv *, jclass, jlong);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaTermIsFpValue
 * Signature: (J)Z
 */
JNIEXPORT jboolean JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsFpValue
  (JNIEnv *, jclass, jlong);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaTermIsRmValue
 * Signature: (J)Z
 */
JNIEXPORT jboolean JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsRmValue
  (JNIEnv *, jclass, jlong);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaTermIsBool
 * Signature: (J)Z
 */
JNIEXPORT jboolean JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsBool
  (JNIEnv *, jclass, jlong);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaTermIsBv
 * Signature: (J)Z
 */
JNIEXPORT jboolean JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsBv
  (JNIEnv *, jclass, jlong);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaTermIsFp
 * Signature: (J)Z
 */
JNIEXPORT jboolean JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsFp
  (JNIEnv *, jclass, jlong);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaTermIsRm
 * Signature: (J)Z
 */
JNIEXPORT jboolean JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsRm
  (JNIEnv *, jclass, jlong);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaTermIsUninterpreted
 * Signature: (J)Z
 */
JNIEXPORT jboolean JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsUninterpreted
  (JNIEnv *, jclass, jlong);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaTermIsTrue
 * Signature: (J)Z
 */
JNIEXPORT jboolean JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsTrue
  (JNIEnv *, jclass, jlong);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaTermIsFalse
 * Signature: (J)Z
 */
JNIEXPORT jboolean JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsFalse
  (JNIEnv *, jclass, jlong);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaTermIsBvValueZero
 * Signature: (J)Z
 */
JNIEXPORT jboolean JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsBvValueZero
  (JNIEnv *, jclass, jlong);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaTermIsBvValueOne
 * Signature: (J)Z
 */
JNIEXPORT jboolean JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsBvValueOne
  (JNIEnv *, jclass, jlong);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaTermIsBvValueOnes
 * Signature: (J)Z
 */
JNIEXPORT jboolean JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsBvValueOnes
  (JNIEnv *, jclass, jlong);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaTermIsBvValueMinSigned
 * Signature: (J)Z
 */
JNIEXPORT jboolean JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsBvValueMinSigned
  (JNIEnv *, jclass, jlong);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaTermIsBvValueMaxSigned
 * Signature: (J)Z
 */
JNIEXPORT jboolean JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsBvValueMaxSigned
  (JNIEnv *, jclass, jlong);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaTermIsFpValuePosZero
 * Signature: (J)Z
 */
JNIEXPORT jboolean JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsFpValuePosZero
  (JNIEnv *, jclass, jlong);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaTermIsFpValueNegZero
 * Signature: (J)Z
 */
JNIEXPORT jboolean JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsFpValueNegZero
  (JNIEnv *, jclass, jlong);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaTermIsFpValuePosInf
 * Signature: (J)Z
 */
JNIEXPORT jboolean JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsFpValuePosInf
  (JNIEnv *, jclass, jlong);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaTermIsFpValueNegInf
 * Signature: (J)Z
 */
JNIEXPORT jboolean JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsFpValueNegInf
  (JNIEnv *, jclass, jlong);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaTermIsFpValueNan
 * Signature: (J)Z
 */
JNIEXPORT jboolean JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsFpValueNan
  (JNIEnv *, jclass, jlong);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaTermIsRmValueRna
 * Signature: (J)Z
 */
JNIEXPORT jboolean JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsRmValueRna
  (JNIEnv *, jclass, jlong);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaTermIsRmValueRne
 * Signature: (J)Z
 */
JNIEXPORT jboolean JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsRmValueRne
  (JNIEnv *, jclass, jlong);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaTermIsRmValueRtn
 * Signature: (J)Z
 */
JNIEXPORT jboolean JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsRmValueRtn
  (JNIEnv *, jclass, jlong);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaTermIsRmValueRtp
 * Signature: (J)Z
 */
JNIEXPORT jboolean JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsRmValueRtp
  (JNIEnv *, jclass, jlong);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaTermIsRmValueRtz
 * Signature: (J)Z
 */
JNIEXPORT jboolean JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermIsRmValueRtz
  (JNIEnv *, jclass, jlong);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaDumpFormula
 * Signature: (JLjava/lang/String;Ljava/lang/String;)V
 */
JNIEXPORT void JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaDumpFormula
  (JNIEnv *, jclass, jlong, jstring, jstring);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaTermToString
 * Signature: (J)Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaTermToString
  (JNIEnv *, jclass, jlong);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaSortToString
 * Signature: (J)Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaSortToString
  (JNIEnv *, jclass, jlong);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaRmToString
 * Signature: (I)Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaRmToString
  (JNIEnv *, jclass, jint);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaResultToString
 * Signature: (I)Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaResultToString
  (JNIEnv *, jclass, jint);

/*
 * Class:     io_ksmt_solver_bitwuzla_bindings_Native
 * Method:    bitwuzlaKindToString
 * Signature: (I)Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL Java_io_ksmt_solver_bitwuzla_bindings_Native_bitwuzlaKindToString
  (JNIEnv *, jclass, jint);

#ifdef __cplusplus
}
#endif
#endif
