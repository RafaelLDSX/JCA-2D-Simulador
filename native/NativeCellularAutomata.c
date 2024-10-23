 
#include <jni.h>
#include <kernel.cuh>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

 
JNIEXPORT jboolean JNICALL Java_NativeCellularAutomata_setup
  (JNIEnv *env, jobject obj, jint w, jint h, 
                             jint bx, jint by, 
                             jint tx, jint ty, 
                             jint ts, jdoubleArray inprobs){

    mMesh.width = w;
    mMesh.height = h;
    mMesh.steps = ts;

    mBlocks.x  = bx;
    mBlocks.y  = by;
    mBlocks.z  = 1;

    mThreads.x = tx;
    mThreads.y = ty;
    mBlocks.z  = 1;
    mProbability = NULL;
    jdouble *probs = (*env)->GetDoubleArrayElements(env, inprobs, NULL);
    if (probs == NULL){
      sprintf(mErrorMessage, "JNI ERROR: Java_NativeCellularAutomata_setup(...): probs parameter is NULL");
      return 0;
    }
    jsize size = (*env)->GetArrayLength(env, inprobs);

    mProbability = (double *) malloc(size * sizeof(double));
    memcpy(mProbability, inprobs, size * sizeof(double));

    return 1;
}


JNIEXPORT jboolean JNICALL Java_NativeCellularAutomata_initCondition
  (JNIEnv *env, jobject obj){
  return (jboolean)  init_condition();
}

JNIEXPORT jboolean JNICALL Java_NativeCellularAutomata_update
  (JNIEnv *env, jobject obj, jint ts){
    
    return (jboolean) update(ts);
}

JNIEXPORT jboolean JNICALL Java_NativeCellularAutomata_finalCondition
  (JNIEnv *env, jobject obj){
   return (jboolean)  final_condition();
}

JNIEXPORT jshortArray JNICALL Java_NativeCellularAutomata_getBuffer
  (JNIEnv *env, jobject obj){
    long size = mMesh.width * mMesh.height;
    
    jshortArray outJNIArray = (*env)->NewShortArray(env, size);  // allocate
    if (NULL == outJNIArray) return NULL;

    short *buff = (short *) malloc(size * sizeof(short));
    cudaMemcpy(buff, mMesh.buff0, size * sizeof(short), cudaMemcpyDeviceToHost);

    (*env)->SetShortArrayRegion(env, outJNIArray, 0 , size, buff);  // copy
    

    free(buff);
    return outJNIArray;
}

JNIEXPORT jstring JNICALL Java_NativeCellularAutomata_getErrorMessage
  (JNIEnv *env, jobject obj){

  return (*env)->NewStringUTF(env, mErrorMessage);
}
  
JNIEXPORT jstring JNICALL Java_NativeCellularAutomata_getModelDescription
  (JNIEnv *env, jobject obj){
    char *descri = "Cellular Automata 2D - Game of Life computed by GPU";

    return (*env)->NewStringUTF(env, descri);
  }