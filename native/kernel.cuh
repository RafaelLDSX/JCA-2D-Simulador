/*
 *
 *
 *
 */
#ifndef _KERNEL_H_
#define _KERNEL_H_
#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
/* */
#define ALIGN       64
#define STRINGSIZE 1024
#define TRUE       1
#define FALSE      0
/* */
struct stMesh{
    short *buff0;
    short *buff1;
    long width;
    long height;
    long steps;
};
typedef struct stMesh tpMesh;

/* */
extern tpMesh mMesh;
extern double *mProbability;
extern dim3   mBlocks;
extern dim3   mThreads;
extern char   mErrorMessage[STRINGSIZE];

#ifdef __cplusplus
extern "C" {
#endif
int init_condition(void);
int update(int);
int final_condition(void);
#ifdef __cplusplus
}
#endif
#endif