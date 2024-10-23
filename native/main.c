//export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:`pwd`
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <kernel.cuh>
void printState(void){
    short *buff = (short*)malloc(mMesh.width * mMesh.height * sizeof(short));
    cudaMemcpy(buff, mMesh.buff0, mMesh.width * mMesh.height * sizeof(short), cudaMemcpyDeviceToHost);
    printf("\n");
    for (int i = 0; i < mMesh.height; i++){
        printf("\t");
        for (int j = 0; j < mMesh.width; j++){
            printf("%d ", buff[i * mMesh.width + j]);
        }//for (int j = 0; j < mMesh.width; j++){
        printf("\n");
    }//for (int i = 0; i < mMesh.height; i++){
    printf("\n");
    printf("Timestep\n");

    free(buff);
}


int main (int ac, char **av){
    //struct cudaDeviceProp deviceProp;
   
    //cudaGetDeviceProperties(&deviceProp, 0);
    long const_C = 22;
    mMesh.width  = 1024 * const_C;
    mMesh.height = 1024 * const_C;
    mMesh.steps  =  100000;
    //mBlocks      = 128;
    //mThreads     = 128;   

    mThreads.x = 32;
    mThreads.y = 32;
    mThreads.z = 1;

    mBlocks.x = (int) mMesh.width / mThreads.x;
    mBlocks.y = (int) mMesh.height / mThreads.y;
    mBlocks.z = 1;

   


    printf("\n\nCUDA EXAMPLE\n");
    /*
    fprintf(stdout, "\n                    Device: %s", deviceProp.name);
    fprintf(stdout, "\n Number of multiprocessors: %d", deviceProp.multiProcessorCount);
    fprintf(stdout, "\n         Number of threads: %d", deviceProp.maxThreadsDim[0]);
    fprintf(stdout, "\nBlocks(%d) and Threads(%d)\n", mBlocks, mThreads);
    fprintf(stdout, "\n----------------------------------------------------------------------------------\n\n");
    */
    fprintf(stdout, "Domain information:\n");
    fprintf(stdout, "\tCA(%ld, %ld, %ld)\n", mMesh.width, mMesh.height, mMesh.steps);
    fprintf(stdout, "\tCUDA DOMAIN:\n");
    fprintf(stdout, "\t\t  Blocks (%.4d, %.4d, %d)\n", mBlocks.x, mBlocks.y, mBlocks.z);
    fprintf(stdout, "\t\t Threads (%.4d, %.4d, %d)\n", mThreads.x, mThreads.y, mThreads.z);
    init_condition();
    int ts = 0;
    do{
        fprintf(stdout, "\n\t Step: %d", ts);
    }while(update(++ts));
    
    //printState();
    final_condition();
    
    printf("\n\nFim do programa\n\n");
    return EXIT_SUCCESS;
}
