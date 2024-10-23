#include <kernel.cuh>
tpMesh mMesh;
double *mProbability = NULL;
dim3   mBlocks;
dim3   mThreads;
char   mErrorMessage[STRINGSIZE];
//int    mBlocks  = -1;
//int    mThreads = -1;


/**
 * GPU kernel Game of Life iwth 0 in boundary condition
 *
 * @param *buff0 is the current cellular automata mesh states - The t0 timestep state of all cells
 *        *buff1 is the next cellular automata state
 */
__global__ 
void GPU_Game_of_Life(short *buff1, short *buff0){
	
	 //	blockIdx.x    --> �ndice do bloco.
	 //  blockDim.x    --> tamanho do bloco.
	 //  threadIdx.x   --> �ndice da thread.
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    short  cell = 0;
    short  myCell = buff0[y * blockDim.x * gridDim.x + x] ;
    short  newC = 0;
	//buff1[y * blockDim.x * gridDim.x + x] = 1 - buff0[y * blockDim.x * gridDim.x + x];

    if ((x > 0) && ( x < ((blockDim.x * gridDim.x) - 1)) && (y > 0) && (y < ((blockDim.y * gridDim.y) - 1))){
        cell = buff0[(y+1) * blockDim.x * gridDim.x + (x-1)] +  \
               buff0[(y+1) * blockDim.x * gridDim.x + (x)]   +  \
               buff0[(y+1) * blockDim.x * gridDim.x + (x+1)] +  \
               buff0[(y-1) * blockDim.x * gridDim.x + (x-1)] +  \
               buff0[(y-1) * blockDim.x * gridDim.x + (x)]   +  \
               buff0[(y-1) * blockDim.x * gridDim.x + (x+1)] +  \
               buff0[y * blockDim.x * gridDim.x + (x-1)] +  \
               buff0[y * blockDim.x * gridDim.x + (x+1)] ;

    }


    if ((cell == 3) && (myCell == 0))
        newC = 1;
    
    if ((cell >= 2) && (cell <= 3) && (myCell == 1))
        newC = 1;


    buff1[y  * blockDim.x * gridDim.x  +  x] = newC;
	
	
	
}

/**
 * Cellular Automata initial condition 
 * @return TRUE to success or FALSE otherwise
 */
extern "C" int init_condition(void){
	cudaError_t cudaStatus;
     
    //posix_memalign((void**)&mMesh.buff0, ALIGN, mMesh.width * mMesh.height * sizeof(int));
    //posix_memalign((void**)&mMesh.buff1, ALIGN, mMesh.width * mMesh.height * sizeof(int));

	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess){
		sprintf(mErrorMessage, "CUDA ERROR: cudaDeviceReset(): %s", cudaGetErrorName(cudaStatus));
        return FALSE;
	}
    short *buff = (short*)malloc(mMesh.width * mMesh.height * sizeof(short));


    cudaStatus = cudaMalloc((void**) &mMesh.buff0, mMesh.width * mMesh.height * sizeof(short));
    if (cudaStatus != cudaSuccess){
		sprintf(mErrorMessage, "CUDA ERROR: cudaMalloc() in buffer0: %s", cudaGetErrorName(cudaStatus));
        return FALSE;
	}
    
    cudaStatus = cudaMalloc((void**) &mMesh.buff1, mMesh.width * mMesh.height * sizeof(short));
    if (cudaStatus != cudaSuccess){
		sprintf(mErrorMessage, "CUDA ERROR: cudaMalloc() in buffer1: %s", cudaGetErrorName(cudaStatus));
        return FALSE;
	}

    

    
    

    bzero(buff, (mMesh.width * mMesh.height * sizeof(short)));
    srand(time(NULL));
    for (long i = 0; i < mMesh.height; i++){
        for (long j = 0; j < mMesh.width; j++){
            double  r = (rand() / (double) RAND_MAX);
            if (r <= mProbability[0])
                buff[i * mMesh.width + j]  =  1;
        }//for (int j = 0; j < mMesh.width; j++){
        
    }//for (int i = 0; i < mMesh.height; i++){

	
	
	
	cudaMemcpy(mMesh.buff1, buff, mMesh.width * mMesh.height * sizeof(short), cudaMemcpyHostToDevice);
	free(buff);

    sprintf(mErrorMessage, "NO ERROR init_condition()");
    return TRUE;
}


/**
 * Cellular Automata update
 * @param ts is the current timestep
 * @return TRUE to success or FALSE otherwise
 */
extern "C"  int update(int ts){
	cudaError_t cudaStatus;

	if (ts >= mMesh.steps){
        strcpy(mErrorMessage, "NO ERROR");
        return FALSE;
    } 
	//Executando kernel.
	//short *buff = (short*)malloc(mMesh.width * mMesh.height * sizeof(short));
	/*
	short *b0 = NULL, *b1 = NULL;
	cudaMalloc((void**) &b0, mMesh.width * mMesh.height * sizeof(short));
    cudaMalloc((void**) &b1, mMesh.width * mMesh.height * sizeof(short));
*/

	GPU_Game_of_Life<<<mBlocks, mThreads>>>(mMesh.buff0, mMesh.buff1);
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess){
		sprintf(mErrorMessage, "CUDA ERROR: cudaDeviceSynchronize(): %s", cudaGetErrorName(cudaStatus));
        return FALSE;
	}
	//cudaMemcpy(buff, mMesh.buff0, mMesh.width * mMesh.height * sizeof(short), cudaMemcpyDeviceToHost);

//	cudaFree(b0);
//	cudaFree(b1);
	short *ptr = mMesh.buff1;
	mMesh.buff1 = mMesh.buff0;
	mMesh.buff0 = ptr;
	
	
	
    sprintf(mErrorMessage, "NO ERROR update(%d)", ts);
	return TRUE;
}

/**
 * Cellular Automata final condition 
 * @return TRUE to success or FALSE otherwise
 */
extern "C" int final_condition(void)
{
    cudaError_t cudaStatus;

	cudaStatus = cudaFree(mMesh.buff0);
	if (cudaStatus != cudaSuccess){
        sprintf(mErrorMessage, "CUDA ERROR: cudaFree(mMesh.buff0): %s", cudaGetErrorName(cudaStatus));
        return FALSE;        
	}

    cudaStatus = cudaFree(mMesh.buff1);
	if (cudaStatus != cudaSuccess){
        sprintf(mErrorMessage, "CUDA ERROR: cudaFree(mMesh.buff1): %s", cudaGetErrorName(cudaStatus));
        return FALSE;        
	}

	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess){
        sprintf(mErrorMessage, "CUDA ERROR: cudaDeviceReset(): %s", cudaGetErrorName(cudaStatus));
        return FALSE;        
	}

    sprintf(mErrorMessage, "NO ERROR final_condition()");
    return TRUE;
}


