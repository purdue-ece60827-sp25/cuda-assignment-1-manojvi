
#include "cudaLib.cuh"
#include "cpuLib.h"
#include <curand_kernel.h>

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

__global__ 
void saxpy_gpu (float* x, float* y, float scale, int size) {
	//	Insert GPU SAXPY kernel code here
	
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size) {
		y[idx] = scale * x[idx] + y[idx];
	}
}



int runGpuSaxpy(int vectorSize) {

	dim3 gridDim(1024, 1, 1);  // 30 blocks in the x-dimension
	dim3 blockDim((vectorSize + 1023)/1024, 1, 1); // 1024 threads in the x-dimension
	std::cout << "Hello GPU Saxpy!\n";
	int numDev = 0;
	gpuAssert(cudaGetDeviceCount(&numDev), __FILE__, __LINE__, true);
	if (numDev < 1) {
		std::cout << "CUDA device missing!\n";
		return -1;
	}
	else {
		std::cout << "CUDA device found " << numDev << "\n";
	}
	// CPU Ops
	float * a, * b, * c;
	int vecbytes = vectorSize * sizeof(float);

	a = (float *) malloc(vecbytes);
	b = (float *) malloc(vecbytes);
	c = (float *) malloc(vecbytes);

	vectorInit(a, vectorSize);
	vectorInit(b, vectorSize);
	memcpy(c, b, vecbytes);

	// Device Ops
	float * d_a, * d_b;
	cudaMalloc((void **) &d_a, vecbytes);
	cudaMalloc((void **) &d_b, vecbytes);
	cudaMemcpy(d_a, a, vecbytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, vecbytes, cudaMemcpyHostToDevice);
	auto tStart = std::chrono::high_resolution_clock::now();
	saxpy_gpu<<<gridDim, blockDim>>>(d_a, d_b, 2.0, vectorSize);
	auto tEnd= std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> time_span = (tEnd- tStart);
	std::cout << "It took " << time_span.count() << " seconds. \n";
	cudaMemcpy(b, d_b, vecbytes, cudaMemcpyDeviceToHost);

	printf(" a = { ");
	printVector(a, vectorSize);
	printf(" c = { ");
	printVector(c, vectorSize);
	printf(" b = { ");
	printVector(b, vectorSize);

	int errorCount = verifyVector(a, c, b, 2.0, vectorSize);
	std::cout << "Found " << errorCount << " / " << vectorSize << " errors \n";
	return 0;
}

/* 
 Some helpful definitions

 generateThreadCount is the number of threads spawned initially. Each thread is responsible for sampleSize points. 
 *pSums is a pointer to an array that holds the number of 'hit' points for each thread. The length of this array is pSumSize.

 reduceThreadCount is the number of threads used to reduce the partial sums.
 *totals is a pointer to an array that holds reduced values.
 reduceSize is the number of partial sums that each reduceThreadCount reduces.

*/

__global__
void generatePoints (uint64_t * pSums, uint64_t pSumSize, uint64_t sampleSize) {
	//	Insert code here
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx < pSumSize) {
		uint64_t count = 0;
		curandState state;
        curand_init(1234, idx, 0, &state);
		for (uint64_t i = 0; i < sampleSize; i++) {
			float x = (float)curand_uniform(&state);
			float y = (float)curand_uniform(&state);
			if (x * x + y * y <= 1) {
				count++;
			}
		}
		pSums[idx] = count;
	}
}


__global__ 
void reduceCounts (uint64_t * pSums, uint64_t * totals, uint64_t pSumSize, uint64_t reduceSize) {
	//	Insert code here
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < pSumSize) {
		uint64_t sum = 0;
		for (uint64_t i = 0; i < reduceSize; i++) {
			sum += pSums[idx * reduceSize + i];
		}
		totals[idx] = sum;
	}
}

int runGpuMCPi (uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {

	//  Check CUDA device presence
	int numDev;
	cudaGetDeviceCount(&numDev);
	if (numDev < 1) {
		std::cout << "CUDA device missing!\n";
		return -1;
	}

	auto tStart = std::chrono::high_resolution_clock::now();
		
	float approxPi = estimatePi(generateThreadCount, sampleSize, 
		reduceThreadCount, reduceSize);
	
	std::cout << "Estimated Pi = " << approxPi << "\n";

	auto tEnd= std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> time_span = (tEnd- tStart);
	std::cout << "It took " << time_span.count() << " seconds.";



	return 0;
}

double estimatePi(uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {
	
	double approxPi = 0;
	// Generate points
	uint64_t * pSums, * totals, * dev_pSums, * dev_totals;
	uint64_t pSumSize = generateThreadCount;
	uint64_t totalSize = reduceThreadCount;
	dev_totals = (uint64_t *) malloc(totalSize * sizeof(uint64_t));
	cudaMalloc((void **) &pSums, pSumSize * sampleSize * sizeof(uint64_t));
	cudaMalloc((void **) &totals, totalSize * sizeof(uint64_t));
	generatePoints<<<generateThreadCount, 1>>>(pSums, pSumSize, sampleSize);
	

	reduceCounts<<<reduceThreadCount, 1>>>(pSums, totals, pSumSize, reduceSize);
	
	cudaMemcpy(dev_totals, totals, totalSize * sizeof(uint64_t), cudaMemcpyDeviceToHost);
	approxPi = 4.0*((double)dev_totals[0]/(double)(reduceSize*sampleSize));
	//      Insert code here
	std::cout << " dev_totals[0]..." << dev_totals[0] << "\n";
	//std::cout << "Compute pi, you must!\n";
	return approxPi;
}
