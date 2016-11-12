/* CUDA timing example

   To compile: nvcc -o testprog2 testprog2.cu
    Coding by Oscar
 */
#include <iostream>
#include <cuda.h>

#define THREADS_PER_BLOCK 1024

long powlong(long n, long k)
/* Evaluate n**k where both are long integers */
{
    long p = 1;
    for (long i = 0; i < k; ++i) p *= n;
    return p;
}


__global__ void cudaCountIn(int* dev_counter, long ndim,  long halfb,  double rsquare, long base, long ntotal)
{
    __shared__  int tmp_array[THREADS_PER_BLOCK];//long* tmp_array = new long[ntotal];

    int tid = blockDim.x * blockIdx.x + threadIdx.x;



    if( tid < ntotal)
    {
        tmp_array[tid] = 0;
        long* index = new long[ndim];
        for (long i = 0; i < ndim; ++i) index[i] = 0;


        long idx = 0;
        int num = tid;
        while (num != 0) {
            long rem = num % base;
            num = num / base;
            index[idx] = rem;
            ++idx;
        }

        double rtestsq = 0;
        for (long k = 0; k < ndim; ++k) {
            double xk = index[k] - halfb;
            rtestsq += xk * xk;
        }


        if (rtestsq < rsquare)
        {

            tmp_array[tid] = 1;
        }

        __syncthreads();
        if(tid == 0)
        {

            int counter = 0;
            for(int i=0; i<ntotal; i++)
                counter += tmp_array[i];
            atomicAdd(dev_counter, counter);
        }
    }



}



int main(void)
{

    const long ntrials = 1;

    for (long n = 0; n < ntrials; ++n)
    {
        const double radius = 25.5;
        const long  ndim = 1;
        std::cout << "### " << n << " " << radius << " " << ndim << " ... " << std::endl;

        const long halfb = static_cast<long>(floor(radius));
        const long base = 2 * halfb + 1;
        const double rsquare = radius * radius;
        const long ntotal = powlong(base, ndim);

        size_t size =  sizeof(int);



        std::cout << "ntotal -> " << ntotal << " " << std::endl;
        // CUDA event types used for timing execution
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);




        // Allocate in HOST memory
        int* host_counter = (int*)malloc(size);


        // Allocate in DEVICE memory
        int *dev_counter;
        cudaMalloc(&dev_counter, size);
        cudaMemset(dev_counter, 0, size);

        // Set up layout of kernel grid
        int threadsPerBlock = THREADS_PER_BLOCK;
        int blocksPerGrid = (ntotal + threadsPerBlock - 1) / threadsPerBlock;


        cudaEventRecord(start, 0);

        cudaCountIn<<<blocksPerGrid, threadsPerBlock>>>(dev_counter, ndim, halfb, rsquare, base, ntotal);


        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        float time;  // Must be a float
        cudaEventElapsedTime(&time, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        std::cout << "Kernel took: " << time << " ms" << std::endl;


        cudaMemcpy(host_counter, dev_counter, size, cudaMemcpyDeviceToHost);


        std::cout << " the numbers of integer coordinate points inside the sphere -> " << host_counter[0] << " " << std::endl;
        cudaFree(dev_counter);

        free(host_counter);





    }
}
