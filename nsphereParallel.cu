/* CUDA timing example

   To compile: nvcc -o testprog2 testprog2.cu
    Coding by Oscar
 */
#include <iostream>
#include <cuda.h>



//const long MAXDIM = 10;
//const double RMIN = 2.0;
//const double RMAX = 8.0;

long powlong(long n, long k)
/* Evaluate n**k where both are long integers */
{
    long p = 1;
    for (long i = 0; i < k; ++i) p *= n;
    return p;
}


__global__ void cudaCountIn(long* dev_array, long ndim,  long halfb,  double rsquare, long base, long ntotal)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;



    if( tid < ntotal)
    {
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

            dev_array[tid] = 1;
        }

        else
        {
            dev_array[tid] = 0;
        }

    }

}



int main(void)
{

    const long ntrials = 1;

    for (long n = 0; n < ntrials; ++n)
    {
        const double radius = 1.5;//drand48() * (RMAX - RMIN) + RMIN;
        const long  ndim = 3;//lrand48() % (MAXDIM - 1) + 1;
        std::cout << "### " << n << " " << radius << " " << ndim << " ... " << std::endl;

        const long halfb = static_cast<long>(floor(radius));
        const long base = 2 * halfb + 1;
        const double rsquare = radius * radius;
        const long ntotal = powlong(base, ndim);

        size_t size = ntotal * sizeof(long);



        std::cout << "ntotal -> " << ntotal << " " << std::endl;
        // CUDA event types used for timing execution
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);




        // Allocate in HOST memory
        long* host_array = (long*)malloc(size);


        // Allocate in DEVICE memory
        long *dev_array;
        cudaMalloc(&dev_array, size);


        //cudaMemcpy(dev_array, host_array, size, cudaMemcpyHostToDevice);
        //cudaMemcpy(dev_index, host_index, index_size, cudaMemcpyHostToDevice);
        cudaMemset(dev_array, 0, size);

        // Set up layout of kernel grid
        int threadsPerBlock = 1024;
        int blocksPerGrid = (ntotal + threadsPerBlock - 1) / threadsPerBlock;

        std::cout << "###  blocksPerGrid ########"  << blocksPerGrid<<std::endl;

        cudaEventRecord(start, 0);

        cudaCountIn<<<blocksPerGrid, threadsPerBlock>>>(dev_array, ndim, halfb, rsquare, base, ntotal);


        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        float time;  // Must be a float
        cudaEventElapsedTime(&time, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        std::cout << "Kernel took: " << time << " ms" << std::endl;


        cudaMemcpy(host_array, dev_array, size, cudaMemcpyDeviceToHost);

        long counter = 0;


        for (long i=0; i< ntotal ; i++)
        {
            std::cout<<"host_array["<<i<<"]="<<host_array[i]<<" ";
            counter += host_array[i];
            //std::cout<<"host_index["<<i<<"]="<<host_array[i]<<" ";
        }


        std::cout << " -> " << counter << " " << std::endl;
        cudaFree(dev_array);

        free(host_array);





    }
}
