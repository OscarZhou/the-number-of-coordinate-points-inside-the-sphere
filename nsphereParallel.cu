
#include <cstdlib>
#include <cmath>

#include <iostream>
#include <string>

#include <vector>


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


__global__ void cudaShoot(long* dev_array/*, long* index */,const long ndim, const long base, const double rsquare, const long halfb, const long ntotal)
{

    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    long* index = new long[ndim];
    //std::cout<< "--------------------------------------------index.size==="<<ndim<<std::endl;
    for (long i = 0; i < ndim; ++i) index[i] = 0;
    long idx = 0;
    int num = tid;
    while (num != 0) {
        long rem = num % base;
        num = num / base;
        index[idx /*+ tid*ndim*/] = rem;
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
        dev_array[tid] = 5;
    }

}



int main(void)
{

    const long ntrials = 1;

    for (long n = 0; n < ntrials; ++n)
    {
        const double radius = 3.5;//drand48() * (RMAX - RMIN) + RMIN;
        const long  ndim = 3;//lrand48() % (MAXDIM - 1) + 1;
        std::cout << "### " << n << " " << radius << " " << ndim << " ... " << std::endl;

        const long halfb = static_cast<long>(floor(radius));
        const long base = 2 * halfb + 1;
        const double rsquare = radius * radius;
        const long ntotal = powlong(base, ndim);
        size_t size = ntotal * sizeof(long);
        //size_t index_size = ntotal * ndim * sizeof(long);



        // CUDA event types used for timing execution
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);




        // Allocate in HOST memory
        long* host_array = (long*)malloc(size);
        // Initialize vectors
        for (int i = 0; i < ntotal; ++i) {
            host_array[i] = 0;
        }


        // Allocate in DEVICE memory
        long *dev_array;//, *dev_index;
        cudaMalloc(&dev_array, size);
        //cudaMalloc(&dev_index, index_size);



        // Set up layout of kernel grid
        int threadsPerBlock = 1024;
        int blocksPerGrid = (ntotal + threadsPerBlock - 1) / threadsPerBlock;

        std::cout << "###  blocksPerGrid ########"  << blocksPerGrid<<std::endl;

        cudaEventRecord(start, 0);

        cudaShoot<<<blocksPerGrid, threadsPerBlock>>>(dev_array/*, dev_index*/, ndim, base, rsquare, halfb, ntotal);


        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        float time;  // Must be a float
        cudaEventElapsedTime(&time, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaMemcpy(host_array, dev_array, size, cudaMemcpyDeviceToHost);


        long counter = 0;
        for (int i=0; i< ntotal; i++)
        {
            counter += host_array[i];
        }


        std::cout << " -> " << counter << " " << std::endl;
        cudaFree(dev_array);
        //cudaFree(dev_index);

        free(host_array);





    }
}