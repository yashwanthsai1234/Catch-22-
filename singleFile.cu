#include <cuda_runtime.h>
#include <cufft.h>
#include <stdio.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <float.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/count.h>
#include <thrust/extrema.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include "splinefit.h"
#include "splinefit.c"
#include "stats.h"
#include<bits/stdc++.h>
#include <vector>
//#include "Histogram5.cu"

// #include "stats.h"
// #include "fft.h"
// #include "histcounts.h"

// #include "helper_functions.h"
static int compare (const void * a, const void * b)
{
    if (*(double*)a < *(double*)b) {
        return -1;
    } else if (*(double*)a > *(double*)b) {
        return 1;
    } else {
        return 0;
    }
}
void sort(double y[], int size)
{
    qsort(y, size, sizeof(*y), compare);
}
double quantile(const double y[], const int size, const double quant)
{   
    double quant_idx, q, value;
    int idx_left, idx_right;
    double *tmp = (double*)malloc(size * sizeof(*y));
    memcpy(tmp, y, size * sizeof(*y));
    sort(tmp, size);
    // out of range limit?
    q = 0.5 / size;
    if (quant < q) {
        value = tmp[0]; // min value
        free(tmp);
        return value; 
    } else if (quant > (1 - q)) {
        value = tmp[size - 1]; // max value
        free(tmp);
        return value; 
    }
    
    quant_idx = size * quant - 0.5;
    idx_left = (int)floor(quant_idx);
    idx_right = (int)ceil(quant_idx);
    value = tmp[idx_left] + (quant_idx - idx_left) * (tmp[idx_right] - tmp[idx_left]) / (idx_right - idx_left);
    free(tmp);
    return value;
}
void linspace(double start, double end, int num_groups, double out[])
{
    double step_size = (end - start) / (num_groups - 1);
    for (int i = 0; i < num_groups; i++) {
        out[i] = start;
        start += step_size;
    }
    return;
}
double mean(const double a[], const int size)
{
    double m = 0.0;
    for (int i = 0; i < size; i++) {
        m += a[i];
    }
    m /= size;
    return m;
}
double max_(const double a[], const int size)
{
    double m = a[0];
    for (int i = 1; i < size; i++) {
        if (a[i] > m) {
            m = a[i];
        }
    }
    return m;
}
double min_(const double a[], const int size)
{
    double m = a[0];
    for (int i = 1; i < size; i++) {
        if (a[i] < m) {
            m = a[i];
        }
    }
    return m;
}
double stddev(const double a[], const int size)
{
    double m = mean(a, size);
    double sd = 0.0;
    for (int i = 0; i < size; i++) {
        sd += pow(a[i] - m, 2);
    }
    sd = sqrt(sd / (size - 1));
    return sd;
}
double cov(const double x[], const double y[], const int size){
    double covariance = 0, meanX = mean(x, size), meanY = mean(y, size);
    for(int i = 0; i < size; i++)covariance += (x[i] - meanX) * (y[i] - meanY);
    return covariance/(size-1);
}
int linreg(const int n, const double x[], const double y[], double* m, double* b) //, double* r)
{
    double   sumx = 0.0;                      /* sum of x     */
    double   sumx2 = 0.0;                     /* sum of x**2  */
    double   sumxy = 0.0;                     /* sum of x * y */
    double   sumy = 0.0;                      /* sum of y     */
    double   sumy2 = 0.0;                     /* sum of y**2  */
    for (int i=0;i<n;i++){
        sumx  += x[i];
        sumx2 += x[i] * x[i];
        sumxy += x[i] * y[i];
        sumy  += y[i];
        sumy2 += y[i] * y[i];
    }
    double denom = (n * sumx2 - sumx * sumx);
    if (denom == 0) {
        *m = 0;
        *b = 0;
        //if (r) *r = 0;
        return 1;
    }
    *m = (n * sumxy  -  sumx * sumy) / denom;
    *b = (sumy * sumx2  -  sumx * sumxy) / denom;
    return 0;
}
double norm_(const double a[], const int size)
{
    double out = 0.0;
    for (int i = 0; i < size; i++){out += a[i]*a[i];}
    out = sqrt(out);
    return out;
}
void sb_coarsegrain(const double y[], const int size, const char how[], const int num_groups, int labels[])
{
    int i, j;
    if (strcmp(how, "quantile") == 1) {
        fprintf(stdout, "ERROR in sb_coarsegrain: unknown coarse-graining method\n");
        exit(1);
    }
    
    double *th = (double*)malloc((num_groups + 1) * 2 * sizeof(th));
    double *ls = (double*)malloc((num_groups + 1) * 2 * sizeof(th));
    linspace(0, 1, num_groups + 1, ls);
    for (i = 0; i < num_groups + 1; i++) {
        //double quant = quantile(y, size, ls[i]);
        th[i] = quantile(y, size, ls[i]);
    }
    th[0] -= 1;
    for (i = 0; i < num_groups; i++) {
        for (j = 0; j < size; j++) {
            if (y[j] > th[i] && y[j] <= th[i + 1]) {
                labels[j] = i + 1;
            }
        }
    }
    free(th);
    free(ls);
}
__global__ void mean_kernel(const double* d_input, double* d_partial_sums, int size) {
    extern __shared__ double sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (i < size) ? d_input[i] : 0;
    __syncthreads();
    for(unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) d_partial_sums[blockIdx.x] = sdata[0];
}

double compute_mean_cuda(const double* h_input, int size) {
    double *d_input, *d_partial_sums, *h_partial_sums;
    int threadsPerBlock = 256;
    int blocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    int partialSumsSize = blocks * sizeof(double);

    cudaError_t cudaStatus = cudaMalloc(&d_input, size * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for d_input: %s\n", cudaGetErrorString(cudaStatus));
        return 0;
    }

    cudaStatus = cudaMalloc(&d_partial_sums, partialSumsSize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for d_partial_sums: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(d_input);
        return 0;
    }

    h_partial_sums = (double*)malloc(partialSumsSize);
    if (!h_partial_sums) {
        fprintf(stderr, "Failed to allocate host memory for h_partial_sums\n");
        cudaFree(d_input);
        cudaFree(d_partial_sums);
        return 0;
    }

    cudaStatus = cudaMemcpy(d_input, h_input, size * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(d_input);
        cudaFree(d_partial_sums);
        free(h_partial_sums);
        return 0;
    }

    mean_kernel<<<blocks, threadsPerBlock, threadsPerBlock * sizeof(double)>>>(d_input, d_partial_sums, size);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "mean_kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(d_input);
        cudaFree(d_partial_sums);
        free(h_partial_sums);
        return 0;
    }

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching mean_kernel: %s\n", cudaStatus, cudaGetErrorString(cudaStatus));
        cudaFree(d_input);
        cudaFree(d_partial_sums);
        free(h_partial_sums);
        return 0;
    }

    cudaStatus = cudaMemcpy(h_partial_sums, d_partial_sums, partialSumsSize, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(d_input);
        cudaFree(d_partial_sums);
        free(h_partial_sums);         
        return 0;
    }

    double totalSum = 0;
    for (int i = 0; i < blocks; i++) {
        totalSum += h_partial_sums[i];
    }

    cudaFree(d_input);
    cudaFree(d_partial_sums);
    free(h_partial_sums);
    return totalSum / size;
}

__global__ void subtract_mean_and_pad(double *d_y, int size, int nFFT, double mean) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_y[idx] -= mean;
    }
    else if (idx < nFFT) {
        d_y[idx] = 0.0;
    }
}

__global__ void complex_conjugate_multiply(cufftDoubleComplex *data, int nFFT) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nFFT) {
        cufftDoubleComplex val = data[idx];
        cufftDoubleComplex conjVal = cuConj(val);
        data[idx] = cuCmul(val, conjVal);
    }
}

__device__ double cufftComplex_abs(cufftDoubleComplex z) {
    return sqrt(z.x * z.x + z.y * z.y);
}

int nextpow2(int n) {
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}

__global__ void normalize(double *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] /= data[0];
    }
}

double *cuda_co_autocorrs(const double *y, const int size) {
    int nFFT =  nextpow2(size);
    double *d_y, *autocorr;
    cufftDoubleComplex *d_freqDomain;
    cufftHandle plan_f, plan_i;
    cudaError_t cudaStatus;
    cufftResult cufftStatus;

    autocorr = (double *)malloc(nFFT * sizeof(double));
    if (!autocorr) {
        fprintf(stderr, "Failed to allocate host memory for autocorr\n");
        return NULL;
    }

    cudaStatus = cudaMalloc(&d_y, nFFT * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for d_y: %s\n", cudaGetErrorString(cudaStatus));
        free(autocorr);
        return NULL;
    }

    cudaStatus = cudaMalloc(&d_freqDomain, nFFT * sizeof(cufftDoubleComplex));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for d_freqDomain: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(d_y);
        free(autocorr);
        return NULL;
    }

    cudaStatus = cudaMemcpy(d_y, y, size * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(d_y);
        cudaFree(d_freqDomain);
        free(autocorr);
        return NULL;
    }

    double *zeroPad = (double *)calloc(nFFT - size, sizeof(double));
    cudaMemcpy(d_y + size, zeroPad, (nFFT - size) * sizeof(double), cudaMemcpyHostToDevice);
    free(zeroPad);

    cufftStatus = cufftPlan1d(&plan_f, nFFT, CUFFT_D2Z, 1);
    if (cufftStatus != CUFFT_SUCCESS) {
        fprintf(stderr, "cufftPlan1d failed with error code %d\n", cufftStatus);
        cudaFree(d_y);
        cudaFree(d_freqDomain);
        free(autocorr);
        return NULL;
    }

    cufftStatus = cufftPlan1d(&plan_i, nFFT, CUFFT_Z2D, 1);
    if (cufftStatus != CUFFT_SUCCESS) {
        fprintf(stderr, "cufftPlan1d failed with error code %d\n", cufftStatus);
        cudaFree(d_y);
        cudaFree(d_freqDomain);
        free(autocorr);
        return NULL;
    }

    cufftStatus = cufftExecD2Z(plan_f, d_y, d_freqDomain);
    if (cufftStatus != CUFFT_SUCCESS) {
        fprintf(stderr, "cufftExecD2Z failed with error code %d\n", cufftStatus);
        cudaFree(d_y);
        cudaFree(d_freqDomain);
        free(autocorr);
        return NULL;
    }

    int threadsPerBlock = 256;
    int blocks = (nFFT + threadsPerBlock - 1) / threadsPerBlock;
    complex_conjugate_multiply<<<blocks, threadsPerBlock>>>(d_freqDomain, nFFT);
    cudaDeviceSynchronize();

    cufftStatus = cufftExecZ2D(plan_i, d_freqDomain, d_y);
    if (cufftStatus != CUFFT_SUCCESS) {
        fprintf(stderr, "cufftExecZ2D failed with error code %d\n", cufftStatus);
        cufftDestroy(plan_i);
        free(autocorr);
        return NULL;
    }

    int blocksPerGrid = (nFFT + threadsPerBlock - 1) / threadsPerBlock;
    normalize<<<blocksPerGrid, threadsPerBlock>>>(d_y, nFFT);
    cudaDeviceSynchronize();

    cudaStatus = cudaMemcpy(autocorr, d_y, nFFT * sizeof(double), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(d_y);
        cudaFree(d_freqDomain);
        free(autocorr);
        return NULL;
    }

   // cufftDestroy(plan_f);
    //cufftDestroy(plan_i);
    //cudaFree(d_y);
    //cudaFree(d_freqDomain);
    return autocorr;
}

__global__ void find_first_min_kernel(const double *autocorrs, int size, int *minIndex) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (i < size - 1 && autocorrs[i] < autocorrs[i - 1] && autocorrs[i] < autocorrs[i + 1]) {
        atomicMin(minIndex, i);
    }
}

int CO_FirstMin_ac_cuda(const double y[], const int size, double *autocorrs) {
    //double *autocorrs = cuda_co_autocorrs(y, size);
    double *d_autocorrs;
    int *d_minIndex;
    int h_minIndex = size;
    cudaMalloc(&d_autocorrs, size * sizeof(double));
    cudaMalloc(&d_minIndex, sizeof(int));
    cudaMemcpy(d_autocorrs, autocorrs, size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_minIndex, &h_minIndex, sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    find_first_min_kernel<<<blocks, threadsPerBlock>>>(d_autocorrs, size, d_minIndex);

    cudaMemcpy(&h_minIndex, d_minIndex, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_autocorrs);
    cudaFree(d_minIndex);
    //free(autocorrs);
    return h_minIndex;
}

__global__ void findThresholdCrossing(const double* autocorr, int size, double thresh, int* crossingIndex) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > 0 && idx < size) {
        if (autocorr[idx] < thresh && autocorr[idx - 1] >= thresh) {
            atomicMin(crossingIndex, idx);
        }
    }
}

double CO_f1ecac_CUDA(const double* y, int size) {
    double* autocorr_d = nullptr;
    int* crossingIndex_d = nullptr;
    int crossingIndex_h = INT_MAX;
    cudaMalloc((void**)&autocorr_d, size * sizeof(double));
    cudaMemcpy(autocorr_d, y, size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&crossingIndex_d, sizeof(int));
    cudaMemcpy(crossingIndex_d, &crossingIndex_h, sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    double thresh = 1.0 / exp(1);
    findThresholdCrossing<<<blocksPerGrid, threadsPerBlock>>>(autocorr_d, size, thresh, crossingIndex_d);

    cudaError_t cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching findThresholdCrossing!\n", cudaStatus);
    }

    cudaMemcpy(&crossingIndex_h, crossingIndex_d, sizeof(int), cudaMemcpyDeviceToHost);
    double out = (double)size;
    if (crossingIndex_h != INT_MAX && crossingIndex_h > 0 && crossingIndex_h < size) {
        double autocorr_values[2];
        cudaMemcpy(autocorr_values, &autocorr_d[crossingIndex_h - 1], 2 * sizeof(double), cudaMemcpyDeviceToHost);
        double m = autocorr_values[1] - autocorr_values[0];
        double dy = thresh - autocorr_values[0];
        double dx = dy / m;
        out = crossingIndex_h - 1 + dx;
    } else {
        printf("Threshold crossing not found.\n");
    }

    //cudaFree(autocorr_d);
    //cudaFree(crossingIndex_d);
    return out;
}

__global__ void compute_cubed_differences(const double *y, double *cubed_diffs, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size - 1) {
        double diff = y[idx + 1] - y[idx];
        cubed_diffs[idx] = diff * diff * diff;
    }
}

double CO_trev_1_num_cuda(const double *y, int size) {
    double *d_y, *d_cubed_diffs;
    cudaMalloc(&d_y, size * sizeof(double));
    cudaMemcpy(d_y, y, size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMalloc(&d_cubed_diffs, (size - 1) * sizeof(double));
    int threadsPerBlock = 256;
    int blocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    compute_cubed_differences<<<blocks, threadsPerBlock>>>(d_y, d_cubed_diffs, size);
    double mean_cubed_diffs = compute_mean_cuda(d_cubed_diffs, size - 1);
    //cudaFree(d_y);
    //cudaFree(d_cubed_diffs);
    return mean_cubed_diffs;
}
int co_firstzero(const double y[], const int size, const int maxtau, double *autocorrs)
{
    //double * autocorrs = cuda_co_autocorrs(y, size);
    int zerocrossind = 0;
    while(autocorrs[zerocrossind] > 0 && zerocrossind < maxtau)zerocrossind += 1;
    free(autocorrs);
    return zerocrossind;
}


__global__ void compute_d(const int tau, const int size, double *d_d, const double *d_y){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx<(size-tau-1)){
        d_d[idx] = (d_y[idx+1]-d_y[idx])*(d_y[idx+1]-d_y[idx]) + (d_y[idx+tau]-d_y[idx+tau+1])*(d_y[idx+tau]-d_y[idx+tau+1]);
    }
    
}

__global__ void compute_histCountsNorms(const int nBins, double *d_histCountsNorm, const int tau, const int size, const int *histCounts){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx<nBins){
        d_histCountsNorm[idx] = (double)histCounts[idx]/(double)(size-tau-1);
    }
}

__global__ void compute_d_expfit_diff(const int nBins, const double l, const double* binEdges, double *d_d_expfit_diff, double *d_histCountsNorm){
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx<nBins){
        double expf = exp(-(binEdges[idx] + binEdges[idx+1])*0.5/l)/l;
        if (expf < 0){expf = 0;}
        d_d_expfit_diff[idx] = fabs(d_histCountsNorm[idx]-expf);
    }
}

int num_bins_auto(const double y[], const int size){
    
    double maxVal = max_(y, size);
    double minVal = min_(y, size);
    
    if (stddev(y, size) < 0.001){return 0;}
    return ceil((maxVal-minVal)/(3.5*stddev(y, size)/pow(size, 1/3.)));
    
}
int histcounts_preallocated(const double y[], const int size, int nBins, int * binCounts, double * binEdges)
{
    int i = 0;
    // check min and max of input array
    double minVal = DBL_MAX, maxVal=-DBL_MAX;
    for(int i = 0; i < size; i++)
    {
        // printf("histcountInput %i: %1.3f\n", i, y[i]);        
        if (y[i] < minVal){minVal = y[i];}
        if (y[i] > maxVal){maxVal = y[i];}
    }
    
    // and derive bin width from it
    double binStep = (maxVal - minVal)/nBins;
    
    // variable to store counted occurances in
    for(i = 0; i < nBins; i++){binCounts[i] = 0;}
    
    for(i = 0; i < size; i++)
    {
        int binInd = (y[i]-minVal)/binStep;
        if(binInd < 0)binInd = 0;
        if(binInd >= nBins)binInd = nBins-1;
        //printf("histcounts, i=%i, binInd=%i, nBins=%i\n", i, binInd, nBins);
        binCounts[binInd] += 1;
    }
    for(i = 0; i < nBins+1; i++){binEdges[i] = i * binStep + minVal;}
    return 0;
    
}
double CO_Embed2_Dist_tau_d_expfit_meandiff(const double y[], const int size, double *autocorr_d)
{
    int threadsPerBlock = 256, blocks = (size + threadsPerBlock - 1) / threadsPerBlock; 
    // NaN check
    for(int i = 0; i < size; i++){if(isnan(y[i])){return NAN;}}
    int tau = co_firstzero(y, size, size, autocorr_d);
    if(tau > (double)size/10){tau = floor((double)size/10);}

    double *d_d, *d_y;// = malloc((size-tau) * sizeof(double));
    double *h_d = (double*)malloc((size-tau)*sizeof(double));
    cudaMalloc(&d_d, (size-tau) * sizeof(double));
    cudaMalloc(&d_y, size*sizeof(double));
    cudaMemcpy(d_y, y, size*sizeof(double), cudaMemcpyHostToDevice);
    compute_d<<<blocks, threadsPerBlock>>>(tau, size, d_d, d_y);
    cudaMemcpy(h_d, d_d, size*sizeof(double), cudaMemcpyDeviceToHost);
    //for(int i=0;i<size-tau;i++)printf("%.5f\n", h_d[i]);
    for(int i = 0; i < size-tau-1; i++){if(isnan(h_d[i])){free(h_d);return NAN;}}
    
    // mean for exponential fit
    double l = mean(h_d, size-tau-1);

    // count histogram bin contents
    int nBins = num_bins_auto(h_d, size-tau-1);
    
    if(nBins == 0){return 0;}
    int *histCounts = (int*)malloc(nBins * sizeof(double));
    double *binEdges = (double*)malloc((nBins + 1) * sizeof(double));
    histcounts_preallocated(h_d, size-tau-1, nBins, histCounts, binEdges);
    //for(int i=0;i<=nBins;i++){printf("%.5f\n", binEdges[i]);}
    // normalise to probability
    int *d_histCounts;
    cudaMalloc(&d_histCounts, nBins*sizeof(int));
    cudaMemcpy(d_histCounts, histCounts, nBins*sizeof(int), cudaMemcpyHostToDevice);
    double *d_histCountsNorm, *h_histCountsNorm = (double*)malloc(nBins*sizeof(double));
    cudaMalloc(&d_histCountsNorm, nBins*sizeof(double));
    compute_histCountsNorms<<<blocks, threadsPerBlock>>>(nBins, d_histCountsNorm, tau, size, d_histCounts);
    cudaMemcpy(h_histCountsNorm, d_histCountsNorm, nBins*sizeof(double), cudaMemcpyDeviceToHost);
    // for(int i=0;i<nBins;i++){
    //     printf("%.5f\n", h_histCountsNorm[i]);
    // }

    //parallelizing calculation of d_expfit_diff
    double *d_d_expfit_diff, *h_d_expfit_diff = (double*)malloc(nBins*sizeof(double));
    cudaMalloc(&d_d_expfit_diff, nBins*sizeof(double));
    double *d_binEdges;
    cudaMalloc(&d_binEdges, (nBins+1)*sizeof(double));
    cudaMemcpy(d_binEdges, binEdges, (nBins+1)*sizeof(double), cudaMemcpyHostToDevice);
    compute_d_expfit_diff<<<blocks, threadsPerBlock>>>(nBins, l, d_binEdges, d_d_expfit_diff, d_histCountsNorm);
    cudaMemcpy(h_d_expfit_diff, d_d_expfit_diff, nBins*sizeof(double), cudaMemcpyDeviceToHost);
    // for(int i=0;i<nBins;i++){
    //     printf("%.5f\n", h_d_expfit_diff[i]);
    // }
    double out = mean(h_d_expfit_diff, nBins);

    // arrays created dynamically in function histcounts

    cudaFree(d_d_expfit_diff);free(binEdges);cudaFree(d_histCountsNorm);free(histCounts);
    return out;
}

__global__ void compute_support_regression(double *d_xReg, const int size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx<size){
        //printf("%d\n", idx);
        d_xReg[idx] = idx+1;
    }
}
__global__ void compute_tau(double tauStep, double linLow, int nTauSteps, int *d_tau){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nTauSteps) {
        d_tau[idx] = round(exp(linLow + idx*tauStep));
    }
}

__global__ void vector_to_cumu_sum(const double *y, int sizeCS, double *d_yCS, const int lag){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx==0){
        d_yCS[0] = y[0];
    }
    else if(idx<sizeCS-1){
        d_yCS[idx+1] = d_yCS[idx] + y[(idx+1)*lag];
    }
}
__global__ void parallel_buffer(const int i, double *d_buffer, const double m1, const double b1, const double *d_logtt, const double *d_logFF){
    int idx = blockDim.x*blockIdx.x + threadIdx.x;
    //buffer[j] = logtt[j] * m1 + b1 - logFF[j];
    if(idx<i){
        d_buffer[idx] = d_logtt[idx]*m1 + b1-d_logFF[idx];
    }
}
__global__ void parallel_buffer2(const int i, const int ntt, double *d_buffer, const double m2, const double b2, const double *d_logtt, const double *d_logFF){
    int idx = blockDim.x*blockIdx.x + threadIdx.x;
    //logtt[j+i-1] * m2 + b2 - logFF[j+i-1];
    if(idx<ntt){
        d_buffer[idx] = d_logtt[idx+i-1]*m2 + b2-d_logFF[idx+i-1];
    }
}
__global__ void calc_logtt_logFF(const int *d_tau, const int nTau, const double *d_F, double *d_logtt, double *d_logFF){
    //logtt[i] = log(h_tau[i]);logFF[i] = log(F[i]);
    int idx = blockDim.x*blockIdx.x + threadIdx.x;
    if(idx<nTau){
        d_logtt[idx] = log(d_tau[idx]);
        d_logFF[idx] = log(d_F[idx]);
    }
}
double SC_FluctAnal_2_50_1_logi_prop_r1(const double y[], const int size, const int lag, const char how[])
{
    int threadsPerBlock = 256;
    int blocks = (size + threadsPerBlock - 1) / threadsPerBlock;    
    
    // NaN check
    for(int i = 0; i < size; i++){if(isnan(y[i])){return NAN;}}
    
    // generating log spaced tau vector
    int nTauSteps = 50;
    double linLow = log(5), linHigh = log(size/2), tauStep = (linHigh - linLow) / (nTauSteps-1);
    
    int *d_tau, *h_tau= (int*)malloc(50*sizeof(int));// = (int*)malloc(50*sizeof(int));
    cudaMalloc(&d_tau, 50*sizeof(int));

    compute_tau<<<blocks, threadsPerBlock>>>(tauStep, linLow, nTauSteps, d_tau);
    cudaMemcpy(h_tau, d_tau, 50*sizeof(int), cudaMemcpyDeviceToHost);
    //cudaFree(d_tau);
    //h_tau has log spaced tau vector

    //Not parallelizing the part to check for ascending code because
    //there are too many if else checks happening. It is gonna make the code extremely slow
    //to execute them simultaneously without the threads knowing of outcomes of other threads.

    int nTau = nTauSteps;
    for(int i = 0; i < nTauSteps-1; i++)
    {
        while (h_tau[i] == h_tau[i+1] && i < nTau-1)
        {
            for(int j = i+1; j < nTauSteps-1; j++){h_tau[j] = h_tau[j+1];}
            // lost one
            nTau -= 1;
        }
    }

    // fewer than 12 points -> leave.
    if(nTau < 12){return 0;}

    int sizeCS = size/lag;
    double *d_yCS, *h_yCS = (double*)malloc(sizeCS*sizeof(double)), *d_y, *h_y = (double*)malloc(size*sizeof(double));
    for(int i=0;i<size;i++){h_y[i] = y[i];}
    cudaMalloc(&d_y, size*sizeof(double));
    cudaMalloc(&d_yCS, sizeCS*sizeof(double));
    cudaMemcpy(d_y, h_y, size*sizeof(double), cudaMemcpyHostToDevice);
    vector_to_cumu_sum<<<blocks, threadsPerBlock>>>(d_y, sizeCS, d_yCS, lag);
    cudaMemcpy(h_yCS, d_yCS, sizeCS*sizeof(double), cudaMemcpyDeviceToHost);
    //cudaFree(d_yCS);
    //h_yCS has cumulative sum vector

    //Debugging using print statements
//    for(int i=0;i<50;i++)printf("%d %d \n", i, h_tau[i]);
//     printf("\n");
//     for(int i=0;i<sizeCS;i++)printf("%f \n", h_yCS[i]); 

    //generate a support for regression (detrending)
    double *d_xReg, *h_xReg = (double*)malloc(h_tau[nTau-1]*sizeof(double));
    cudaMalloc(&d_xReg, h_tau[nTau-1]*sizeof(double));
    compute_support_regression<<<blocks, threadsPerBlock>>>(d_xReg, h_tau[nTau-1]);
    cudaMemcpy(h_xReg, d_xReg,  h_tau[nTau-1]*sizeof(double), cudaMemcpyDeviceToHost);
    //cudaFree(d_xReg);
    // for(int i=0;i<h_tau[nTau-1];i++)printf("%f \n", h_xReg[i]);
    // double * xReg = (double*)malloc(h_tau[nTau-1] * sizeof * xReg);
    // for(int i = 0; i < h_tau[nTau-1]; i++){xReg[i] = i+1;}//printf("%f\n", xReg[i]);

    double *F = (double*)malloc(nTau * sizeof(double)), *d_F;
    for(int i = 0; i < nTau; i++)
    {
        int nBuffer = sizeCS/h_tau[i];
        double * buffer = (double*)malloc(h_tau[i] * sizeof * buffer);
        double m = 0.0, b = 0.0;
        
        F[i] = 0;
        for(int j = 0; j < nBuffer; j++)
        {
            linreg(h_tau[i], h_xReg, h_yCS+j*h_tau[i], &m, &b);
            for(int k = 0; k < h_tau[i]; k++){buffer[k] = h_yCS[j*h_tau[i]+k] - (m * (k+1) + b);}
            if (strcmp(how, "rsrangefit") == 0) {
                F[i] += pow(max_(buffer, h_tau[i]) - min_(buffer, h_tau[i]), 2);
            }
            else if (strcmp(how, "dfa") == 0) {
                for(int k = 0; k<h_tau[i]; k++){
                    F[i] += buffer[k]*buffer[k];
                }
            }
            else{
                return 0.0;
            }
        }
        
        if (strcmp(how, "rsrangefit") == 0) {
            F[i] = sqrt(F[i]/nBuffer);
        }
        else if (strcmp(how, "dfa") == 0) {
            F[i] = sqrt(F[i]/(nBuffer*h_tau[i]));
        }
        free(buffer);
    }

    cudaMalloc(&d_F, nTau*sizeof(double));
    cudaMemcpy(d_F, F, nTau*sizeof(double), cudaMemcpyHostToDevice);
    double *h_logtt = (double*)malloc(nTau * sizeof(double)), *d_logtt;
    double *h_logFF = (double*)malloc(nTau * sizeof(double)), *d_logFF;
    int ntt = nTau;
    cudaMalloc(&d_logtt, nTau*sizeof(double));
    cudaMalloc(&d_logFF, nTau*sizeof(double));
    //to_parallelize
    calc_logtt_logFF<<<blocks, threadsPerBlock>>>(d_tau, nTau, d_F, d_logtt, d_logFF);
    //for (int i = 0; i < nTau; i++){logtt[i] = log(h_tau[i]);logFF[i] = log(F[i]);}
    cudaMemcpy(h_logFF, d_logFF, nTau*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_logtt, d_logtt, nTau*sizeof(double), cudaMemcpyDeviceToHost);
    int minPoints = 6, nsserr = (ntt - 2*minPoints + 1);
    double *sserr = (double*)malloc(nsserr * sizeof * sserr);
    double *h_buffer = (double*)malloc((ntt - minPoints + 1) * sizeof(double)), *d_buffer;
    cudaMalloc(&d_buffer, (ntt - minPoints + 1)*sizeof(double));
    // for(int i=0;i<nTau;i++){
    //     printf("%.5f %.5f\n", h_logFF[i], h_logtt[i]);
    // }
    for (int i = minPoints; i < ntt - minPoints + 1; i++)
    {
        // this could be done with less variables of course
        double m1 = 0.0, b1 = 0.0;
        double m2 = 0.0, b2 = 0.0;
        sserr[i - minPoints] = 0.0;
        
        linreg(i, h_logtt, h_logFF, &m1, &b1);
        linreg(ntt-i+1, h_logtt+i-1, h_logFF+i-1, &m2, &b2);
        parallel_buffer<<<blocks, threadsPerBlock>>>(i, d_buffer, m1, b1, d_logtt, d_logFF);
        cudaMemcpy(h_buffer, d_buffer, (ntt - minPoints + 1)*sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(d_buffer, &h_buffer, (ntt - minPoints + 1)*sizeof(double), cudaMemcpyHostToDevice);
        //for(int j = 0; j < i; j++){buffer[j] = logtt[j] * m1 + b1 - logFF[j];}
        sserr[i - minPoints] += norm_(h_buffer, i);
        parallel_buffer2<<<blocks, threadsPerBlock>>>(i, (ntt-i+1), d_buffer, m2, b2, d_logtt, d_logFF);
        cudaMemcpy(h_buffer, d_buffer, (ntt - minPoints + 1)*sizeof(double), cudaMemcpyDeviceToHost);
        //for(int j = 0; j < ntt-i+1; j++){buffer[j] = logtt[j+i-1] * m2 + b2 - logFF[j+i-1];}
        sserr[i - minPoints] += norm_(h_buffer, ntt-i+1);
    }
    
    double firstMinInd = 0.0;double minimum = min_(sserr, nsserr);
    for(int i = 0; i < nsserr; i++){if(sserr[i] == minimum){firstMinInd = i + minPoints - 1;break;}}
    //free everything
    return 2*(firstMinInd+1)/ntt;
}

double SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1(const double y[], const int size){
    return SC_FluctAnal_2_50_1_logi_prop_r1(y, size, 2, "dfa");
}
double SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1(const double y[], const int size){
    return SC_FluctAnal_2_50_1_logi_prop_r1(y, size, 1, "rsrangefit");
}

__global__ void autocorr_lag_kernel(const double *x, double *result, const int size, const int lag) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size - lag) {
        result[idx] = (x[idx] - x[idx + lag]) / (double)size;
    }
}

__global__ void corr_kernel(const double *x, const double *y, double *nom, double *denomX, double *denomY, const int size, const double meanX, const double meanY) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        nom[idx] = (x[idx] - meanX) * (y[idx] - meanY);
        denomX[idx] = (x[idx] - meanX) * (x[idx] - meanX);
        denomY[idx] = (y[idx] - meanY) * (y[idx] - meanY);
    }
}

// __global__ void mean_kernel(const double *a, double *result, const int size) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx < size) {
//         *result += a[idx];
//     }
// }

double IN_AutoMutualInfoStats_40_gaussian_fmmi(const double *y, const int size) {
    // NaN check
    for (int i = 0; i < size; i++) {
        if (isnan(y[i])) {
            return NAN;
        }
    }

    // maximum time delay
    int tau = 40;

    // don't go above half the signal length
    if (tau > ceil((double)size / 2)) {
        tau = ceil((double)size / 2);
    }

    // Allocate device memory
    double *d_y, *d_ami;
    cudaMalloc((void**)&d_y, size * sizeof(double));
    cudaMalloc((void**)&d_ami, size * sizeof(double));

    // Copy input data to device
    cudaMemcpy(d_y, y, size * sizeof(double), cudaMemcpyHostToDevice);

    // Launch autocorrelations kernel
    autocorr_lag_kernel<<<(size - tau + 255) / 256, 256>>>(d_y, d_ami, size, tau);

    // Copy results back to host
    double *ami = (double*)malloc(size * sizeof(double));
    cudaMemcpy(ami, d_ami, size * sizeof(double), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_y);
    cudaFree(d_ami);

    // compute automutual information
    double fmmi = tau;
    for (int i = 1; i < tau - 1; i++) {
        if (ami[i] < ami[i - 1] && ami[i] < ami[i + 1]) {
            fmmi = i;
            break;
        }
    }

    free(ami);

    return fmmi;
}

__global__ void prepareWindowedSignal(const double *y, const double *window, cufftDoubleComplex *d_F, int windowWidth, int NFFT, int offset, double m) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < windowWidth) {
        double windowedValue = (y[idx + offset] - m) * window[idx];
        d_F[idx].x = windowedValue;  
        d_F[idx].y = 0.0;            
    } else if (idx < NFFT) {
        d_F[idx].x = 0.0;  
        d_F[idx].y = 0.0;  
    }
}

void computeFFT(cufftHandle plan, cufftDoubleComplex *d_F, int NFFT) {
    cufftExecZ2Z(plan, d_F, d_F, CUFFT_FORWARD);
}

// __global__ void mean_kernel(const double* d_input, double* d_partial_sums, int size) {
//     extern __shared__ double sdata[];
//     unsigned int tid = threadIdx.x;
//     unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
//     sdata[tid] = (i < size) ? d_input[i] : 0;
//     __syncthreads();
//     for(unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
//         if (tid < s) {
//             sdata[tid] += sdata[tid + s];
//         }
//         __syncthreads();
//     }
//     if (tid == 0) d_partial_sums[blockIdx.x] = sdata[0];
// }

// double compute_mean_cuda(const double* h_input, int size) {
//     double *d_input, *d_partial_sums, *h_partial_sums;
//     int threadsPerBlock = 256;
//     int blocks = (270+size + threadsPerBlock - 1) / threadsPerBlock;
//     int partialSumsSize = blocks * sizeof(double);
//     cudaMalloc(&d_input, size * sizeof(double));
//     cudaMalloc(&d_partial_sums, partialSumsSize);
//     h_partial_sums = (double*)malloc(partialSumsSize);
//     cudaMemcpy(d_input, h_input, size * sizeof(double), cudaMemcpyHostToDevice);
//     mean_kernel<<<blocks, threadsPerBlock, threadsPerBlock * sizeof(double)>>>(d_input, d_partial_sums, size);
//     cudaDeviceSynchronize();
//     cudaMemcpy(h_partial_sums, d_partial_sums, partialSumsSize, cudaMemcpyDeviceToHost);
//     double totalSum = 0;
//     for (int i = 0; i < blocks; i++) {
//         totalSum += h_partial_sums[i];
//     }
//     cudaFree(d_input);
//     cudaFree(d_partial_sums);
//     free(h_partial_sums);
//     return totalSum / size;
// }

int welchCuda(const double y[], const int size, const int NFFT, const double Fs, const double window[], const int windowWidth, double **d_Pxx, double **d_f) {
    double dt = 1.0 / Fs;
    double df = Fs / NFFT;
    int k = floor((double)size / ((double)windowWidth / 2.0)) - 1;
    double *d_y, *d_window;
    double m = compute_mean_cuda(y, size);
    cufftDoubleComplex *d_F;
    cudaMalloc(&d_y, size * sizeof(double));
    cudaMemcpy(d_y, y, size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMalloc(&d_window, windowWidth * sizeof(double));
    cudaMemcpy(d_window, window, windowWidth * sizeof(double), cudaMemcpyHostToDevice);
    cudaMalloc(&d_F, NFFT * sizeof(cufftDoubleComplex));
    cufftHandle plan;
    cufftPlan1d(&plan, NFFT, CUFFT_Z2Z, 1);
    double *P = (double *)malloc(NFFT * sizeof(double));
    for (int i = 0; i < NFFT; i++) {
        P[i] = 0;
    }
    for (int i = 0; i < k; i++) {
        int offset = i * windowWidth / 2;
        int threadsPerBlock = 256;
        int blocksPerGrid = (NFFT + threadsPerBlock - 1) / threadsPerBlock;
        prepareWindowedSignal<<<blocksPerGrid, threadsPerBlock>>>(d_y, d_window, d_F, windowWidth, NFFT, offset, m);
        computeFFT(plan, d_F, NFFT);
        cufftDoubleComplex *F = (cufftDoubleComplex *)malloc(NFFT * sizeof(cufftDoubleComplex));
        cudaMemcpy(F, d_F, NFFT * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);
        for (int l = 0; l < NFFT; l++) {
            P[l] += (F[l].x * F[l].x + F[l].y * F[l].y);
        }    
        free(F);
    }
    cufftDestroy(plan);
    int Nout = (NFFT / 2 + 1);
    cudaMalloc((void**)d_Pxx, Nout * sizeof(double));
    cudaMalloc((void**)d_f, Nout * sizeof(double));
    double* temp_f = (double*)malloc(Nout * sizeof(double));
    double* temp_Pxx = (double*)malloc(Nout * sizeof(double));
    for (int i = 0; i < Nout; i++) {
        temp_f[i] = i * df;
        temp_Pxx[i] = P[i] / (k * windowWidth) * dt;
        if (i > 0 && i < Nout - 1) temp_Pxx[i] *= 2;
    }
    cudaMemcpy(*d_Pxx, temp_Pxx, Nout * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(*d_f, temp_f, Nout * sizeof(double), cudaMemcpyHostToDevice);
    free(P);
    free(temp_f);
    free(temp_Pxx);
    cudaFree(d_y);
    cudaFree(d_window);
    cudaFree(d_F);
    return Nout;
}

__global__ void calculateAngularFrequencyAndSpectrum(const double* f, double* w, double* Sw, int nWelch, double PI) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < nWelch) {
        w[idx] = 2 * PI * f[idx];
        Sw[idx] = Sw[idx] / (2 * PI);
    }
}

void cumsum(const double a[], const int size, double b[]) {
    b[0] = a[0];
    for (int i = 1; i < size; i++) {
        b[i] = a[i] + b[i-1];
    }   
}

double SP_Summaries_welch_rect_cuda(const double y[], int size, const char what[]) {
    for (int i = 0; i < size; ++i) {
        if (isnan(y[i])) return NAN;
    }
    double Fs = 1.0;
    int NFFT = pow(2, ceil(log2(size)));
    int windowWidth = size;
    double* window = (double*)malloc(windowWidth * sizeof(double));
    for (int i = 0; i < windowWidth; ++i) {
        window[i] = 1.0;
    }
    double* d_Pxx = nullptr;
    double* d_f = nullptr;
    double* d_window;
    cudaMalloc(&d_window, windowWidth * sizeof(double));
    cudaMemcpy(d_window, window, windowWidth * sizeof(double), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_Pxx, NFFT / 2 + 1 * sizeof(double));
    cudaMalloc((void**)&d_f, NFFT / 2 + 1 * sizeof(double));  
    int nWelch = welchCuda(y, size, NFFT, Fs, d_window, windowWidth, &d_Pxx, &d_f);
    free(window);
    cudaFree(d_window);
    double* Pxx = (double*)malloc(nWelch * sizeof(double));
    double* f = (double*)malloc(nWelch * sizeof(double));
    cudaMemcpy(Pxx, d_Pxx, nWelch * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(f, d_f, nWelch * sizeof(double), cudaMemcpyDeviceToHost);
    double PI = 3.14159265359;
    double* w = (double*)malloc(nWelch * sizeof(double));
    double* Sw = (double*)malloc(nWelch * sizeof(double));
    for (int i = 0; i < nWelch; i++) {
        w[i] = 2 * PI * f[i];
        Sw[i] = Pxx[i] / (2 * PI);
    }
    double dw = w[1] - w[0];
    double* csS = (double*)malloc(nWelch * sizeof(double));
    cumsum(Sw, nWelch, csS);
    double output = 0.0;
    if (strcmp(what, "centroid") == 0) {
        double csSThres = csS[nWelch - 1] * 0.5;
        double centroid = 0.0;
        for (int i = 0; i < nWelch; i++) {
            if (csS[i] > csSThres) {
                centroid = w[i];
                break;
            }
        }
        output = centroid;
        free(csS); 
    } else if (strcmp(what, "area_5_1") == 0) {
        int limit = nWelch / 5;
        double area_5_1 = 0.0;
        for (int i = 0; i < limit; i++) {
            area_5_1 += Sw[i];
        }
        area_5_1 *= dw;
        output = area_5_1;
    }
    cudaFree(d_Pxx);
    cudaFree(d_f);
    free(Pxx);
    free(f);
    free(w);
    free(Sw);
    return output;
}

__global__ void countZeroStretches(const int *yBin, int size, int *blockResults, int *blockStarts, int *blockEnds) {
	extern __shared__ int ldata[];
	int tid = threadIdx.x;
	int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
	ldata[tid] = (globalIdx < size) ? yBin[globalIdx] : 1;
	__syncthreads();
	int count = 0, maxCount = 0;
	for (int i = 0; i < blockDim.x; i++) {
		if (ldata[i] == 0) {
			count++;
			maxCount = max(maxCount, count);
		} else {
			count = 0;
		}
	}
	if (tid == 0) {
		blockResults[blockIdx.x] = maxCount;
	}
	if (tid == 0) {
		int startCount = 0;
		while (startCount < blockDim.x && ldata[startCount] == 0) {
			startCount++;
		}
		blockStarts[blockIdx.x] = startCount;
		int endCount = 0, i = blockDim.x - 1;
		while (i >= 0 && ldata[i] == 0) {
			endCount++;
			i--;
		}
		blockEnds[blockIdx.x] = endCount;
	}
}


__global__ void binarizeDiffs(const double *y, int *yBin, int size) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size - 1) {
		double diffTemp = y[i + 1] - y[i];
		yBin[i] = diffTemp < 0 ? 0 : 1;
	}
}
double SB_BinaryStats_diff_longstretch0_CUDA(const double y[], int size) {
	double *d_y;
	int *d_yBin, *d_blockResults, *d_blockStarts, *d_blockEnds;
	size_t bytes = size * sizeof(double);
	size_t bytesBin = (size - 1) * sizeof(int);
	cudaMalloc(&d_y, bytes);
	cudaMalloc(&d_yBin, bytesBin);
	int numBlocks = (size + 255) / 256;
	cudaMalloc(&d_blockResults, numBlocks * sizeof(int));
	cudaMalloc(&d_blockStarts, numBlocks * sizeof(int));
	cudaMalloc(&d_blockEnds, numBlocks * sizeof(int));
	cudaMemcpy(d_y, y, bytes, cudaMemcpyHostToDevice);
	binarizeDiffs<<<numBlocks, 256>>>(d_y, d_yBin, size);
	countZeroStretches<<<numBlocks, 256, 256 * sizeof(int)>>>(d_yBin, size - 1, d_blockResults, d_blockStarts, d_blockEnds);
	int *blockResults = (int *)malloc(numBlocks * sizeof(int));
	int *blockStarts = (int *)malloc(numBlocks * sizeof(int));
	int *blockEnds = (int *)malloc(numBlocks * sizeof(int));
	cudaMemcpy(blockResults, d_blockResults, numBlocks * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(blockStarts, d_blockStarts, numBlocks * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(blockEnds, d_blockEnds, numBlocks * sizeof(int), cudaMemcpyDeviceToHost);
	// printf("Block Results:\n");
	// for (int i = 0; i < numBlocks; ++i) {
	// 	printf("Block %d: Max Stretch = %d, Start Stretch = %d, End Stretch = %d\n", i, blockResults[i], blockStarts[i], blockEnds[i]);
	// }
	int globalMaxStretch = 0, currentStretch = blockResults[0];
	for (int i = 1; i < numBlocks; ++i) {
		if (blockEnds[i - 1] > 0 && blockStarts[i] > 0) {
			currentStretch += blockStarts[i] + blockEnds[i - 1] - 1;
		} else {
			globalMaxStretch = max(globalMaxStretch, currentStretch);
			currentStretch = blockResults[i];
		}
		globalMaxStretch = max(globalMaxStretch, blockResults[i]);
	}
	globalMaxStretch = max(globalMaxStretch, currentStretch);
	cudaFree(d_y);
	cudaFree(d_yBin);
	cudaFree(d_blockResults);
	cudaFree(d_blockStarts);
	cudaFree(d_blockEnds);
	free(blockResults);
	free(blockStarts);
	free(blockEnds);
	return (double)globalMaxStretch;
}

__global__ void countOneStretches(const int *yBin, int size, int *maxStretches, int *startStretches, int *endStretches) {
	extern __shared__ int l_data[];
	int tid = threadIdx.x;
	int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if (globalIdx < size) {
		l_data[tid] = yBin[globalIdx];
	} else {
		l_data[tid] = 0;
	}
	__syncthreads();
	int localMaxStretch = 0;
	int count = (tid == 0 && globalIdx > 0) ? (yBin[globalIdx - 1] == 1) : 0;
	for (int i = 0; i < blockDim.x && globalIdx + i < size; i++) {
		if (l_data[i] == 1) {
			count++;
			localMaxStretch = max(localMaxStretch, count);
		} else {
			count = 0;
		}
	}
	if (tid == 0) {
		maxStretches[blockIdx.x] = localMaxStretch;
	}
	if (tid == 0) {
		int startStretch = 0;
		if (globalIdx > 0 && yBin[globalIdx - 1] == 1) {
			for (int i = 0; i < blockDim.x && i < size - globalIdx; i++) {
				if (l_data[i] == 1) startStretch++;
				else break;
			}
		}
		startStretches[blockIdx.x] = startStretch;
		int endStretch = 0;
		if (globalIdx + blockDim.x < size && yBin[globalIdx + blockDim.x] == 1) {
			for (int i = blockDim.x - 1; i >= 0 && globalIdx + i < size; i--) {
				if (l_data[i] == 1) endStretch++;
				else break;
			}
		}
		endStretches[blockIdx.x] = endStretch;
	}
}


__global__ void binarizeDiffs_1(const double *y, int *yBin, int size, double mean) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size - 1) {
		yBin[i] = (y[i] - mean <= 0) ? 0 : 1;
	}
}
double SB_BinaryStats_diff_longstretch1_CUDA(const double y[], int size) {	
	double *d_y;
	int *d_yBin, *d_blockResults, *d_blockStarts, *d_blockEnds;
	size_t bytes = size * sizeof(double);
	size_t bytesBin = (size - 1) * sizeof(int);
	double mean = compute_mean_cuda(y, size);
	cudaMalloc(&d_y, bytes);
	cudaMalloc(&d_yBin, bytesBin);
	int numBlocks = (size + 255) / 256;
	cudaMalloc(&d_blockResults, numBlocks * sizeof(int));
	cudaMalloc(&d_blockStarts, numBlocks * sizeof(int));
	cudaMalloc(&d_blockEnds, numBlocks * sizeof(int));
	cudaMemcpy(d_y, y, bytes, cudaMemcpyHostToDevice);
	binarizeDiffs_1<<<numBlocks, 256>>>(d_y, d_yBin, size, mean);
	countOneStretches<<<numBlocks, 256, 256 * sizeof(int)>>>(d_yBin, size - 1, d_blockResults, d_blockStarts, d_blockEnds);
	int *blockResults = (int *)malloc(numBlocks * sizeof(int));
	int *blockStarts = (int *)malloc(numBlocks * sizeof(int));
	int *blockEnds = (int *)malloc(numBlocks * sizeof(int));
	cudaMemcpy(blockResults, d_blockResults, numBlocks * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(blockStarts, d_blockStarts, numBlocks * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(blockEnds, d_blockEnds, numBlocks * sizeof(int), cudaMemcpyDeviceToHost);
	// printf("Block Results:\n");
	// for (int i = 0; i < numBlocks; ++i) {
	// 	printf("Block %d: Max Stretch = %d, Start Stretch = %d, End Stretch = %d\n", i, blockResults[i], blockStarts[i], blockEnds[i]);
	// }
	int globalMaxStretch = 0, currentStretch = blockResults[0];
	for (int i = 1; i < numBlocks; ++i) {
		if (blockEnds[i - 1] > 0 && blockStarts[i] > 0) {
			currentStretch += blockStarts[i] + blockEnds[i - 1] - 1;
		} else {
			globalMaxStretch = max(globalMaxStretch, currentStretch);
			currentStretch = blockResults[i];
		}
		globalMaxStretch = max(globalMaxStretch, blockResults[i]);
	}
	globalMaxStretch = max(globalMaxStretch, currentStretch);
	cudaFree(d_y);
	cudaFree(d_yBin);
	cudaFree(d_blockResults);
	cudaFree(d_blockStarts);
	cudaFree(d_blockEnds);
	free(blockResults);
	free(blockStarts);
	free(blockEnds);
	return (double)globalMaxStretch;
}


void diff(const double a[], const int size, double b[]) {
	for (int i = 1; i < size; i++) {
		b[i - 1] = a[i] - a[i - 1];
	}
}

// CUDA kernel to calculate differences between consecutive elements
__global__ void cudaDiffKernel(const double* y, int size, double* Dy) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < size - 1) {
		Dy[tid] = y[tid + 1] - y[tid];	
	}
}

// CUDA kernel to calculate PNN40
__global__ void calculatePNN40Kernel(double* Dy, int size, double pNNx, double* pnn40) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < size - 1) {
		pnn40[tid] = (fabs(Dy[tid]) * 1000 > pNNx) ? 1.0 : 0.0;
	}
	//printf("Thread %d: Dy[%d] = %f, pnn40[%d] = %f\n", tid, tid, Dy[tid], tid, pnn40[tid]);
}

// CUDA kernel for reduction
__global__ void reductionKernel(double* pnn40, int size, double* result) {	
	extern __shared__ double shared_data[];
	int tid = threadIdx.x;
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	shared_data[tid] = (idx < size) ? pnn40[idx] : 0.0;
	__syncthreads();
	// Parallel reduction to calculate the sum of 1s
	for (int stride = 1; stride < blockDim.x; stride *= 2) {
		int index = 2 * stride * tid;
		if (index < blockDim.x) {
			shared_data[index] += shared_data[index + stride];
		}
	__syncthreads();
	}
	if (tid == 0) {
		atomicAdd(reinterpret_cast<unsigned long long*>(result),
		__double_as_longlong(shared_data[0]));
	}
}
double MD_hrv_classic_pnn40(const double y[], const int size) {
	const int numThreadsPerBlock = 256;
	const int numBlocksPerGrid = (size + numThreadsPerBlock - 1) / numThreadsPerBlock;
	const double pNNx = 40.0;
	double* d_y, *d_Dy;
	cudaMalloc((void**)&d_y, size * sizeof(double));
	cudaMalloc((void**)&d_Dy, (size - 1) * sizeof(double));
	cudaMemcpy(d_y, y, size * sizeof(double), cudaMemcpyHostToDevice);
	cudaDiffKernel<<<numBlocksPerGrid, numThreadsPerBlock>>>(d_y, size, d_Dy);
	cudaDeviceSynchronize();
	double* d_pnn40, h_pnn40 = 0.0;
	cudaMalloc((void**)&d_pnn40, (size - 1) * sizeof(double));
	calculatePNN40Kernel<<<numBlocksPerGrid, numThreadsPerBlock>>>(d_Dy, size, pNNx, d_pnn40);
	double* d_h_pnn40;
	cudaMalloc((void**)&d_h_pnn40, sizeof(double));

	// reductionKernel<<<1, numThreadsPerBlock, numThreadsPerBlock * sizeof(double)>>>(d_pnn40, size - 1, &h_pnn40);
	// Call the reduction kernel
	reductionKernel<<<1, numThreadsPerBlock, numThreadsPerBlock * sizeof(double)>>>(d_pnn40, size - 1, d_h_pnn40);
	// Copy the result back from device to host
	cudaMemcpy(&h_pnn40, d_h_pnn40, sizeof(double), cudaMemcpyDeviceToHost);
	// Print the result
	double result;
	result = h_pnn40/(size-1);
	//printf("Final Result from Host: %f\n", h_pnn40);
	printf("Result: %f\n", result);
	// Copy data back from device to host
	cudaMemcpy(&h_pnn40, d_pnn40, sizeof(double), cudaMemcpyDeviceToHost);
	// Free device memory
	cudaFree(d_h_pnn40);
	cudaFree(d_y);
	cudaFree(d_Dy);
	cudaFree(d_pnn40);
	// free(h_Dy);
	return h_pnn40 / (size - 1);
}

__device__ double atomicAddDouble(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}


// Kernel to calculate the sum of products for covariance using shared memory
__global__ void cuda_cov_mean(const double *x, const double *y, const int size, double *answer) {
    extern __shared__ double ans[];
    int idx = blockDim.x * blockIdx.x + threadIdx.x, tid = threadIdx.x;
    double prod = 0;
    if (idx < size) {
        prod = x[idx] * y[idx];
    }
    ans[tid] = prod;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            ans[tid] += ans[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAddDouble(answer, ans[0] / size);
    }
}

// Kernel to compute ySub
__global__ void compute_ySub(double *d_ySub, const double *y, const double *ySpline, const int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_ySub[idx] = y[idx] - ySpline[idx];
    }
}

// Kernel to compute troughs and peaks
__global__ void compute_troughs_peaks(const int acmax, const double *d_acf, double *d_troughs, double *d_peaks, int *d_nTroughs, int *d_nPeaks) {
    __shared__ int shared_nTroughs;
    __shared__ int shared_nPeaks;
    double slopeIn = 0, slopeOut = 0;
    int idx = blockIdx.x * blockDim.x + threadIdx.x + 1;

    if (idx < acmax - 1) {
        slopeIn = d_acf[idx] - d_acf[idx - 1];
        slopeOut = d_acf[idx + 1] - d_acf[idx];
        if (threadIdx.x == 0) {
            shared_nPeaks = 0;
            shared_nTroughs = 0;
        }

        __syncthreads();

        if (slopeIn < 0 && slopeOut > 0) {
            int index = atomicAdd(&shared_nTroughs, 1);
            d_troughs[index] = idx;
        } else if (slopeIn > 0 && slopeOut < 0) {
            int index = atomicAdd(&shared_nPeaks, 1);
            d_peaks[index] = idx;
        }

        __syncthreads();

        if (threadIdx.x == 0) {
            atomicAdd(d_nTroughs, shared_nTroughs);
            atomicAdd(d_nPeaks, shared_nPeaks);
        }
    }
}

// Host function to launch covariance kernel
void launch_cuda_cov_mean(const double *x, const double *y, int size, double *d_answer) {
    int threadsPerBlock = 256;
    int blocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    cuda_cov_mean<<<blocks, threadsPerBlock, threadsPerBlock * sizeof(double)>>>(x, y, size, d_answer);
}

// Host function to compute autocovariance with lag
void compute_autocov_lag(double *d_acf, const int acmax, const int size, const double *d_ySub, double *d_answer) {
    for (int tau = 1; tau <= acmax; ++tau) {
        cudaMemset(d_answer, 0, sizeof(double));
        launch_cuda_cov_mean(d_ySub, &d_ySub[tau], size - tau, d_answer);
        cudaMemcpy(&d_acf[tau], d_answer, sizeof(double), cudaMemcpyDeviceToDevice);
    }
}

int periodicity_wang(const double * y, const int size) {
    int threadsPerBlock = 256;
    int blocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    for (int i = 0; i < size; i++) {
        if (isnan(y[i])) {
            return 0;
        }
    }

    const double th = 0.01;
    double *ySpline = (double*)malloc(size * sizeof(double));
    splinefit(y, size, ySpline);
    //for(int i=0;i<size;i++)printf("%.5f\n", ySpline[i]);
    double *d_ySub, *h_ySub = (double*)malloc(size * sizeof(double));
    cudaMalloc(&d_ySub, size * sizeof(double));
    compute_ySub<<<blocks, threadsPerBlock>>>(d_ySub, y, ySpline, size);
    cudaMemcpy(h_ySub, d_ySub, size * sizeof(double), cudaMemcpyDeviceToHost);

    int acmax = (int)ceil((double)size / 3);

    double *d_acf, *h_acf = (double*)malloc(acmax * sizeof(double));
    cudaMalloc(&d_acf, acmax * sizeof(double));
    double h_answer = 0;
    double* d_answer;
    cudaMalloc((void**)&d_answer, sizeof(double));
    cudaMemcpy(d_answer, &h_answer, sizeof(double), cudaMemcpyHostToDevice);
    compute_autocov_lag(d_acf, acmax, size, d_ySub, d_answer);
    cudaMemcpy(h_acf, d_acf, acmax * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_answer, d_answer, sizeof(double), cudaMemcpyDeviceToHost);

    double *d_troughs, *h_troughs = (double*)malloc(acmax * sizeof(double));
    double *d_peaks, *h_peaks = (double*)malloc(acmax * sizeof(double));
    int *d_nTroughs, h_nTroughs = 0, *d_nPeaks, h_nPeaks = 0;
    cudaMalloc(&d_nTroughs, sizeof(int));
    cudaMalloc(&d_nPeaks, sizeof(int));
    cudaMalloc(&d_troughs, acmax * sizeof(double));
    cudaMalloc(&d_peaks, acmax * sizeof(double));

    compute_troughs_peaks<<<blocks, threadsPerBlock>>>(acmax, d_acf, d_troughs, d_peaks, d_nPeaks, d_nTroughs);

    cudaMemcpy(h_troughs, d_troughs, acmax * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_peaks, d_peaks, acmax * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_nTroughs, d_nTroughs, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_nPeaks, d_nPeaks, sizeof(int), cudaMemcpyDeviceToHost);

    int iPeak = 0, iTrough = 0, out = 0;
    double thePeak = 0, theTrough = 0;
    for (int i = 0; i < h_nPeaks; i++) {
        iPeak = h_peaks[i];
        thePeak = h_acf[iPeak];

        int j = -1;
        while (h_troughs[j + 1] < iPeak && (j + 1) < h_nTroughs) j++;
        if (j == -1) continue;
        iTrough = h_troughs[j];
        theTrough = h_acf[iTrough];

        if (thePeak - theTrough < th) continue;
        if (thePeak < 0) continue;

        out = iPeak;
        break;
    }
    free(ySpline);
    free(h_ySub);
    free(h_acf);
    free(h_troughs);
    free(h_peaks);
    cudaFree(d_ySub);
    cudaFree(d_acf);
    cudaFree(d_answer);
    cudaFree(d_troughs);
    cudaFree(d_peaks);
    cudaFree(d_nTroughs);
    cudaFree(d_nPeaks);
    return out;
}

__global__ void markOutliers(const double *data, int *outlierFlags, double threshold, int size, int sign) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        outlierFlags[idx] = sign > 0 ? data[idx] >= threshold : data[idx] <= -threshold;
    }
}

__global__ void calculateDistances(const int *indices, double *distances, int numIndices) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numIndices - 1) {
        distances[idx] = indices[idx + 1] - indices[idx];
    }
}

double DN_OutlierInclude_np_001_mdrmd_CUDA(const double *y_host, int size, int sign) {
    double *data_dev = nullptr;
    cudaMalloc(&data_dev, size * sizeof(double));
    cudaMemcpy(data_dev, y_host, size * sizeof(double), cudaMemcpyHostToDevice);

    int *outlierFlags_dev = nullptr;
    cudaMalloc(&outlierFlags_dev, size * sizeof(int));

    double maxValue = 0;
    if (sign > 0) {
        thrust::device_ptr<double> dev_ptr(data_dev);
        maxValue = *thrust::max_element(dev_ptr, dev_ptr + size);
    } else {
        thrust::device_ptr<double> dev_ptr(data_dev);
        double minValue = *thrust::min_element(dev_ptr, dev_ptr + size);
        maxValue = fabs(minValue);
    }

    double inc = 0.01;
    int nThresh = static_cast<int>(maxValue / inc) + 1;

    thrust::device_vector<double> msDti1(nThresh), msDti4(nThresh);
    thrust::host_vector<double> msDti3(nThresh);

    int blockSize = 256;
    for (int j = 0; j < nThresh; ++j) {
        double threshold = j * inc;

        markOutliers<<<(size + blockSize - 1) / blockSize, blockSize>>>(data_dev, outlierFlags_dev, threshold, size, sign);
        cudaDeviceSynchronize();

        thrust::device_vector<int> indices(size);
        thrust::sequence(indices.begin(), indices.end());

        thrust::device_vector<int> outlierIndices(size);
        auto end = thrust::copy_if(thrust::device, indices.begin(), indices.end(), outlierFlags_dev, outlierIndices.begin(), [] __device__(int flag) { return flag == 1; });
        int numOutliers = end - outlierIndices.begin();

        if (numOutliers > 1) {
            thrust::device_vector<double> distances(numOutliers - 1);
            calculateDistances<<<(numOutliers - 1 + blockSize - 1) / blockSize, blockSize>>>(thrust::raw_pointer_cast(outlierIndices.data()), thrust::raw_pointer_cast(distances.data()), numOutliers);
            cudaDeviceSynchronize();

            msDti1[j] = thrust::reduce(distances.begin(), distances.end()) / (numOutliers - 1);
            msDti4[j] = (outlierIndices[numOutliers / 2] - size / 2.0) / (size / 2.0);
        } else {
            msDti1[j] = 0;
            msDti4[j] = 0;
        }

        msDti3[j] = static_cast<double>(numOutliers) / size * 100.0;
    }

    thrust::sort(msDti4.begin(), msDti4.end());
    double outputScalar = msDti4[nThresh / 2];

    cudaFree(data_dev);
    cudaFree(outlierFlags_dev);

    return outputScalar;
}
// __device__ double atomicAddDouble(double* address, double val)
// {
//     unsigned long long int* address_as_ull = (unsigned long long int*)address;
//     unsigned long long int old = *address_as_ull, assumed;
//     do {
//         assumed = old;
//         old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
//     } while (assumed != old);
//     return __longlong_as_double(old);
// }

__global__ void histogramKernel(double *data, int size, double minVal, double maxVal, int nBins, double *histCounts, double binStep) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int binIdx = (int)((data[idx] - minVal) / binStep);
        if(binIdx >= 0 && binIdx < nBins) {
            atomicAddDouble(&(histCounts[binIdx]), 1.0);
        }
    }
}

double hist5(double *h_data, const int size){
    //int size = 270; // or read from file to set size dynamically
    //double *h_data = (double*)malloc(size * sizeof(double));
    double h_minVal = DBL_MAX, h_maxVal = -DBL_MAX;
    for (int i = 0; i < size; ++i) {
        if (h_data[i] < h_minVal) h_minVal = h_data[i];
            if (h_data[i] > h_maxVal) h_maxVal = h_data[i];
    }

    int nBins = 5; // Set No. of bins
    double binStep = (h_maxVal - h_minVal) / nBins;

    double *h_histCounts = (double*)malloc(nBins * sizeof(double));
    for (int i = 0; i < nBins; ++i) h_histCounts[i] = 0;

    double *d_data, *d_histCounts;
    cudaMalloc((void**)&d_data, size * sizeof(double));
    cudaMalloc((void**)&d_histCounts, nBins * sizeof(double));

    cudaMemcpy(d_data, h_data, size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_histCounts, h_histCounts, nBins * sizeof(double), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    histogramKernel<<<gridSize, blockSize>>>(d_data, size, h_minVal, h_maxVal, nBins, d_histCounts, binStep);

    cudaMemcpy(h_histCounts, d_histCounts, nBins * sizeof(double), cudaMemcpyDeviceToHost);
    double maxCount = 0;
    int numMaxs = 1;
    double mode = 0;
    for (int i = 0; i < nBins; ++i) {
        if (h_histCounts[i] > maxCount) {
            maxCount = h_histCounts[i];
            numMaxs = 1;
            mode = h_minVal + binStep * (i + 0.5);
        }else if (h_histCounts[i] == maxCount) {
            numMaxs += 1;
            mode += h_minVal + binStep * (i + 0.5);
        }
    }
    mode = mode / numMaxs;

    // Printing the mode
    printf("The mode is: %f\n", mode);
    
    // cudaFree(d_data);
    // cudaFree(d_histCounts);
    // free(h_data);
    // free(h_histCounts);
    return 0;
}
double hist10(double *h_data, const int size){
    //int size = 270; // or read from file to set size dynamically
    //double *h_data = (double*)malloc(size * sizeof(double));
    double h_minVal = DBL_MAX, h_maxVal = -DBL_MAX;
    for (int i = 0; i < size; i++) {
        if (h_data[i] < h_minVal) h_minVal = h_data[i];
            if (h_data[i] > h_maxVal) h_maxVal = h_data[i];
    }

    int nBins = 10; // Set No. of bins
    double binStep = (h_maxVal - h_minVal) / nBins;

    double *h_histCounts = (double*)malloc(nBins * sizeof(double));
    for (int i = 0; i < nBins; ++i) h_histCounts[i] = 0;

    double *d_data, *d_histCounts;
    cudaMalloc((void**)&d_data, size * sizeof(double));
    cudaMalloc((void**)&d_histCounts, nBins * sizeof(double));

    cudaMemcpy(d_data, h_data, size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_histCounts, h_histCounts, nBins * sizeof(double), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    histogramKernel<<<gridSize, blockSize>>>(d_data, size, h_minVal, h_maxVal, nBins, d_histCounts, binStep);

    cudaMemcpy(h_histCounts, d_histCounts, nBins * sizeof(double), cudaMemcpyDeviceToHost);
    double maxCount = 0;
    int numMaxs = 1;
    double mode = 0;
    for (int i = 0; i < nBins; ++i) {
        if (h_histCounts[i] > maxCount) {
            maxCount = h_histCounts[i];
            numMaxs = 1;
            mode = h_minVal + binStep * (i + 0.5);
        }else if (h_histCounts[i] == maxCount) {
            numMaxs += 1;
            mode += h_minVal + binStep * (i + 0.5);
        }
    }
    mode = mode / numMaxs;

    // Printing the mode
    printf("The mode is: %f\n", mode);
    
    // cudaFree(d_data);
    // cudaFree(d_histCounts);
    // free(h_data);
    // free(h_histCounts);
    return 0;
}

__global__ void findPotentialZeroCrossings(const double *autocorrs, int *zeroCrossingFlags, int size, int maxtau) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < maxtau && idx < size) {
		zeroCrossingFlags[idx] = (autocorrs[idx] <= 0) ? idx : maxtau;
	}
}

__global__ void reduceMinIndex(int *input, int size) {
	int idx = threadIdx.x;
	for (int offset = 1; offset < blockDim.x; offset *= 2) {
		if (idx % (2 * offset) == 0 && idx + offset < size) {
			if (input[idx + offset] < input[idx]) {
				input[idx] = input[idx + offset];
			}
		}
		__syncthreads();  // Ensure all threads have updated their values before next iteration
	}
}

__global__ void complexConjugateMultiply(cufftDoubleComplex *data, int n) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n) {
		cufftDoubleComplex val = data[idx];
		data[idx] = cuCmul(val, make_cuDoubleComplex(val.x, -val.y)); // element * its conjugate
								        }
}

__global__ void normalizeAutocorr(double *data, int n) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n) {
		data[idx] /= data[0]; // Normalize by the first element (maximum value)
	}
}


double* cudaComputeAutocorrs(const double *y, int size) {
	int nFFT = nextpow2(size); // Assuming nextPow2 function is defined elsewhere
	// Allocate memory
	double *d_y, *autocorr;
	cufftDoubleComplex *d_freqDomain;
	cudaMalloc(&d_y, nFFT * sizeof(double));
	cudaMalloc(&d_freqDomain, nFFT * sizeof(cufftDoubleComplex));
	cudaMemset(d_y, 0, nFFT * sizeof(double)); // Zero padding
	cudaMemcpy(d_y, y, size * sizeof(double), cudaMemcpyHostToDevice);
	// Create CUFFT plans
	cufftHandle planForward, planInverse;
	cufftPlan1d(&planForward, nFFT, CUFFT_D2Z, 1);
	cufftPlan1d(&planInverse, nFFT, CUFFT_Z2D, 1);
	// Perform forward FFT
	cufftExecD2Z(planForward, d_y, d_freqDomain);
	// Compute complex conjugate multiplication
	int blockSize = 256;
	int numBlocks = (nFFT + blockSize - 1) / blockSize;
	complexConjugateMultiply<<<numBlocks, blockSize>>>(d_freqDomain, nFFT);
	// Perform inverse FFT
	cufftExecZ2D(planInverse, d_freqDomain, d_y);
	// Normalize the result
	normalizeAutocorr<<<numBlocks, blockSize>>>(d_y, nFFT);
	// Copy the result back to the host
	autocorr = (double*)malloc(nFFT * sizeof(double));
	cudaMemcpy(autocorr, d_y, nFFT * sizeof(double), cudaMemcpyDeviceToHost);
	// Cleanup
	cufftDestroy(planForward);
	cufftDestroy(planInverse);
	cudaFree(d_y);
	cudaFree(d_freqDomain);
	return autocorr;
}


__global__ void nanCheckKernel(const double *y, int size, int *nanFlag) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size && isnan(y[idx])) {
		atomicExch(nanFlag, 1);  // Set nanFlag to 1 if NaN is found
	}
}
__global__ void computeResKernel(const double *y, int size, int train_length, double *res) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size - train_length) {
		double yest = 0.0;
		for (int j = 0; j < train_length; j++) {
			yest += y[idx + j];
		}
		yest /= train_length;
		res[idx] = y[idx + train_length] - yest;
	}
}
__global__ void computeMeanKernel(const double *res, int size, double *mean) {
	extern __shared__ double sharedData[];
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int tid = threadIdx.x;
	// Load data into shared memory
	sharedData[tid] = (idx < size) ? res[idx] : 0;
	__syncthreads();
	// Reduction to compute the sum
	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (tid < s) {
			sharedData[tid] += sharedData[tid + s];
		}
		__syncthreads();
	}
	// Compute the mean in the first thread of each block
	if (tid == 0) {
		 atomicAddDouble(mean, sharedData[0] / size);
	}
}

// Kernel to compute the variance of the residuals
__global__ void computeVarianceKernel(const double *res, int size, double mean, double *variance) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size) {
		double diff = res[idx] - mean;
		atomicAddDouble(variance, diff * diff / size);
	}
}


int co_firstzero_cuda(const double y[], const int size, const int maxtau) {
	double *d_autocorrs;
	int *d_zeroCrossingFlags;
	int minIndex = maxtau;
	double *x = cudaComputeAutocorrs(y,size);  		        // Allocate memory on the device
	cudaMalloc((void **)&d_autocorrs, size * sizeof(double));
	cudaMalloc((void **)&d_zeroCrossingFlags, size * sizeof(int));
	// Copy data to the device
	cudaMemcpy(d_autocorrs, x, size * sizeof(double), cudaMemcpyHostToDevice);
	// Initialize zeroCrossingFlags with maxtau
	cudaMemset(d_zeroCrossingFlags, maxtau, size * sizeof(int));
	// Launch the findPotentialZeroCrossings kernel
	int threadsPerBlock = 256;
	int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
	findPotentialZeroCrossings<<<blocksPerGrid, threadsPerBlock>>>(d_autocorrs, d_zeroCrossingFlags, size, maxtau);
	// Reduce to find the minimum index
	int numThreadsForReduce = 1024;  // Choose based on your device's capability
	reduceMinIndex<<<1, numThreadsForReduce>>>(d_zeroCrossingFlags, size);
	// Copy the result back to the host
	cudaMemcpy(&minIndex, d_zeroCrossingFlags, sizeof(int), cudaMemcpyDeviceToHost);
	// Free device memory
	cudaFree(d_autocorrs);
	cudaFree(d_zeroCrossingFlags);
	return minIndex;
}

// Kernel to compute the variance of the residuals

double FC_LocalSimple_mean_tauresrat_CUDA(const double y[], const int size, const int train_length) {
	double *d_y, *d_res;
	int *d_nanFlag;
	int nanFlag = 0;
	// Allocate memory on the device
	cudaMalloc((void **)&d_y, size * sizeof(double));
	cudaMalloc((void **)&d_res, (size - train_length) * sizeof(double));
	cudaMalloc((void **)&d_nanFlag, sizeof(int));
	// Copy data to the device
	cudaMemcpy(d_y, y, size * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemset(d_nanFlag, 0, sizeof(int));  // Initialize nanFlag to 0
	// Launch the nanCheckKernel
	int threadsPerBlock = 256;
	int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
	nanCheckKernel<<<blocksPerGrid, threadsPerBlock>>>(d_y, size, d_nanFlag);
	// Copy the nanFlag back to host and check
	cudaMemcpy(&nanFlag, d_nanFlag, sizeof(int), cudaMemcpyDeviceToHost);
	if (nanFlag) {
		cudaFree(d_y);
		cudaFree(d_res);
		cudaFree(d_nanFlag);
		return NAN;  // Return if any NaN found
	}
	// Launch the computeResKernel
	computeResKernel<<<blocksPerGrid, threadsPerBlock>>>(d_y, size, train_length, d_res);
	// Copy res back to host
	double *res = (double *)malloc((size - train_length) * sizeof(double));
	cudaMemcpy(res, d_res, (size - train_length) * sizeof(double), cudaMemcpyDeviceToHost);
	// Continue with the sequential part...
	double resAC1stZ = co_firstzero_cuda(res, size - train_length, size - train_length);
	double yAC1stZ = co_firstzero_cuda(y, size, size);  // This can be optimized further if y is large
	double output = resAC1stZ / yAC1stZ;
	// This includes calls to co_firstzero and calculation of the final output.
	// Free device memory
	// cudaFree(d_y);
	// cudaFree(d_res);
	// cudaFree(d_nanFlag);
	// Free host memory
	// free(res);
	// Return the final output (placeholder, replace with actual computation)
	return output;
}
double FC_LocalSimple_mean_stderr_CUDA(const double y[], const int size, const int train_length) {
	double *d_y, *d_res, *d_mean, *d_variance;
	int *d_nanFlag;
	double mean, variance, stddev;
	int nanFlag = 0;
	// Allocate memory on the device
	cudaMalloc((void **)&d_y, size * sizeof(double));
	cudaMalloc((void **)&d_res, (size - train_length) * sizeof(double));
	cudaMalloc((void **)&d_mean, sizeof(double));
	cudaMalloc((void **)&d_variance, sizeof(double));
	cudaMalloc((void **)&d_nanFlag, sizeof(int));
	// Copy data to the device
	cudaMemcpy(d_y, y, size * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemset(d_nanFlag, 0, sizeof(int));
	cudaMemset(d_mean, 0, sizeof(double));
	cudaMemset(d_variance, 0, sizeof(double));
	// Define kernel execution configuration
	int threadsPerBlock = 256;
	int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
	int sharedSize = threadsPerBlock * sizeof(double);
	// Launch kernels
	nanCheckKernel<<<blocksPerGrid, threadsPerBlock>>>(d_y, size, d_nanFlag);
	cudaMemcpy(&nanFlag, d_nanFlag, sizeof(int), cudaMemcpyDeviceToHost);
	if (nanFlag) {
		cudaFree(d_y);
		cudaFree(d_res);
		cudaFree(d_mean);
		cudaFree(d_variance);
		cudaFree(d_nanFlag);
		return NAN;  // Return NaN if any NaNs found in the input
	}
	computeResKernel<<<blocksPerGrid, threadsPerBlock>>>(d_y, size, train_length, d_res);
	// Compute the mean of residuals
	computeMeanKernel<<<blocksPerGrid, threadsPerBlock, sharedSize>>>(d_res, size - train_length, d_mean);
	// Compute the variance of residuals
  // Host variable for mean
	cudaMemcpy(&mean, d_mean, sizeof(double), cudaMemcpyDeviceToHost);
	computeVarianceKernel<<<blocksPerGrid, threadsPerBlock>>>(d_res, size - train_length, mean, d_variance);
	// Copy the variance back to host and compute standard deviation
	cudaMemcpy(&variance, d_variance, sizeof(double), cudaMemcpyDeviceToHost);
	stddev = sqrt(variance);
	// Clean up
	cudaFree(d_y);
	cudaFree(d_res);
	cudaFree(d_mean);
	cudaFree(d_variance);
	cudaFree(d_nanFlag);
	return stddev;
}
__global__ void compute_y1_y2(double *d_y1, double *d_y2, const double *d_y, const int tau, const int n){
    int idx = blockDim.x*blockIdx.x + threadIdx.x;
    if(idx<n){
        d_y1[idx] = d_y[idx];
        d_y2[idx] = d_y[idx+tau];
    }
}

int * histbinassign(const double y[], const int size, const double binEdges[], const int nEdges)
{
    // variable to store counted occurances in
    int * binIdentity = (int*)malloc(size * sizeof(int));
    for(int i = 0; i < size; i++)
    {
        // if not in any bin -> 0
        binIdentity[i] = 0;
        // go through bin edges
        for(int j = 0; j < nEdges; j++){
            if(y[i] < binEdges[j]){
                binIdentity[i] = j;
                break;
            }
        }
    }
    return binIdentity;
}

int * histcount_edges(const double y[], const int size, const double binEdges[], const int nEdges)
{
    int * histcounts = (int*)malloc(nEdges * sizeof(int));
    for(int i = 0; i < nEdges; i++){histcounts[i] = 0;}
    for(int i = 0; i < size; i++)
    {
        // go through bin edges
        for(int j = 0; j < nEdges; j++){
            if(y[i] <= binEdges[j]){
                histcounts[j] += 1;
                break;
            }
        }
    }
    return histcounts;
}
__global__ void compute_binEdges12(double *d_binEdges12, const int n){
    int idx = threadIdx.x+blockDim.x*blockIdx.x;
    if(idx<n){
        d_binEdges12[idx] = idx+1;
    }
}
__global__ void compute_binEdges(const double minValue, const double binStep, const int n, double *d_binEdges){
    int idx = blockDim.x*blockIdx.x + threadIdx.x;
    if(idx<n){
        //binEdges[i] = minValue + binStep*i - 0.1;
        d_binEdges[idx] = minValue + binStep*idx - 0.1;
    }
}
__global__ void compute_bins12(const int numBins, int *d_bins1, int *d_bins2, double *d_bins12, const int n){
    int idx = blockDim.x*blockIdx.x + threadIdx.x;
    if(idx<n){
        //bins12[i] = (bins1[i]-1)*(numBins+1) + bins2[i];
        d_bins12[idx] = (d_bins1[idx]-1)*(numBins+1) + d_bins2[idx];
    }
}
double CO_HistogramAMI_even_2_5(const double y[], const int size)
{
    // NaN check
    for(int i = 0; i < size; i++){if(isnan(y[i])){return NAN;}}
    int threadsPerBlock = 256, blocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    const int tau = 2, numBins = 5;

    double *y1 = (double*)malloc((size-tau) * sizeof(double));
    double *y2 = (double*)malloc((size-tau) * sizeof(double));

    double *d_y1, *d_y2;
    cudaMalloc(&d_y1, (size-tau) * sizeof(double));
    cudaMalloc(&d_y2, (size-tau) * sizeof(double));
    double *d_y; cudaMalloc(&d_y, size*sizeof(double));cudaMemcpy(d_y, y, size*sizeof(double), cudaMemcpyHostToDevice);
    compute_y1_y2<<<blocks, threadsPerBlock>>>(d_y1, d_y2, d_y, tau, (size-tau));
    cudaMemcpy(y1, d_y1, (size-tau)*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(y2, d_y2, (size-tau)*sizeof(double), cudaMemcpyDeviceToHost);
    // set bin edges
    const double maxValue = max_(y, size), minValue = min_(y, size);

    double binStep = (maxValue - minValue + 0.2)/5;
    //double binEdges[numBins+1] = {0};
	double *binEdges = (double*)malloc((numBins+1)*sizeof(double));
    double *d_binEdges;
    cudaMalloc(&d_binEdges, (numBins+1)*sizeof(double));
    compute_binEdges<<<blocks, threadsPerBlock>>>(minValue, binStep, numBins+1, d_binEdges);
    cudaMemcpy(binEdges, d_binEdges, (numBins+1)*sizeof(double), cudaMemcpyDeviceToHost);
    


    // count histogram bin contents
    int * bins1 = histbinassign(y1, size-tau, binEdges, numBins+1);
    int * bins2 = histbinassign(y2, size-tau, binEdges, numBins+1);

    // joint
    double * bins12 = (double*)malloc((size-tau) * sizeof(double));
	//double binEdges12[(numBins+1)*(numBins+1)] = {0};
    double *binEdges12 = (double*)malloc((numBins+1)*(numBins+1)*sizeof(double));

    double *d_bins12;
    cudaMalloc(&d_bins12, (size-tau)*sizeof(double)); 
    int *d_bins1, *d_bins2;
    cudaMalloc(&d_bins1, (size-tau)*sizeof(double));
    cudaMalloc(&d_bins2, (size-tau)*sizeof(double));
    cudaMemcpy(d_bins1, bins1, (size-tau)*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bins2, bins2, (size-tau)*sizeof(double), cudaMemcpyHostToDevice);
    compute_bins12<<<blocks, threadsPerBlock>>>(numBins, d_bins1, d_bins2, d_bins12, (size-tau));
    cudaMemcpy(bins12, d_bins12, (size-tau)*sizeof(double), cudaMemcpyDeviceToHost);

    double *d_binEdges12; cudaMalloc(&d_binEdges12, (numBins+1)*(numBins+1)*sizeof(double));
    compute_binEdges12<<<blocks, threadsPerBlock>>>(d_binEdges12, (numBins+1)*(numBins+1));
    //for(int i = 0; i < (numBins+1)*(numBins+1); i++){binEdges12[i] = i+1;}
    cudaMemcpy(binEdges12, d_binEdges12, (numBins+1)*(numBins+1)*sizeof(double), cudaMemcpyDeviceToHost);

    // fancy solution for joint histogram here
    int * jointHistLinear = histcount_edges(bins12, size-tau, binEdges12, (numBins + 1) * (numBins + 1));

    // transfer to 2D histogram (no last bin, as in original implementation)
    double pij[numBins][numBins];
    int sumBins = 0;
    for(int i = 0; i < numBins; i++){
        for(int j = 0; j < numBins; j++){
            pij[j][i] = jointHistLinear[i*(numBins+1)+j];
            sumBins += pij[j][i];
        }
    }

    // normalise
    for(int i = 0; i < numBins; i++){for(int j = 0; j < numBins; j++){pij[j][i] /= sumBins;}}

    // marginals
    //double pi[numBins] = {0};
	double pi[5] = {0};
    //double pj[numBins] = {0};
	double pj[5] = {0};
    for(int i = 0; i < numBins; i++){
        for(int j = 0; j < numBins; j++){
            pi[i] += pij[i][j];
            pj[j] += pij[i][j];
            // printf("pij[%i][%i]=%1.3f, pi[%i]=%1.3f, pj[%i]=%1.3f\n", i, j, pij[i][j], i, pi[i], j, pj[j]);
        }
    }

    // mutual information
    double ami = 0;
    for(int i = 0; i < numBins; i++){
        for(int j = 0; j < numBins; j++){
            if(pij[i][j] > 0){
                //printf("pij[%i][%i]=%1.3f, pi[%i]=%1.3f, pj[%i]=%1.3f, logarg=, %1.3f, log(...)=%1.3f\n",
                //       i, j, pij[i][j], i, pi[i], j, pj[j], pij[i][j]/(pi[i]*pj[j]), log(pij[i][j]/(pi[i]*pj[j])));
                ami += pij[i][j] * log(pij[i][j]/(pj[j]*pi[i]));
            }
        }
    }

    // free(bins1);
    // free(bins2);
    // free(jointHistLinear);

    // free(y1);
    // free(y2);
    // free(bins12);

    return ami;
}


__global__ void calc_words_len1(int *d_yt, int *d_r1, int *d_sizes_r1, int size, int alphabet_sizes){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i<alphabet_sizes){
        int r_idx = 0;
        for (int j = 0; j < size; j++) {
            if (d_yt[j] == i + 1) {
                d_r1[i * size + r_idx] = j;
                r_idx++;
            }
        }
        d_sizes_r1[i] = r_idx;
    }

}
__global__ void calc_words_len2(int alphabet_size, int *d_sizes_r2, int size, int *d_sizes_r1, int *d_yt, int *d_r1, int *d_r2, double *d_out2){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i<alphabet_size){
        for(int j=0;j<alphabet_size;j++){
            d_sizes_r2[i*alphabet_size + j] = 0;
            int dynamic_idx = 0;

            for(int k=0;k<d_sizes_r1[i];k++){
                int tmp_idx = d_yt[d_r1[i*size + k + 1]];

                if(tmp_idx == (j+1)){
                    d_r2[i*alphabet_size*d_sizes_r1[i] + j*d_sizes_r1[i] + dynamic_idx] = d_r1[i*size + k];
                    dynamic_idx++;

                    d_sizes_r2[i*alphabet_size + j]++;
                }
            }

            double tmp = (double)d_sizes_r2[i*alphabet_size+j] / ((double)(size)-(double)(1.0));
            d_out2[i*alphabet_size + j] = tmp;
        }
    }
}
double f_entropy(const double a[], const int size)
{
    double f = 0.0;
    for (int i = 0; i < size; i++) {
        if (a[i] > 0) {
            f += a[i] * log(a[i]);
        }
    }
    return -1 * f;
}
double SB_MotifThree_quantile_hh(const double y[], const int size){
    int threadsPerBlock1 = 256;
    int blocks1 = (size + threadsPerBlock1 - 1) / threadsPerBlock1;
    int *h_yt = (int*)malloc(size * sizeof(int)), *d_yt;
    cudaMalloc(&d_yt, size*sizeof(int));
    double hh = 0;// answer to be returned
    
    //coarsegrain
    sb_coarsegrain(y, size, "quantile", 3, h_yt);
    cudaMemcpy(d_yt, h_yt, size*sizeof(int), cudaMemcpyHostToDevice);
    //cudaMemcpy(&h_ySub, d_ySub, size*sizeof(double), cudaMemcpyDeviceToHost);

    int alphabet_size = 3, array_size = alphabet_size;
    //int tmp_idx, r_idx, dynamic_idx;
    //declaring the 2d arrays
    //dim3 threadsPerBlock2(16, 16);
    //dim3 blocks2((alphabet_size + threadsPerBlock2.x - 1) / threadsPerBlock2.x, (size + threadsPerBlock2.y - 1) / threadsPerBlock2.y);

    int *d_r1;
    int lengthd_r1 = array_size*size*sizeof(int);
    cudaMalloc(&d_r1, lengthd_r1);
    int *h_r1 = (int*)malloc(lengthd_r1);// for(int i=0;i<array_size;i++)h_r1[i] = (int*)malloc(size*sizeof(int));

    //declaring sizes_r1
    int *h_sizes_r1 = (int*)malloc(array_size*sizeof(int));
    int *d_sizes_r1; cudaMalloc(&d_sizes_r1, array_size*sizeof(int));

    calc_words_len1<<<blocks1, threadsPerBlock1>>>(d_yt, d_r1, d_sizes_r1, size, alphabet_size);
    
    cudaMemcpy(h_r1, d_r1, lengthd_r1, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_sizes_r1, d_sizes_r1, array_size*sizeof(int), cudaMemcpyDeviceToHost);
    // for(int i=0;i<lengthd_r1;i++){
    //     printf("%d\t", h_r1[i]);
    // }

    //array_size *= alphabet_size;
    for(int i=0;i<alphabet_size;i++){
        if(h_sizes_r1[i]!=0 && h_r1[i*size+(h_sizes_r1[i]-1)] == (size-1)){
            int *tmp = (int*)malloc(h_sizes_r1[i]*sizeof(int));
            for(int j=0;j<h_sizes_r1[i];j++){
                tmp[j] = h_r1[i*size + j]; 
            }

            for(int j=0;j<(h_sizes_r1[i]-1);j++){
                h_r1[i*size + j] = tmp[j];
            }

            h_sizes_r1[i]--;
        }
    }
    // int *nd_sizes_r1; cudaMalloc(&nd_sizes_r1, );
    cudaMemcpy(d_sizes_r1, h_sizes_r1, array_size*sizeof(int), cudaMemcpyHostToDevice);
    // for(int i=0;i<lengthd_r1;i++){
    //     printf("%d\t", h_r1[i]);
    // }


    //Declaring 3d matrices as 1d
    int lengthd_r2 = alphabet_size*alphabet_size*size*sizeof(int);
    int lengthd_sizes_r2 = alphabet_size*alphabet_size*sizeof(int);
    int lengthd_out2 = alphabet_size*alphabet_size*sizeof(double);

    int *h_r2 = (int*)malloc(lengthd_r2), *d_r2; cudaMalloc(&d_r2, lengthd_r2);
    int *h_sizes_r2 = (int*)malloc(lengthd_sizes_r2), *d_sizes_r2; cudaMalloc(&d_sizes_r2, lengthd_sizes_r2);
    double *h_out2 = (double*)malloc(lengthd_out2), *d_out2; cudaMalloc(&d_out2, lengthd_out2);

    calc_words_len2<<<blocks1, threadsPerBlock1>>>(alphabet_size, d_sizes_r2, size, d_sizes_r1, d_yt, d_r1, d_r2, d_out2);

    cudaMemcpy(h_out2, d_out2, lengthd_out2, cudaMemcpyDeviceToHost);
    // for(int i=0;i<lengthd_out2;i++){
    //     printf("%.2f\t", h_out2[i]);
    // }

    for(int i=0;i<alphabet_size;i++){
        for(int j=0;j<alphabet_size;j++){
            double ele = h_out2[i*alphabet_size + j];
            if(ele>0){
                hh+=2*(ele)*log(ele);
            }
        }
    }
    return (-1)*hh;

}
__global__ void yfiltInit(double *d_y, double *d_yfilt, int size){
    int i = threadIdx.x+blockDim.x *blockIdx.x;
    if(i<size){
        d_yfilt[i] = d_y[i];
    }
}
__global__ void ynDownCalc(int nDown, double *d_yfilt, double *d_yDown, int tau){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i<nDown){
        d_yDown[i] = d_yfilt[i*tau];
    }
}
__global__ void calcT(int *d_yCG, double *d_T, int nDown, int numGroups){
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    if(j<nDown-1){
        d_T[(d_yCG[j]-1)*numGroups + (d_yCG[j+1]-1)]++;
    }
}
__global__ void divNdown(int numGroups, int nDown, double *d_T){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i<numGroups){
        for(int j=0;j<numGroups;j++){
            d_T[i*numGroups + j]/=(nDown-1);
        }
    }
}
double SB_TransitionMatrix_3ac_sumdiagcov(const double y[], const int size){
	double *d_y; cudaMalloc(&d_y, size*sizeof(double));
    cudaMemcpy(d_y, y, size*sizeof(double), cudaMemcpyHostToDevice);
    int *d_nanFlag;
	int nanFlag = 0;
	// Allocate memory on the device
	cudaMalloc((void **)&d_nanFlag, sizeof(int));
	// Copy data to the device
	cudaMemset(d_nanFlag, 0, sizeof(int));
	// Define kernel execution configuration
	int threadsPerBlock = 256;
	int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
	//int sharedSize = threadsPerBlock * sizeof(double);

	// Launch kernels

	nanCheckKernel<<<blocksPerGrid, threadsPerBlock>>>(d_y, size, d_nanFlag);
	cudaMemcpy(&nanFlag, d_nanFlag, sizeof(int), cudaMemcpyDeviceToHost);
	if (nanFlag) {
		cudaFree(d_nanFlag);
		return NAN;  // Return NaN if any NaNs found in the input
	}

    int numGroups = 3;
    int tau = co_firstzero_cuda(y, size, size);
    //co_firstzero_cuda

    //yfilt
    double *h_yfilt = (double*)malloc(size*sizeof(double)), *d_yfilt; 
    cudaMalloc(&d_yfilt, size*sizeof(double));
    yfiltInit<<<blocksPerGrid, threadsPerBlock>>>(d_y, d_yfilt, size);
    cudaMemcpy(h_yfilt, d_yfilt, size*sizeof(double), cudaMemcpyDeviceToHost);
    // for(int i=0;i<size;i++){
    //     printf("%.2f", h_yfilt[i]);
    // }

    int nDown = (size-1)/tau+1;
    //
    //printf("%d\n", nDown);
    double *h_yDown = (double*)malloc(nDown * sizeof(double)), *d_yDown;
    cudaMalloc(&d_yDown, nDown*sizeof(double));
    ynDownCalc<<<blocksPerGrid, threadsPerBlock>>>(nDown, d_yfilt, d_yDown, tau);

    cudaMemcpy(h_yDown, d_yDown, nDown*sizeof(double), cudaMemcpyDeviceToHost);
    // for(int i=0;i<nDown;i++){
    //     printf("%.2f", h_yDown[i]);
    // }
    int *h_yCG = (int*)malloc(nDown * sizeof(int)), *d_yCG;
    cudaMalloc(&d_yCG, nDown * sizeof(int));
    sb_coarsegrain(h_yDown, nDown, "quantile", numGroups, h_yCG);

    cudaMemcpy(d_yCG, h_yCG, nDown * sizeof(int), cudaMemcpyHostToDevice);
    // for(int i=0;i<nDown;i++){
    //     printf("%d\t", h_yCG[i]);
    // }
    //t calculation
    //numGroups = 3;
    double *h_T = (double*)malloc((numGroups*numGroups)*sizeof(double)), *d_T;
    cudaMalloc(&d_T, (numGroups*numGroups)*sizeof(double));
    calcT<<<blocksPerGrid, threadsPerBlock>>>(d_yCG, d_T, nDown, numGroups);
    cudaMemcpy(h_T, d_T, (numGroups*numGroups)*sizeof(double), cudaMemcpyDeviceToHost);
    // for(int i=0;i<(numGroups*numGroups);i++){
    //     printf("%.2f\t", h_T[i]);
    // }
    divNdown<<<blocksPerGrid, threadsPerBlock>>>(numGroups, nDown, d_T);
    cudaMemcpy(h_T, d_T, (numGroups*numGroups)*sizeof(double), cudaMemcpyDeviceToHost);
    // for(int i=0;i<(numGroups*numGroups);i++){
    //     printf("%.9f\t", h_T[i]);
    // }
    double column1[3] = {0};
    double column2[3] = {0};
    double column3[3] = {0};
    
    for(int i = 0; i < numGroups; i++){
        column1[i] = h_T[i*numGroups + 0];
        column2[i] = h_T[i*numGroups + 1];
        column3[i] = h_T[i*numGroups + 2];
        //printf("%d\t%d\t%d\n", i*numGroups + 0, i*numGroups + 1, i*numGroups + 2);
    }
    // for(int i=0;i<numGroups;i++){
    //     printf("%f\t%f\t%f\n", column1[i], column2[i], column3[i]);
    // }
    double *columns[3];
    columns[0] = &(column1[0]);
    columns[1] = &(column2[0]);
    columns[2] = &(column3[0]);
    
    double COV[3][3];
    double covTemp = 0;
    for(int i = 0; i < numGroups; i++){
        for(int j = i; j < numGroups; j++){
            covTemp = cov(columns[i], columns[j], 3);
            COV[i][j] = covTemp;
            COV[j][i] = covTemp;
        }
    }
    
    double sumdiagcov = 0;
    for(int i = 0; i < numGroups; i++){
        sumdiagcov += COV[i][i];
    }
    // free(yFilt);free(yDown);free(yCG);
    return sumdiagcov;

}

int main() {
    //cudaEvent_t start, stop;
    //cudaEventCreate(&start);
    //cudaEventCreate(&stop);
    //cudaEventRecord(start);
    double durationUs = 0.0;
    auto start = std::chrono::high_resolution_clock::now();
    double *y = NULL;
    int size = 0;
    FILE *fp = fopen("/home/aru/Catch-22-Matrix-Profile/modified_data.txt", "rb");
    if (!fp) {
        fprintf(stderr, "Failed to open the file.\n");
        return 1;
    }

    double temp;
    while (fscanf(fp, "%lf", &temp) == 1) {
        size++;
    }

    fseek(fp, 0, SEEK_SET);
    y = (double *)malloc(size * sizeof(double));
    if (!y) {
        fprintf(stderr, "Failed to allocate memory.\n");
        fclose(fp);
        return 1;
    }

    for (int i = 0; i < size; i++) {
        if (fscanf(fp, "%lf", &y[i]) != 1) {
            fprintf(stderr, "Failed to read data from file.\n");
            free(y);
            fclose(fp);
            return 1;
        }
    }

    fclose(fp);
    double *autocorr_d = cuda_co_autocorrs(y, size);

    int firstMinIndex = CO_FirstMin_ac_cuda(y, size, autocorr_d); //result 22
    printf("CO_First_min: %d\n", firstMinIndex);
    float result_1 = CO_f1ecac_CUDA(autocorr_d, size);
    printf("CO_F1ecac : %f\n", result_1);
    float result_2 = CO_trev_1_num_cuda(y, size);
    printf("CO_trev_num1 : %.14f\n", result_2);
    double result_3 = CO_Embed2_Dist_tau_d_expfit_meandiff(y, size, autocorr_d);
    printf("CO_Embed2_Dist_tau_d_expfit_meandiff : %f\n", result_3);
    double result_21 = CO_HistogramAMI_even_2_5(y, size);
	printf("CO_HistogramAMI_even_2_5 %f\n", result_21);


    double result_5 = SC_FluctAnal_2_50_1_logi_prop_r1(y, size, 2, "dfa");
    printf("SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1 : %f\n", result_5);
    double result_6 = SC_FluctAnal_2_50_1_logi_prop_r1(y, size, 1, "rsrangefit");
    printf("SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1 : %f\n", result_6);
    double result_7 = IN_AutoMutualInfoStats_40_gaussian_fmmi(y, size);
    printf("IN_AutoMutualInfoStats_40_gaussian_fmmi : %f\n", result_7);
    double result_8 = SP_Summaries_welch_rect_cuda(y, size, "area_5_1");
    printf("Area under the first 1/5th of the spectrum: %f\n", result_8);
    double result_9 = SP_Summaries_welch_rect_cuda(y, size, "centroid");
    printf("Spectral Centroid: %f\n", result_9);
    double result_10 = SB_BinaryStats_diff_longstretch0_CUDA(y, size);
	printf("SB_BinaryStats_diff_longstretch0_CUDA %f\n", result_10);
	double result_11 = SB_BinaryStats_diff_longstretch1_CUDA(y, size);
	printf("SB_BinaryStats_diff_longstretch1_CUDA %f\n", result_11);
    double result_12 = MD_hrv_classic_pnn40(y, size); //No need to print as it is already done in the function. The result will be shown below by the virue of the function.
    //printf("MD_hrv_classic_pnn40 %f\n", result_12);
    double result_19 = FC_LocalSimple_mean_tauresrat_CUDA(y, size, 1);
	printf("FC_LocalSimple_mean_tauresrat %f\n", result_19);
	double result_20= FC_LocalSimple_mean_stderr_CUDA(y, size, 3);
	printf("FC_LocalSimple_mean_stderr_CUDA %f\n", result_20);
    double result_14 = SB_MotifThree_quantile_hh(y, size);
	printf("motifthree %f\n", result_14);


    double result_15 = hist5(y, size);
	printf("hist5 %f\n", result_15);
    double result_16 = hist10(y, size);
	printf("hist10 %f\n", result_16);

    double result_4 = SB_TransitionMatrix_3ac_sumdiagcov(y, size);
    printf("SB_TransitionMatrix_3ac_sumdiagcov : %.20f\n", result_4);

    double result_13 = periodicity_wang(y, size);
	printf("periodicity_wang %f\n", result_13);
    // double sign = -1.0; //n
    // double result_17 = DN_OutlierInclude_np_001_mdrmd_CUDA(y, size, sign);
    // printf("The OutlierInclude_np_001_mdrmd is %f\n", result_17);
    // sign = 1.0; //p
    // double result_18 = DN_OutlierInclude_np_001_mdrmd_CUDA(y, size, sign);
    // printf("The OutlierInclude_np_001_mdrmd is %f\n", result_17);

    auto end = std::chrono::high_resolution_clock::now();
    durationUs = (std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count() / 10000.0);
    printf("%lf\n", durationUs);
    //cudaEventRecord(stop);
    //cudaEventSynchronize(stop);
    //float milliseconds = 0;
    //cudaEventElapsedTime(&milliseconds, start, stop);
    //printf("Elapsed time: %f ms\n", milliseconds);

    //cudaEventDestroy(start);
    //cudaEventDestroy(stop);
    free(y);
    return 0;
}