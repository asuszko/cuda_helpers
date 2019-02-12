#include <cuda.h>
#include "cu_errchk.h"
#include "cu_transpose.h"

#define TILE_DIM 32
#define BLOCK_ROWS 8


template <typename T>
__global__ void transpose_real(T *data, int ncols, int nrows)
{
    int index = blockIdx.x * TILE_DIM + threadIdx.x;

    if(index < ncols*nrows) {
        int iy = index/ncols;
        int ix = index-iy*ncols;
        int k = ix+iy*ncols;
        int idx = k;
        do { // calculate index in the original array
            idx = (idx % nrows) * ncols + (idx / nrows);
        } while(idx < k); // make sure we don't swap elements twice
        data[k] = data[idx];
    }
}


template <typename T>
__global__ void ccc_transpose_real(T *data)
{
    __shared__ T tile_s[TILE_DIM][TILE_DIM+1];
    __shared__ T tile_d[TILE_DIM][TILE_DIM+1];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = gridDim.x * TILE_DIM;

    if (blockIdx.y > blockIdx.x) { // handle off-diagonal case
        int dx = blockIdx.y * TILE_DIM + threadIdx.x;
        int dy = blockIdx.x * TILE_DIM + threadIdx.y;

        for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
            tile_s[threadIdx.y+j][threadIdx.x] = data[(y+j)*width + x];
        }
        for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
            tile_d[threadIdx.y+j][threadIdx.x] = data[(dy+j)*width + dx];
        }
        __syncthreads();
        for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
            data[(dy+j)*width + dx] = tile_s[threadIdx.x][threadIdx.y + j];
        }
        for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
            data[(y+j)*width + x] = tile_d[threadIdx.x][threadIdx.y + j];
        }
    }

    else if (blockIdx.y==blockIdx.x) { // handle on-diagonal case
        for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
            tile_s[threadIdx.y+j][threadIdx.x] = data[(y+j)*width + x];
        }
        __syncthreads();
        for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
            data[(y+j)*width + x] = tile_s[threadIdx.x][threadIdx.y + j];
        }
    }
}


template <typename T>
__global__ void transpose_complex(T *data)
{
    __shared__ T tile_s[TILE_DIM][TILE_DIM+1];
    __shared__ T tile_d[TILE_DIM][TILE_DIM+1];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = gridDim.x * TILE_DIM;

    if (blockIdx.y > blockIdx.x) { // handle off-diagonal case
        int dx = blockIdx.y * TILE_DIM + threadIdx.x;
        int dy = blockIdx.x * TILE_DIM + threadIdx.y;

        for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
            tile_s[threadIdx.y+j][threadIdx.x].x = data[(y+j)*width + x].x;
            tile_s[threadIdx.y+j][threadIdx.x].y = data[(y+j)*width + x].y;
        }
        for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
            tile_d[threadIdx.y+j][threadIdx.x].x = data[(dy+j)*width + dx].x;
            tile_d[threadIdx.y+j][threadIdx.x].y = data[(dy+j)*width + dx].y;
        }
        __syncthreads();
        for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
            data[(dy+j)*width + dx].x = tile_s[threadIdx.x][threadIdx.y + j].x;
            data[(dy+j)*width + dx].y = tile_s[threadIdx.x][threadIdx.y + j].y;
        }
        for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
            data[(y+j)*width + x].x = tile_d[threadIdx.x][threadIdx.y + j].x;
            data[(y+j)*width + x].y = tile_d[threadIdx.x][threadIdx.y + j].y;
        }
    }

    else if (blockIdx.y==blockIdx.x) { // handle on-diagonal case
        for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
            tile_s[threadIdx.y+j][threadIdx.x].x = data[(y+j)*width + x].x;
            tile_s[threadIdx.y+j][threadIdx.x].y = data[(y+j)*width + x].y;
        }
        __syncthreads();
        for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
            data[(y+j)*width + x].x = tile_s[threadIdx.x][threadIdx.y + j].x;
            data[(y+j)*width + x].y = tile_s[threadIdx.x][threadIdx.y + j].y;
        }
    }
}



void cu_transpose(void *dev_ptr, int nrows, int ncols, int dtype_id,
                  cudaStream_t *stream)
{
    dim3 blockSize(TILE_DIM, TILE_DIM);
    dim3 gridSize((ncols-1)/TILE_DIM+1, (nrows-1)/TILE_DIM+1);

    cudaStream_t stream_id;
    (stream == NULL) ? stream_id = NULL : stream_id = *stream;

    switch(dtype_id) {
        case 0: {
            transpose_real<<<gridSize,blockSize,0,stream_id>>>((float *)dev_ptr, ncols, nrows);
            break;
        }
        case 1: {
            transpose_real<<<gridSize,blockSize,0,stream_id>>>((double *)dev_ptr, ncols, nrows);
            break;
        }
        case 2: {
            transpose_complex<<<gridSize,blockSize,0,stream_id>>>((float2 *)dev_ptr);
            break;
        }
        case 3: {
            transpose_complex<<<gridSize,blockSize,0,stream_id>>>((double2 *)dev_ptr);
            break;
        }
    }
    return;
}
