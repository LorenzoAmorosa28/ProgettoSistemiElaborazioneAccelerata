#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

#define ALPHA 0.05f
#define K 2.5f
#define BLOCK_X 16
#define BLOCK_Y 16
#define RADIUS 2

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
        exit(1); \
    } \
}

__global__ void bgSubGaussian(
    const unsigned char* frame,
    float* mean,
    float* var,
    unsigned char* mask,
    int rows,
    int cols
) {
    int x = blockIdx.x * BLOCK_X + threadIdx.x;
    int y = blockIdx.y * BLOCK_Y + threadIdx.y;

    // Coordinate nella shared
    int sx = threadIdx.x + RADIUS;
    int sy = threadIdx.y + RADIUS;

    //Shared memory
    __shared__ unsigned char tile[BLOCK_Y + 2*RADIUS][BLOCK_X + 2*RADIUS];

    //pixel centrale
    if (x < cols && y < rows)
        tile[sy][sx] = frame[y * cols + x];
    else
        tile[sy][sx] = 0;

    //sinistra
    if (threadIdx.x < RADIUS) {
        int xx = max(x - RADIUS, 0);
        tile[sy][sx - RADIUS] = frame[y * cols + xx];
    }

    //destra
    if (threadIdx.x >= BLOCK_X - RADIUS) {
        int xx = min(x + RADIUS, cols - 1);
        tile[sy][sx + RADIUS] = frame[y * cols + xx];
    }

    //sopra
    if (threadIdx.y < RADIUS) {
        int yy = max(y - RADIUS, 0);
        tile[sy - RADIUS][sx] = frame[yy * cols + x];
    }

    //sotto
    if (threadIdx.y >= BLOCK_Y - RADIUS) {
        int yy = min(y + RADIUS, rows - 1);
        tile[sy + RADIUS][sx] = frame[yy * cols + x];
    }

    __syncthreads(); 

    if (x >= cols || y >= rows) return;

    const int kernel[5][5] = {
        {1, 4, 6, 4, 1},
        {4,16,24,16,4},
        {6,24,36,24,6},
        {4,16,24,16,4},
        {1, 4, 6, 4, 1}
    };

    float sum = 0.0f;

    for (int dy = -RADIUS; dy <= RADIUS; dy++) {
        for (int dx = -RADIUS; dx <= RADIUS; dx++) {
            sum += kernel[dy + RADIUS][dx + RADIUS] *
                   tile[sy + dy][sx + dx];
        }
    }

    int idx = y * cols + x;

    float pixel = sum / 256.0f;
    float diff  = pixel - mean[idx];
    float stddev = sqrtf(var[idx] + 1e-6f);

    mask[idx] = (fabsf(diff) > K * stddev) ? 255 : 0;

    mean[idx] = (1.0f - ALPHA) * mean[idx] + ALPHA * pixel;
    var[idx]  = (1.0f - ALPHA) * var[idx]  + ALPHA * diff * diff;
}


int main(int argc, char** argv) {
    if (argc < 2) {
        cout << "Uso: ./bg_cuda video.mp4\n";
        return -1;
    }

    VideoCapture cap(argv[1]);
    if (!cap.isOpened()) {
        cout << "Errore apertura video\n";
        return -1;
    }

    Mat frame, gray;
    cap >> frame;
    cvtColor(frame, gray, COLOR_BGR2GRAY);

    int rows = gray.rows;
    int cols = gray.cols;
    int size = rows * cols;

    size_t bytesU8 = size * sizeof(unsigned char);
    size_t bytesF  = size * sizeof(float);

    // Host
    Mat mask(rows, cols, CV_8UC1);
    vector<float> h_mean(size), h_var(size, 15.0f);

    for (int i = 0; i < size; i++)
        h_mean[i] = gray.data[i];

    // Device
    unsigned char *d_frame, *d_mask;
    float *d_mean, *d_var;

    cudaMalloc(&d_frame, bytesU8);
    cudaMalloc(&d_mask,  bytesU8);
    cudaMalloc(&d_mean,  bytesF);
    cudaMalloc(&d_var,   bytesF);

    cudaMemcpy(d_mean, h_mean.data(), bytesF, cudaMemcpyHostToDevice);
    cudaMemcpy(d_var,  h_var.data(),  bytesF, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid(
        (cols + block.x - 1) / block.x,
        (rows + block.y - 1) / block.y
    );


    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        cvtColor(frame, gray, COLOR_BGR2GRAY);
        cudaMemcpy(d_frame, gray.data, bytesU8, cudaMemcpyHostToDevice);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);

        bgSubGaussian<<<grid, block>>>(
            d_frame, d_mean, d_var, d_mask, rows, cols
        );

        cudaEventRecord(stop);
        CHECK_CUDA(cudaEventSynchronize(stop));

        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        cout << "Tempo kernel CUDA: " << ms << " ms\n";

        cudaMemcpy(mask.data, d_mask, bytesU8, cudaMemcpyDeviceToHost);

        //imshow("Mask CUDA", mask);
        //if (waitKey(1) >= 0) break;
    }

    cudaFree(d_frame);
    cudaFree(d_mask);
    cudaFree(d_mean);
    cudaFree(d_var);

    return 0;
}
