// reduce_sum.cu
#include <cstdio>

__global__ void reduce_sum_kernel(const float* __restrict__ in,
                                  float* out,
                                  int n) {
    extern __shared__ float sdata[];   // 动态分配的共享内存

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    // 每个线程先在全局内存里累加自己负责的 2 个元素
    float sum = 0.0f;
    if (i < n) {
        sum += in[i];
    }
    if (i + blockDim.x < n) {
        sum += in[i + blockDim.x];
    }

    // 写入 shared memory
    sdata[tid] = sum;
    __syncthreads();

    // 共享内存内做归约（标准二分归约）
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // 每个 block 的线程 0 把结果写回到全局（atomicAdd 汇总）
    if (tid == 0) {
        atomicAdd(out, sdata[0]);
    }
}

// 一个简单的 host 封装，演示如何调用
float gpu_reduce_sum(const float* h_in, int n) {
    float *d_in = nullptr, *d_out = nullptr;
    float h_out = 0.0f;

    // 分配显存
    cudaMalloc(&d_in, n * sizeof(float));
    cudaMalloc(&d_out, sizeof(float));

    // 拷贝输入，并初始化输出为 0
    cudaMemcpy(d_in, h_in, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, sizeof(float));

    int blockSize = 256;
    // 这里选择2，是想让每个线程处理 2 个元素，充分利用全局内存带宽，
    // 在寄存器里先做一部分归约, 减少最后在shared memory里做归约的计算量。
    int gridSize = (n + blockSize * 2 - 1) / (blockSize * 2);
    size_t shmSize = blockSize * sizeof(float);

    reduce_sum_kernel<<<gridSize, blockSize, shmSize>>>(d_in, d_out, n);
    cudaDeviceSynchronize();

    // 把结果拷回 host
    cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);

    return h_out;
}

int main() {
    const int N = 1 << 20; // 1M
    float* h_in = new float[N];

    for (int i = 0; i < N; ++i) {
        h_in[i] = 1.0f;   // 随便填点数据，这里都设为 1
    }

    float sum = gpu_reduce_sum(h_in, N);
    printf("sum = %f\n", sum);

    delete[] h_in;
    return 0;
}
