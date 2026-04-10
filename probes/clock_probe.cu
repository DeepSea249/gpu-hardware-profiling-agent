/**
 * GPU Clock Frequency & SM Count Probe
 *
 * Measures the actual GPU SM clock frequency under sustained load 
 * using clock64() cycle counter and CUDA event wall-clock timing.
 * Also detects the number of active SMs using inline PTX.
 *
 * This is critical for anti-hacking resilience: the GPU may be
 * frequency-locked at a non-standard rate, making API-reported
 * values incorrect.
 *
 * Output format:
 *   CLOCK_MHZ=<f>
 *   NUM_ACTIVE_SMS=<n>
 *   CLOCK_CYCLES=<n> ELAPSED_MS=<f>
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

/**
 * Sustained FMA compute kernel.
 * Many dependent FMA operations to keep the SM busy and at boost clock.
 * Returns both the cycle count (via clock64) and a dummy result.
 */
__global__ void sustained_compute_kernel(float* output,
                                          long long* cycles_out,
                                          int iterations) {
    float x = 1.0f;
    float a = 1.0000001f;
    float b = 0.0000001f;

    // Warmup to ramp up to boost clock
    for (int i = 0; i < 100000; i++) {
        x = fmaf(x, a, b);
    }

    // Timed measurement
    long long start = clock64();
    #pragma unroll 1
    for (int i = 0; i < iterations; i++) {
        x = fmaf(x, a, b);
    }
    long long end = clock64();

    output[threadIdx.x] = x;  // Prevent optimization
    if (threadIdx.x == 0) {
        *cycles_out = end - start;
    }
}

/**
 * Detect active SMs using inline PTX to read the %smid register.
 * Launches many blocks; each block reports its SM ID.
 */
__global__ void sm_detect_kernel(int* sm_ids, int* counter) {
    if (threadIdx.x == 0) {
        int smid;
        asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
        int idx = atomicAdd(counter, 1);
        sm_ids[idx] = smid;
    }
}

int detect_active_sms() {
    int max_blocks = 4096;
    int* d_sm_ids;
    int* d_counter;
    CUDA_CHECK(cudaMalloc(&d_sm_ids, max_blocks * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_counter, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_counter, 0, sizeof(int)));

    // Launch many blocks with minimal threads
    sm_detect_kernel<<<max_blocks, 32>>>(d_sm_ids, d_counter);
    CUDA_CHECK(cudaDeviceSynchronize());

    int count;
    CUDA_CHECK(cudaMemcpy(&count, d_counter, sizeof(int), cudaMemcpyDeviceToHost));
    if (count > max_blocks) count = max_blocks;

    int* h_sm_ids = (int*)malloc(count * sizeof(int));
    CUDA_CHECK(cudaMemcpy(h_sm_ids, d_sm_ids, count * sizeof(int), cudaMemcpyDeviceToHost));

    // Count unique SM IDs
    int seen[512] = {0};
    int unique = 0;
    int max_smid = 0;
    for (int i = 0; i < count; i++) {
        int id = h_sm_ids[i];
        if (id >= 0 && id < 512 && !seen[id]) {
            seen[id] = 1;
            unique++;
            if (id > max_smid) max_smid = id;
        }
    }

    printf("SM_ID_MAX=%d SM_ID_UNIQUE=%d\n", max_smid, unique);

    free(h_sm_ids);
    CUDA_CHECK(cudaFree(d_sm_ids));
    CUDA_CHECK(cudaFree(d_counter));

    return unique;
}

int main(int argc, char** argv) {
    int num_trials = 5;
    int iterations = 50000000;  // 50M FMA ops

    if (argc > 1) iterations = atoi(argv[1]);
    if (argc > 2) num_trials = atoi(argv[2]);

    printf("CLOCK_PROBE_START\n");

    // ==================== SM Count Detection ====================
    int num_sms = detect_active_sms();
    printf("NUM_ACTIVE_SMS=%d\n", num_sms);

    // ==================== Clock Frequency Measurement ====================
    float* d_output;
    long long* d_cycles;
    CUDA_CHECK(cudaMalloc(&d_output, 32 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cycles, sizeof(long long)));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warmup to reach stable boost clock
    sustained_compute_kernel<<<1, 1>>>(d_output, d_cycles, iterations);
    CUDA_CHECK(cudaDeviceSynchronize());

    double clock_measurements[32];

    for (int trial = 0; trial < num_trials; trial++) {
        CUDA_CHECK(cudaEventRecord(start));
        sustained_compute_kernel<<<1, 1>>>(d_output, d_cycles, iterations);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

        long long cycles;
        CUDA_CHECK(cudaMemcpy(&cycles, d_cycles, sizeof(long long), cudaMemcpyDeviceToHost));

        // clock_freq = cycles / time_in_seconds
        // clock_mhz = cycles / (ms * 1000)  [since 1 MHz = 1e6 Hz]
        double clock_mhz = (double)cycles / ((double)ms * 1000.0);
        clock_measurements[trial] = clock_mhz;

        printf("TRIAL=%d CYCLES=%lld ELAPSED_MS=%.4f CLOCK_MHZ=%.2f\n",
               trial, cycles, ms, clock_mhz);
    }

    // Compute median clock frequency
    for (int i = 0; i < num_trials - 1; i++) {
        for (int j = i + 1; j < num_trials; j++) {
            if (clock_measurements[j] < clock_measurements[i]) {
                double tmp = clock_measurements[i];
                clock_measurements[i] = clock_measurements[j];
                clock_measurements[j] = tmp;
            }
        }
    }
    double median_clock = clock_measurements[num_trials / 2];
    printf("CLOCK_MHZ=%.2f\n", median_clock);

    // Also try to read reported clock from device properties
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("REPORTED_CLOCK_KHZ=%d\n", prop.clockRate);
    printf("REPORTED_MEM_CLOCK_KHZ=%d\n", prop.memoryClockRate);
    printf("REPORTED_SM_COUNT=%d\n", prop.multiProcessorCount);
    printf("REPORTED_DEVICE_NAME=%s\n", prop.name);
    printf("REPORTED_COMPUTE_CAP=%d.%d\n", prop.major, prop.minor);

    // Compare measured vs reported
    double reported_mhz = (double)prop.clockRate / 1000.0;
    double deviation_pct = 100.0 * (median_clock - reported_mhz) / reported_mhz;
    printf("CLOCK_DEVIATION_PCT=%.2f\n", deviation_pct);

    if (num_sms != prop.multiProcessorCount) {
        printf("ANOMALY=SM_MASKING measured=%d reported=%d\n", num_sms, prop.multiProcessorCount);
    }
    if (deviation_pct < -5.0 || deviation_pct > 5.0) {
        printf("ANOMALY=FREQ_LOCKING measured=%.0f reported=%.0f\n", median_clock, reported_mhz);
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_cycles));

    printf("CLOCK_PROBE_END\n");
    return 0;
}
