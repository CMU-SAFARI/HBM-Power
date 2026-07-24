// hbm_bw.cu (v3) — HBM3e memory-bandwidth stress test / benchmark for NVIDIA H200 (Hopper, sm_90)
//
// Build:
//     nvcc -O3 -arch=sm_90 -o hbm_bw hbm_bw.cu
//
// Run:
//     ./hbm_bw [gib] [iters] [stressSec] [stressKernel] [fill] [altRunWords] [constHex]
//       gib          GiB per buffer            (default 4)
//       iters        timed iterations          (default 50)
//       stressSec    sustained-stress seconds  (default 0 = off)
//       stressKernel read | write | copy | triad   (default read)
//                      read =1R:0W, write =0R:1W, copy =1R:1W, triad =2R:1W
//                      (sweeping the R:W ratio lets you regress out E_read vs E_write)
//       fill         rand | const | alt        (default rand)
//       altRunWords  run length for 'alt', in 32-bit words (default 1)
//       constHex     value for 'const' fill, hex ok (default 0x00000000)
//
//   Examples:
//     ./hbm_bw 8 200 120 read rand                  # random data (your current run)
//     ./hbm_bw 8 200 120 read const 1 0x00000000    # all-0s: low static level, no toggle
//     ./hbm_bw 8 200 120 read const 1 0xFFFFFFFF    # all-1s: high static level, no toggle
//                                                   #   00 vs FF -> static/DBI, NOT toggling
//     ./hbm_bw 8 200 120 read alt 1                 # alternate 0x0/0xFFFFFFFF every word
//     ./hbm_bw 8 200 120 read alt 8                 # alternate in runs of 8 words (sweep this)
//     ./hbm_bw 8 200 120 write alt 1                # pure-write toggle ceiling (write energy)
//     ./hbm_bw 8 200 120 copy rand                  # 1R:1W mix
//
// Why the fill modes:
//   Memory power = DRAM access energy (scales with bandwidth) + active PHY/controller
//   power (partly data-dependent: DQ-line toggling) + refresh/bias floor. Holding
//   bandwidth ~constant and varying ONLY the data pattern isolates the data-dependent
//   PHY term:
//     const (all-zeros) -> minimal toggling  -> PHY active baseline (low end)
//     rand              -> ~50% lines flip    -> mid
//     alt  0x0/0xFF...  -> aims for max flips  -> toggle ceiling (high end)
//   Sweep altRunWords (1,2,4,8,16,...) because the logical pattern may not map 1:1 to
//   physical DQ beats after address interleaving; the run length that MAXIMIZES measured
//   memory power is the empirical worst case. If 'alt' barely beats 'rand', the interface
//   is likely scrambling / using data-bus inversion, and 'rand' is already near the max.

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                          \
    do {                                                                         \
        cudaError_t err__ = (call);                                              \
        if (err__ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,        \
                    cudaGetErrorString(err__));                                  \
            exit(EXIT_FAILURE);                                                  \
        }                                                                        \
    } while (0)

enum FillMode { FILL_RAND = 0, FILL_CONST = 1, FILL_ALT = 2 };

__device__ __forceinline__ unsigned int hash_u32(unsigned int x) {
    x ^= x >> 16; x *= 0x7feb352dU;
    x ^= x >> 15; x *= 0x846ca68bU;
    x ^= x >> 16; return x;
}

// One fill kernel, three patterns. p viewed as uint4 (4 x 32-bit words per element).
__global__ void patternFillKernel(uint4* __restrict__ p, size_t n4,
                                  int mode, size_t runWords, unsigned int param) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
         i < n4; i += (size_t)gridDim.x * blockDim.x) {
        if (mode == FILL_RAND) {
            unsigned int b = (unsigned int)(i * 4u) ^ param;
            p[i] = make_uint4(hash_u32(b), hash_u32(b + 1u), hash_u32(b + 2u), hash_u32(b + 3u));
        } else if (mode == FILL_CONST) {
            p[i] = make_uint4(param, param, param, param);
        } else { // FILL_ALT: word w is 0xFFFFFFFF or 0x0, flipping every `runWords` words
            size_t w = i * 4;
            unsigned int m0 = (((w + 0) / runWords) & 1) ? 0xFFFFFFFFu : 0x0u;
            unsigned int m1 = (((w + 1) / runWords) & 1) ? 0xFFFFFFFFu : 0x0u;
            unsigned int m2 = (((w + 2) / runWords) & 1) ? 0xFFFFFFFFu : 0x0u;
            unsigned int m3 = (((w + 3) / runWords) & 1) ? 0xFFFFFFFFu : 0x0u;
            p[i] = make_uint4(m0, m1, m2, m3);
        }
    }
}

__global__ void readKernelU(const uint4* __restrict__ a, size_t n4, unsigned int* __restrict__ out) {
    uint4 acc = make_uint4(0u, 0u, 0u, 0u);
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
         i < n4; i += (size_t)gridDim.x * blockDim.x) {
        uint4 v = a[i];
        acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w;
    }
    if (acc.x == 0xDEADBEEFu && acc.y == 0xCAFEBABEu && acc.z == 0x0u && acc.w == 0x1u)
        out[threadIdx.x] = acc.x ^ acc.y ^ acc.z ^ acc.w;   // never taken
}

__global__ void copyKernel(float4* __restrict__ c, const float4* __restrict__ a, size_t n4) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
         i < n4; i += (size_t)gridDim.x * blockDim.x) c[i] = a[i];
}
__global__ void scaleKernel(float4* __restrict__ b, const float4* __restrict__ c, float s, size_t n4) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
         i < n4; i += (size_t)gridDim.x * blockDim.x) {
        float4 v = c[i]; v.x *= s; v.y *= s; v.z *= s; v.w *= s; b[i] = v;
    }
}
__global__ void addKernel(float4* __restrict__ c, const float4* __restrict__ a,
                          const float4* __restrict__ b, size_t n4) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
         i < n4; i += (size_t)gridDim.x * blockDim.x) {
        float4 va = a[i], vb = b[i];
        c[i] = make_float4(va.x + vb.x, va.y + vb.y, va.z + vb.z, va.w + vb.w);
    }
}
__global__ void triadKernel(float4* __restrict__ a, const float4* __restrict__ b,
                            const float4* __restrict__ c, float s, size_t n4) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
         i < n4; i += (size_t)gridDim.x * blockDim.x) {
        float4 vb = b[i], vc = c[i];
        a[i] = make_float4(vb.x + s * vc.x, vb.y + s * vc.y, vb.z + s * vc.z, vb.w + s * vc.w);
    }
}

template <typename LaunchFn>
static double bestTimeMs(LaunchFn launch, int iters) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop));
    launch(); CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    double best = 1e30;
    for (int it = 0; it < iters; ++it) {
        CUDA_CHECK(cudaEventRecord(start));
        launch();
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.f; CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        if (ms < best) best = ms;
    }
    CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop));
    return best;
}
static void report(const char* name, double bytesMoved, double ms) {
    printf("  %-7s %10.3f ms   %10.1f GB/s\n", name, ms, bytesMoved / (ms * 1e-3) / 1e9);
}

int main(int argc, char** argv) {
    double gibPerBuf = (argc > 1) ? atof(argv[1]) : 4.0;
    int    iters     = (argc > 2) ? atoi(argv[2]) : 50;
    int    stressSec = (argc > 3) ? atoi(argv[3]) : 0;
    const char* sk   = (argc > 4) ? argv[4] : "read";
    const char* fs   = (argc > 5) ? argv[5] : "rand";
    size_t altRun    = (argc > 6) ? (size_t)strtoull(argv[6], nullptr, 10) : 1;
    // const fill value (arg 7, hex ok). const has ZERO inter-beat DQ toggle for ANY
    // value, so 0x00000000 vs 0xFFFFFFFF isolates the STATIC level + data-bus-inversion
    // (DBI)/termination asymmetry from the toggling term. rand uses per-buffer seeds.
    unsigned int constParam = (argc > 7) ? (unsigned int)strtoul(argv[7], nullptr, 0) : 0x00000000u;
    if (iters < 1) iters = 1;
    if (altRun < 1) altRun = 1;
    // stress kernel: read (1R) | write (1W) | copy (1R+1W) | triad (2R+1W)
    enum { K_READ = 0, K_TRIAD = 1, K_WRITE = 2, K_COPY = 3 };
    int stressKern = K_READ;
    if      (!strcmp(sk, "triad")) stressKern = K_TRIAD;
    else if (!strcmp(sk, "write")) stressKern = K_WRITE;
    else if (!strcmp(sk, "copy"))  stressKern = K_COPY;
    int fillMode = FILL_RAND;
    if (!strcmp(fs, "const")) fillMode = FILL_CONST;
    else if (!strcmp(fs, "alt")) fillMode = FILL_ALT;

    int dev = 0;
    CUDA_CHECK(cudaSetDevice(dev));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    // memoryClockRate/memoryBusWidth were removed from cudaDeviceProp in CUDA 13;
    // query via attributes (stable across versions). theoGBs is informational only.
    int memClockKHz = 0, busWidthBits = 0;
    cudaDeviceGetAttribute(&memClockKHz, cudaDevAttrMemoryClockRate, dev);
    cudaDeviceGetAttribute(&busWidthBits, cudaDevAttrGlobalMemoryBusWidth, dev);
    double theoGBs = 2.0 * (double)memClockKHz * 1e3 * (busWidthBits / 8.0) / 1e9;

    printf("Device         : %s\n", prop.name);
    printf("Total memory   : %.1f GiB   L2: %.1f MB   SMs: %d\n",
           prop.totalGlobalMem/(1024.0*1024.0*1024.0), prop.l2CacheSize/(1024.0*1024.0),
           prop.multiProcessorCount);
    printf("Peak (props)   : %.0f GB/s  (HBM3e real peak higher; H200 spec ~4800 GB/s)\n",
           theoGBs);
    printf("NOTE: GB/s below are DECIMAL (bytes/1e9). 4500 GB/s = 4.5 TB/s ~= 4.09 TiB/s.\n\n");

    size_t bufBytes = ((size_t)(gibPerBuf * 1024.0*1024.0*1024.0)) & ~((size_t)15);
    size_t freeB = 0, totalB = 0; CUDA_CHECK(cudaMemGetInfo(&freeB, &totalB));
    size_t maxPerBuf = ((size_t)((double)freeB * 0.9 / 3.0)) & ~((size_t)15);
    if (bufBytes > maxPerBuf) { bufBytes = maxPerBuf;
        printf("Clamped buffer to %.2f GiB (headroom).\n", bufBytes/(1024.0*1024.0*1024.0)); }
    size_t n4 = bufBytes / sizeof(float4);
    double movedPerBuf = (double)n4 * 16.0;

    char fillName[64];
    if      (fillMode==FILL_RAND)  snprintf(fillName, sizeof fillName, "rand");
    else if (fillMode==FILL_CONST) snprintf(fillName, sizeof fillName, "const(0x%08X)", constParam);
    else                           snprintf(fillName, sizeof fillName, "alt(0x0/0xFFFFFFFF)");
    const char* kernName = (stressKern==K_READ)?"read":(stressKern==K_TRIAD)?"triad":
                           (stressKern==K_WRITE)?"write":"copy";
    printf("Buffer         : %.2f GiB each x3   iters=%d\n",
           movedPerBuf/(1024.0*1024.0*1024.0), iters);
    printf("Fill           : %s\n", fillName);
    if (fillMode==FILL_ALT) printf("Alt run length : %zu word(s)\n", altRun);
    printf("Stress kernel  : %s\n\n", kernName);

    float4 *a=nullptr,*b=nullptr,*c=nullptr; unsigned int* out=nullptr;
    CUDA_CHECK(cudaMalloc(&a, bufBytes)); CUDA_CHECK(cudaMalloc(&b, bufBytes));
    CUDA_CHECK(cudaMalloc(&c, bufBytes)); CUDA_CHECK(cudaMalloc(&out, 1024*sizeof(unsigned int)));

    int threads = 256, blocks = prop.multiProcessorCount * 32;
    uint4 *au_ = reinterpret_cast<uint4*>(a), *bu_ = reinterpret_cast<uint4*>(b), *cu_ = reinterpret_cast<uint4*>(c);
    unsigned int seedA = (fillMode==FILL_RAND)?0x12345678u:constParam;
    unsigned int seedB = (fillMode==FILL_RAND)?0x9e3779b9u:constParam;
    unsigned int seedC = (fillMode==FILL_RAND)?0xa5a5a5a5u:constParam;
    patternFillKernel<<<blocks,threads>>>(au_, n4, fillMode, altRun, seedA);
    patternFillKernel<<<blocks,threads>>>(bu_, n4, fillMode, altRun, seedB);
    patternFillKernel<<<blocks,threads>>>(cu_, n4, fillMode, altRun, seedC);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());

    const float scalar = 3.0f;
    const uint4* au = reinterpret_cast<const uint4*>(a);

    printf("Benchmark (best of %d):\n", iters);
    double t;
    t = bestTimeMs([&]{ readKernelU<<<blocks,threads>>>(au, n4, out);          }, iters); report("Read",  1.0*movedPerBuf, t);
    t = bestTimeMs([&]{ patternFillKernel<<<blocks,threads>>>(au_, n4, fillMode, altRun, seedA); }, iters); report("Write", 1.0*movedPerBuf, t);
    t = bestTimeMs([&]{ copyKernel <<<blocks,threads>>>(c, a, n4);             }, iters); report("Copy",  2.0*movedPerBuf, t);
    t = bestTimeMs([&]{ scaleKernel<<<blocks,threads>>>(b, c, scalar, n4);     }, iters); report("Scale", 2.0*movedPerBuf, t);
    t = bestTimeMs([&]{ addKernel  <<<blocks,threads>>>(c, a, b, n4);          }, iters); report("Add",   3.0*movedPerBuf, t);
    t = bestTimeMs([&]{ triadKernel<<<blocks,threads>>>(a, b, c, scalar, n4);  }, iters); report("Triad", 3.0*movedPerBuf, t);

    if (stressSec > 0) {
        // STREAM kernels overwrote the buffers; restore the chosen pattern in all three
        // so read/write operands carry the intended data pattern every iteration.
        unsigned int pa = (fillMode==FILL_RAND)?0xdeadbeefu:constParam;
        unsigned int pb = (fillMode==FILL_RAND)?seedB:constParam;
        unsigned int pc = (fillMode==FILL_RAND)?seedC:constParam;
        patternFillKernel<<<blocks,threads>>>(au_, n4, fillMode, altRun, pa);
        patternFillKernel<<<blocks,threads>>>(bu_, n4, fillMode, altRun, pb);
        patternFillKernel<<<blocks,threads>>>(cu_, n4, fillMode, altRun, pc);
        CUDA_CHECK(cudaDeviceSynchronize());
        // bytes moved per launch: read 1x, write 1x, copy 2x (1R+1W), triad 3x (2R+1W)
        double mult = (stressKern==K_TRIAD)?3.0:(stressKern==K_COPY)?2.0:1.0;
        printf("\nSustained %s stress (%s) for %d s  (watch: log_power.py)\n",
               kernName, fillName, stressSec);
        using clk = std::chrono::steady_clock;
        auto t0 = clk::now(); auto last = t0; unsigned long long n = 0;
        const int batch = 50; const double bpl = mult*movedPerBuf;
        while (std::chrono::duration<double>(clk::now()-t0).count() < stressSec) {
            for (int i=0;i<batch;++i){
                switch (stressKern) {
                    case K_READ:  readKernelU<<<blocks,threads>>>(au, n4, out);                          break;
                    case K_TRIAD: triadKernel<<<blocks,threads>>>(a, b, c, scalar, n4);                  break;
                    case K_WRITE: patternFillKernel<<<blocks,threads>>>(au_, n4, fillMode, altRun, pa);  break;
                    case K_COPY:  copyKernel<<<blocks,threads>>>(c, a, n4);                              break;
                }
                ++n;
            }
            CUDA_CHECK(cudaDeviceSynchronize());
            auto now = clk::now();
            if (std::chrono::duration<double>(now-last).count() >= 1.0) {
                double s = std::chrono::duration<double>(now-t0).count();
                printf("  t=%5.1fs   launches=%-8llu   avg %.1f GB/s\n", s, n, n*bpl/s/1e9);
                last = now;
            }
        }
        CUDA_CHECK(cudaGetLastError()); printf("Done.\n");
    }
    CUDA_CHECK(cudaFree(a)); CUDA_CHECK(cudaFree(b)); CUDA_CHECK(cudaFree(c)); CUDA_CHECK(cudaFree(out));
    return 0;
}
