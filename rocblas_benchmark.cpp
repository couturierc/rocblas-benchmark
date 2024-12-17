#include "tensor.h"
#include <chrono>
#include <cstdint>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand.h>
#include <iomanip>
#include <iostream>
#include <memory>
#include <rocblas/rocblas.h>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <vector>

#define CHECK_ROCBLAS_ERROR(error)                                             \
  if (error != rocblas_status_success) {                                       \
    std::stringstream ss;                                                      \
    ss << "rocBLAS error " << error << " at line " << __LINE__ << std::endl;   \
    throw std::runtime_error(ss.str());                                        \
  }

struct ProblemConfig {
  int m, n, k;
  bool a_t, b_t, enable_tune;
  int batch_count;
  int64_t stride_a, stride_b, stride_c, stride_d;
  std::string a_type, b_type, c_type, d_type, compute_type;
  std::string name; // Add a name field to identify the configuration
};

std::vector<ProblemConfig> gemm_configs;
std::vector<ProblemConfig> gemv_configs;

// GEMM problem sizes
std::vector<int> gemm_sizes = {1024, 2048, 4096, 8192};

// GEMV problem sizes
std::vector<int> gemv_sizes = {1024, 2048, 4096, 8192, 16384, 32768};

void initialize_problem_configs() {
  // GEMM configurations
  for (const auto &size : gemm_sizes) {
    int64_t stride = size * size;

    // HHS
    gemm_configs.push_back({size, size, size, false, false, false, 1, 0, 0, 0, 0,
                            "f16_r", "f16_r", "f16_r", "f16_r", "f32_r", "HHS"});
    // HHS batched
    gemm_configs.push_back({size, size, size, false, false, false, 5, 0, 0, 0, 0,
                            "f16_r", "f16_r", "f16_r", "f16_r", "f32_r", "HHS Batched"});
    // HHS strided batched
    gemm_configs.push_back({size, size, size, false, false, false, 5, stride, stride, stride, stride,
                            "f16_r", "f16_r", "f16_r", "f16_r", "f32_r", "HHS Strided Batched"});
    // I8II
    gemm_configs.push_back({size, size, size, false, false, false, 1, 0, 0, 0, 0,
                            "i8_r", "i8_r", "i32_r", "i32_r", "i32_r", "I8II"});
    // I8II batched
    gemm_configs.push_back({size, size, size, false, false, false, 5, 0, 0, 0, 0,
                            "i8_r", "i8_r", "i32_r", "i32_r", "i32_r", "I8II Batched"});
    // I8II strided batched
    gemm_configs.push_back({size, size, size, false, false, false, 5, stride, stride, stride, stride,
                            "i8_r", "i8_r", "i32_r", "i32_r", "i32_r", "I8II Strided Batched"});
  }

  // GEMV configurations
  for (const auto &size : gemv_sizes) {
    // GEMV
    gemv_configs.push_back({size, 1, size, false, false, false, 1, 0, 0, 0, 0,
                            "f16_r", "f16_r", "f16_r", "f16_r", "f32_r", "GEMV"});
  }
}

template <typename T1, typename T2>
int time_gemm(Tensor<T1> A, Tensor<T1> B, Tensor<T2> C, const ProblemConfig &config,
              rocblas_handle rocblas_handle) {
  int m = config.m;
  int n = config.n;
  int k = config.k;
  int lda = m;
  int ldb = k;
  int ldc = m;
  int warp_up_iters = 1;
  int numRepeats = 10;

  const T1 alpha = 1;
  const T1 beta = 1;

  rocblas_status stat;
  rocblas_operation transA =
      config.a_t ? rocblas_operation_transpose : rocblas_operation_none;
  rocblas_operation transB =
      config.b_t ? rocblas_operation_transpose : rocblas_operation_none;

  // Warmup
  for (int i = 0; i < warp_up_iters; ++i) {
    stat = rocblas_gemm_ex(
        rocblas_handle, transA, transB, m, n, k, &alpha, A.begin(),
        rocblas_datatype_f16_r, lda, B.begin(), rocblas_datatype_f16_r, ldb,
        &beta, C.begin(), rocblas_datatype_f16_r, ldc, C.begin(),
        rocblas_datatype_f16_r, ldc, rocblas_datatype_f32_r,
        rocblas_gemm_algo_standard, 0, 0);
    CHECK_ROCBLAS_ERROR(stat);
  }

  auto start = std::chrono::steady_clock::now();

  // Measure time for a single iteration
  stat = rocblas_gemm_ex(
      rocblas_handle, transA, transB, m, n, k, &alpha, A.begin(),
      rocblas_datatype_f16_r, lda, B.begin(), rocblas_datatype_f16_r, ldb,
      &beta, C.begin(), rocblas_datatype_f16_r, ldc, C.begin(),
      rocblas_datatype_f16_r, ldc, rocblas_datatype_f32_r,
      rocblas_gemm_algo_standard, 0, 0);
  CHECK_ROCBLAS_ERROR(stat);

  hipDeviceSynchronize();

  auto end = std::chrono::steady_clock::now();
  auto single_iteration_time_us = std::chrono::duration<double, std::micro>(end - start).count();

  // Calculate the number of iterations needed to reach at least 3 minutes (180 seconds)
  int min_duration_us = 180 * 1000000; // 120 seconds in microseconds
  numRepeats = static_cast<int>(min_duration_us / single_iteration_time_us);

  start = std::chrono::steady_clock::now();

  // Timing loop
  for (int i = 0; i < numRepeats; ++i) {
    stat = rocblas_gemm_ex(
        rocblas_handle, transA, transB, m, n, k, &alpha, A.begin(),
        rocblas_datatype_f16_r, lda, B.begin(), rocblas_datatype_f16_r, ldb,
        &beta, C.begin(), rocblas_datatype_f16_r, ldc, C.begin(),
        rocblas_datatype_f16_r, ldc, rocblas_datatype_f32_r,
        rocblas_gemm_algo_standard, 0, 0);
    CHECK_ROCBLAS_ERROR(stat);
  }

  hipDeviceSynchronize();

  end = std::chrono::steady_clock::now();
  return static_cast<int>(
      std::chrono::duration<double, std::micro>(end - start).count() /
      numRepeats);
}

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <gpu_index>" << std::endl;
    return 1;
  }

  int gpu_index = std::stoi(argv[1]);
  hipSetDevice(gpu_index);

  initialize_problem_configs();

  int deviceCount = 1;
  hipGetDeviceCount(&deviceCount);
  if (gpu_index >= deviceCount) {
    std::cerr << "Invalid GPU index. Available GPUs: 0 to " << deviceCount - 1 << std::endl;
    return 1;
  }

  hiprandGenerator_t hiprand_gen;
  hiprandCreateGenerator(&hiprand_gen, HIPRAND_RNG_PSEUDO_DEFAULT);
  hiprandSetPseudoRandomGeneratorSeed(hiprand_gen, 123ULL);

  rocblas_handle rocblas_handle;
  CHECK_ROCBLAS_ERROR(rocblas_create_handle(&rocblas_handle));

  std::cout << "Running GEMM benchmarks on GPU " << gpu_index << "..." << std::endl;

  for (const auto &config : gemm_configs) {
    Tensor<half> A({config.m, config.k});
    Tensor<half> B({config.k, config.n});
    Tensor<half> C({config.m, config.n});

    hiprandGenerateUniformHalf(hiprand_gen, A.begin(), A.size());
    hiprandGenerateUniformHalf(hiprand_gen, B.begin(), B.size());
    hiprandGenerateUniformHalf(hiprand_gen, C.begin(), C.size());

    int time_us = time_gemm<half, half>(A, B, C, config, rocblas_handle);

    std::cout << "GEMM," << config.name << "," << config.m << "," << config.n << ","
              << config.k << "," << config.a_type << "," << config.b_type << ","
              << config.c_type << ",Time (ms): " << time_us / 1000.0 << std::endl;
  }

  std::cout << "Running GEMV benchmarks on GPU " << gpu_index << "..." << std::endl;

  for (const auto &config : gemv_configs) {
    Tensor<half> A({config.m, config.k});
    Tensor<half> x({config.k, 1});
    Tensor<half> y({config.m, 1});

    hiprandGenerateUniformHalf(hiprand_gen, A.begin(), A.size());
    hiprandGenerateUniformHalf(hiprand_gen, x.begin(), x.size());
    hiprandGenerateUniformHalf(hiprand_gen, y.begin(), y.size());

    int time_us = time_gemm<half, half>(A, x, y, config, rocblas_handle);

    std::cout << "GEMV," << config.name << "," << config.m << "," << config.n << ","
              << config.k << "," << config.a_type << "," << config.b_type << ","
              << config.c_type << ",Time (ms): " << time_us / 1000.0 << std::endl;
  }

  rocblas_destroy_handle(rocblas_handle);
  hiprandDestroyGenerator(hiprand_gen);

  return 0;
}