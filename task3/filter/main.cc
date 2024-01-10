#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120

#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.hpp>
#else
#include <CL/cl2.hpp>
#endif

#include <array>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>

#include "filter.hh"
#include "linear-algebra.hh"
#include "reduce-scan.hh"

using clock_type = std::chrono::high_resolution_clock;
using duration = clock_type::duration;
using time_point = clock_type::time_point;

double bandwidth(int n, time_point t0, time_point t1) {
    using namespace std::chrono;
    const auto dt = duration_cast<microseconds>(t1-t0).count();
    if (dt == 0) { return 0; }
    return ((n+n+n)*sizeof(float)*1e-9)/(dt*1e-6);
}

void print(const char* name, std::array<duration,5> dt) {
    using namespace std::chrono;
    std::cout << std::setw(19) << name;
    for (size_t i=0; i<5; ++i) {
        std::stringstream tmp;
        tmp << duration_cast<microseconds>(dt[i]).count() << "us";
        std::cout << std::setw(20) << tmp.str();
    }
    std::cout << '\n';
}

void print_column_names() {
    std::cout << std::setw(19) << "function";
    std::cout << std::setw(20) << "OpenMP";
    std::cout << std::setw(20) << "OpenCL total";
    std::cout << std::setw(20) << "OpenCL copy-in";
    std::cout << std::setw(20) << "OpenCL kernel";
    std::cout << std::setw(20) << "OpenCL copy-out";
    std::cout << '\n';
}

struct OpenCL {
    cl::Platform platform;
    cl::Device device;
    cl::Context context;
    cl::Program program;
    cl::CommandQueue queue;
};

void profile_filter(int n, OpenCL& opencl) {
    auto input = random_std_vector<float>(n);
    int local_sz = 256;
    std::vector<int> offsets(n / local_sz);
    std::vector<float> result;
    result.reserve(n);
    std::vector<float> result_gpu(n);
    cl::Kernel cnt_pos_kernel(opencl.program, "count_positive");
    cl::Kernel comp_partial_kernel(opencl.program, "compute_partial");
    cl::Kernel scan_finish_kernel(opencl.program, "finish_scan");
    cl::Kernel filter_kernel(opencl.program, "filter");
    auto t0 = clock_type::now();
    filter(input, result, [] (float x) { return x > 0; }); // filter positive numbers
    auto t1 = clock_type::now();
    cl::Buffer d_input(opencl.queue, begin(input), end(input), false);
    cl::Buffer d_offsets(opencl.context, CL_MEM_READ_WRITE, offsets.size()*sizeof(int));
    cl::Buffer d_result(opencl.context, CL_MEM_READ_WRITE, input.size()*sizeof(float));
    opencl.queue.finish();
    cnt_pos_kernel.setArg(0, d_input);
    cnt_pos_kernel.setArg(1, d_offsets);
    opencl.queue.flush();
    auto t2 = clock_type::now();
    opencl.queue.enqueueNDRangeKernel(cnt_pos_kernel, cl::NullRange, cl::NDRange(n), cl::NDRange(local_sz));
    opencl.queue.finish();

    comp_partial_kernel.setArg(0, d_offsets);
    opencl.queue.enqueueNDRangeKernel(comp_partial_kernel, cl::NullRange, cl::NDRange(n / local_sz), cl::NDRange(local_sz));
    opencl.queue.finish();

    scan_finish_kernel.setArg(0, d_offsets);
    opencl.queue.enqueueNDRangeKernel(scan_finish_kernel, cl::NullRange, cl::NDRange(n / local_sz), cl::NDRange(local_sz));
    opencl.queue.finish();

    filter_kernel.setArg(0, d_input);
    filter_kernel.setArg(1, d_offsets);
    filter_kernel.setArg(2, d_result);
    opencl.queue.enqueueNDRangeKernel(filter_kernel, cl::NullRange, cl::NDRange(n), cl::NDRange(local_sz));
    auto t3 = clock_type::now();
    cl::copy(opencl.queue, d_offsets, std::begin(offsets), std::end(offsets));
    int sz = offsets.back();
    cl::copy(opencl.queue, d_result, std::begin(result_gpu), std::begin(result_gpu) + sz);
    result_gpu.resize(sz);

    auto t4 = clock_type::now();

    verify_vector(result, result_gpu);
    print("filter", {t1-t0,t4-t1,t2-t1,t3-t2,t4-t3});
}

void opencl_main(OpenCL& opencl) {
    using namespace std::chrono;
    print_column_names();
    profile_filter(1024*1024, opencl);
}

const std::string src = R"(
#define BUFFSIZE 1024
kernel void filter(global const float *input,
                    global const int *offsets,
                    global float *result) {
    const int m = get_local_size(0);
    int k = get_group_id(0);
    int t = get_local_id(0);
    local float buff[BUFFSIZE];

    buff[t] = input[k * m + t];
    barrier(CLK_LOCAL_MEM_FENCE);
    
    if (t == 0) {
        int idx = 0;
        if (k > 0)
            idx = offsets[k - 1];
        for (int j = 0; j < m; j++) {
            if (buff[j] > 0) {
                result[idx] = buff[j];
                idx++;
            }
        }
    }
}

kernel void count_positive(global const float *a,
                    global int *result) {
    const int m = get_local_size(0);
    int k = get_group_id(0);
    int t = get_local_id(0);
    local float buff[BUFFSIZE];

    buff[t] = a[k * m + t];
    barrier(CLK_LOCAL_MEM_FENCE);

    if (t == 0) {
        int cnt = 0;
        for (int j = 0; j < m; j++) {
            if (buff[j] > 0)
                cnt++;
        }
        result[k] = cnt;
    }
}

kernel void compute_partial(global int *result) {
    const int i = get_global_id(0);
    const int t = get_local_id(0);
    const int m = get_local_size(0);

    // move parts of array into local
    local int buff[BUFFSIZE];
    buff[t] = result[i];
    barrier(CLK_LOCAL_MEM_FENCE);

    int sum = buff[t];
    // compute in local
    for (int offset = 1; offset < m; offset *= 2) {
        if (t >= offset) {
            sum += buff[t - offset];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        buff[t] = sum;
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    result[i] = buff[t];
}

kernel void finish_scan(global int *result) {
    const int k = get_group_id(0);
    const int n = get_global_size(0);
    const int t = get_local_id(0);
    const int m = get_local_size(0);
    if (k == 0) {
        for (int j = 1; j < n / m; j++) {
            result[j * m + t] += result[j * m - 1];
            barrier(CLK_GLOBAL_MEM_FENCE);
        }
    }
}
)";

int main() {
    try {
        // find OpenCL platforms
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        if (platforms.empty()) {
            std::cerr << "Unable to find OpenCL platforms\n";
            return 1;
        }
        cl::Platform platform = platforms[0];
        std::clog << "Platform name: " << platform.getInfo<CL_PLATFORM_NAME>() << '\n';
        // create context
        cl_context_properties properties[] =
            { CL_CONTEXT_PLATFORM, (cl_context_properties)platform(), 0};
        cl::Context context(CL_DEVICE_TYPE_GPU, properties);
        // get all devices associated with the context
        std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
        cl::Device device = devices[0];
        std::clog << "Device name: " << device.getInfo<CL_DEVICE_NAME>() << '\n';
        cl::Program program(context, src);
        // compile the programme
        try {
            program.build(devices);
        } catch (const cl::Error& err) {
            for (const auto& device : devices) {
                std::string log = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
                std::cerr << log;
            }
            throw;
        }
        cl::CommandQueue queue(context, device);
        OpenCL opencl{platform, device, context, program, queue};
        opencl_main(opencl);
    } catch (const cl::Error& err) {
        std::cerr << "OpenCL error in " << err.what() << '(' << err.err() << ")\n";
        std::cerr << "Search cl.h file for error code (" << err.err()
            << ") to understand what it means:\n";
        std::cerr << "https://github.com/KhronosGroup/OpenCL-Headers/blob/master/CL/cl.h\n";
        return 1;
    } catch (const std::exception& err) {
        std::cerr << err.what() << std::endl;
        return 1;
    }
    return 0;
}