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

void print(const char* name, std::array<duration,5> dt, std::array<double,2> bw) {
    using namespace std::chrono;
    std::cout << std::setw(19) << name;
    for (size_t i=0; i<5; ++i) {
        std::stringstream tmp;
        tmp << duration_cast<microseconds>(dt[i]).count() << "us";
        std::cout << std::setw(20) << tmp.str();
    }
    for (size_t i=0; i<2; ++i) {
        std::stringstream tmp;
        tmp << bw[i] << "GB/s";
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
    std::cout << std::setw(20) << "OpenMP bandwidth";
    std::cout << std::setw(20) << "OpenCL bandwidth";
    std::cout << '\n';
}

struct OpenCL {
    cl::Platform platform;
    cl::Device device;
    cl::Context context;
    cl::Program program;
    cl::CommandQueue queue;
};

void profile_reduce(int n, OpenCL& opencl) {
    auto a = random_vector<float>(n);
    int localSize = 128;
    Vector<float> result(localSize);
    float expected_result = 0;
    opencl.queue.flush();
    cl::Kernel kernel(opencl.program, "reduce");
    auto t0 = clock_type::now();
    expected_result = reduce(a);
    auto t1 = clock_type::now();
    cl::Buffer d_a(opencl.queue, begin(a), end(a), true);
    opencl.queue.flush();
    auto t2 = clock_type::now();
    auto t3 = t2;
    auto t4 = t2;
    int size = n / localSize;
    while (true) {
        kernel.setArg(0, d_a);
        cl::Buffer d_result(opencl.context, CL_MEM_READ_WRITE, sizeof(float) * size);
        kernel.setArg(1, d_result);
        opencl.queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(size * localSize), cl::NDRange(localSize));
        d_a = d_result;

        if (size % localSize != 0 && size > localSize) { size *= 2; }

        if (size <= localSize) {
            opencl.queue.flush();
            t3 = clock_type::now();
            cl::copy(opencl.queue, d_result, std::begin(result), std::begin(result) + size);
            t4 = clock_type::now();
            break;
        }
        size /= localSize;
    }

    float sum = result[0];

    if (std::abs(expected_result - sum) > 1e3) {
        std::stringstream msg;
        msg << "Got: " << sum << ", expected: " << expected_result;
        throw std::runtime_error(msg.str());
    }
    print("reduce",
          {t1-t0,t4-t1,t2-t1,t3-t2,t4-t3},
          {bandwidth(n*n+n+n, t0, t1), bandwidth(n*n+n+n, t2, t3)});
}

void profile_scan_inclusive(int n, OpenCL& opencl) {
    int localSize = 128;
    auto a = random_vector<float>(n);
    Vector<float> result(a), expected_result(a);
    opencl.queue.flush();
    cl::Kernel kernel(opencl.program, "scan_inclusive");
    cl::Kernel kernel_fin(opencl.program, "finish_scan_inclusive");
    auto t0 = clock_type::now();
    scan_inclusive(expected_result);
    auto t1 = clock_type::now();
    cl::Buffer d_a(opencl.queue, begin(a), end(a), false);
    opencl.queue.finish();
    auto t2 = clock_type::now();
    int size = n;
    int val = 1;
    while (size > 1) {
        kernel.setArg(0, d_a);
        kernel.setArg(1, val);
        if (size < localSize) {
            localSize = size;
        }
        opencl.queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(size), cl::NDRange(localSize));
        size /= localSize;
        val *= localSize;
    }
    kernel_fin.setArg(0, d_a);
    kernel_fin.setArg(1, localSize);
    opencl.queue.enqueueNDRangeKernel(kernel_fin, cl::NullRange, cl::NDRange(n/localSize - 1), cl::NullRange);
    opencl.queue.finish();
    auto t3 = clock_type::now();
    cl::copy(opencl.queue, d_a, begin(result), end(result));
    auto t4 = clock_type::now();
    float res = result[n - 1];
    float expected_res = expected_result[n - 1];
    if (std::abs(expected_res - res) > 1e3) {
        std::stringstream msg;
        msg << "Got: " << res << ", expected: " << expected_res;
        throw std::runtime_error(msg.str());
    }
    print("scan-inclusive",
          {t1-t0,t4-t1,t2-t1,t3-t2,t4-t3},
          {bandwidth(n*n+n*n+n*n, t0, t1), bandwidth(n*n+n*n+n*n, t2, t3)});
}


void opencl_main(OpenCL& opencl) {
    using namespace std::chrono;
    print_column_names();
    profile_reduce(1024*1024*10, opencl);
    profile_scan_inclusive(1024*1024*10, opencl);
}

const std::string src = R"(
#define BUFFSIZE 1024
kernel void reduce(global float* a, global float* result) {
    const int localSize = get_local_size(0);
    int localId = get_local_id(0);
    int groupId = get_group_id(0);
    const int groupNum = get_num_groups(0);
    const int globalId = get_global_id(0);
    local float buff[BUFFSIZE];
    buff[localId] = a[groupId * localSize + localId];
    barrier(CLK_LOCAL_MEM_FENCE);
    const int globalSize = get_global_size(0);

    for (int offset = localSize / 2; offset > 0; offset /= 2) {
        if (localId < offset) { buff[localId] += buff[localId + offset]; }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    if (localId == 0) { result[groupId] = buff[0]; }
    
    if (globalId == 0 && globalSize / localSize <= localSize) {
        float sum = 0;
        for (int i = 0; i < groupNum; i++)
            sum += result[i];
        result[0] = sum;
    }
}

kernel void scan_inclusive(global float* a, int step) {
    const int globalId = get_global_id(0);
    const int localId = get_local_id(0);
    const int localSize = get_local_size(0);
    local float buff[BUFFSIZE];
    buff[localId] = a[(step - 1) + globalId * step];
    barrier(CLK_LOCAL_MEM_FENCE);
    float sum = buff[localId];

    for (int offset = 1; offset < localSize; offset *= 2) {
        if (localId >= offset) {
            sum += buff[localId - offset];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        buff[localId] = sum;
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    a[(step - 1) + globalId * step] = buff[localId];
}

kernel void finish_scan_inclusive(global float* a, int step) {
    const int globalId = get_global_id(0);

    for (int i = 0; i < step - 1; i++) {
        a[(globalId + 1) * step + i] += a[(globalId + 1) * step - 1];
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