#ifndef UTILS_H
#define UTILS_H

#include <CL/cl2.hpp>
#include <vector>
#include <string>
#include <fstream>

// Helper to load kernel files
static std::string readKernelFile(const std::string &path)
{
    std::ifstream f(path);
    if (!f)
        throw std::runtime_error("Can't open file");
    std::string content(
        (std::istreambuf_iterator<char>(f)),
        std::istreambuf_iterator<char>());
    return content;
}

// Get first platform (simplified)
static cl::Platform getFirstPlatform()
{
    std::vector<cl::Platform> plats;
    cl::Platform::get(&plats);
    if (plats.empty())
        throw std::runtime_error("No platforms");
    return plats[0];
}

// Timing struct (basic)
struct TimeData
{
    cl_ulong start, end;
    double ms() { return (end - start) * 1e-6; }
};

// Get kernel time from event
static TimeData getTime(cl::Event &evt)
{
    evt.wait();
    TimeData t;
    t.start = evt.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    t.end = evt.getProfilingInfo<CL_PROFILING_COMMAND_END>();
    return t;
}

#endif