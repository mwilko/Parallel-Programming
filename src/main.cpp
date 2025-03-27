#include <iostream>
#include <vector>
#include "CImg.h"  // Image loader
#include "Utils.h" // Helpers
#include <CL/cl2.hpp>

int main()
{
    int w, h;
    std::vector<cl_uchar> img;
    try // Import image
    {
        img = readPGM("images/test.pgm", w, h);
    }
    catch (...) // If image cant be imported
    {
        std::cerr << "Failed loading image\n";
        return 1;
    }
    size_t total_px = img.size(); // Image size
    std::cout << "Loaded image: " << w << "x" << h << "\n";

    // OpenCL setup
    cl::Platform plat = getFirstPlatform();
    cl::Device dev = getDevice(plat);
    cl::Context ctx(dev);
    cl::CommandQueue queue(ctx, dev);

    // Build kernels
    cl::Program prog(ctx, readKernelFile("kernels/my_kernels.cl"));
    if (prog.build({dev}) != CL_SUCCESS)
    {
        std::cerr << "Build failed:\n"
                  << prog.getBuildInfo<CL_PROGRAM_BUILD_LOG>(dev) << "\n";
        return 1;
    }

    // Make buffers
    cl::Buffer img_buf(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                       total_px * sizeof(cl_uchar), img.data());
    cl::Buffer hist_buf(ctx, CL_MEM_READ_WRITE, 256 * sizeof(uint));
    cl::Buffer cumul_buf(ctx, CL_MEM_READ_WRITE, 256 * sizeof(uint));
    cl::Buffer lut_buf(ctx, CL_MEM_WRITE_ONLY, 256 * sizeof(uchar));
    cl::Buffer out_buf(ctx, CL_MEM_WRITE_ONLY, total_px * sizeof(uchar));

    // Run histogram
    cl::Kernel hist(prog, "hist_kernel");
    hist.setArg(0, img_buf);
    hist.setArg(1, hist_buf);
    hist.setArg(2, 256 * sizeof(uint), nullptr);
    queue.enqueueNDRangeKernel(hist, cl::NullRange, cl::NDRange(total_px), cl::NDRange(256));

    // Run scan
    cl::Kernel scan(prog, "scan_kernel");
    scan.setArg(0, hist_buf);
    scan.setArg(1, cumul_buf);
    scan.setArg(2, 256 * sizeof(uint), nullptr);
    queue.enqueueNDRangeKernel(scan, cl::NullRange, cl::NDRange(256), cl::NDRange(256));

    // Find min/max
    std::vector<uint> cumul(256);
    queue.enqueueReadBuffer(cumul_buf, CL_TRUE, 0, 256 * sizeof(uint), cumul.data());
    uint minv = cumul[0], maxv = cumul[255];

    // Make LUT
    cl::Kernel lut(prog, "lut_kernel");
    lut.setArg(0, cumul_buf);
    lut.setArg(1, lut_buf);
    lut.setArg(2, minv);
    lut.setArg(3, maxv);
    queue.enqueueNDRangeKernel(lut, cl::NullRange, cl::NDRange(256), cl::NullRange);

    // Apply LUT
    cl::Kernel apply(prog, "apply_kernel");
    apply.setArg(0, img_buf);
    apply.setArg(1, out_buf);
    apply.setArg(2, lut_buf);
    queue.enqueueNDRangeKernel(apply, cl::NullRange, cl::NDRange(total_px), cl::NullRange);

    // Save result
    std::vector<uchar> output(total_px);
    queue.enqueueReadBuffer(out_buf, CL_TRUE, 0, total_px * sizeof(uchar), output.data());
    writePGM("output.pgm", w, h, output.data());

    std::cout << "Finished, output saved to: output.pgm";
    return 0;
}