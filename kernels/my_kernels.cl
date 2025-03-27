__kernel void hist_kernel(
    __global uchar* pixels,
    __global uint* hist,
    __local uint* tmp
) {
    int lid = get_local_id(0);
    int lsize = get_local_size(0);

    // Clear tmp memory first
    for(int i=lid; i<256; i+=lsize) tmp[i] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Tally pixels (atomic adds are slow but needed)
    uchar val = pixels[get_global_id(0)];
    atomic_inc(&tmp[val]);
    barrier(CLK_LOCAL_MEM_FENCE);

    // Combine results
    for(int i=lid; i<256; i+=lsize)
        atomic_add(&hist[i], tmp[i]);
}

// Add up histogram values
__kernel void scan_kernel(
    __global uint* hist,
    __global uint* cumul,
    __local uint* temp
) {
    int lid = get_local_id(0);
    temp[lid] = (lid == 0) ? 0 : hist[lid-1];
    barrier(CLK_LOCAL_MEM_FENCE);

    // Prefix sum magic
    for(int s=1; s<get_local_size(0); s*=2) {
        if(lid >= s) temp[lid] += temp[lid-s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    cumul[get_global_id(0)] = temp[lid];
}

// Make lookup table from cumul hist
__kernel void lut_kernel(
    __global uint* cumul,
    __global uchar* lut,
    uint minv,
    uint maxv
) {
    int i = get_global_id(0);
    lut[i] = maxv != minv ? (cumul[i]-minv)*255/(maxv-minv) : 0;
}

// Apply LUT to image
__kernel void apply_kernel(
    __global uchar* in,
    __global uchar* out,
    __global uchar* lut
) {
    int idx = get_global_id(0);
    out[idx] = lut[in[idx]]; // Simple remap
}