#include <metal_stdlib>
using namespace metal;

// Utils
METAL_FUNC uint get_strided_index(
    uint idx,
    constant size_t &num_dims,
    constant size_t *dims,
    constant size_t *strides
) {
    uint strided_i = 0;
    for (uint d = 0; d < num_dims; d++) {
        uint dim_idx = num_dims - 1 - d;
        strided_i += (idx % dims[dim_idx]) * strides[dim_idx];
        idx /= dims[dim_idx];
    }
    return strided_i;
}

template<uint Y>
constexpr uint div_ceil(uint x) {
    return x / Y + (x % Y > 0);
}

template<uint X, uint Y>
constexpr uint div_ceil() {
    return X / Y + (X % Y > 0);
}

template<typename T>
constexpr uint work_per_thread() {
    return div_ceil<8, sizeof(T)>();
}

// Kernels
template <
    typename T,
    typename U,
    typename IR = T,
    int W = work_per_thread<T>()
>
[[kernel]] void cast_kernel(
    constant size_t &dim,
    device const T* input,
    device U* output,
    uint tid [[thread_position_in_grid]]
) {
    const uint step = div_ceil<W>(dim);
    #pragma clang loop unroll(full)
    for (uint i = tid; i < dim; i += step) {
        output[i] = static_cast<U>(static_cast<IR>(input[i]));
    }
}

template <typename T, typename U, typename IR = T>
[[kernel]] void cast_kernel_strided(
    constant size_t &dim,
    constant size_t &num_dims,
    constant size_t *dims,
    constant size_t *strides,
    constant const T *input,
    device U *output,
    uint tid [[ thread_position_in_grid ]]
) {
    if (tid >= dim) return;
    output[tid] = static_cast<U>(
        static_cast<IR>(input[get_strided_index(tid, num_dims, dims, strides)])
    );
}

// Vectorized half<->float cast: 4 elements per thread via SIMD.
// Adjacent threads access adjacent memory (coalesced), and the
// half4<->float4 conversion is a single SIMD instruction.
[[host_name("cast_f16_f32_vec4")]]
[[kernel]] void cast_f16_f32_vec4(
    constant size_t &dim,
    device const half *input,
    device float *output,
    uint tid [[thread_position_in_grid]]
) {
    const uint base = tid * 4;
    if (base + 4 <= dim) {
        half4 h = *reinterpret_cast<device const half4 *>(input + base);
        *reinterpret_cast<device float4 *>(output + base) = float4(h);
    } else {
        for (uint i = base; i < min(base + 4u, uint(dim)); i++) {
            output[i] = static_cast<float>(input[i]);
        }
    }
}

[[host_name("cast_f32_f16_vec4")]]
[[kernel]] void cast_f32_f16_vec4(
    constant size_t &dim,
    device const float *input,
    device half *output,
    uint tid [[thread_position_in_grid]]
) {
    const uint base = tid * 4;
    if (base + 4 <= dim) {
        float4 f = *reinterpret_cast<device const float4 *>(input + base);
        *reinterpret_cast<device half4 *>(output + base) = half4(f);
    } else {
        for (uint i = base; i < min(base + 4u, uint(dim)); i++) {
            output[i] = static_cast<half>(input[i]);
        }
    }
}

#if defined(__HAVE_BFLOAT__)
[[host_name("cast_bf16_f32_vec4")]]
[[kernel]] void cast_bf16_f32_vec4(
    constant size_t &dim,
    device const bfloat *input,
    device float *output,
    uint tid [[thread_position_in_grid]]
) {
    const uint base = tid * 4;
    if (base + 4 <= dim) {
        // bfloat4 -> float4 via component-wise cast
        bfloat4 b = *reinterpret_cast<device const bfloat4 *>(input + base);
        *reinterpret_cast<device float4 *>(output + base) = float4(b);
    } else {
        for (uint i = base; i < min(base + 4u, uint(dim)); i++) {
            output[i] = static_cast<float>(input[i]);
        }
    }
}

[[host_name("cast_f32_bf16_vec4")]]
[[kernel]] void cast_f32_bf16_vec4(
    constant size_t &dim,
    device const float *input,
    device bfloat *output,
    uint tid [[thread_position_in_grid]]
) {
    const uint base = tid * 4;
    if (base + 4 <= dim) {
        float4 f = *reinterpret_cast<device const float4 *>(input + base);
        *reinterpret_cast<device bfloat4 *>(output + base) = bfloat4(f);
    } else {
        for (uint i = base; i < min(base + 4u, uint(dim)); i++) {
            output[i] = static_cast<bfloat>(input[i]);
        }
    }
}
#endif

// Macros to help initialize kernels
#define init_kernel(name, func, ...) \
  template [[host_name(name)]] [[kernel]] decltype(func<__VA_ARGS__>) func<__VA_ARGS__>;

#define init_cast(tname, t, uname, u)                                           \
    init_kernel("cast_" #tname "_" #uname, cast_kernel, t, u)                   \
    init_kernel("cast_" #tname "_" #uname "_strided", cast_kernel_strided, t, u)

#if defined(__HAVE_BFLOAT__)
#define init_cast_all(tname, t)         \
    init_cast(tname, t, f32, float)     \
    init_cast(tname, t, f16, half)      \
    init_cast(tname, t, bf16, bfloat)   \
    init_cast(tname, t, i64, int64_t)   \
    init_cast(tname, t, u32, uint32_t)  \
    init_cast(tname, t, u8, uint8_t)
#else
#define init_cast_all(tname, t)         \
    init_cast(tname, t, f32, float)     \
    init_cast(tname, t, f16, half)      \
    init_cast(tname, t, i64, int64_t)   \
    init_cast(tname, t, u32, uint32_t)  \
    init_cast(tname, t, u8, uint8_t)
#endif


init_cast_all(f32, float);
init_cast_all(f16, half);
#if defined(__HAVE_BFLOAT__)
init_cast_all(bf16, bfloat);
#endif
init_cast_all(i64, int64_t);
init_cast_all(u32, uint32_t);
init_cast_all(u8, uint8_t);