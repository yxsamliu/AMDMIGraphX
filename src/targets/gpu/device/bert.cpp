#include <migraphx/gpu/device/bert.hpp>
#include <migraphx/gpu/device/reduce.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {


// m = x - mean(x)
// sqrt(mean(m ^ 2) + 1e-12) / m
void bert(hipStream_t stream, const argument& result, const argument& arg)
{
    auto relements = arg.get_shape().lens().back();
    hip_visit_all(result, arg)([&](auto output, auto input) {
        using value_type = typename decltype(input)::value_type;
        auto nelements = result.get_shape().elements() / relements;

        const std::size_t max_block_size = 256;
        const std::size_t block_size     = compute_block_size(relements, max_block_size);
        gs_launch(stream, nelements * block_size, block_size)([=](auto i, auto idx) __device__ {
            const auto out_idx  = i / block_size;
            const auto base_idx = out_idx * relements;
            value_type x[4];
            idx.local_stride(relements, [&](auto j) __device__ {
                x[j] = input.data()[base_idx + j];
            });
            auto m = block_reduce<max_block_size>(idx, sum{}, 0, relements, [&](auto j) __device__ {
                return x[j];
            }) / relements;
            idx.local_stride(relements, [&](auto j) __device__ {
                x[j] = x[j] - m;
            });

            auto r = block_reduce<max_block_size>(idx, sum{}, 0, relements, [&](auto j) __device__ {
                return x[j];
            }) / relements;

            idx.local_stride(relements, [&](auto j) __device__ {
                output.data()[base_idx + j] = ::sqrt(r + 1e-12) / (x[j]);
            });
        });
    });

}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
