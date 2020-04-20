#include <migraphx/shape.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/gpu/device/nary.hpp>
#include <migraphx/gpu/device/pad.hpp>
#include <migraphx/gpu/device/tensor.hpp>
#include <migraphx/gpu/device/launch.hpp>
#include <migraphx/float_equal.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

argument
pad_constant(hipStream_t stream, argument result, const std::vector<argument>& args, const std::vector<std::int64_t>& pads)
{
    hip_visit_all(result, arg[0])([&](auto output, auto input) {
        using hip_index = typename decltype(output)::hip_index;
        
        if (args.size() == 2)
        {
            args[1].visit([&](auto val) {
                const auto* val_ptr = device_cast(val.data());
                gs_launch(stream, result.get_shape().elements())(
                    [=](auto i) __device__ { output.data()[i] = val_ptr[0]; });
            })
        }
        else {
            gs_launch(stream, result.get_shape().elements())(
                [=](auto i) __device__ { output.data()[i] = 0; });
        }

        std::size_t nelements = arg[0].get_shape().elements();
        hip_index offsets;
        std::copy(pads.begin(), pads.begin() + offsets.size(), offsets.begin());
        gs_launch(stream, nelements)([=](auto i) __device__ {
            auto idx = input.get_shape().multi(i);
            for(std::size_t j = 0; j < offsets.size(); j++)
            {
                idx[j] += offsets[j];
            }
            output[idx] = input.data()[i];
        });
    });

    return result;
}

argument
pad_edge(hipStream_t stream, argument result, const std::vector<argument>& args, const std::vector<std::int64_t>& pads)
{
    std::size_t nelements = result.get_shape().elements();
    auto in_lens = args[0].get_shape().lens();
    hip_visit_all(result, arg[0], result.get_shape())([&](auto output, auto input, auto out_s) {
        gs_launch(stream, nelements)([=](auto i) __device__ {
            auto idx = out_s.multi(i);
            auto in_idx = idx;
            for(std::size_t j = 0; j < idx.size(); ++j)
            {
                if (in_idx[j] < pads[j])
                {
                    in_idx[j] = 0;
                }
                else if (in_idx[j] >= pads[j] and in_idx[j] < pads[j] + in_lens[j])
                {
                    in_idx[j] = idx[j] - pads[j];
                }
                else
                {
                    in_idx[j] = in_lens[j] - 1;
                }
            }
            output[idx] = input[in_idx];
        });
    });

    return result;
}

argument
pad_reflect(hipStream_t stream, argument result, const std::vector<argument>& args, const std::vector<std::int64_t>& pads)
{
    auto reflect_idx = [](auto& idx,
                          const std::vector<std::size_t>& in_loc_start,
                          const std::vector<std::size_t>& in_loc_end)
    {
        std::vector<std::size_t> vec_dims(in_loc_end.size());
        std::transform(in_loc_end.begin(),
                       in_loc_start.end(),
                       in_loc_start.begin(),
                       vec_dims.begin(),
                       [](auto i, auto j) { return i - j; });
        std::size_t n_dim = in_loc_start.size();
        for(size_t i = 0; i < n_dim; ++)
        {
            if(vec_dims[i] == 1)
            {
                idx[i] = 0;
            }
            else
            {
                std::size_t size = vec_dims[i] - 1;
                if(idx[i] < in_loc_start[i])
                {
                    auto delta = in_loc_start[i] - idx[i];
                    auto index = delta % (2 * size);
                    if(index > size)
                    {
                        idx[i] = 2 * size - index;
                    }
                    else
                    {
                        idx[i] = index;
                    }
                }
                else if(idx[i] >= in_loc_end[i])
                {
                    auto delta = idx[i] - (in_loc_end[i] - 1);
                    auto index = delta % (2 * size);
                    if(index < size)
                    {
                        idx[i] = size - index;
                    }
                    else
                    {
                        idx[i] = index - size;
                    }
                }
                // inside input
                else
                {
                    idx[i] -= in_loc_start[i];
                }
            }
        }
    };

    std::size_t nelements = result.get_shape().elements();
    auto in_lens = args[0].get_shape().lens();
    std::vector<std::size_t> in_loc_start(in_lens.size());
    std::copy(pads.begin(), pads.begin() + in_lens.size(), in_loc_start.begin());
    std::transform(in_loc_start.begin(), in_loc_start.end(),
                   in_lens.begin(), in_loc_end.begin(), 
                   [](auto i, auto j) { return i + j; });
    hip_visit_all(result, arg[0], result.get_shape())([&](auto output, auto input, auto out_s) {
        gs_launch(stream, nelements)([=](auto i) __device__ {
            auto idx = out_s.multi(i);
            auto in_idx = idx;
            reflect_idx(in_idx, in_loc_start, in_loc_end);
            output[idx] = input[in_idx];
        });
    });
    
    return result;
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
