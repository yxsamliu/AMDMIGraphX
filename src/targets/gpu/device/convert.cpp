#include <migraphx/gpu/device/convert.hpp>
#include <migraphx/gpu/device/nary.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

void convert(hipStream_t stream,
             const argument& result,
             const argument& arg,
             float scale,
             float shift,
             shape::type_t target_type)
{
    result.visit([&](auto output) {
        arg.visit([&](auto input) {
            const auto* input_ptr = device_cast(input.data());
            auto* output_ptr      = device_cast(output.data());
            if(target_type == shape::int8_type)
            {
                // gs_launch(stream, result.get_shape().elements())(
                //     [=](auto i) {
                //         output_ptr[i] = ::min(::max(to_hip_type(-128.0), to_hip_type(input_ptr[i]
                //         * scale + shift)), to_hip_type(127));
                //     });
                gs_launch(stream, result.get_shape().elements())(
                    [=](auto i) { output_ptr[i] = input_ptr[i] * scale + shift; });
            }
            else
            {
                gs_launch(stream, result.get_shape().elements())(
                    [=](auto i) { output_ptr[i] = input_ptr[i] * scale + shift; });
            }
        });
    });
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
