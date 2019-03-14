#include <migraphx/gpu/split.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/device/split.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

shape hip_split::compute_shape(std::vector<shape> inputs) const
{
    inputs.pop_back();
    check_shapes{inputs, *this}.has(2);
    auto input_shape = inputs[0];
    std::vector<std::size_t> out_dims;
    out_dims.push_back(input_shape.elements());
    return {input_shape.type(), out_dims};
}

argument hip_split::compute(context& ctx,
                            const shape& output_shape,
                            const std::vector<argument>& args) const
{
    auto arg0 = args[0];
    return device::split(ctx.get_stream().get(), output_shape, args);
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
