#include <migraphx/gpu/pad.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/device/pad.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

shape hip_pad::compute_shape(std::vector<shape> inputs) const
{
    inputs.pop_back();
    // it could be 1 or 2 inputs
    check_shapes{inputs, *this}.standard();
    return op.compute_shape(inputs);
}

argument hip_pad::compute(context& ctx, const shape&, const std::vector<argument>& args) const
{
    // the input pad value is availabel
    std:vector<argument> inputs(args);
    inputs.pop_back();

    if (op.mode == constant_pad)
    {
        return device::pad_constant(ctx.get_stream().get(), args.back(), inputs, op.pads);
    }
    else if (op.mode == reflect_pad)
    {
        return device::pad_reflect(ctx.get_stream().get(), args.back(), inputs, op.pads);
    }
    else
    {
        return device::pad_edge(ctx.get_stream().get(), args.back(), inputs, op.pads);
    }
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
