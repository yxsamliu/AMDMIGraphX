#include <migraphx/shape.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/gpu/device/concat.hpp>
#include <migraphx/gpu/device/tensor.hpp>
#include <migraphx/gpu/device/launch.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

template <class F>
void visit_concat_sizes(std::size_t n, F f)
{
    switch(n)
    {
    case 2:
    {
        f(std::integral_constant<std::size_t, 2>{});
        break;
    }
    case 3:
    {
        f(std::integral_constant<std::size_t, 3>{});
        break;
    }
    case 4:
    {
        f(std::integral_constant<std::size_t, 4>{});
        break;
    }
    case 5:
    {
        f(std::integral_constant<std::size_t, 5>{});
        break;
    }
    case 6:
    {
        f(std::integral_constant<std::size_t, 6>{});
        break;
    }
    default: throw std::runtime_error("Unknown concat size");
    }
}

argument concat(hipStream_t stream,
                const migraphx::shape&,
                std::vector<migraphx::argument> args,
                std::vector<std::size_t> offsets)
{
    auto ninputs = args.size() - 1;
    assert(offsets.size() == ninputs);
    if(ninputs > 6 or ninputs < 2)
    {
        for(std::size_t j = 0; j < ninputs; j++)
        {
            auto&& arg            = args[j];
            std::size_t nelements = arg.get_shape().elements();
            auto offset           = offsets[j];
            shape arg_shape{arg.get_shape().type(), arg.get_shape().lens()};
            hip_visit_all(args.back(), arg, arg_shape)(
                [&](auto output, auto input, auto input_shape) {
                    gs_launch(stream, nelements)([=](auto i) {
                        auto input_idx              = input_shape.multi(i);
                        auto idx                    = output.get_shape().index(input_idx);
                        output.data()[idx + offset] = input[input_idx];
                    });
                });
        }
    }
    else
    {
        auto nelements =
            std::max_element(
                args.begin(),
                args.end(),
                [](auto x, auto y) { return x.get_shape().elements() < y.get_shape().elements(); })
                ->get_shape()
                .elements();
        visit_concat_sizes(ninputs, [&](auto n) {
            hip_array<argument, n> dargs;
            std::copy(args.begin(), args.end(), dargs.begin());
            hip_array<std::size_t, n> doffsets;
            std::copy(offsets.begin(), offsets.end(), doffsets.begin());
            hip_array<shape, n> arg_shapes;
            std::transform(args.begin(), args.end(), arg_shapes.begin(), [](auto&& arg) {
                return shape{arg.get_shape().type(), arg.get_shape().lens()};
            });
            hip_visit_all(args.back(), dargs, arg_shapes)(
                [&](auto output, auto inputs, auto input_shapes) {
                    launch(stream, nelements, 256)([=](auto idx) {
                        for(std::size_t j = 0; j < n; j++)
                        {
                            auto&& arg                 = args[j];
                            auto&& input_shape         = input_shapes[j];
                            auto&& input               = inputs[j];
                            auto offset                = offsets[j];
                            std::size_t local_elements = arg.get_shape().elements();
                            idx.global_stride(local_elements, [&](auto i) {
                                auto input_idx = input_shape.multi(i);
                                auto out_idx   = output.get_shape().index(input_idx);
                                output.data()[out_idx + offset] = input[input_idx];
                            });
                        }
                    });
                });
        });
    }
    return args.back();
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
