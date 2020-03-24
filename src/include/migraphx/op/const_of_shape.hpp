#ifndef MIGRAPHX_GUARD_OPERATORS_CONST_OF_SHAPE_HPP
#define MIGRAPHX_GUARD_OPERATORS_CONST_OF_SHAPE_HPP

#include <migraphx/operation.hpp>
#include <migraphx/check_shapes.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/streamutils.hpp>
#include <migraphx/shape_for_each.hpp>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct const_of_shape
{
    std::string name() const { return "constant_of_shape"; }

    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(2);
        auto type = inputs[0].type();
        if (inputs[1].elements() == 0)
        {
            return {type};
        }
        else
        {
            return {type, inputs[1].lens()};           
        }
    }

    argument compute(const shape& output_shape, std::vector<argument> args) const
    {
        argument result{output_shape};
        result.visit([&](auto out) {
            using type = typename decltype(output)::value_type;
            type val = args[1].at<type>();
            std::fill(out.begin(), out.end(), val);
        });

        return result;
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
