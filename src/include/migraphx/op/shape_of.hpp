#ifndef MIGRAPHX_GUARD_OPERATORS_SHAPE_OF_HPP
#define MIGRAPHX_GUARD_OPERATORS_SHAPE_OF_HPP

#include <migraphx/operation.hpp>
#include <migraphx/check_shapes.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/streamutils.hpp>
#include <migraphx/shape_for_each.hpp>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct shape_of
{
    std::string name() const { return "shape"; }

    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(1);
        return {shape::int64_type, {inputs[0].lens().size()}};
    }

    argument compute(const shape& output_shape, std::vector<argument> args) const
    {
        auto lens = args[0].get_shape().lens();
        argument result{output_shape};
        result.visit([&](auto output) {
            std::copy(lens.begin(), lens.end(), output.begin());
        });

        return result;
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
