#ifndef MIGRAPHX_GUARD_OPERATORS_SELU_HPP
#define MIGRAPHX_GUARD_OPERATORS_SELU_HPP

#include <array>
#include <migraphx/op/unary.hpp>
#include <migraphx/operation.hpp>
#include <migraphx/check_shapes.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/streamutils.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/shape_for_each.hpp>
#include <migraphx/config.hpp>
#include <cmath>
#include <utility>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct selu : unary<selu>
{
    auto apply() const
    {
        float alpha = 1.67326324;
        float scale = 1.05070098;
        return [&](auto x) { return x < 0 ? scale * alpha * std::exp(x) : scale * x; };
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
