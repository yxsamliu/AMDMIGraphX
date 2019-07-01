#ifndef MIGRAPHX_GUARD_OPERATORS_EXP_HPP
#define MIGRAPHX_GUARD_OPERATORS_EXP_HPP

#include <migraphx/op/unary.hpp>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct exp : unary<exp>
{
    auto apply() const
    {
        return [](auto x) { return std::exp(x); };
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
