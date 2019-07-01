#ifndef MIGRAPHX_GUARD_OPERATORS_COS_HPP
#define MIGRAPHX_GUARD_OPERATORS_COS_HPP

#include <migraphx/op/unary.hpp>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct cos : unary<cos>
{
    auto apply() const
    {
        return [](auto x) { return std::cos(x); };
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
