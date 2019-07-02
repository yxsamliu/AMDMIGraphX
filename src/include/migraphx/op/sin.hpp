#ifndef MIGRAPHX_GUARD_OPERATORS_SIN_HPP
#define MIGRAPHX_GUARD_OPERATORS_SIN_HPP

#include <migraphx/op/unary.hpp>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct sin : unary<sin>
{
    auto apply() const
    {
        return [](auto x) { return std::sin(x); };
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
