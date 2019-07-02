#ifndef MIGRAPHX_GUARD_OPERATORS_SINH_HPP
#define MIGRAPHX_GUARD_OPERATORS_SINH_HPP

#include <migraphx/op/unary.hpp>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct sinh : unary<sinh>
{
    auto apply() const
    {
        return [](auto x) { return std::sinh(x); };
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
