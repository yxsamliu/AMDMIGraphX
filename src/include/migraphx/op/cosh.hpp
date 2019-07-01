#ifndef MIGRAPHX_GUARD_OPERATORS_COSH_HPP
#define MIGRAPHX_GUARD_OPERATORS_COSH_HPP

#include <migraphx/op/unary.hpp>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct cosh : unary<cosh>
{
    auto apply() const
    {
        return [](auto x) { return std::cosh(x); };
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
