#ifndef MIGRAPHX_GUARD_OPERATORS_NEG_HPP
#define MIGRAPHX_GUARD_OPERATORS_NEG_HPP

#include <migraphx/op/unary.hpp>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct neg : unary<neg>
{
    auto apply() const
    {
        return [](auto x) { return -x; };
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
