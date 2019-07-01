#ifndef MIGRAPHX_GUARD_OPERATORS_ACOS_HPP
#define MIGRAPHX_GUARD_OPERATORS_ACOS_HPP

#include <migraphx/op/unary.hpp>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct acos : unary<acos>
{
    auto apply() const
    {
        return [](auto x) { return std::acos(x); };
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
