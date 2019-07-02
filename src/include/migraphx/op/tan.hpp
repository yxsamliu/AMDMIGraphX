#ifndef MIGRAPHX_GUARD_OPERATORS_TAN_HPP
#define MIGRAPHX_GUARD_OPERATORS_TAN_HPP

#include <migraphx/op/unary.hpp>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct tan : unary<tan>
{
    auto apply() const
    {
        return [](auto x) { return std::tan(x); };
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
