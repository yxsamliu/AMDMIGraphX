#ifndef MIGRAPHX_GUARD_OPERATORS_ATAN_HPP
#define MIGRAPHX_GUARD_OPERATORS_ATAN_HPP

#include <migraphx/op/unary.hpp>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct atan : unary<atan>
{
    auto apply() const
    {
        return [](auto x) { return std::atan(x); };
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
