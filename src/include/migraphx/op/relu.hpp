#ifndef MIGRAPHX_GUARD_OPERATORS_RELU_HPP
#define MIGRAPHX_GUARD_OPERATORS_RELU_HPP

#include <migraphx/op/unary.hpp>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct relu : unary<relu>
{
    auto apply() const
    {
        return [](auto x) { return std::max(decltype(x){0}, x); };
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
