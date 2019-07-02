#ifndef MIGRAPHX_GUARD_OPERATORS_SIGMOID_HPP
#define MIGRAPHX_GUARD_OPERATORS_SIGMOID_HPP

#include <migraphx/op/unary.hpp>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct sigmoid : unary<sigmoid>
{
    auto apply() const
    {
        return [](auto x) { return 1.f / (1.f + std::exp(-x)); };
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
