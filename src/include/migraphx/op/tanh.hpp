#ifndef MIGRAPHX_GUARD_OPERATORS_TANH_HPP
#define MIGRAPHX_GUARD_OPERATORS_TANH_HPP

#include <migraphx/op/unary.hpp>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct tanh : unary<tanh>
{
    auto apply() const
    {
        return [](auto x) { return std::tanh(x); };
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
