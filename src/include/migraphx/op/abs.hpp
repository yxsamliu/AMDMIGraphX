#ifndef MIGRAPHX_GUARD_OPERATORS_ABS_HPP
#define MIGRAPHX_GUARD_OPERATORS_ABS_HPP

#include <migraphx/op/unary.hpp>
#include <migraphx/config.hpp>
#include <migraphx/make_signed.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct abs : unary<abs>
{
    auto apply() const
    {
        return [](auto x) { return std::abs(make_signed(x)); };
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
