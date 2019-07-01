#ifndef MIGRAPHX_GUARD_OPERATORS_ASIN_HPP
#define MIGRAPHX_GUARD_OPERATORS_ASIN_HPP

#include <migraphx/op/unary.hpp>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct asin : unary<asin>
{
    auto apply() const
    {
        return [](auto x) { return std::asin(x); };
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
