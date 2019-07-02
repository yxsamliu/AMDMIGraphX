#ifndef MIGRAPHX_GUARD_OPERATORS_LOG_HPP
#define MIGRAPHX_GUARD_OPERATORS_LOG_HPP

#include <migraphx/op/unary.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct log : unary<log>
{
    auto apply() const
    {
        return [](auto x) { return std::log(x); };
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
