#ifndef MIGRAPHX_GUARD_OPERATORS_MIN_HPP
#define MIGRAPHX_GUARD_OPERATORS_MIN_HPP

#include <migraphx/op/binary.hpp>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct min : binary<min>
{
    auto apply() const
    {
        return [](auto x, auto y) { return std::min(x, y); };
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
