#ifndef MIGRAPHX_GUARD_OPERATORS_MAX_HPP
#define MIGRAPHX_GUARD_OPERATORS_MAX_HPP

#include <migraphx/op/binary.hpp>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct max : binary<max>
{
    auto apply() const
    {
        return [](auto x, auto y) { return std::max(x, y); };
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
