#ifndef MIGRAPHX_GUARD_OPERATORS_SUB_HPP
#define MIGRAPHX_GUARD_OPERATORS_SUB_HPP

#include <migraphx/op/binary.hpp>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct sub : binary<sub>
{
    auto apply() const
    {
        return [](auto x, auto y) { return x - y; };
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
