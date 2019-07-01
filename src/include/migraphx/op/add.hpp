#ifndef MIGRAPHX_GUARD_OPERATORS_ADD_HPP
#define MIGRAPHX_GUARD_OPERATORS_ADD_HPP

#include <migraphx/op/binary.hpp>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct add : binary<add>
{
    auto apply() const
    {
        return [](auto x, auto y) { return x + y; };
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
