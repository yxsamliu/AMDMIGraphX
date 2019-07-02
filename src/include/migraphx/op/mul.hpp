#ifndef MIGRAPHX_GUARD_OPERATORS_MUL_HPP
#define MIGRAPHX_GUARD_OPERATORS_MUL_HPP

#include <migraphx/op/binary.hpp>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct mul : binary<mul>
{
    auto apply() const
    {
        return [](auto x, auto y) { return x * y; };
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
