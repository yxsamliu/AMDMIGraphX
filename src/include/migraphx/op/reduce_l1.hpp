#ifndef MIGRAPHX_GUARD_OPERATORS_REDUCE_L1_HPP
#define MIGRAPHX_GUARD_OPERATORS_REDUCE_L1_HPP

#include <migraphx/op/reduce_op.hpp>
#include <migraphx/make_signed.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct reduce_l1 : reduce_op<reduce_l1>
{
    reduce_l1() {}
    reduce_l1(std::vector<int64_t> ax) : reduce_op(std::move(ax)) {}

    auto op() const
    {
        return [=](auto x, auto y) { return x + y; };
    }

    auto input() const
    {
        return [=](auto val) { return std::abs(make_signed(val)); };
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
