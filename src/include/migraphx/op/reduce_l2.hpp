#ifndef MIGRAPHX_GUARD_OPERATORS_REDUCE_L2_HPP
#define MIGRAPHX_GUARD_OPERATORS_REDUCE_L2_HPP

#include <migraphx/op/reduce_op.hpp>
#include <migraphx/make_signed.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct reduce_l2 : reduce_op<reduce_l2>
{
    reduce_l2() {}
    reduce_l2(std::vector<int64_t> ax) : reduce_op(std::move(ax)) {}

    auto op() const
    {
        return [=](auto x, auto y) { return x + y; };
    }

    auto input() const
    {
        return [=](auto val) { return val * val; };
    }

    auto output(const shape&) const
    {
        return [=](auto val) { return std::sqrt(make_signed(val)); };
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
