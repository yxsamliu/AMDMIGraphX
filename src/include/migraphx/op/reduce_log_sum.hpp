#ifndef MIGRAPHX_GUARD_OPERATORS_REDUCE_LOG_SUM_HPP
#define MIGRAPHX_GUARD_OPERATORS_REDUCE_LOG_SUM_HPP

#include <migraphx/op/reduce_op.hpp>
#include <migraphx/make_signed.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct reduce_log_sum : reduce_op<reduce_log_sum>
{
    reduce_log_sum() {}
    reduce_log_sum(std::vector<int64_t> ax) : reduce_op(std::move(ax)) {}

    auto op() const
    {
        return [=](auto x, auto y) { return x + y; };
    }

    auto output(const shape&) const
    {
        return [=](auto val) { return std::log(make_signed(val)); };
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
