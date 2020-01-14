#ifndef MIGRAPHX_GUARD_RTGLIB_REDUCE_LOG_SUM_EXP_HPP
#define MIGRAPHX_GUARD_RTGLIB_REDUCE_LOG_SUM_EXP_HPP

#include <migraphx/op/reduce_log_sum_exp.hpp>
#include <migraphx/gpu/reduce_op.hpp>
#include <migraphx/gpu/device/reduce_log_sum_exp.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct context;

struct hip_reduce_log_sum_exp
    : reduce_op<hip_reduce_log_sum_exp, op::reduce_log_sum_exp, device::reduce_log_sum_exp>
{
    hip_reduce_log_sum_exp() {}
    hip_reduce_log_sum_exp(const op::reduce_log_sum_exp& op_ref) : reduce_op(op_ref) {}
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
