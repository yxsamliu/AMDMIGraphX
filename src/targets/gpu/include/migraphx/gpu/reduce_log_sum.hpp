#ifndef MIGRAPHX_GUARD_RTGLIB_REDUCE_LOG_SUM_HPP
#define MIGRAPHX_GUARD_RTGLIB_REDUCE_LOG_SUM_HPP

#include <migraphx/op/reduce_log_sum.hpp>
#include <migraphx/gpu/reduce_op.hpp>
#include <migraphx/gpu/device/reduce_log_sum.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct context;

struct hip_reduce_log_sum : reduce_op<hip_reduce_log_sum, op::reduce_log_sum, device::reduce_log_sum>
{
    hip_reduce_log_sum() {}
    hip_reduce_log_sum(const op::reduce_log_sum& op_ref) : reduce_op(op_ref) {}
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
