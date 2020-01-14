#ifndef MIGRAPHX_GUARD_RTGLIB_REDUCE_SUM_SQUARE_HPP
#define MIGRAPHX_GUARD_RTGLIB_REDUCE_SUM_SQUARE_HPP

#include <migraphx/op/reduce_sum_square.hpp>
#include <migraphx/gpu/reduce_op.hpp>
#include <migraphx/gpu/device/reduce_sum_square.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct context;

struct hip_reduce_sum_square : reduce_op<hip_reduce_sum_square, op::reduce_sum_square, device::reduce_sum_square>
{
    hip_reduce_sum_square() {}
    hip_reduce_sum_square(const op::reduce_sum_square& op_ref) : reduce_op(op_ref) {}
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
