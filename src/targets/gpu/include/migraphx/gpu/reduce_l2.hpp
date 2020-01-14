#ifndef MIGRAPHX_GUARD_RTGLIB_REDUCE_L2_HPP
#define MIGRAPHX_GUARD_RTGLIB_REDUCE_L2_HPP

#include <migraphx/op/reduce_l2.hpp>
#include <migraphx/gpu/reduce_op.hpp>
#include <migraphx/gpu/device/reduce_l2.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct context;

struct hip_reduce_l2 : reduce_op<hip_reduce_l2, op::reduce_l2, device::reduce_l2>
{
    hip_reduce_l2() {}
    hip_reduce_l2(const op::reduce_l2& op_ref) : reduce_op(op_ref) {}
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
