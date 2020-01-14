#ifndef MIGRAPHX_GUARD_RTGLIB_REDUCE_L1_HPP
#define MIGRAPHX_GUARD_RTGLIB_REDUCE_L1_HPP

#include <migraphx/op/reduce_l1.hpp>
#include <migraphx/gpu/reduce_op.hpp>
#include <migraphx/gpu/device/reduce_l1.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct context;

struct hip_reduce_l1 : reduce_op<hip_reduce_l1, op::reduce_l1, device::reduce_l1>
{
    hip_reduce_l1() {}
    hip_reduce_l1(const op::reduce_l1& op_ref) : reduce_op(op_ref) {}
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
