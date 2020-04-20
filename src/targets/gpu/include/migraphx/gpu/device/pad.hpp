
#ifndef MIGRAPHX_GUARD_RTGLIB_DEVICE_PAD_HPP
#define MIGRAPHX_GUARD_RTGLIB_DEVICE_PAD_HPP

#include <migraphx/argument.hpp>
#include <migraphx/config.hpp>
#include <hip/hip_runtime_api.h>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

argument pad_constant(hipStream_t stream,
             argument result,
             const std::vector<argument>& args,
             const std::vector<std::int64_t>& pads);
argument pad_edge(hipStream_t stream,
             argument result,
             const std::vector<argument>& args,
             const std::vector<std::int64_t>& pads);
argument pad_reflect(hipStream_t stream,
             argument result,
             const std::vector<argument>& args,
             const std::vector<std::int64_t>& pads);
} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
