#include <migraphx/gpu/device/erf_factor.hpp>
#include <migraphx/gpu/device/nary.hpp>
#include <migraphx/gpu/device/types.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

void erf_factor(hipStream_t stream, const argument& result, const argument& arg, float factor)
{
    nary(stream, result, arg)([=](auto x) { return ::erf(to_hip_type(x * factor)); });
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
