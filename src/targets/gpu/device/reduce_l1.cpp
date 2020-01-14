#include <migraphx/gpu/device/reduce_l1.hpp>
#include <migraphx/gpu/device/reduce.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

void reduce_l1(hipStream_t stream, const argument& result, const argument& arg)
{

    reduce(stream, result, arg, sum{}, 0, abs{}, id{});
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
