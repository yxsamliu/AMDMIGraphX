#ifndef MIGRAPHX_GUARD_RTGLIB_LOGSOFTMAX_HPP
#define MIGRAPHX_GUARD_RTGLIB_LOGSOFTMAX_HPP

#include <migraphx/op/logsoftmax.hpp>
#include <migraphx/config.hpp>
#include <migraphx/gpu/context.hpp>
#include <utility>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct hip_logsoftmax
{
    op::logsoftmax op;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return migraphx::reflect(self.op, f);
    }

    std::string name() const { return "gpu::logsoftmax"; }
    shape compute_shape(const std::vector<shape>& inputs) const;
    argument
    compute(context& ctx, const shape& output_shape, const std::vector<argument>& args) const;
    std::ptrdiff_t output_alias(const std::vector<shape>& shapes) const
    {
        return shapes.size() - 1;
    }
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
