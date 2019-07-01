#ifndef MIGRAPHX_GUARD_OPERATORS_IDENTITY_HPP
#define MIGRAPHX_GUARD_OPERATORS_IDENTITY_HPP

#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct identity
{
    std::string name() const { return "identity"; }
    shape compute_shape(std::vector<shape> inputs) const { return inputs.at(0); }
    argument compute(shape output_shape, std::vector<argument> args) const
    {
        return {std::move(output_shape), std::move(args.at(0).data)};
    }
    std::ptrdiff_t output_alias(const std::vector<shape>&) const { return 0; }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
