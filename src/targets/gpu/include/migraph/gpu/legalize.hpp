#ifndef MIGRAPH_GUARD_RTGLIB_MIOPEN_LEGALIZE_HPP
#define MIGRAPH_GUARD_RTGLIB_MIOPEN_LEGALIZE_HPP

#include <migraph/program.hpp>
#include <migraph/gpu/context.hpp>
#include <migraph/gpu/context.hpp>
#include <migraph/gpu/convolution.hpp>

namespace migraph {
namespace gpu {

struct legalize
{
    context* ctx = nullptr;
    std::string name() const { return "gpu::legalize"; }
    void apply(program& p) const;
};

} // namespace gpu

} // namespace migraph

#endif
