#ifndef MIGRAPHX_GUARD_RTGLIB_PROPOGATE_COPY_HPP
#define MIGRAPHX_GUARD_RTGLIB_PROPOGATE_COPY_HPP

#include <string>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct program;

/**
 * Propagate copies when they are only used once
 */
struct propagate_copy
{
    std::string copy;
    std::string name() const { return "propagate_copy"; }
    void apply(program& p) const;
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
