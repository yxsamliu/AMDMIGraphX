#ifndef MIGRAPH_GUARD_RTGLIB_CONTEXT_HPP
#define MIGRAPH_GUARD_RTGLIB_CONTEXT_HPP

#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPH_INLINE_NS {
namespace cpu {

struct context
{
    void finish() {}
    void set_stream(int) {}
    void add_stream() {}
    int  create_event() { return -1; }
    void record_event(int, int) {}
    void wait_event(int, int) {}
    void destroy(){}
};

} // namespace cpu
} // namespace MIGRAPH_INLINE_NS
} // namespace migraphx

#endif
