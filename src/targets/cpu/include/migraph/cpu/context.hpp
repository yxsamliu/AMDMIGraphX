#ifndef MIGRAPH_GUARD_RTGLIB_CONTEXT_HPP
#define MIGRAPH_GUARD_RTGLIB_CONTEXT_HPP

namespace migraph {
namespace cpu {

struct context
{
    void finish() const {}
    void set_handle_ndx(int ndx) { handle_ndx = ndx;}
    int  create_event() { return -1; }
    void record_event(int event, int stream) {}
    void wait_event(int stream, int event) {}
    int handle_ndx = 0;
};

} // namespace cpu
} // namespace migraph

#endif
