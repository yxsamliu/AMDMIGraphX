#ifndef MIGRAPH_GUARD_RTGLIB_CONTEXT_HPP
#define MIGRAPH_GUARD_RTGLIB_CONTEXT_HPP

#include <migraph/gpu/miopen.hpp>
#include <migraph/gpu/rocblas.hpp>
#include <migraph/gpu/hip.hpp>

namespace migraph {
namespace gpu {

static int default_stream = 0;
struct context
{
    std::vector<shared<miopen_handle>> handle;
    shared<rocblas_handle_ptr> rbhandle;
    argument scratch;
    std::vector<argument> literals{};
    int handle_ndx = default_stream;
    void finish() const { gpu_sync(); }
    void set_handle_ndx(int ndx) {handle_ndx = ndx;}
    
};
} // namespace gpu
} // namespace migraph

#endif
