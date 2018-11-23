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
    std::vector<hipEvent_t> events{};
    void finish() const
    {
        gpu_sync();
#if 0        
        for (auto i = 0; i < events.size(); ++i)
            hipEventDestroy(events[i]);
#endif        
    }
    void set_handle_ndx(int ndx) {handle_ndx = ndx;}
    int create_event()
    {
        hipEvent_t event;
        hipEventCreateWithFlags(&event, hipEventDisableTiming);
        events.push_back(event);
        return (events.size() - 1);
    }
    void record_event(int event, int stream)
    {
        hipStream_t s;
        miopenGetStream(&(*(handle[stream])), &s);
        hipEventRecord(events[event], s);
    }

    void wait_event(int stream, int event)
    {
        hipStream_t s;
        miopenGetStream(&(*(handle[stream])), &s);
        hipStreamWaitEvent(s, events[event], 0);
    }
    
};
} // namespace gpu
} // namespace migraph

#endif
