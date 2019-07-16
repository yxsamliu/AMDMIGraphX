#ifndef MIGRAPHX_GUARD_RTGLIB_TIME_HPP
#define MIGRAPHX_GUARD_RTGLIB_TIME_HPP

#include <chrono>
#include <migraphx/config.hpp>
#include <sys/time.h>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

template <class Duration, class F>
auto time(F f)
{
    auto start = std::chrono::steady_clock::now();
    f();
    auto finish = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<Duration>(finish - start).count();
}

using namespace std::chrono;
struct ms_timer
{
    void start()
    {
        //t1 = high_resolution_clock::now();
        gettimeofday(&t1, nullptr);
    }

    void end()
    {
        //t2 = high_resolution_clock::now();
        gettimeofday(&t2, nullptr);
    }

    float get_ms()
    {
        //duration<float, std::milli> fp_ms = t2 - t1;
        //return fp_ms.count();
        return 1000.0f * (t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec) / 1000.0f;
    }

private:
    // high_resolution_clock::time_point t1, t2;
    timeval t1, t2;
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
