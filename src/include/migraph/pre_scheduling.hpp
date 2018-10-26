#ifndef MIGRAPH_GUARD_RTGLIB_PRE_SCHEDULING_HPP
#define MIGRAPH_GUARD_RTGLIB_PRE_SCHEDULING_HPP

#include <string>
#include <migraph/instruction_ref.hpp>

namespace migraph {
struct program;

struct pre_scheduling
{
    std::function<float(std::string&)> weight_func;
    std::string name() const { return "pre scheduling"; }
    void apply(program& p) const;
};
} // namespace migraph

#endif
