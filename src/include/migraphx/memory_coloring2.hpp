#ifndef MIGRAPH_GUARD_RTGLIB_MEMORY_COLORING2_HPP
#define MIGRAPH_GUARD_RTGLIB_MEMORY_COLORING2_HPP

#include <string>
#include <migraph/instruction_ref.hpp>

namespace migraph {
struct program;

struct memory_coloring2
{
    std::string allocation_op{};
    std::string name() const { return "memory_coloring2"; }
    void apply(program& p) const;
};

} // namespace migraph

#endif
