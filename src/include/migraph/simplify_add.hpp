#ifndef MIGRAPH_GUARD_RTGLIB_SIMPLIFY_ADD_HPP
#define MIGRAPH_GUARD_RTGLIB_SIMPLIFY_ADD_HPP

#include <string>
#include <migraph/instruction_ref.hpp>

namespace migraph {

struct program;

struct simplify_add
{
    std::string name() const { return "simplify_add"; }
    void apply(program& p) const;
};

} // namespace migraph

#endif
