#ifndef MIGRAPH_GUARD_CONTEXT_HPP
#define MIGRAPH_GUARD_CONTEXT_HPP

#include <cassert>
#include <string>
#include <functional>
#include <memory>
#include <type_traits>
#include <utility>

namespace migraph {

#ifdef DOXYGEN

/// A context is used to store internal data for a `target`. A context is
/// constructed by a target during compilation and passed to the operations
/// during `eval`.
struct context
{
    /// Wait for any tasks in the context to complete
    void finish() const;
    void set_handle_ndx(int);
    int create_event();
    void record_event(int, int);
    void wait_event(int, int);
};

#else

<%
interface('context',
    virtual('finish', returns='void', const=True)
    virtual('set_handle_ndx', returns='void', input='int', const=False)
)
%>

#endif

} // namespace migraph

#endif
