#include <migraph/gpu/legalize.hpp>

namespace migraph {
namespace gpu {

void legalize::apply(program& prog) const
{
    for(auto it = prog.begin(); it != prog.end(); it++)
    {
        if (it->get_stream() == default_stream)
            continue;
        if(it->name() == "gpu::convolution")
        {
            auto op = any_cast<miopen_convolution>(it->get_operator());

        }
    }
};

} // namespace gpu
} // namespace migraph
