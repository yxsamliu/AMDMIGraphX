#include <migraph/gpu/legalize.hpp>

namespace migraph {
namespace gpu {

void legalize::apply(program& prog) const
{
    //    if(!enabled(MIGRAPH_DISABLE_PRE_SCHEDULING{}))
        return;
    
    for(auto it = prog.begin(); it != prog.end(); it++)
    {
        int stream = it->get_stream();
        if (stream == default_stream)
            continue;
        
        for(auto&& arg : it->inputs())
        {
            int arg_s = arg->get_stream();
            if (arg_s == default_stream)
                continue;
            if (arg_s != stream) {
                hipStream_t hip_s;
                miopenGetStream(&(*(ctx->handle[stream])), &hip_s);
                prog.insert_instruction(it, hip_stream_sync{hip_s});
                break;
            }
        }
    }
};

} // namespace gpu
} // namespace migraph
