#include <migraph/gpu/legalize.hpp>
#include <migraph/pass_config.hpp>
#include <migraph/iterator_for.hpp>
namespace migraph {
namespace gpu {

void legalize::apply(program& prog) const
{
    if(enabled(MIGRAPH_DISABLE_PRE_SCHEDULING{}))
        return;
    return;
    int ndx = 0;
    std::unordered_map<int, int> stream2_ndx;
    std::unordered_map<instruction_ref, int> ins2_ndx;
    for(auto it : iterator_for(prog))
    {
        ins2_ndx[it] = ndx;
        int stream = it->get_stream();
        for(auto&& arg : it->inputs())
        {
            int arg_s = arg->get_stream();
            if ((arg_s == default_stream) || (arg_s == stream))
                continue;
            int arg_ndx = ins2_ndx[arg];
            if (stream2_ndx.find(arg_s) != stream2_ndx.end())
            {
                int last_sync = stream2_ndx[arg_s];
                if (last_sync >= arg_ndx)
                    continue;
            }
            hipStream_t hip_s;
            miopenGetStream(&(*(ctx->handle[arg_s])), &hip_s);
            prog.insert_instruction(it, hip_stream_sync{hip_s});
            stream2_ndx[arg_s] = ndx;
        }
        ndx++;
    }
    //    std::cout << prog << std::endl;
};

} // namespace gpu
} // namespace migraph
