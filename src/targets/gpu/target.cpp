#include <migraph/gpu/target.hpp>
#include <migraph/gpu/lowering.hpp>
#include <migraph/memory_coloring.hpp>
#include <migraph/gpu/lowering_memory_coloring.hpp>
#include <migraph/gpu/write_literals.hpp>
#include <migraph/gpu/context.hpp>
#include <migraph/gpu/eliminate_workspace.hpp>
#include <migraph/eliminate_allocation.hpp>
#include <migraph/gpu/fuse_ops.hpp>
#include <migraph/check_context.hpp>
#include <migraph/auto_contiguous.hpp>
#include <migraph/dead_code_elimination.hpp>
#include <migraph/simplify_reshapes.hpp>
#include <migraph/eliminate_contiguous.hpp>
#include <migraph/fwd_conv_batchnorm_rewrite.hpp>
#include <migraph/pre_scheduling.hpp>
#include <migraph/gpu/machine_model.hpp>
#include <migraph/gpu/legalize.hpp>

namespace migraph {
namespace gpu {

std::vector<pass> target::get_passes(migraph::context& gctx) const
{
    auto& ctx                                      = any_cast<context>(gctx);
    std::function<std::pair<int,int>(std::string&)> weight_func = op_info();
    int num_of_streams = stream_info().num_of_streams();
    // clang-format off
    return
    {
        dead_code_elimination{},
        fwd_conv_batchnorm_rewrite{},
        dead_code_elimination{},
        auto_contiguous{},
        simplify_reshapes{},
        dead_code_elimination{},
        pre_scheduling{weight_func, num_of_streams},            
        lowering{ctx},
            //        fuse_ops{&ctx},
        dead_code_elimination{},
        eliminate_contiguous{},
        dead_code_elimination{},
        legalize{&ctx},            
        memory_coloring{"hip::allocate"},
        lowering_memory_coloring{&ctx},
        write_literals{&ctx},
        eliminate_workspace{},
        eliminate_allocation{"hip::allocate"},
        check_context<context>{},
        dead_code_elimination{}
    };
    // clang-format on
}

std::string target::name() const { return "miopen"; }

migraph::context target::get_context() const
{
    std::function<std::pair<int,int>(std::string&)> weight_func = op_info();
    int num_of_streams = stream_info().num_of_streams();
    std::vector<shared<miopen_handle>> handles;
    handles.push_back(share(make_obj<miopen_handle>(&miopenCreate)));
    for (int i = 0; i < num_of_streams; ++i)
    {
        hipStream_t s = nullptr;
        handles.push_back(share(make_obj<miopen_handle>(&miopenCreateWithStream, s)));
    }
    return context{
        handles, share(create_rocblas_handle_ptr()), {}};
}
} // namespace gpu
} // namespace migraph
