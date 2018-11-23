
#include <migraph/onnx.hpp>

#include <migraph/gpu/target.hpp>
#include <migraph/gpu/hip.hpp>
#include <migraph/generate.hpp>
#include <migraph/verify.hpp>
#include <chrono>
#include <chrono>
using namespace std::chrono;

migraph::program::parameter_map create_param_map(const migraph::program& p, bool gpu = true)
{
    migraph::program::parameter_map m;
    for(auto&& x : p.get_parameter_shapes())
    {
        if(gpu)
            m[x.first] = migraph::gpu::to_gpu(migraph::generate_argument(x.second));
        else
            m[x.first] = migraph::generate_argument(x.second);
    }
    return m;
}

int main(int argc, char const* argv[])
{
    if(argc > 1)
    {
        std::string file = argv[1];
        std::size_t n    = argc > 2 ? std::stoul(argv[2]) : 50;
        auto p           = migraph::parse_onnx(file);
        std::cout << "Compiling ... " << std::endl;
        
        p.compile(migraph::gpu::target{});
        std::cout << "Allocating params ... " << std::endl;
        auto m = create_param_map(p);
        std::cout << "Running performance report ... " << std::endl;
        auto start = std::chrono::steady_clock::now();
#if 0
        p.perf_report(std::cout, n, m);
#else
        for (auto i = 0; i < 10; i++)
            p.eval(m);
#endif        
        auto finish = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<milliseconds>(finish - start).count();
        std:: cout << "elapsed time: " << duration << std::endl; 
    }
}
