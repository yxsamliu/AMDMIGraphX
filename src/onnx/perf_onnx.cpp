
#include <migraphx/onnx.hpp>

#include <migraphx/gpu/target.hpp>
#include <migraphx/gpu/hip.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/quantization.hpp>
#include <migraphx/verify.hpp>

migraphx::program::parameter_map create_param_map(const migraphx::program& p, bool gpu = true)
{
    migraphx::program::parameter_map m;
    for(auto&& x : p.get_parameter_shapes())
    {
        if(gpu)
            m[x.first] = migraphx::gpu::to_gpu(migraphx::generate_argument(x.second));
        else
            m[x.first] = migraphx::generate_argument(x.second);
    }
    return m;
}

int main(int argc, char const* argv[])
{
    if (argc < 2)
    {
        std::cout << "Usage: " << argv[0] << " onnx n quant_flag" << std::endl;
        std::cout << "quant_flag: " << std::endl;
        std::cout << "      fp32   default" << std::endl;
        std::cout << "      fp16   fp16 quantization" << std::endl;
        std::cout << "      int8   int8 quantization" << std::endl;

        return 0;
    }

    std::string file = argv[1];
    std::size_t n    = argc > 2 ? std::stoul(argv[2]) : 50;
    auto p           = migraphx::parse_onnx(file);

    std::string quant_flag("fp32");
    if (argc == 4)
    {
        std::string quant_flag = argv[3];
    }

    if (quant_flag == "fp16")
    {
        std::cout << "Quantize to fp16 ... " << std::endl;
        migraphx::quantize(p);
    }
    else if (quant_flag == "int8")
    {
        std::cout << "Quantize to int8 ... " << std::endl;
        std::cout << "First, capture arguments to calculate scale ... " << std::endl;
        auto cap_p = p;
        migraphx::capture_arguments(cap_p);
    }
    std::cout << "Compiling ... " << std::endl;
    p.compile(migraphx::gpu::target{});
    std::cout << "Allocating params ... " << std::endl;
    auto m = create_param_map(p);
    std::cout << "Running performance report ... " << std::endl;
    p.perf_report(std::cout, n, m);
}
