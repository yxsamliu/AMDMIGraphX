
#include <migraphx/onnx.hpp>

#include <migraphx/gpu/target.hpp>
#include <migraphx/cpu/target.hpp>
#include <migraphx/gpu/hip.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/quantization.hpp>
#include <migraphx/verify.hpp>

migraphx::program::parameter_map create_param_map(const migraphx::program& p)
{
    migraphx::program::parameter_map m;
    for(auto&& x : p.get_parameter_shapes())
    {
        m[x.first] = migraphx::generate_argument(x.second);
    }
    return m;
}

int main(int argc, char const* argv[])
{
    if(argc < 2)
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
    if(argc == 4)
    {
        quant_flag = argv[3];
    }

    if(!quant_flag.compare("fp16"))
    {
        std::cout << "Quantize to fp16 ... " << std::endl;
        migraphx::quantize_fp16(p);
    }
    else if(!quant_flag.compare("int8"))
    {
        std::cout << "Quantize to int8 ... " << std::endl;
        auto m                                             = create_param_map(p);
        migraphx::target t                                 = migraphx::gpu::target{};
        std::vector<migraphx::program::parameter_map> cali = {m};
        migraphx::quantize_int8(p, t, cali);
    }

    std::cout << "Compiling ... " << std::endl;
    p.compile(migraphx::gpu::target{});
    std::cout << "Allocating params ... " << std::endl;
    auto m = create_param_map(p);
    std::cout << "Running performance report ... " << std::endl;
    p.perf_report(std::cout, n, m);
}
