
#include <migraph/onnx.hpp>
#include <migraph/gpu/target.hpp>

int main(int argc, char const* argv[])
{
    if(argc > 1)
    {
        std::string file = argv[1];
        auto prog        = migraph::parse_onnx(file);
        std::cout << prog << std::endl;
        prog.compile(migraph::gpu::target{});
        std::cout << prog << std::endl;
    }
}
