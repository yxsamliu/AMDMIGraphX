#ifndef MIGRAPH_GUARD_MIGRAPHLIB_MIOPEN_MACHINE_MODEL_HPP
#define MIGRAPH_GUARD_MIGRAPHLIB_MIOPEN_MACHINE_MODEL_HPP
#include <string>
#include <unordered_map>

namespace migraph {
namespace gpu {

struct op_weight
{
    op_weight()
    {
        weight_map["convolution"] = 4.0f;
        weight_map["pooling"]     = 2.0f;
        weight_map["gemm"]        = 2.0f;
        weight_map["@param"]      = 0.0f;
        weight_map["@literal"]    = 4.0f;
        weight_map["hip::allocate"] = 0.0f;
        weight_map["@outline"] = 0.0f;
        weight_map["gpu::convolution"] = 4.0f;
        weight_map["gpu::pooling"]     = 2.0f;
        weight_map["gpu::gemm"]        = 2.0f;
        weight_map["hip::add_relu"]    = 2.0f;
    }

    float operator()(const std::string& op)
    {
        if(weight_map.find(op) != weight_map.end())
        {
            return weight_map[op];
        }
        else
        {
            return 1.0f;
        }
    }
    std::unordered_map<std::string, float> weight_map;
};
} // namespace gpu
} // namespace migraph

#endif
