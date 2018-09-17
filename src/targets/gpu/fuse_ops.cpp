#include <migraph/gpu/fuse_ops.hpp>
#include <migraph/iterator_for.hpp>
#include <migraph/gpu/device/add_relu.hpp>
#include <migraph/gpu/device/add.hpp>
#include <migraph/instruction.hpp>

namespace migraph {

namespace gpu {

struct hip_add_relu
{
    std::string name() const { return "hip::add_relu"; }
    shape compute_shape(const std::vector<shape>& inputs) const
    {
        check_shapes{inputs, *this}.has(3);
        return inputs.front();
    }
    argument compute(context&, const shape&, const std::vector<argument>& args) const
    {
        device::add_relu(args.at(2), args.at(0), args.at(1));
        return args.at(2);
    }
};

struct hip_triadd
{
    std::string name() const { return "hip::triadd"; }
    shape compute_shape(const std::vector<shape>& inputs) const
    {
        check_shapes{inputs, *this}.has(4);
        return inputs.front();
    }
    argument compute(context&, const shape&, const std::vector<argument>& args) const
    {
        device::add(args.at(3), args.at(0), args.at(1), args.at(2));
        return args.at(3);
    }
};

struct hip_triadd_relu
{
    std::string name() const { return "hip::triadd_relu"; }
    shape compute_shape(const std::vector<shape>& inputs) const
    {
        check_shapes{inputs, *this}.has(4);
        return inputs.front();
    }
    argument compute(context&, const shape&, const std::vector<argument>& args) const
    {
        device::add_relu(args.at(3), args.at(0), args.at(1), args.at(2));
        return args.at(3);
    }
};

void fuse_ops::apply(program& p) const
{
    for(auto ins : iterator_for(p))
    {
        if(ins->name() == "gpu::add")
        {
            instruction_ref add_ins;
            instruction_ref input;
            if(ins->inputs()[0]->name() == "gpu::add")
            {
                add_ins = ins->inputs()[0];
                input   = ins->inputs()[1];
            }
            else if(ins->inputs()[1]->name() == "gpu::add")
            {
                add_ins = ins->inputs()[1];
                input   = ins->inputs()[0];
            }
            else
            {
                continue;
            }
            auto is_broadcasted = [](auto arg) { return arg->get_shape().broadcasted(); };
            auto args           = add_ins->inputs();
            if(std::count_if(args.begin(), args.end(), is_broadcasted) > 1)
                continue;
            args.insert(args.begin(), input);
            // Ensure the last arguments is the broadcasted one
            auto it = std::find_if(args.begin(), args.end(), is_broadcasted);
            if(it != args.end())
                std::swap(*it, *std::prev(args.end(), 2));
            args.back() = ins->inputs().back();
            p.replace_instruction(ins, hip_triadd{}, args);
        }
        if(ins->name() == "gpu::relu")
        {
            auto add_ins = ins->inputs().front();
            if(add_ins->name() == "gpu::add") 
            {
                auto args = add_ins->inputs();
                // Use the allocation from the relu operator
                args.back() = ins->inputs().back();
                p.replace_instruction(ins, hip_add_relu{}, args);
            }
            if(add_ins->name() == "hip::triadd")
            {
                auto args = add_ins->inputs();
                // Use the allocation from the relu operator
                args.back() = ins->inputs().back();
                p.replace_instruction(ins, hip_triadd_relu{}, args);
            }
        }
    }
}

} // namespace gpu

} // namespace migraph
