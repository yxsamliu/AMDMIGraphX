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
        if(ins->op.name() == "gpu::add") 
        {
            instruction_ref add_ins;
            instruction_ref input;
            if(ins->arguments[0]->op.name() == "gpu::add") 
            {
                add_ins = ins->arguments[0];
                input = ins->arguments[1];
            }
            else if(ins->arguments[1]->op.name() == "gpu::add") 
            {
                add_ins = ins->arguments[1];
                input = ins->arguments[0];
            }
            else
            {
                continue;
            }
            auto is_broadcasted = [](auto arg) { return arg->get_shape().broadcasted(); };
            auto args = add_ins->arguments;
            if(std::count_if(args.begin(), args.end(), is_broadcasted) > 1)
                continue;
            args.insert(args.begin(), input);
            // Ensure the last arguments is the broadcasted one
            auto it = std::find_if(args.begin(), args.end(), is_broadcasted);
            if(it != args.end())
                std::swap(*it, *std::prev(args.end(), 2));
            p.replace_instruction(ins, hip_triadd{}, args);
        }
        if(ins->op.name() == "gpu::relu") 
        {
            auto add_ins = ins->arguments.front();
            if(add_ins->op.name() == "gpu::add")
                p.replace_instruction(ins, hip_add_relu{}, add_ins->arguments);
            if(add_ins->op.name() == "hip::triadd") {
                p.replace_instruction(ins, hip_triadd_relu{}, add_ins->arguments);
            }
        }
    }
}

} // namespace gpu

} // namespace migraph
