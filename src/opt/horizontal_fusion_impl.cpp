#include "horizontal_fusion_impl.hpp"
#include <migraphx/iterator_for.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

static unsigned opcode_bits = 16;
static unsigned hash_id_bits = 16;
static unsigned filter_bits = 8;
static unsigned kernel_bits = 8;

// Register a single operation.
// 1st arg: operation name. 2nd arg: encoding function.
void horizontal_fusion_impl::register_op(std::string name, Encoder func, int flag)
{
    op_registry[name] = func;
    op_flag[name] = flag;
}
           
// Register operations.
void horizontal_fusion_impl::register_all()
{
    register_op("gpu::convolution", EncodeConvCommon, 1);
    register_op("gpu::conv_bias_relu", EncodeConvCommon, 1);
    register_op("hip::add_relu", EncodeCommon, 0);
    register_op("convolution", EncodeConvCommon, 1);
    register_op("add", EncodeCommon, 0);
    register_op("relu", EncodeCommon, 0);
}

static unsigned opcode_shift_count()
{
    return ((sizeof(key_type) * 8) - opcode_bits);
}

static unsigned hash_id_shift_count()
{
    return (opcode_shift_count() - hash_id_bits);
}

static unsigned filter_shift_count()
{
    return (hash_id_shift_count() - filter_bits);
}

static unsigned kernel_shift_count()
{
    return (filter_shift_count() - kernel_bits);
}

// Encode common fields.
//
// |----- 16 bits -----|----- 16 bits -------|----- 32 bits -----|
// |      opcode       | 1st operand hash id |
encode_info EncodeCommon(instruction_ref ins, Ins2Val& instr2_value, unsigned opcode)
{
    if (opcode >= ( 1 << opcode_bits))
        return encode_info(0, false);
    key_type encode = (static_cast<key_type>(opcode) << opcode_shift_count());
    instruction_ref op1 = ins->inputs().front();
    assert(instr2_value.find(op1) != instr2_value.end());
    hash_value_ptr op1_val = instr2_value[op1];
    if (op1_val->id >= ( 1 << hash_id_bits))
        return encode_info(0, false);
    encode |= (static_cast<key_type>(op1_val->id) << hash_id_shift_count());
    encode_info info(encode, true);
    info.add_input(op1_val);
    return info;
}

// Encode common fields in convolution:
//
// |----- 16 bits -----|----- 16 bits -------|----- 8 bits -----|----- 8 bits-----|----- 16 bits -----|
// |     opcode        | 1st operand hash id |   filter size    |  kernel size    |     0x0000        |

encode_info EncodeConvCommon(instruction_ref ins, Ins2Val& instr2_value, unsigned opcode)
{
    encode_info info = EncodeCommon(ins, instr2_value, opcode);
    if (!info.is_valid())
        return info;
    key_type encode = info.get_key();
    instruction_ref op2 = ins->inputs().at(1);
    auto lens = op2->get_shape().lens();
    auto filter = lens[lens.size() - 2];
    auto kernel = lens[lens.size() - 1];
    if ((filter < ( 1 << filter_bits)) && (kernel < ( 1 << kernel_bits)))
    {
        encode |= (filter << filter_shift_count());
        encode |= (kernel << kernel_shift_count());
        info.set_key(encode);
        return info;
    } else {
        return encode_info(0, false);
    }
}

bool horizontal_fusion_impl::is_conv(instruction_ref ins)
{
    if (op_flag.find(ins->name()) == op_flag.end())
        return false;
    return (op_flag[ins->name()] == 1);
}

// Hash given instruction.           
hash_value_ptr horizontal_fusion_impl::hash(instruction_ref ins)
{
    if (op_registry.find(ins->name()) == op_registry.end())
        return nullptr;

    Encoder encode_func = op_registry.at(ins->name());
    unsigned opcode = hash_opcode(ins);
    encode_info encode_val = encode_func(ins, instr2_value, opcode);
    if (!encode_val.is_valid())
    {
        std::cout << "warning: value hash fails" << std::endl;
        return nullptr;
    }
    key_type key = encode_val.get_key();
    hash_value_ptr hash_val = nullptr;

    if (encode2_value.find(key) != encode2_value.end()) {
        hash_val = encode2_value[key];
        add_instr(hash_val->id);
        instr2_value[ins] = hash_val;
    } else {
        hash_val = &(create_value(ins));
        encode2_value[key] = hash_val;
    }
    for (auto&& input : encode_val.get_inputs())
    {
        add_input(hash_val->id, input);
        add_output(input->id, hash_val);
    }
    return hash_val;
}

hash_value& horizontal_fusion_impl::create_value(instruction_ref ins)
{
    unsigned id = static_cast<unsigned>(values.size());
    values.push_back(hash_value{id, cur_point});
    hash_value& val = get_value(id);
    add_instr(id);
    instr2_value[ins] = &val;
    return val;
}
           
void horizontal_fusion_impl::process(instruction_ref ins)
{
    // Do not hash literals.
    if (ins->name() == "@literal")
        return;
    if (instr2_hash.find(ins) != instr2_hash.end())
    {
        // Hash this instruction.
        if (hash(ins) != nullptr)
        {
            for (auto&& output : ins->outputs())
            {
                instr2_hash[output] = true;
            }
            return;
        }
    }
    std::unordered_map<std::string, int> op2_cnt;
    bool hash_child = false;
    // Do hash if at least two outputs have same operations.
    for (auto&& output : ins->outputs())
    {
        const std::string& str = output->name();
        if (op2_cnt.find(str) == op2_cnt.end())
            op2_cnt[str] = 1;
        else {
            op2_cnt[str] += 1;
            hash_child = true;
            break;
        }
    }
    if (hash_child)
    {            
        // Create a value for this instruction.
        hash_value& value = create_value(ins);
        value.set_root();
        // Flag children to be hashed.
        for (auto&& output : ins->outputs())
        {
            if (op2_cnt[output->name()] > 1)
                instr2_hash[output] = true;
        }
    }
}

// Find the first axis that matches given dim.
int horizontal_fusion_impl::find_axis(instruction_ref ins, int dim)
{
    int ndx = 0;
    for (auto&& size : ins->get_shape().lens())
    {
        if (size == dim)
            return ndx;
        ndx++;
    }
    return -1;
}

// Check whether ins1 and ins2 match in all dimensions excluding axis.
bool horizontal_fusion_impl::match_dim(instruction_ref ins1, instruction_ref ins2, int axis)
{
    auto lens1 = ins1->get_shape().lens();
    auto lens2 = ins2->get_shape().lens();
    if (lens1.size() != lens2.size())
        return false;
    int ndx = 0;
    for (auto&& size : lens1)
    {
        if ((size != lens2.at(ndx)) && (ndx != axis))
            return false;
        ndx++;
    }
    return true;
}
    
bool horizontal_fusion_impl::compare_inputs(std::vector<instruction_ref>& input1, std::vector<instruction_ref>& input2, instruction_ref base_ins, int base_axis)
{
    if (input1.size() != input2.size())
        return false;
    int ndx = 0;
    bool check_conv_kernel = is_conv(base_ins);
    
    for (auto && ins1 : input1)
    {
        instruction_ref ins2 = input2.at(ndx++);
        if (ins1->name() != ins2->name())
            return false;

        auto base_lens = base_ins->get_shape().lens();
        int base_dim = base_lens.at(base_axis);
        if (check_conv_kernel || (ins1->outputs().at(0)->name() == "broadcast"))
        {
            // Find concat axis for convolution's kernel.
            int axis = find_axis(ins2, base_dim);
            if (axis == -1)
                return false;
            if (!match_dim(ins1, ins2, axis))
                return false;
        } 
    }

    return true;
}

void horizontal_fusion_impl::concat(std::vector<instruction_ref>& instrs, std::unordered_map<instruction_ref, instruction_ref>& root, int root_axis)
{
    instruction_ref ins0 = instrs.at(0);
    shape s = ins0->get_shape();
    std::vector<std::size_t> new_lens = s.lens();
    instruction_ref base = root[ins0];
    int axis = root_axis;
    std::vector<std::size_t> root_lens = base->get_shape().lens();
    
    if (is_conv(base) || (ins0->outputs().at(0)->name() == "broadcast"))
    {        
        int dim = base->get_shape().lens().at(root_axis);
        axis = find_axis(ins0, dim);
    }
    int sum = 0;
    int root_sum = 0;
    for (auto&& ins : instrs)
    {
        sum += ins->get_shape().lens().at(axis);
        root_sum += root[ins]->get_shape().lens().at(root_axis);
    }
    new_lens[axis] = sum;
    shape new_s = shape(s.type(), new_lens);
    unsigned long long unit_slice = 1;
    int ndx = 0;
    unsigned long long new_elements = 1;
    for (auto&& len : s.lens())
    {
        if (ndx > axis)
            unit_slice *= len;
        if (ndx != axis)
            new_elements *= len;
        ndx++;
    }
    new_elements *= sum;
    unsigned type_size = s.type_size();

    if (ins0->name() == "@literal")
    {
        // concat literals.
        unsigned long long total_bytes = new_elements * type_size;
        std::vector<char> input(total_bytes, 0);
        std::vector<unsigned long long> bytes_per_slice;

        for (auto&& ins : instrs)
            bytes_per_slice.push_back(ins->get_shape().lens().at(axis) * unit_slice * type_size);

        unsigned out_ndx = 0;
        int slice_ndx = 0;
        while (out_ndx < total_bytes)
        {
            unsigned ins_ndx = 0;
            for (auto && ins : instrs)
            {
                unsigned long long bytes = bytes_per_slice[ins_ndx];
                for (auto i = 0; i < bytes ; ++i)
                    input[out_ndx++] = ins->get_literal().data()[slice_ndx * bytes + i];
                ins_ndx++;
            }
            slice_ndx++;
        }
        shape new_shape{s.type(), new_lens};
        auto new_literal = p_program->add_literal(literal{new_shape, input});
        assert(ins0->outputs().size() == 1);
        instruction_ref output = ins0->outputs().at(0);

        if (output == base)
        {
            if (root_lens[root_axis] != root_sum)
            {
                // change root's shape.
                root_lens[root_axis] = root_sum;
                output->set_shape({base->get_shape().type(), root_lens});
            }
        }
        else
            assert(false);
        instruction::replace_argument(output, ins0, new_literal);
    }
    else
        assert(false);
    
}

// If ins and input only diff in one axis, return that axis.
int horizontal_fusion_impl::find_unique_axis(instruction_ref ins, instruction_ref input)
{
    auto lens1 = ins->get_shape().lens();
    auto lens2 = input->get_shape().lens();
    if (lens1.size() != lens2.size())
        return false;
    int count = 0;
    int ndx = 0;
    int ret = -1;
    for (auto && size : lens1)
    {
        if (size != lens2.at(ndx))
        {            
            count++;
            ret = ndx;
        }
        ndx++;
    }
    return (count == 1) ? ret : -1;

}

int horizontal_fusion_impl::find_axis(instruction_ref ins, std::unordered_map<instruction_ref, bool>& is_common)
{
    int axis = -1;
    for (auto&& input : ins->inputs())
    {
        if (is_common.find(input) != is_common.end())
        {
            int cur_axis = find_unique_axis(ins, input);
            if ((cur_axis == -1) || ((axis != -1) && (cur_axis != axis)))
                return -1;
            axis = cur_axis;
        }
    }
    return axis;
}
                                                       
void horizontal_fusion_impl::transform()
{
    for (auto && val : values)
    {
        unsigned id = val.id;
        if ((hash_instrs.find(id) == hash_instrs.end())
            || (hash_instrs[id].size() <= 1))
            continue;
        std::vector<unsigned> cluster;
        cluster.push_back(id);
        unsigned cur = id;
        int size = hash_instrs[id].size();
        // Find a sub-tree of the hash tree to be fused together.
        // Every node in the sub-tree contain the same amount of instructions.
        while ((hash_outputs.find(cur) != hash_outputs.end())
               && (hash_outputs[cur].size() == 1))
        {
            unsigned output = (*(hash_outputs[cur].begin()))->id;
            if ((hash_instrs.find(output) != hash_instrs.end())
                && (hash_instrs[output].size() == size))
                {
                cluster.push_back(output);
                cur = output;
            } else
                break;
        }

        std::unordered_map<instruction_ref, bool> visited;
        std::unordered_map<instruction_ref, instruction_ref> root;
        for (auto && hash_id : cluster)
        {
            assert(hash_inputs.find(hash_id) != hash_inputs.end());
            bool doit = true;
            // Flag common inputs which will not be concated.
            for (auto && input : hash_inputs[hash_id])
            {
                std::vector<instruction_ref> instrs = get_instrs(input->id);
                if (instrs.size() != 1) {
                    doit = false;
                    break;
                }
                visited[instrs.at(0)] = true;
            }
            if (!doit)
                continue;

            // collect and compare inputs to be concated.
            std::vector<std::vector<instruction_ref>> all_inputs;
            int axis = -1;
            std::vector<instruction_ref> base_instrs = get_instrs(hash_id);
            for (auto && ins : base_instrs)
             {
                 // Find concat axis for ins.
                 axis = (axis == -1) ? find_axis(ins, visited) : axis;
                 if (axis == -1)
                 {
                     doit = false;
                     break;
                 }
                 std::vector<instruction_ref> inputs = walk(ins, visited);
                 if (inputs.empty() || (!all_inputs.empty() && !compare_inputs(all_inputs.at(0), inputs, ins, axis)))
                 {
                     doit = false;
                     break;
                 }
                 else
                 {
                     for (auto&& input : inputs)
                         root[input] = ins;
                     all_inputs.push_back(inputs);
                 }
             }
             if (!doit)
                 continue;
             
             std::vector<instruction_ref> input0 = all_inputs.at(0);
             // concat inputs.
             for (int ndx = 0; ndx < input0.size(); ndx++)
             {
                 std::vector<instruction_ref> instrs;
                 for (auto&& input : all_inputs)
                 {
                     instrs.push_back(input.at(ndx));
                 }
                 concat(instrs, root, axis);
             }
             // remove redundant inputs.
             for (auto&& input : all_inputs)
             {                        
                 for (int ndx = 0; ndx < input.size(); ndx++)
                 {
                     instruction_ref ins = input.at(ndx);
                     bool is_literal = (ins->name() == "@literal");
                     if ((ndx == 0) && !is_literal)
                         continue;
                     if (is_literal)
                         ins->get_literal().delete_data();
                     p_program->remove_instruction(ins);
                 }
             }
             // remove redundant roots.
             instruction_ref root_ins = base_instrs.at(0);
             for (int ndx = 1; ndx < base_instrs.size(); ndx++)
             {
                 instruction_ref base = base_instrs.at(ndx);
                 std::vector<instruction_ref> outputs;
                 for (auto && output : base->outputs())
                     outputs.push_back(output);
                 for (auto && output : outputs)
                     instruction::replace_argument(output, base, root_ins, false);
                 p_program->remove_instruction(base);
             }
             // update hash tree.
             unsigned first = *(hash_instrs[id].begin());
             hash_instrs[id].clear();
             hash_instrs[id].insert(first);
             std::cout << *p_program << std::endl;
             dump_hash_tree();
        }
    }
}

std::vector<instruction_ref> horizontal_fusion_impl::walk(instruction_ref ins, std::unordered_map<instruction_ref, bool>& visited)
{
    
    std::stack<instruction_ref> stk;
    for (auto && input : ins->inputs())
    {
        if (visited.find(input) == visited.end())
            stk.push(input);
    }

    std::vector<instruction_ref> ret;
    while (!stk.empty())
    {
        instruction_ref top = stk.top();
        if ((top->inputs().size() > 1) || (top->outputs().size() > 1)
            || (top->inputs().empty() && (top->name() != "@literal")))
        {
            ret.clear();
            return ret;
        }
        else if (top->inputs().empty() || (visited.find(top) != visited.end()))
        {
            ret.push_back(top);
            stk.pop();
        } 
        else {
            instruction_ref input = top->inputs().at(0);
            stk.push(input);
            visited[top] = true;
        }
    }
    return ret;
}
               
void horizontal_fusion_impl::run()
{

    MIGRAPHX_DEBUG(dump("---Before horizontal fusion---"));
    MIGRAPHX_DEBUG(dump_program());
    std::cout << *p_program << std::endl;
    for (auto ins : iterator_for(*p_program))
    {
        process(ins);
        point2_instr[cur_point] = ins;
        ins->id = cur_point;
        cur_point++;
    }
    dump_hash_tree();
    transform();
}

#ifdef MIGRAPHX_DEBUG_H_FUSION

void horizontal_fusion_impl::dump_program() { std::cout << *p_program << std::endl; }

void horizontal_fusion_impl::dump_hash_value(hash_value& val)
{
    unsigned id = val.id;
    std::cout << "id: " << id << " @" << val.cur_point;
    if (hash_inputs.find(id) != hash_inputs.end())
    {
        std::cout << " input: ";
        for (auto && input : hash_inputs[id])
            std::cout << " " << input->id;
    }

    if (hash_outputs.find(id) != hash_outputs.end())
    {
        std::cout << " output: ";
        for (auto && output : hash_outputs[id])
            std::cout << " " << output->id;
    }
    if (hash_instrs.find(id) != hash_instrs.end())
    {
        std::cout << " instrs: ";
        for (auto && point : hash_instrs[id])
        {
            if (point2_instr.find(point) != point2_instr.end())
            {
                int ins_id = point2_instr[point]->id;
                if (ins_id > 0)
                    std::cout << " (" << ins_id << ")";
            }
            std::cout << " @" << point;
        }
    }
    std::cout << std::endl;
}

void horizontal_fusion_impl::dump_hash_tree()
{
    for (auto && val : values)
    {
        dump_hash_value(val);
     }

}
#endif

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
