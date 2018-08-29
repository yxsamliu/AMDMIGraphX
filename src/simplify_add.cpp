#include <migraph/simplify_add.hpp>
#include <migraph/program.hpp>
#include <migraph/instruction.hpp>
#include <migraph/operators.hpp>
#include <migraph/iterator_for.hpp>
#include <migraph/functional.hpp>

namespace migraph {

void simplify_add::apply(program& p) const
{
    for(auto ins : iterator_for(p))
    {
        if(ins->op.name() != "add")
            continue;
        auto is_cop = [](auto x) { return x->op.name() == "broadcast" or x->op.name() == "@literal"; };
        auto is_not_cop = [&](auto x) { return not is_cop(x); };
        if(!std::all_of(ins->arguments.begin(), ins->arguments.end(), [&](auto x) { 
            return x->op.name() == "add" and std::count_if(x->arguments.begin(), x->arguments.end(), is_cop) == 1; 
        }))
            continue;
        auto add1 = ins->arguments.at(0);
        auto add2 = ins->arguments.at(1);

        auto x = *std::find_if(add1->arguments.begin(), add1->arguments.end(), is_not_cop);
        auto a = *std::find_if(add1->arguments.begin(), add1->arguments.end(), is_cop);

        auto y = *std::find_if(add2->arguments.begin(), add2->arguments.end(), is_not_cop);
        auto b = *std::find_if(add2->arguments.begin(), add2->arguments.end(), is_cop);

        if(a->op.name() != b->op.name())
            continue;
        instruction_ref sumab;
        // TODO: Make broadcast unary
        if(a->op.name() == "broadcast") {
            if(a->arguments.at(1)->get_shape() != b->arguments.at(1)->get_shape())
                continue;
            auto op = a->op;
            auto presum = p.insert_instruction(ins, add{}, a->arguments.at(1), b->arguments.at(1));
            sumab = p.insert_instruction(ins, op, a->arguments.at(0), presum);
        }
        else
        {
            sumab = p.insert_instruction(ins, add{}, a, b);
        }

        auto sumxy = p.insert_instruction(ins, add{}, x, y);
        p.replace_instruction(ins, add{}, sumxy, sumab);
    }

    // Propogate constant adds
    for(auto ins : iterator_for(p))
    {
        if(ins->op.name() != "add")
            continue;
        if(!std::all_of(ins->arguments.begin(), ins->arguments.end(), [&](auto x) { 
            return x->op.name() == "@literal"; 
        }))
            continue;
        auto arg1 = ins->arguments.at(0)->lit;
        auto arg2 = ins->arguments.at(1)->lit;

        auto sum = p.add_literal(transform(arg1, arg2, [](auto x, auto y) { return x + y; }));
        p.replace_instruction(ins, sum);
    }
}

} // namespace migraph
