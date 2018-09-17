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
        if(ins->name() != "add")
            continue;
        auto is_cop = [](auto x) {
            return x->name() == "broadcast" or x->name() == "@literal";
        };
        auto is_not_cop = [&](auto x) { return not is_cop(x); };
        if(!std::all_of(ins->inputs().begin(), ins->inputs().end(), [&](auto x) {
               return x->name() == "add" and
                      std::count_if(x->inputs().begin(), x->inputs().end(), is_cop) == 1;
           }))
            continue;
        auto add1 = ins->inputs().at(0);
        auto add2 = ins->inputs().at(1);

        auto x = *std::find_if(add1->inputs().begin(), add1->inputs().end(), is_not_cop);
        auto a = *std::find_if(add1->inputs().begin(), add1->inputs().end(), is_cop);

        auto y = *std::find_if(add2->inputs().begin(), add2->inputs().end(), is_not_cop);
        auto b = *std::find_if(add2->inputs().begin(), add2->inputs().end(), is_cop);

        if(a->name() != b->name())
            continue;
        instruction_ref sumab;
        // TODO: Make broadcast unary
        if(a->name() == "broadcast")
        {
            if(a->inputs().at(1)->get_shape() != b->inputs().at(1)->get_shape())
                continue;
            auto op     = a->get_operator();
            auto presum = p.insert_instruction(ins, add{}, a->inputs().at(1), b->inputs().at(1));
            sumab       = p.insert_instruction(ins, op, a->inputs().at(0), presum);
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
        if(ins->name() != "add")
            continue;
        if(!std::all_of(ins->inputs().begin(), ins->inputs().end(), [&](auto x) {
               return x->name() == "@literal";
           }))
            continue;
        auto arg1 = ins->inputs().at(0)->get_literal();
        auto arg2 = ins->inputs().at(1)->get_literal();

        auto sum = p.add_literal(transform(arg1, arg2, [](auto x, auto y) { return x + y; }));
        p.replace_instruction(ins, sum);
    }
}

} // namespace migraph
