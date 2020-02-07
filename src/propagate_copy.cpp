#include <migraphx/propagate_copy.hpp>
#include <migraphx/program.hpp>
#include <migraphx/matcher.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/functional.hpp>
#include <unordered_set>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

bool single_use(instruction_ref alias, instruction_ref ins)
{
    if (not is_context_free(ins->get_operator()))
        return ins->outputs().size() < 2;
    return false;

}

void propagate_copy::apply(program& p) const
{
    for(auto ins : iterator_for(p))
    {
        if(ins->name() != copy)
            continue;
        auto input = ins->inputs().front();
        auto i = instruction::get_output_alias(input);
        if (not single_use(i, input))
            continue;
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
