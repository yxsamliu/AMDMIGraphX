#include <migraph/memory_coloring2.hpp>
#include <migraph/program.hpp>
#include <migraph/operators.hpp>
#include <migraph/instruction.hpp>
#include <migraph/iterator_for.hpp>
#include <migraph/functional.hpp>
#include <migraph/ranges.hpp>
#include <migraph/stringutils.hpp>
#include <unordered_set>
#include <unordered_map>
#include <map>
#include <set>

namespace migraph {

using instruction_set     = std::unordered_set<instruction_ref>;
using instruction_set_map = std::unordered_map<instruction_ref, instruction_set>;

template <class F>
void liveness(const program& p, F f)
{
    instruction_set live_set;
    auto rp = reverse(p);
    for(auto rins : iterator_for(rp))
    {
        auto ins = std::prev(rins.base());
        // Add live variables
        for(auto input : ins->inputs())
        {
            auto i = instruction::get_output_alias(input);
            // Skip builtins
            if(starts_with(i->name(), "@"))
                continue;
            live_set.insert(i);
        }
        // Remove last usage
        auto it = live_set.find(ins);
        if(it != live_set.end())
        {
            f(ins, live_set);
            live_set.erase(it);
        }
    }
}

instruction_set_map build_conflict_table(const program& p, std::string allocation_op)
{
    instruction_set_map conflict_table;
    liveness(p, [&](auto, auto live_set) {
        for(auto i : live_set)
        {
            conflict_table[i];
            if(i->name() != allocation_op)
                continue;
            for(auto j : live_set)
            {
                if(j->name() != allocation_op)
                    continue;
                if(i == j)
                    continue;
                conflict_table[i].insert(j);
                conflict_table[j].insert(i);
            }
        }
    });
    return conflict_table;
}

struct allocation_color
{
    std::unordered_map<instruction_ref, int> ins2color;
    std::map<int, instruction_set> color2ins;

    void add_color(instruction_ref ins, int color)
    {
        assert(color >= 0);
        this->remove(ins);
        ins2color[ins] = color;
        color2ins[color].insert(ins);
    }

    int get_color(instruction_ref ins) const
    {
        auto it = ins2color.find(ins);
        if(it == ins2color.end())
            return -1;
        return it->second;
    }

    void remove(instruction_ref ins)
    {
        auto it = ins2color.find(ins);
        if(it != ins2color.end())
        {
            color2ins[it->second].erase(ins);
            ins2color.erase(it);
        }
    }

    std::size_t max_bytes(int color) const
    {
        auto&& is = color2ins.at(color);
        auto it   = std::max_element(is.begin(), is.end(), [](auto x, auto y) {
            return x->get_shape().bytes() < y->get_shape().bytes();
        });
        return (*it)->get_shape().bytes();
    }
};

int next_color(const std::set<int>& colors)
{
    int i = 0;
    // TODO: Use adjacent_find
    for(auto color : colors)
    {
        if(color < 0)
            continue;
        if(color != i)
            return i;
        i++;
    }
    return i;
}

void memory_coloring2::apply(program& p) const
{
    auto conflict_table = build_conflict_table(p, allocation_op);
    allocation_color ac{};
    std::vector<instruction_ref> conflict_queue;
    std::transform(conflict_table.begin(),
                   conflict_table.end(),
                   std::back_inserter(conflict_queue),
                   [](auto&& pp) { return pp.first; });
    std::sort(conflict_queue.begin(), conflict_queue.end(), [&](auto x, auto y) {
        return conflict_table.at(x).size() < conflict_table.at(y).size();
    });
    for(auto parent : conflict_queue)
    {
        auto&& children = conflict_table[parent];
        std::set<int> colors;
        auto parent_color = ac.get_color(parent);
        colors.insert(parent_color);
        std::transform(children.begin(),
                       children.end(),
                       std::inserter(colors, colors.end()),
                       [&](auto child) { return ac.get_color(child); });
        // Color parent if needed
        if(parent_color < 0)
        {
            parent_color = next_color(colors);
            ac.add_color(parent, parent_color);
            colors.insert(parent_color);
        }
        for(auto child : children)
        {
            auto color = ac.get_color(child);
            if(color < 0)
            {
                color = next_color(colors);
                ac.add_color(child, color);
                colors.insert(color);
            }
        }
    }

    const std::size_t alignment = 32;
    // Compute offsets
    std::size_t n = 0;
    std::map<int, int> color2offset;
    for(auto&& pp : ac.color2ins)
    {
        auto color = pp.first;
        // auto&& allocations = pp.second;
        color2offset.emplace(color, n);
        std::size_t size    = ac.max_bytes(color);
        std::size_t padding = (alignment - (size % alignment)) % alignment;
        n += size + padding;
    }

    // Replace allocations
    auto mem = p.add_parameter("scratch", shape{shape::int8_type, {n}});
    for(auto&& pp : ac.color2ins)
    {
        auto color         = pp.first;
        auto&& allocations = pp.second;
        auto offset        = color2offset.at(color);
        for(auto ins : allocations)
        {
            auto s = ins->get_shape();
            p.replace_instruction(ins, op::load{s, std::size_t(offset)}, mem);
        }
    }
}

} // namespace migraph
