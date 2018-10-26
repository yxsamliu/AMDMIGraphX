#include "pre_scheduling_impl.hpp"
#include <migraph/iterator_for.hpp>
#include <stack>
namespace migraph {


void pre_scheduling_impl::compute_weights()
{
    int ndx = 0;
    std::unordered_map<dag_node*, bool> visited;
    for(auto ins : iterator_for(*p_program))
    {
        dag_node& node = nodes[ndx];
        std::string name = ins->name();
        float weight = weight_func(name);
        node.weight = weight;
        node.weight_sum += weight;
        visited.clear();

        for(auto&& arg : ins->inputs()) {
            assert(instr2_node.find(arg) != instr2_node.end());
            dag_node* def_node = instr2_node[arg];
            if (visited.find(def_node) == visited.end()) {
                node.weight_sum += def_node->weight_sum;
                visited[def_node] = true;
            }
        }
        if (ins->outputs().empty()) {
            exit_nodes.push_back(&node);
        } 
        node.ins = ins;
        node.ins_ndx = ndx++;
        instr2_node[ins] = &node;
    }
    int size = exit_nodes.size();
    if (size > 1) {
        std::sort(exit_nodes.begin(), exit_nodes.end(), compare_exit_nodes);
    }
}

void pre_scheduling_impl::reorder()
{
    std::list<dag_node*> sorted_nodes;
    std::stack<dag_node*> stack;
    std::priority_queue<dag_node*, std::vector<dag_node*>, ordering> child_queue;
    std::unordered_map<dag_node*, bool> visited;
    std::unordered_map<dag_node*, bool> dequeued;
    
    for (auto&& node : exit_nodes)
    {
        stack.push(node);
        while (!stack.empty()) {
            auto cur = stack.top();
            if (dequeued.find(cur) != dequeued.end()) {
                stack.pop();
                continue;
            } else if ((visited.find(cur) != visited.end()) || cur->ins->inputs().empty()) {
                stack.pop();
                sorted_nodes.push_back(cur);
                dequeued[cur] = true;
                continue;
            }
            // sort child nodes.
            for(auto&& arg : cur->ins->inputs()) {
                dag_node* child_node = instr2_node[arg];
                if (dequeued.find(child_node) == dequeued.end()) {
                    child_queue.push(child_node);
                }
            }
            while (!child_queue.empty()) {
                dag_node * child = child_queue.top();
                stack.push(child);
                child_queue.pop();
            }
            visited[cur] = true;
        }
    }
    
#ifdef MIGRAPH_DEBUG_OPT
    MIGRAPH_DEBUG(dump("---After weighted topology sort---"));
    MIGRAPH_DEBUG(dump(sorted_nodes));
#endif
    
    splice(sorted_nodes);
#ifdef MIGRAPH_DEBUG_OPT
    MIGRAPH_DEBUG(dump("---After pre-scheduling---"));
    MIGRAPH_DEBUG(dump_program());
    verify();
#endif
    
}

void pre_scheduling_impl::splice(std::list<dag_node*>& sorted_nodes)
{
    auto begin = sorted_nodes.begin();
    auto iter = sorted_nodes.end();
    instruction_ref insert_before = (*(--iter))->ins;
    do {
        iter--;
        insert_before = p_program->move_instruction((*iter)->ins, insert_before);
    } while (iter != begin);
}
    
void pre_scheduling_impl::run()
{
    std::size_t num_of_instrs = p_program->size();
    if(num_of_instrs == 0)
        return;
    MIGRAPH_DEBUG(dump("---Before pre-scheduling---"));
    MIGRAPH_DEBUG(dump_program());
    nodes.resize(num_of_instrs);
    compute_weights();
    reorder();
}

#ifdef MIGRAPH_DEBUG_OPT
void pre_scheduling_impl::dump(const std::string& str)
{
    std::cout << str << std::endl;
}

void pre_scheduling_impl::dump_program()
{
    std::cout << *p_program << std::endl;
}

void pre_scheduling_impl::dump(std::list<dag_node*>& sorted_nodes)
{
    for (auto&& node : sorted_nodes)
    {
        node->dump();
        if (!node->ins->inputs().empty()) {
            std::cout << " inputs: ";
            for(auto&& arg : node->ins->inputs()) {
                dag_node* def_node = instr2_node[arg];
                std::cout << " @" << def_node->ins_ndx;
            }
            std::cout << std::endl;
        }
    }
}

void pre_scheduling_impl::verify()
{
    std::unordered_map<instruction_ref, bool> visited;
    for(auto ins : iterator_for(*p_program))
    {
        for(auto&& arg : ins->inputs()) {
            assert(visited.find(arg) != visited.end());
        }
        visited[ins] = true;
    }
}

void dag_node::dump()
{
    std::cout << " @" << ins_ndx;
    std::cout << " name: " << ins->name();
    std::cout << " weight: " << weight;
    std::cout << " weight_sum: " << weight_sum;
    std::cout << std::endl;
}
#endif    
    
} // namespace migraph
