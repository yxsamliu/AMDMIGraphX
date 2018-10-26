#ifndef MIGRAPH_GUARD_RTGLIB_PRE_SCHEDULING_IMPL_HPP
#define MIGRAPH_GUARD_RTGLIB_PRE_SCHEDULING_IMPL_HPP
#include "common_header.hpp"
#include <migraph/instruction_ref.hpp>

namespace migraph {

struct dag_node
{
    dag_node()
    {
        weight = 0.0f;
        weight_sum = 0.0f;
        ins_ndx = -1;
    }
    float weight;
    float weight_sum;
    instruction_ref ins;
    int ins_ndx;
    bool is_mem() const
    {
        return (ins->name() == "@literal");
    }
    
#ifdef MIGRAPH_DEBUG_OPT
    void dump();
#endif    
};

struct dag_partition
{
    dag_partition()
    {
        last_node = nullptr;
    }
    dag_node* last_node;
};
    
struct pre_scheduling_impl
{
    pre_scheduling_impl(program* p, std::function<float(std::string&)> w)
        : p_program(p), weight_func(std::move(w))
    {
        instr2_node.clear();
    }
    void run();
    void compute_weights();
    void reorder();
    void splice(std::list<dag_node*>&);
    static bool compare_exit_nodes(dag_node* d1, dag_node* d2)
    {
        return (d1->weight_sum > d2->weight_sum);
    }

    struct ordering
    {
        bool operator()(const dag_node* d1, const dag_node* d2) const
        {
            if (d1->is_mem() && !d2->is_mem())
            {
                // mem is is placed on top of queue.
                return false;
            } else if (!d1->is_mem() && d2->is_mem())
            {
                return true;
            } 
            else if (d1->weight_sum < d2->weight_sum) {
                // smaller weigth_sum is placed on top of the queue.
                return false;
            }
            else if (d1->weight_sum > d2->weight_sum) {
                return true;
            }
            else {
                // smaller instrution index is placed on top of the queue,
                return d1->ins_ndx > d2->ins_ndx;
            }
        }
    };

#ifdef MIGRAPH_DEBUG_OPT
    void dump(const std::string&);
    void dump_program();
    void dump(std::list<dag_node*>&);
    void verify();
#endif    
    private:
    program* p_program;
    std::function<float(std::string&)> weight_func;
    std::vector<dag_node> nodes;
    std::vector<dag_node*> exit_nodes;
    std::unordered_map<instruction_ref, dag_node*> instr2_node;
};

} // namespace migraph
#endif
