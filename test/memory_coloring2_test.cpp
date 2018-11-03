#include <migraph/memory_coloring2.hpp>
#include <migraph/dead_code_elimination.hpp>
#include <migraph/operators.hpp>
#include <migraph/generate.hpp>
#include <basic_ops.hpp>
#include <test.hpp>

struct memory_coloring_target
{
    std::string name() const { return "memory_coloring2"; }
    std::vector<migraph::pass> get_passes(migraph::context&) const
    {
        return {migraph::memory_coloring2{"allocate"}, migraph::dead_code_elimination{}};
    }
    migraph::context get_context() const { return {}; }
};

struct allocate
{
    migraph::shape s{};
    std::string name() const { return "allocate"; }
    migraph::shape compute_shape(const std::vector<migraph::shape>& inputs) const
    {
        migraph::check_shapes{inputs, *this}.has(1);
        return inputs.front();
    }
    migraph::argument compute(migraph::context&,
                              const migraph::shape& output_shape,
                              const std::vector<migraph::argument>&) const
    {
        return {output_shape};
    }
};

void test1()
{
    migraph::program p;
    auto a0 = p.add_outline(migraph::shape{migraph::shape::float_type, {8}});
    auto a1 = p.add_instruction(allocate{}, a0);
    auto p1 = p.add_instruction(pass_op{}, a1);
    auto a2 = p.add_outline(migraph::shape{migraph::shape::float_type, {40}});
    auto p2 = p.add_instruction(allocate{}, a2);
    p.add_instruction(pass_op{}, p2, p1);
    p.compile(memory_coloring_target{});
    EXPECT(p.get_parameter_shape("scratch").bytes() == 192);
}

int main() { test1(); }
