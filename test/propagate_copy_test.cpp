#include <migraphx/propagate_copy.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/eliminate_identity.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/op/add.hpp>
#include <migraphx/op/scalar.hpp>
#include <migraphx/op/mul.hpp>
#include <basic_ops.hpp>
#include <test.hpp>

// y = f(x, mem);
// z = copy(y, mem2);
// =>
// y = f(x, mem2);

// y = f(x, mem);
// y2 = g(y);
// z = copy(y2, mem2);
// =>
// y = f(x, mem2);

struct copy
{
    std::string name() const { return "copy"; }
    migraphx::shape compute_shape(std::vector<migraphx::shape> inputs) const
    {
        migraphx::check_shapes{inputs}.has(2);
        return inputs.at(1);
    }
    std::ptrdiff_t output_alias(const std::vector<migraphx::shape>&) const { return 1; }
};

struct allocate
{
    migraphx::shape s{migraphx::shape::int32_type, {1}, {0}};

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return migraphx::pack(f(self.s, "shape"));
    }

    std::string name() const { return "allocate"; }
    migraphx::shape compute_shape(const std::vector<migraphx::shape>& inputs) const
    {
        migraphx::check_shapes{inputs, *this}.has(0);
        return s;
    }
};

struct buffered
{
    std::string name() const { return "buffered"; }
    migraphx::shape compute_shape(std::vector<migraphx::shape> inputs) const
    {
        migraphx::check_shapes{inputs, *this}.has(2);
        auto s = inputs.at(0);
        if(s.packed())
        {
            return s;
        }
        else
        {
            return {s.type(), s.lens()};
        }
    }
    std::ptrdiff_t output_alias(const std::vector<migraphx::shape>& shapes) const
    {
        return shapes.size() - 1;
    }
};

void run_pass(migraphx::program& p)
{
    migraphx::run_passes(p,
                         {migraphx::propagate_copy{"copy"},
                          migraphx::dead_code_elimination{},
                          migraphx::eliminate_identity{}});
}

TEST_CASE(simple)
{
    migraphx::program p1;
    {
        auto one = p1.add_literal(1);
        auto m   = p1.add_instruction(allocate{});
        auto x   = p1.add_instruction(buffered{}, one, m);
        auto m2  = p1.add_instruction(allocate{});
        auto y   = p1.add_instruction(copy{}, x, m2);
        p1.add_instruction(pass_op{}, y);
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto one = p2.add_literal(1);
        auto m   = p2.add_instruction(allocate{});
        auto x   = p2.add_instruction(buffered{}, one, m);
        p2.add_instruction(pass_op{}, x);
    }
    EXPECT(p1 == p2);
}

TEST_CASE(as_shape_input)
{
    migraphx::program p1;
    {
        auto one = p1.add_literal(1);
        auto m   = p1.add_instruction(allocate{});
        auto x   = p1.add_instruction(buffered{}, one, m);
        auto m2  = p1.add_instruction(allocate{});
        auto g   = p1.add_instruction(pass_op{}, x);
        auto y   = p1.add_instruction(copy{}, g, m2);
        p1.add_instruction(pass_op{}, y);
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto one = p2.add_literal(1);
        auto m   = p2.add_instruction(allocate{});
        auto x   = p2.add_instruction(buffered{}, one, m);
        auto g   = p2.add_instruction(pass_op{}, x);
        p2.add_instruction(pass_op{}, g);
    }
    EXPECT(p1 == p2);
}

// TEST_CASE(pass_input)
// {
//     migraphx::program p1;
//     {
//         auto one = p1.add_literal(1);
//         auto m = p1.add_instruction(allocate{});
//         auto x = p1.add_instruction(buffered{}, one, m);
//         auto m2 = p1.add_instruction(allocate{});
//         auto g = p1.add_instruction(pass_op{}, x);
//         auto y = p1.add_instruction(copy{}, g, m2);
//         p1.add_instruction(pass_op{}, y);
//     }
//     run_pass(p1);

//     migraphx::program p2;
//     {
//         auto one = p2.add_literal(1);
//         auto m = p2.add_instruction(allocate{});
//         auto x = p2.add_instruction(buffered{}, one, m);
//         auto g = p2.add_instruction(pass_op{}, x);
//         p2.add_instruction(pass_op{}, g);
//     }
//     EXPECT(p1 == p2);
// }

int main(int argc, const char* argv[]) { test::run(argc, argv); }
