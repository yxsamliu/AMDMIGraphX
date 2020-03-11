#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/target.hpp>
#include <migraphx/onnx.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/auto_contiguous.hpp>
#include <migraphx/decompose.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/simplify_reshapes.hpp>
#include <migraphx/simplify_algebra.hpp>
#include <migraphx/propagate_constant.hpp>
#include <migraphx/eliminate_contiguous.hpp>
#include <migraphx/eliminate_common_subexpression.hpp>
#include <migraphx/eliminate_pad.hpp>
#include <migraphx/rewrite_batchnorm.hpp>
#include <migraphx/rewrite_rnn.hpp>
#include <migraphx/rewrite_pooling.hpp>
#include <migraphx/eliminate_concat.hpp>
#include <migraphx/eliminate_identity.hpp>

struct gen_target
{
    std::string name() const
    {
        return "cpu-generate";
    }
    std::vector<migraphx::pass> get_passes(migraphx::context& ctx, const migraphx::compile_options&) const
    {
        return 
        {
            migraphx::decompose{},
            migraphx::dead_code_elimination{},
            migraphx::simplify_reshapes{},
            migraphx::dead_code_elimination{},
            migraphx::eliminate_identity{},
            migraphx::eliminate_pad{},
            migraphx::dead_code_elimination{},
            migraphx::rewrite_batchnorm{},
            migraphx::dead_code_elimination{},
            migraphx::rewrite_rnn{},
            migraphx::rewrite_pooling{},
            migraphx::dead_code_elimination{},
            migraphx::eliminate_common_subexpression{},
            migraphx::dead_code_elimination{},
            migraphx::simplify_algebra{},
            migraphx::dead_code_elimination{},
            migraphx::auto_contiguous{},
            migraphx::simplify_reshapes{},
            migraphx::dead_code_elimination{},
            migraphx::propagate_constant{},
            migraphx::dead_code_elimination{}
        };
    }
    migraphx::context get_context() const { return migraphx::context{}; }

    migraphx::argument copy_to(const migraphx::argument& arg) const { return arg; }
    migraphx::argument copy_from(const migraphx::argument& arg) const { return arg; }
    migraphx::argument allocate(const migraphx::shape& s) const
    {
        return {};
    }
};

std::string preamble = R"CODE(


)CODE";


template<class R, class F>
void delim(std::ostream& os, std::string d, R&& r, F f)
{
    bool first = true;
    for(auto&& x:r)
    {
        if (first)
            first = false;
        else
            os << d;
        os << f(x);
    }
}

struct generator
{
    migraphx::program * prog;
    std::unordered_map<std::string, std::function<std::string(migraphx::instruction_ref)>> apply_map{};

    std::stringstream fs{};
    std::size_t function_count = 0;

    void init()
    {

    }

    struct param
    {
        std::string name;
        std::string type;
    };

    std::string create_function(std::vector<param> params, std::string return_type, std::string content)
    {
        function_count++;
        std::string name = std::to_string(function_count);
        fs << return_type << " " << name << "(";
        char delim = '(';
        for(auto&& p:params)
        {
            fs << delim << p.type << " " << p.name;
            delim = ',';
        }
        fs << ") {\n" << content << "\n}\n";
        return name;
    }

    std::string to_array_type(std::string name, std::size_t n)
    {
        return "std::array<" + name + ", " + std::to_string(n) + ">";
    }


    std::string to_array_type(const migraphx::shape& s)
    {
        return to_array_type(s.type_string(), s.elements());
    }


    std::string create_pointwise(std::vector<migraphx::shape> inputs, migraphx::shape output, std::string op)
    {
        if (inputs.empty())
            return "";
        std::size_t n = inputs.front().elements();
        std::stringstream ss;
        ss << to_array_type(output) << " result;\n";
        ss << "for(std::size_t i = 0; i < " + std::to_string(n) + "; i++) {\n";
        std::size_t x = 0;
        for(auto input:inputs)
        {
            std::string index = "i";
            // if (not input.standard())
            // {
            // }
            ss << input.type_string() << " x" << x << " = input" << x << "[" << index << "];\n";
            x++;
        }
        ss << "result[i] = " << op << ";\n";
        ss << "}\n";
        ss << "return result;\n";
        std::vector<param> params;
        std::size_t input = 0;
        std::transform(inputs.begin(), inputs.end(), std::back_inserter(params), [&](auto&& s) {
            param p{"input" + std::to_string(input), to_array_type(s)};
            input++;
            return p;
        });
        return create_function(params, to_array_type(output), ss.str());
    }

    std::string generate()
    {
        init();
        std::unordered_map<migraphx::instruction_ref, std::string> names;
        std::stringstream ss;

        auto outputs = prog->get_output_shapes();
        assert(outputs.size() == 1);
        std::string output_type = to_array_type(outputs.front());

        ss << output_type << " run(";

        auto params = prog->get_parameter_shapes();
        delim(ss, ", ", params, [&](auto&& p) {
            auto&& name = p.first;
            auto&& s = p.second;

            return to_array_type(s) + " " + name;
        });

        ss << ") {\n";

        auto return_ins = std::prev(prog->end());

        for(auto ins:migraphx::iterator_for(*prog))
        {
            if (ins->name() == "@return")
            {
                assert(ins->inputs().size() == 1);
                return_ins = ins->inputs().front();
            }
            std::string name = "z" + std::to_string(names.size());
            if (ins->name() == "@param")
            {
                name = migraphx::any_cast<migraphx::builtin::param>(ins->get_operator()).parameter;   
            }
            names[ins] = name;
            if (apply_map.count(ins->name()) > 0)
            {
                std::string f = apply_map[ins->name()](ins);
                ss << "auto " << name << " = " << f;
                char delim = '(';
                for(auto input:ins->inputs())
                {
                    ss << delim << names.at(input);
                }
                ss << ");\n";
            }
            else if (ins->name() == "@literal")
            {
                ss << to_array_type(ins->get_shape()) << " " << name << " = {" << ins->get_literal() << "};\n";
            }
        }

        ss << "return " << names.at(return_ins) << ";\n";

        ss << "}\n";

        return fs.str() + ss.str();
    }

};

int main(int argc, const char* argv[])
{
    std::vector<std::string> args(argv + 1, argv + argc);
    if(args.empty())
        return 0;
    auto p = migraphx::parse_onnx(args.front());
    p.compile(gen_target{});
    std::string s = generator{&p}.generate();
    std::cout << s << std::endl;
    return 0;
}
