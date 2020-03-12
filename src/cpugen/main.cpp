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
#include <migraphx/op/convolution.hpp>
#include <migraphx/op/pooling.hpp>
#include <migraphx/op/dot.hpp>

struct gen_target
{
    std::string name() const { return "cpu-generate"; }
    std::vector<migraphx::pass> get_passes(migraphx::context& ctx,
                                           const migraphx::compile_options&) const
    {
        return {migraphx::decompose{},
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
                migraphx::dead_code_elimination{}};
    }
    migraphx::context get_context() const { return migraphx::context{}; }

    migraphx::argument copy_to(const migraphx::argument& arg) const { return arg; }
    migraphx::argument copy_from(const migraphx::argument& arg) const { return arg; }
    migraphx::argument allocate(const migraphx::shape& s) const { return {}; }
};

std::string preamble = R"CODE(
inline auto dfor()
{
    return [](auto f) { f(); };
}

template <class T, class... Ts>
auto dfor(T x, Ts... xs)
{
    return [=](auto f) {
        for(T i = 0; i < x; i++)
        {
            dfor(xs...)([&](Ts... is) { f(i, is...); });
        }
    };
}

inline std::size_t index(std::vector<std::size_t> strides, std::vector<std::size_t> idx)
{
    return std::inner_product(idx.begin(), idx.end(), strides().begin(), std::size_t{0})
}
)CODE";

template <class R, class F>
void delim(std::ostream& os, std::string d, R&& r, F f)
{
    bool first = true;
    for(auto&& x : r)
    {
        if(first)
            first = false;
        else
            os << d;
        os << f(x);
    }
}

inline void replace_string(std::string& subject, const std::string& search, const std::string& replace)
{
    size_t pos = 0;
    while((pos = subject.find(search, pos)) != std::string::npos)
    {
        subject.replace(pos, search.length(), replace);
        pos += replace.length();
    }
}

template<class T>
inline void replace_string(std::string& subject, const std::string& search, T x)
{
    replace_string(subject, search, std::to_string(x));
}

template <class Strings>
inline std::string join_strings(Strings strings, const std::string& delim)
{
    auto it = strings.begin();
    if(it == strings.end())
        return "";

    auto nit = std::next(it);
    return std::accumulate(nit, strings.end(), *it, [&](std::string x, std::string y) {
        return std::move(x) + delim + std::move(y);
    });
}

struct generator
{
    migraphx::program* prog;
    std::unordered_map<std::string, std::function<std::string(migraphx::instruction_ref)>>
        apply_map{};

    std::stringstream fs{};
    std::size_t function_count = 0;

    void init()
    {
        add_pointwise("add", "x0 + x1");
        add_pointwise("mul", "x0 * x1");
        add_pointwise("relu", "std::max<decltype(x0)>(0, x0)");
        apply_map["convolution"] = [=](auto ins) {
            auto op = migraphx::any_cast<migraphx::op::convolution>(ins->get_operator());
            return create_convolution(to_shapes(ins->inputs()), ins->get_shape(), op);
        };
        apply_map["pooling"] = [=](auto ins) {
            auto op = migraphx::any_cast<migraphx::op::pooling>(ins->get_operator());
            return create_pooling(to_shapes(ins->inputs()), ins->get_shape(), op);
        };
        apply_map["dot"] = [=](auto ins) {
            auto op = migraphx::any_cast<migraphx::op::dot>(ins->get_operator());
            return create_dot(to_shapes(ins->inputs()), ins->get_shape(), op);
        };
    }

    struct param
    {
        std::string name;
        std::string type;
    };

    std::vector<migraphx::shape> to_shapes(std::vector<migraphx::instruction_ref> v)
    {
        std::vector<migraphx::shape> shapes;
        std::transform(v.begin(), v.end(), std::back_inserter(shapes), [](auto i) {
            return i->get_shape();
        });
        return shapes;
    }

    std::string
    create_function(std::vector<param> params, std::string return_type, std::string content)
    {
        function_count++;
        std::string name = "f" + std::to_string(function_count);
        fs << return_type << " " << name << "(";
        char delim = '(';
        for(auto&& p : params)
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

    template<class T>
    static void replace_array(std::string& s, const std::string& search, std::vector<T> v)
    {
        std::vector<std::string> strings;
        std::transform(v.begin(), v.end(), std::back_inserter(strings), [](auto x) {
            return std::to_string(x);
        });
        replace_string(s, search, "{" + join_strings(strings, ", ") + "}");
    }

    static void replace_shape(std::string& s, const std::string& search, migraphx::shape shp)
    {
        replace_array(s, search + ".lens", shp.lens());
        replace_array(s, search + ".strides", shp.strides());
    }

    std::string create_dot(std::vector<migraphx::shape> inputs, migraphx::shape output, migraphx::op::dot op)
    {
        std::string body = R"CODE(
            $result_type cmat;
            gemm(amat, bmat, cmat);
            return cmat;
        )CODE";
        replace_string(body, "$result_type", to_array_type(output));
        return create_function({{"amat", to_array_type(inputs.at(0))}, {"bmat", to_array_type(inputs.at(1))}}, to_array_type(output), body);
    }

    std::string create_pooling(std::vector<migraphx::shape> inputs, migraphx::shape output, migraphx::op::pooling op)
    {
        std::string body = R"CODE(
            auto input_lens = $input.lens;
            auto in_h  = input_lens[2];
            auto in_w  = input_lens[3];

            auto output_lens = $output.lens;

            dfor(output_lens[0],
                     output_lens[1],
                     output_lens[2],
                     output_lens[3])(
                [&](std::size_t o, std::size_t w, std::size_t i, std::size_t j) {
                    const int start_x0 = i * $stride[0] - $padding[0];
                    const int start_y0 = j * $stride[1] - $padding[1];

                    const int hend = std::min(start_x0 + $lengths[0], in_h);
                    const int wend = std::min(start_y0 + $lengths[1], in_w);

                    const int start_x = std::max(start_x0, 0);
                    const int start_y = std::max(start_y0, 0);

                    const int w_h       = (hend - start_x);
                    const int w_w       = (wend - start_y);
                    const int pool_size = std::max(w_h * w_w, 1);

                    double acc = std::numeric_limits<double>::lowest();
                    dfor(w_h, w_w)([&](int x, int y) {
                        const int in_x = start_x + x;
                        const int in_y = start_y + y;
                        if(in_x >= 0 && in_x < in_h && in_y >= 0 && in_y < in_w)
                        {
                            auto input_idx = index($input.strides, {o, w, in_x, in_y});
                            acc = std::max<double>(acc, input[input_idx]]);
                        }
                    });
                    auto output_idx = index($output.strides, {o, w, i, j});
                    output[output_idx] = acc;
                });
        )CODE";
        replace_shape(body, "$input", inputs.at(0));
        replace_shape(body, "$output", output);
        replace_string(body, "$padding[0]", op.padding[0]);
        replace_string(body, "$padding[1]", op.padding[1]);
        replace_string(body, "$stride[0]", op.stride[0]);
        replace_string(body, "$stride[1]", op.stride[1]);
        replace_string(body, "$lengths[0]", op.lengths[0]);
        replace_string(body, "$lengths[1]", op.lengths[1]);
        replace_string(body, "$result_type", to_array_type(output));
        return create_function({{"input", to_array_type(inputs.at(0))}}, to_array_type(output), body);

    }
    std::string create_convolution(std::vector<migraphx::shape> inputs, migraphx::shape output, migraphx::op::convolution op)
    {
        std::string body = R"CODE(
            $result_type output;
            auto in   = $input.lens;
            auto in_h = in[2];
            auto in_w = in[3];

            auto wei   = $weight.lens;
            auto wei_n = wei[0];
            auto wei_c = wei[1];
            auto wei_h = wei[2];
            auto wei_w = wei[3];

            auto output_lens = $output.lens;

            dfor(output_lens[0],
                     output_lens[1],
                     output_lens[2],
                     output_lens[3])(
                [&](std::size_t o, std::size_t w, std::size_t i, std::size_t j) {
                    const auto start_x  = i * $stride[0] - $padding[0];
                    const auto start_y  = j * $stride[1] - $padding[1];
                    const auto group_id = w / (wei_n / $group);

                    double acc = 0;
                    dfor(wei_c, wei_h, wei_w)([&](std::size_t k, std::size_t x, std::size_t y) {
                        const auto in_x  = start_x + x;
                        const auto in_y  = start_y + y;
                        const auto in_ch = group_id * wei_c + k;
                        if(in_x >= 0 && in_x < in_h && in_y >= 0 && in_y < in_w)
                        {
                            auto input_idx = index($input.strides, {o, in_ch, in_x, in_y});
                            auto weights_idx = index($weights.strides, {w, k, x, y});
                            acc += input[input_idx] * weights[weights_idx];
                        }
                    });
                    auto output_idx = index($output.strides, {o, w, i, j});
                    output[output_idx] = acc;
                });
            return output;
        )CODE";
        replace_shape(body, "$input", inputs.at(0));
        replace_shape(body, "$weights", inputs.at(1));
        replace_shape(body, "$output", output);
        replace_string(body, "$padding[0]", op.padding[0]);
        replace_string(body, "$padding[1]", op.padding[1]);
        replace_string(body, "$stride[0]", op.stride[0]);
        replace_string(body, "$stride[1]", op.stride[1]);
        replace_string(body, "$group", op.group);
        replace_string(body, "$result_type", to_array_type(output));
        return create_function({{"input", to_array_type(inputs.at(0))}, {"weights", to_array_type(inputs.at(1))}}, to_array_type(output), body);

    }

    std::string
    create_pointwise(std::vector<migraphx::shape> inputs, migraphx::shape output, std::string op)
    {
        if(inputs.empty())
            return "";
        std::size_t n = inputs.front().elements();
        std::stringstream ss;
        ss << to_array_type(output) << " result;\n";
        ss << "for(std::size_t i = 0; i < " + std::to_string(n) + "; i++) {\n";
        std::size_t x = 0;
        for(auto input : inputs)
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

    void add_pointwise(std::string name, std::string content)
    {
        apply_map[name] = [=](auto ins) {
            return create_pointwise(to_shapes(ins->inputs()), ins->get_shape(), content);
        };
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
            auto&& s    = p.second;

            return to_array_type(s) + " " + name;
        });

        ss << ") {\n";

        auto return_ins = std::prev(prog->end());

        for(auto ins : migraphx::iterator_for(*prog))
        {
            ss << "// " << ins->get_operator() << " -> " << ins->get_shape() << "\n";
            if(ins->name() == "@return")
            {
                assert(ins->inputs().size() == 1);
                return_ins = ins->inputs().front();
            }
            std::string name = "z" + std::to_string(names.size());
            if(ins->name() == "@param")
            {
                name = migraphx::any_cast<migraphx::builtin::param>(ins->get_operator()).parameter;
            }
            names[ins] = name;
            if(apply_map.count(ins->name()) > 0)
            {
                std::string f = apply_map[ins->name()](ins);
                ss << "auto " << name << " = " << f << "(";
                delim(ss, ", ", ins->inputs(), [&](auto input) {
                    return names.at(migraphx::instruction::get_output_alias(input));
                });
                ss << ");\n";
            }
            else if(ins->name() == "@literal")
            {
                ss << to_array_type(ins->get_shape()) << " " << name << " = {" << ins->get_literal()
                   << "};\n";
            }
        }

        ss << "return " << names.at(return_ins) << ";\n";

        ss << "}\n";

        return preamble + fs.str() + ss.str();
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
