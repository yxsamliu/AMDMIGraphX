#include <migraphx/to_shapes.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/instruction.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

std::vector<shape> to_shapes(const std::vector<instruction_ref>& args)
{
    std::vector<shape> shapes(args.size());
    std::transform(
        args.begin(), args.end(), shapes.begin(), [](instruction_ref i) { return i->get_shape(); });
    return shapes;
}

std::vector<shape> to_shapes(const std::vector<argument>& args)
{
    std::vector<shape> shapes(args.size());
    std::transform(
        args.begin(), args.end(), shapes.begin(), [](argument i) { return i.get_shape(); });
    return shapes;
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
