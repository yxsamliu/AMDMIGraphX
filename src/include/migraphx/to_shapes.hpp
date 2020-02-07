#ifndef MIGRAPHX_GUARD_RTGLIB_TO_SHAPES_HPP
#define MIGRAPHX_GUARD_RTGLIB_TO_SHAPES_HPP

#include <migraphx/shape.hpp>
#include <migraphx/instruction_ref.hpp>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct argument;

std::vector<shape> to_shapes(const std::vector<instruction_ref>& args);
std::vector<shape> to_shapes(const std::vector<argument>& args);

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
