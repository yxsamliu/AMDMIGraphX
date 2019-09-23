#ifndef MIGRAPHX_GUARD_OPERATORS_HARD_SIGMOID_HPP
#define MIGRAPHX_GUARD_OPERATORS_HARD_SIGMOID_HPP

#include <array>
#include <migraphx/op/unary.hpp>
#include <migraphx/operation.hpp>
#include <migraphx/check_shapes.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/streamutils.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/shape_for_each.hpp>
#include <migraphx/config.hpp>
#include <cmath>
#include <utility>
#include <limits>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct hard_sigmoid : unary<hard_sigmoid>
{
    float alpha_val = 0.2;
    float beta_val = 0.5;

    hard_sigmoid() {}

    hard_sigmoid(float alpha, float beta) : alpha_val(alpha), beta_val(beta) {}

    auto apply() const
    {
        auto alpha = alpha_val;
        auto beta = beta_val;
        return [alpha, beta](auto x) {
            // using type = decltype(x);
            // return std::min(std::max(type(min), x), type(max));
            return std::max(0, std::min(1, alpha * x + beta));
        };
    }

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.max_val, "max"), f(self.min_val, "min"));
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
