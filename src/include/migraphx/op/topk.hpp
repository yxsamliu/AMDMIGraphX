#ifndef MIGRAPHX_GUARD_OPERATORS_TOPK_HPP
#define MIGRAPHX_GUARD_OPERATORS_TOPK_HPP

#include <migraphx/operation.hpp>
#include <migraphx/check_shapes.hpp>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct topk
{
    int k = 1;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.k, "k"));
    }

    std::string name() const { return "topk"; }
    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{{inputs[0]}}.standard();
        auto dims = inputs[0].lens();
        dims.insert(dims.begin(), 2);

        for(size_t i = 1; i < dims.size(); i++)
        {
            dims[i] = k;
        }
        return shape{inputs[0].type(), dims};
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
