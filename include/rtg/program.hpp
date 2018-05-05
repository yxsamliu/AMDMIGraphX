#ifndef RTG_GUARD_RTGLIB_PROGRAM_HPP
#define RTG_GUARD_RTGLIB_PROGRAM_HPP

#include <list>
#include <unordered_map>
#include <rtg/operation.hpp>
#include <rtg/literal.hpp>
#include <rtg/builtin.hpp>
#include <rtg/instruction_ref.hpp>
#include <algorithm>

namespace rtg {

struct program_impl;

struct program
{
    program();
    program(program&&) noexcept;
    program& operator=(program&&) noexcept;
    ~program() noexcept;

    template <class... Ts>
    instruction_ref add_instruction(operation op, Ts... args)
    {
        return add_instruction(op, {args...});
    }
    instruction_ref add_instruction(operation op, std::vector<instruction_ref> args);
    template <class... Ts>
    instruction_ref add_literal(Ts&&... xs)
    {
        return add_literal(literal{std::forward<Ts>(xs)...});
    }

    instruction_ref add_literal(literal l);

    instruction_ref add_parameter(std::string name, shape s);

    literal eval(std::unordered_map<std::string, argument> params) const;

    // TODO: Change to stream operator
    void print() const;

    bool has_instruction(instruction_ref ins) const;

    private:
    std::unique_ptr<program_impl> impl;
};

} // namespace rtg

#endif
