//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

//

#include "vpux/utils/core/simple_math.hpp"

#include "vpux/utils/core/error.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <stack>

using namespace vpux;

//
// IntOrFloat
//

vpux::IntOrFloat::IntOrFloat(int x) {
    _value.i = x;
    _isInt = true;
}

vpux::IntOrFloat::IntOrFloat(float x) {
    _value.f = x;
    _isInt = false;
}

vpux::IntOrFloat::IntOrFloat(StringRef str) {
    int integer = 0;
    if (!str.getAsInteger(10, integer)) {
        _value.i = integer;
        _isInt = true;
        return;
    }

    double fp = 0.0;
    if (!str.getAsDouble(fp)) {
        _value.f = static_cast<float>(fp);
        _isInt = false;
        return;
    }

    VPUX_THROW("Failed to convert string '{0}' to number", str);
}

namespace {

template <template <typename> class Op>
IntOrFloat applyOp(const IntOrFloat& a, const IntOrFloat& b) {
    if (a.isInt() && b.isInt()) {
        Op<int> op;
        return IntOrFloat(op(a.asInt(), b.asInt()));
    }

    Op<float> op;
    return IntOrFloat(op(a.asFloat(), b.asFloat()));
}

}  // namespace

IntOrFloat vpux::operator+(const IntOrFloat& a, const IntOrFloat& b) {
    return applyOp<std::plus>(a, b);
}

IntOrFloat vpux::operator-(const IntOrFloat& a, const IntOrFloat& b) {
    return applyOp<std::minus>(a, b);
}

IntOrFloat vpux::operator*(const IntOrFloat& a, const IntOrFloat& b) {
    return applyOp<std::multiplies>(a, b);
}

IntOrFloat vpux::operator/(const IntOrFloat& a, const IntOrFloat& b) {
    return applyOp<std::divides>(a, b);
}

IntOrFloat vpux::operator%(const IntOrFloat& a, const IntOrFloat& b) {
    if (a.isInt() && b.isInt()) {
        return IntOrFloat(a.asInt() % b.asInt());
    }

    VPUX_THROW("Can't apply modulus operation to floating point values : '{0} % {1}'", a.asFloat(), b.asFloat());
}

//
// MathExpression
//

void vpux::MathExpression::setVariables(const VarMap& variables) {
    for (const auto& p : variables) {
        // if string converts to float, it also will be able to convert to int
        StringRef var(p.second);
        double fp = 0.0;
        if (!var.getAsDouble(fp)) {
            _vars.emplace(p.first, IntOrFloat(var));
        }
    }
}

namespace {

struct Operator final {
    int priority;
    std::function<IntOrFloat(IntOrFloat, IntOrFloat)> op;
};

using UnaryFunc = std::function<IntOrFloat(IntOrFloat)>;

static const std::unordered_map<char, Operator> operators = {{'+', {0, std::plus<IntOrFloat>()}},
                                                             {'-', {0, std::minus<IntOrFloat>()}},
                                                             {'*', {1, std::multiplies<IntOrFloat>()}},
                                                             {'/', {1, std::divides<IntOrFloat>()}},
                                                             {'%', {1, std::modulus<IntOrFloat>()}}};

static const std::unordered_map<StringRef, UnaryFunc> functions = {{"floor",
                                                                    [](IntOrFloat x) {
                                                                        return IntOrFloat(std::floor(x.asFloat()));
                                                                    }},
                                                                   {"ceil",
                                                                    [](IntOrFloat x) {
                                                                        return IntOrFloat(std::ceil(x.asFloat()));
                                                                    }},
                                                                   {"round",
                                                                    [](IntOrFloat x) {
                                                                        return IntOrFloat(std::round(x.asFloat()));
                                                                    }},
                                                                   {"abs",
                                                                    [](IntOrFloat x) {
                                                                        return IntOrFloat(std::abs(x.asFloat()));
                                                                    }},
                                                                   {"sqrt", [](IntOrFloat x) {
                                                                        return IntOrFloat(std::sqrt(x.asFloat()));
                                                                    }}};

bool isFunction(StringRef token) {
    return functions.find(token) != functions.end();
}

bool isOperator(char token) {
    return operators.find(token) != operators.end();
}

int opPriority(char token) {
    return operators.at(token).priority;
}

}  // namespace

void MathExpression::parse(StringRef expression) {
    _tokens.clear();
    std::stack<StringRef> tokenStack;

    StringRef leftPart = expression;
    while (!leftPart.empty()) {
        const auto curCh = leftPart.front();

        if (curCh == ' ' || curCh == '\t') {
            leftPart = leftPart.drop_front();
            continue;
        }

        // parse number
        if (std::isdigit(curCh)) {
            // parse number and use its length
            char* endPos = nullptr;
            std::strtof(leftPart.data(), &endPos);
            const auto len = endPos - leftPart.data();

            const auto curToken = leftPart.substr(0, len);
            _tokens.emplace_back(TokenType::Value, IntOrFloat(curToken), "");

            leftPart = leftPart.drop_front(curToken.size());
            continue;
        }

        // parse variable/function
        if (std::isalpha(curCh)) {
            const auto curToken = leftPart.take_while([](char c) {
                return std::isalnum(c) || c == '_';
            });

            leftPart = leftPart.drop_front(curToken.size());

            if (isFunction(curToken)) {
                tokenStack.push(curToken);
                continue;
            }

            const auto it = _vars.find(curToken.str());
            if (it != _vars.end()) {
                _tokens.emplace_back(TokenType::Value, it->second, "");
                continue;
            }
        }

        // parse operator
        if (isOperator(curCh)) {
            const auto curToken = leftPart.substr(0, 1);

            while (!tokenStack.empty() &&
                   (isFunction(tokenStack.top()) ||
                    (isOperator(tokenStack.top()[0]) && opPriority(tokenStack.top()[0]) >= opPriority(curCh)))) {
                const auto tokenType = isOperator(tokenStack.top()[0]) ? TokenType::Operator : TokenType::Function;
                _tokens.emplace_back(tokenType, IntOrFloat(0), tokenStack.top());
                tokenStack.pop();
            }

            tokenStack.push(curToken);

            leftPart = leftPart.drop_front(curToken.size());
            continue;
        }

        if (curCh == '(') {
            tokenStack.push("(");

            leftPart = leftPart.drop_front();
            continue;
        }

        if (curCh == ')') {
            while (!tokenStack.empty() && tokenStack.top() != "(") {
                const auto tokenType = isOperator(tokenStack.top()[0]) ? TokenType::Operator : TokenType::Function;
                _tokens.emplace_back(tokenType, IntOrFloat(0), tokenStack.top());
                tokenStack.pop();
            }

            VPUX_THROW_WHEN(tokenStack.empty(), "Mismatched parentheses in '{0}'", expression);
            tokenStack.pop();

            leftPart = leftPart.drop_front();
            continue;
        }

        VPUX_THROW("Unknown token '{0}' in '{1}'", curCh, expression);
    }

    while (!tokenStack.empty()) {
        VPUX_THROW_WHEN(tokenStack.top() == "(", "Mismatched parentheses in '{0}'", expression);

        const auto tokenType = isOperator(tokenStack.top()[0]) ? TokenType::Operator : TokenType::Function;
        _tokens.emplace_back(tokenType, IntOrFloat(0), tokenStack.top());

        tokenStack.pop();
    }
}

IntOrFloat MathExpression::evaluate() const {
    std::stack<IntOrFloat> values;

    for (const auto& token : _tokens) {
        switch (token._type) {
        case TokenType::Value:
            values.push(token._value);
            break;

        case TokenType::Operator: {
            VPUX_THROW_UNLESS(values.size() >= 2, "Illegal expression: not enough values for operator evaluation");

            const auto val2 = values.top();
            values.pop();

            const auto val1 = values.top();
            values.pop();

            values.push(operators.at(token._opName[0]).op(val1, val2));
            break;
        }

        case TokenType::Function: {
            VPUX_THROW_WHEN(values.empty(), "Illegal expression: not enough values for function evaluation");

            const auto val1 = values.top();
            values.pop();

            values.push(functions.at(token._opName)(val1));
            break;
        }

        default:
            VPUX_THROW("Illegal expression: unhandled token");
        }
    }

    VPUX_THROW_UNLESS(values.size() == 1, "Illegal expression: not enough operators");

    return values.top();
}
