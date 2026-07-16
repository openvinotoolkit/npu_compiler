//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/utils/ov/compat_string_parser.hpp"

#include <gtest/gtest.h>

using namespace vpux::compat;

auto parseInput(const std::string& input) {
    // empty legalAttributeNames to allow any attribute in tests
    Parser parser(input, std::array<std::string_view, 0>{});
    return parser.getAttributes();
}

TEST(CompatStringParserTest, SimpleAttributes) {
    Parser parser("key=VALUE;my_id=123", std::array{"key", "my_id"});
    const auto attrs = parser.getAttributes();
    EXPECT_EQ(attrs.size(), 2);
    EXPECT_EQ(attrs.at("key"), "VALUE");
    EXPECT_EQ(attrs.at("my_id"), "123");
}

TEST(CompatStringParserTest, DuplicateAttributes) {
    EXPECT_THROW(parseInput("same=1;same=2"), std::runtime_error);
    EXPECT_THROW(parseInput("same=1;{same=2}"), std::runtime_error);
    EXPECT_NO_THROW(parseInput("same=1;list=[same=2]"));
}

TEST(CompatStringParserTest, OptionalAttributes) {
    auto parser = Parser("key=VAL;{optional=NESTED_VAL}", std::array{"key"});
    const auto& attrs = parser.getAttributes();
    const auto& optionalAttrs = parser.getOptionalAttributes();
    EXPECT_EQ(attrs.size(), 1);
    EXPECT_EQ(optionalAttrs.size(), 1);
    EXPECT_EQ(attrs.at("key"), "VAL");
    EXPECT_EQ(optionalAttrs.at("optional"), "NESTED_VAL");
}

TEST(CompatStringParserTest, NestedOptionalAttributes) {
    auto parser = Parser("{a=1;{b=2};c=3}", std::array{"a", "b", "c"});
    const auto& attrs = parser.getAttributes();
    const auto& optionalAttrs = parser.getOptionalAttributes();
    EXPECT_EQ(attrs.size(), 0);
    EXPECT_EQ(optionalAttrs.size(), 3);
    EXPECT_EQ(optionalAttrs.at("a"), "1");
    EXPECT_EQ(optionalAttrs.at("b"), "2");
    EXPECT_EQ(optionalAttrs.at("c"), "3");
}

TEST(CompatStringParserTest, ListCaptureOpaque) {
    const auto attrs = parseInput("data=[attr=A|attr=B]");
    ASSERT_TRUE(attrs.count("data"));
    EXPECT_EQ(attrs.at("data"), "[attr=A|attr=B]");
}

TEST(CompatStringParserTest, NestedListsAndBracesInList) {
    const auto attrs = parseInput("config=[type=X|meta=X;{id=1;tags=[a=1|b=2]}]");
    EXPECT_EQ(attrs.size(), 1);
    EXPECT_EQ(attrs.at("config"), "[type=X|meta=X;{id=1;tags=[a=1|b=2]}]");
}

TEST(CompatStringParserTest, SplitListHelper) {
    std::string listContent = "[item=1|item=2;extra=[sub=3]|item=[a=1;b=2;{c=[x=1]}]]";
    auto parts = Parser::splitList(listContent);

    ASSERT_EQ(parts.size(), 3);
    EXPECT_EQ(parts[0], "item=1");
    EXPECT_EQ(parts[1], "item=2;extra=[sub=3]");
    EXPECT_EQ(parts[2], "item=[a=1;b=2;{c=[x=1]}]");
}

TEST(CompatStringParserTest, EmptyBlocks) {
    EXPECT_THROW(parseInput(""), std::runtime_error);
    EXPECT_THROW(parseInput("{}"), std::runtime_error);
    EXPECT_THROW(parseInput("list=[]"), std::runtime_error);
}

TEST(CompatStringParserTest, InvalidString) {
    EXPECT_THROW(parseInput(";"), std::runtime_error);
    EXPECT_THROW(parseInput("key"), std::runtime_error);
    EXPECT_THROW(parseInput("=VALUE"), std::runtime_error);
    EXPECT_THROW(parseInput("inner=[a=1|b=2]|c=3"), std::runtime_error);
    EXPECT_THROW(parseInput("list=[a=1||b=2]"), std::runtime_error);
    EXPECT_THROW(parseInput("{"), std::runtime_error);

    EXPECT_THROW(parseInput("key=VALUE;"), std::runtime_error);
    EXPECT_THROW(parseInput("key=VALUE;{}"), std::runtime_error);
}

// ---- Error message tests ----

// Expect error message
#define EXPECT_THROW_MSG(statement, exception_type, msg)                                               \
    do {                                                                                               \
        try {                                                                                          \
            statement;                                                                                 \
            FAIL() << "Expected " #exception_type " to be thrown";                                     \
        } catch (const exception_type& e) {                                                            \
            EXPECT_EQ(std::string(e.what()), msg);                                                     \
        } catch (...) {                                                                                \
            ADD_FAILURE() << "Expected " #exception_type " but a different exception type was thrown"; \
        }                                                                                              \
    } while (false)

TEST(CompatStringParserTest, ErrorMessages) {
    // errorAt() — peek()==0, message appended with " at the end of the string"
    EXPECT_THROW_MSG(parseInput("key="), std::runtime_error, "expected attribute value at the end of the string");
    EXPECT_THROW_MSG(parseInput("key"), std::runtime_error, "expected '=' at the end of the string");
    EXPECT_THROW_MSG(parseInput("{key=1"), std::runtime_error, "expected '}' at the end of the string");
    EXPECT_THROW_MSG(parseInput("key=[a=1"), std::runtime_error, "expected ']' at the end of the string");
    // errorAt() - "Unexpected character '<c>': <message>"
    EXPECT_THROW_MSG(parseInput("=VALUE"), std::runtime_error, "Unexpected character '=': expected attribute name");
    EXPECT_THROW_MSG(parseInput("key=;"), std::runtime_error, "Unexpected character ';': expected attribute value");
    EXPECT_THROW_MSG(parseInput("keyVALUE"), std::runtime_error, "Unexpected character 'V': expected '='");
    EXPECT_THROW_MSG(parseInput("key=VALUE}"), std::runtime_error,
                     "Unexpected character '}': at the end of the string");
    EXPECT_THROW_MSG(parseInput(" "), std::runtime_error, "Unexpected character expected attribute name");
    EXPECT_THROW_MSG(parseInput("a = 1"), std::runtime_error, "Unexpected character expected '='");
    EXPECT_THROW_MSG(parseInput("a=1 "), std::runtime_error, "Unexpected character at the end of the string");
    // Direct runtime_error (no errorAt)
    EXPECT_THROW_MSG(parseInput("dup=1;dup=2"), std::runtime_error, "Duplicate attribute: dup");
    EXPECT_THROW_MSG(parseInput("dup=1;{dup=2}"), std::runtime_error, "Duplicate attribute: dup");
    EXPECT_THROW_MSG(
            parseInput(
                    "a=[b=[c=[d=[e=[f=[g=[h=[i=[j=[k=[l=[m=[n=[o=[p=[q=[r=[s=[t=[u=[v=[w=[a=1]]]]]]]]]]]]]]]]]]]]]]]"),
            std::runtime_error, "Unexpected character '[': Maximum nesting depth exceeded");
    EXPECT_THROW_MSG(parseInput("{{{{{{{{{{{{{{{{{{{{{{a=1}}}}}}}}}}}}}}}}}}}}}}"), std::runtime_error,
                     "Unexpected character '{': Maximum nesting depth exceeded");
}

TEST(CompatStringParserTest, AttributeErrors) {
    Parser parser("key=VALUE", std::array{"key"});
    EXPECT_THROW_MSG(parser.getAttribute("missing"), std::runtime_error, "Attribute not found: missing");
    EXPECT_THROW_MSG((Parser("key=VALUE;bad=1", std::array{"key"})), std::runtime_error, "Illegal attribute: bad");
}

TEST(CompatStringParserTest, ListErrors) {
    EXPECT_THROW_MSG(Parser::splitList("[]"), std::runtime_error, "Unexpected character ']': expected attribute name");
    EXPECT_THROW_MSG(Parser::splitList("[a=1||b=2]"), std::runtime_error,
                     "Unexpected character '|': expected attribute name");
    EXPECT_THROW_MSG(Parser::splitList("[a=1|b=2"), std::runtime_error, "expected ']' at the end of the string");
    EXPECT_THROW_MSG(Parser::splitList("[a=1|b=2];"), std::runtime_error,
                     "Unexpected character ';': at the end of the list");
    EXPECT_THROW_MSG(Parser::splitList("[a=1|b=2]\n"), std::runtime_error,
                     "Unexpected character at the end of the list");
}
