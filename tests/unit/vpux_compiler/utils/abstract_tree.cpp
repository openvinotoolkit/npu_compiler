//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/utils/abstract_tree.hpp"

#include <gtest/gtest.h>

#include <sstream>
#include <stack>

using namespace vpux;

// Note: these tests give good intuition as to what is possible to achieve with
// the abstract tree.

using IntegerTree = utils::AbstractTree<int>;
struct IntegerPrinter final : IntegerTree::Visitor {
    std::vector<std::string> lines;
    size_t indent = 0;

    bool visit(const Node& node) final {
        std::stringstream stream;
        stream << std::string(indent, '.') << node.data();
        lines.push_back(stream.str());
        ++indent;
        return true;
    }

    void endVisit(const Node&) final {
        --indent;
    }
};

TEST(AbstractTreeTest, BasicConstructionAndPrinting) {
    std::vector<IntegerTree::Node> roots = {
            IntegerTree::Node(0,
                              {
                                      IntegerTree::Node(1),
                                      IntegerTree::Node(2, {IntegerTree::Node(3)}),
                              }),
            IntegerTree::Node(4),
    };
    IntegerTree tree(std::move(roots));
    IntegerPrinter printer;
    tree.apply(printer);

    const std::vector<std::string> expected = {
            "0", ".1", ".2", "..3", "4",
    };
    ASSERT_EQ(printer.lines, expected);
}

TEST(AbstractTreeTest, RecursiveConstruction) {
    std::vector<IntegerTree::Node> roots = {
            IntegerTree::Node(0),
            IntegerTree::Node(1),
    };
    size_t count = 0;
    const auto addTwoAndFour = [&](const IntegerTree::Node& curr) -> std::vector<int> {
        ++count;
        if (count > 2) {
            // artificial threshold to stop generating the tree infinitely
            return {};
        }

        const int value = curr.data();
        return {value + 2, value + 4};
    };
    IntegerTree tree(std::move(roots), addTwoAndFour);
    IntegerPrinter printer;
    tree.apply(printer);

    const std::vector<std::string> expected = {
            "0", ".2", "..4", "..6", ".4", "1",
    };

    ASSERT_EQ(printer.lines, expected);
}

// defines the constant operation to perform
enum ConstFoldingOp {
    Add_5,
    MultiplyBy_2,
    Subtract_8,
};
// defines a tree of operations on constants
using ConstFoldingTree = utils::AbstractTree<ConstFoldingOp>;
struct ConstantFolder final : ConstFoldingTree::Visitor {
    // used to mimic DFS algorithm's temporary result(s).
    std::stack<int> intermediateResults;
    // stores all folded constants.
    std::vector<int> results;

    bool visit(const Node& node) final {
        int result = intermediateResults.top();
        switch (node.data()) {
        case Add_5:
            result += 5;
            break;
        case MultiplyBy_2:
            result *= 2;
            break;
        case Subtract_8:
            result -= 8;
            break;
        }

        if (node.children().empty()) {  // leaf node
            results.push_back(result);
            return false;  // prevents endVisit
        }

        // Note: in real constant folding we may want to have a smarter approach
        // here to save RAM. for instance, if parent allocates stack for child
        // (or if child knows that it is a single child), instead of growing the
        // stack, the existing entry could be reused i.e.
        // `intermediateResults.back() = result` because we know that there
        // are no more users of `intermediateResults.back()`. for the sake
        // of tests, however, keep this simple.

        intermediateResults.push(result);
        return true;
    }

    void endVisit(const Node& node) final {
        VPUX_THROW_WHEN(node.children().empty(), "This is supposed to be handled in visit()");
        intermediateResults.pop();
    }
};

TEST(AbstractTreeTest, ConstantFolding) {
    std::vector<ConstFoldingTree::Node> roots = {
            ConstFoldingTree::Node(Add_5,
                                   {
                                           ConstFoldingTree::Node(MultiplyBy_2),
                                           ConstFoldingTree::Node(Subtract_8, {ConstFoldingTree::Node(Add_5)}),
                                   }),
            ConstFoldingTree::Node(MultiplyBy_2,
                                   {
                                           ConstFoldingTree::Node(MultiplyBy_2),
                                           ConstFoldingTree::Node(Subtract_8),
                                   }),
    };

    ConstFoldingTree tree(std::move(roots));
    ConstantFolder folder;
    folder.intermediateResults.push(42);  // starting value
    tree.apply(folder);

    ASSERT_EQ(folder.intermediateResults.size(), 1) << "Starting value must still be present";
    ASSERT_EQ(folder.results.size(), 4) << "There are a total of 4 leaves, so 4 results must be given";

    const std::vector<int> expected = {(42 + 5) * 2, ((42 + 5) - 8) + 5, (42 * 2) * 2, (42 * 2) - 8};
    ASSERT_EQ(folder.results, expected);
}
