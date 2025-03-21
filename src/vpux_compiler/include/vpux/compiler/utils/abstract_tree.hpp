//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/func_ref.hpp"

#include <functional>

namespace vpux::utils {

/** @brief An abstract node tree capable of representing tree-like structures
    found in IR.

    This data structure is a fairly simple representation of a tree with nodes
    being abstract (templated). By carefully choosing which data the node
    stores, one could make the tree "non-owning" (when data is a non-owning
    pointer), or make every node "virtual" (when data is a virtual interface).

    @note As this class stores multiple roots, this is technically not a tree
    but rather a "forest", yet it is called a tree for the sake of simplicity.
 */
template <typename Data>
struct AbstractTree {
    class Node;

    /** @brief A visitor interface that specifies the API to "visit" every node
        in the tree.

        A two-visit visitor abstraction that allows one to visit the node once
        before its children and then visit the same node again "on the trip
        back", when all of the children are already visited twice:
        ```
        parent      // visit for 'parent' happens
        |- node1    // visit + endVisit for 'node1' happen
        |- node2    // visit + endVisit for 'node2' happen
                    // endVisit for 'parent' happens
        ```
    */
    struct Visitor {
        virtual ~Visitor() = default;
        using Base = Visitor;
        using Node = AbstractTree<Data>::Node;

        //! @brief Visits the current node. Returns whether visitation should
        //! continue.
        virtual bool visit(const Node&) = 0;

        //! @brief Visits the current node after fully visiting all children of
        //! the node.
        virtual void endVisit(const Node&) = 0;
    };

    /** @brief Represents an abstract node in the tree.
     */
    class Node {
        Data _data;
        std::vector<Node> _children;

        friend struct AbstractTree;  // allow setting children directly

    public:
        //! @brief Constructs a new node with data and children.
        Node(Data data, std::vector<Node> children = {}): _data(std::move(data)), _children(std::move(children)) {
        }

        //! @brief Applies the specified visitor to itself and all the children.
        //! This procedure is recursive in nature.
        void apply(Visitor& visitor) const {
            if (!visitor.visit(*this)) {
                return;
            }

            for (const auto& node : children()) {
                node.apply(visitor);
            }

            visitor.endVisit(*this);
        }

        //! @brief Returns the children of this node.
        const std::vector<Node>& children() const {
            return _children;
        }

        //! @brief Returns the data of this node by reference.
        const Data& data() const {
            return _data;
        }
    };

    /** @brief Creates a tree with roots as specified.

        @note The roots are assumed to be fully initialized (having children)
        and thus the tree is ready to be used immediately.
    */
    explicit AbstractTree(std::vector<Node> roots): _roots(std::move(roots)) {
    }

    /** @brief Creates a tree with partially initialized roots, adding child
        nodes in the tree recursively.

        @note This is a helper constructor suitable when the tree could be
        created from the roots alone.
     */
    inline explicit AbstractTree(std::vector<Node> roots, FuncRef<std::vector<Data>(const Node&)> collectChildData);

    //! @brief Applies the specified visitor to the whole tree.
    void apply(Visitor& visitor) const {
        for (const auto& root : roots()) {
            root.apply(visitor);
        }
    }

    //! @brief Returns all the roots of this tree.
    const std::vector<Node>& roots() const {
        return _roots;
    }

private:
    std::vector<Node> _roots;
};

/** @brief A visitor wrapper that provides functional-style visitation API.

    @note This reduces the boilerplate when the visitation performed is simple
    enough to be enclosed into a lambda function.
*/
template <typename Data>
struct CallbackVisitor final : AbstractTree<Data>::Visitor {
    using Node = typename AbstractTree<Data>::Node;
    using VisitFunc = std::function<bool(const Node&)>;
    using EndVisitFunc = std::function<void(const Node&)>;

    //! @brief Creates a visitor that supports both visit and endVisit.
    CallbackVisitor(const VisitFunc& v, const EndVisitFunc& ev): _visitCall(v), _endVisitCall(ev) {
    }
    //! @brief Creates a visitor that supports only visit.
    CallbackVisitor(const VisitFunc& v, std::nullptr_t): _visitCall(v) {
    }
    //! @brief Creates a visitor that supports only endVisit.
    CallbackVisitor(std::nullptr_t, const EndVisitFunc& ev): _endVisitCall(ev) {
    }

    bool visit(const Node& node) final {
        return _visitCall(node);
    }
    void endVisit(const Node& node) final {
        _endVisitCall(node);
    }

private:
    VisitFunc _visitCall = [](const Node&) {
        return true;
    };
    EndVisitFunc _endVisitCall = [](const Node&) {};
};

template <typename Data>
inline AbstractTree<Data>::AbstractTree(std::vector<Node> roots,
                                        FuncRef<std::vector<Data>(const Node&)> collectChildData)
        : AbstractTree(std::move(roots)) {
    CallbackVisitor<Data> visitor(
            [&](const Node& node) {
                VPUX_THROW_UNLESS(node.children().empty(), "Cannot initialize already initialized tree node");
                auto allData = collectChildData(node);

                // Note: const_cast to workaround the traversal API that works with
                // const nodes - this avoids the need to maintain non-const version
                // of the same API.
                auto& nodes = const_cast<Node&>(node)._children;

                nodes.reserve(allData.size());
                std::transform(std::move_iterator(allData.begin()), std::move_iterator(allData.end()),
                               std::back_inserter(nodes), [](Data&& data) {
                                   return Node(std::move(data), {});
                               });
                return true;
            },
            nullptr);
    apply(visitor);
}

}  // namespace vpux::utils
