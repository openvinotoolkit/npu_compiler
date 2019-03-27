#ifndef TOPOLOGICAL_SORT_HPP_
#define TOPOLOGICAL_SORT_HPP_

#include "include/mcm/graph/graph.hpp"
#include <set>
#include <vector>
#include <algorithm>

namespace mv
{
    template <typename NodeIterator>
    struct OpItComparatorTemplate2
    {
        bool operator()(NodeIterator lhs, NodeIterator rhs) const
        {
            return (*lhs) < (*rhs);
        }
    };

    template <typename T_node, typename T_edge, typename OpIteratorComp>
    void visit(typename graph<T_node, T_edge>::node_list_iterator root, std::set<typename graph<T_node, T_edge>::node_list_iterator, OpIteratorComp>& unmarkedNodes, std::vector<typename graph<T_node, T_edge>::node_list_iterator>& toReturn, graph<T_node, T_edge>& g)
    {
        if(unmarkedNodes.find(root) == unmarkedNodes.end())
            return;
        for(auto neighbour = root->leftmost_child(); neighbour != g.node_end(); ++neighbour)
            visit(neighbour, unmarkedNodes, toReturn, g);

        unmarkedNodes.erase(root);
        toReturn.push_back(root);
    }

    // NOTE: This graph non member function works only on DAGs
    template <typename T_node, typename T_edge>
    std::vector<typename graph<T_node, T_edge>::node_list_iterator> topologicalSort(graph<T_node, T_edge>& g)
    {
        std::vector<typename graph<T_node, T_edge>::node_list_iterator> toReturn;

        std::set<typename graph<T_node, T_edge>::node_list_iterator, OpItComparatorTemplate2<typename graph<T_node, T_edge>::node_list_iterator>> unmarkedNodes;
        for(auto node = g.node_begin(); node != g.node_end(); ++node)
            unmarkedNodes.insert(node);

        while(!unmarkedNodes.empty())
        {
            auto toVisit = unmarkedNodes.begin();
            visit(*toVisit, unmarkedNodes, toReturn, g);
        }

        std::reverse(toReturn.begin(), toReturn.end());
        return toReturn;
    }
}

#endif
