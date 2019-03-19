#ifndef IS_DAG_HPP_
#define IS_DAG_HPP_

#include "include/mcm/graph/graph.hpp"
#include <map>
#include <vector>

namespace mv
{
    // is_DAG() checks if the graph is acyclic (no cycles).
    // Algorithm used is similar to the one in python networkX https://networkx.github.io/documentation/networkx-1.9/_modules/networkx/algorithms/dag.html#is_directed_acyclic_graph
    // for each of the nodes of graph, using recursive DFS check for back edges (backedges get back to the node already explored)

    template <typename T_node, typename T_edge>
    bool is_DAG(graph<T_node, T_edge>& g)
    {
        int64_t V = g.node_size();
        // 'explored array' stores if the node is explored. recurStack stores the state of the stack (which node being processed)
        std::vector<bool> explored = std::vector<bool>(V, false);
        std::vector<bool> recurStack = std::vector<bool>(V, false);

        for (auto it = g.node_begin(); it != g.node_end(); ++it)
            if (hasCycle(g, it, explored, recurStack))
                  return false;
        return true;
    }

    // hasCycle returns if there are cycles by doing DFS
    template <typename T_node, typename T_edge>
    bool hasCycle(graph<T_node, T_edge>&g, typename graph<T_node, T_edge>::node_list_iterator it, std::vector<bool>& explored, std::vector<bool>& recurStack)
    {
        int64_t v = it->getID();
        // below node_list_iterator object needed as the hasCycle method takes these objects only. so 'child iterator' gets masked as node_list_iterator and passed
        typename graph<T_node, T_edge>::node_list_iterator it1;

        if(!explored[v])
        {
            // Mark the current node as explored and part of recursion stack
            explored[v] = true;
            recurStack[v] = true;
        }

        for (typename graph<T_node, T_edge>::node_child_iterator cit(it); cit != g.node_end(); ++cit)
        {
            it1 = cit;
            // recursive call to hasCycle method
            if (!explored[cit->getID()] && hasCycle(g, it1, explored, recurStack) )
                return true;

            // below condition checks if there are loops
            else if (recurStack[cit->getID()])
                return true;

        }
        // update recursion stack by removing the node (from the stack)
        recurStack[v] = false;

        return false;
    }
}

#endif
