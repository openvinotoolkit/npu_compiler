#ifndef DIJKSTRA_HPP_
#define DIJKSTRA_HPP_

#include <iostream>
#include <queue>
#include <vector>
#include <functional>
#include <set>
#include <algorithm>
#include <map>
#include "include/mcm/graph/graph.hpp"

namespace mv
{

    template <typename T, typename D>
    struct HeapContent
    {
        T id;
        D distance;

        friend bool operator<(const HeapContent<T, D>& a, const HeapContent<T, D>& b)
        {
            return a.distance < b.distance;
        }

        friend bool operator>(const HeapContent<T, D>& a, const HeapContent<T, D>& b)
        {
            return a.distance > b.distance;
        }
    };


    template<typename T, typename D>
    struct DijkstraReturnValue
    {
        std::vector<T> nodes;
        std::vector<D> distances;
    };

    //In dijkstraRT the initial graph consists of only one node. A rule to generate new neighbours must be passed
    //This rule specifies as well the exit condition (no more neighbours)
    template <typename NodeValue, typename DistanceValue>
    DijkstraReturnValue<NodeValue, DistanceValue> dijkstraRT(NodeValue source, NodeValue target, std::function<std::vector<NodeValue>(NodeValue)> generateNeighbours, std::function<DistanceValue(NodeValue, NodeValue)> computeCost)
    {
         // Auxiliary data structures
         std::map<NodeValue, DistanceValue> distances;
         std::map<NodeValue, NodeValue> previous;
         std::set<NodeValue> seen;
         std::set<NodeValue> generatedNodes;
         std::priority_queue<HeapContent<NodeValue, DistanceValue>, std::vector<HeapContent<NodeValue, DistanceValue>>, std::greater<HeapContent<NodeValue, DistanceValue>>> minHeap;
         DistanceValue zeroCost(0);

         // Variable to return
         DijkstraReturnValue<NodeValue, DistanceValue> toReturn;

         // Inserting the source into heap and graph
         distances[source] = zeroCost;
         previous[source] = source;
         HeapContent<NodeValue, DistanceValue> source_heap = {source, distances[source]};
         minHeap.push(source_heap);
         generatedNodes.insert(source);

         while(!minHeap.empty())
         {
            HeapContent<NodeValue, DistanceValue> top = minHeap.top();
            NodeValue u = top.id;
            minHeap.pop();
            if(seen.count(u))
                continue;
            seen.insert(u);
            if(u == target)
                break;
            std::vector<NodeValue> neighbours = generateNeighbours(u);
            int degree = neighbours.size();
            if(degree == 0)
                continue;
            for(int i = 0; i < degree; ++i)
            {
                NodeValue v = neighbours[i];
                DistanceValue cost_u_v = computeCost(u, v);
                if(cost_u_v > zeroCost) //solutions with infinite cost are marked with -1
                {
                    DistanceValue distance = distances[u] + cost_u_v;
                    if(generatedNodes.count(v))
                    {
                        if(distance < distances[v])
                        {
                            distances[v] = distance;
                            previous[v] = u;
                            HeapContent<NodeValue, DistanceValue> toPush = {v, distances[v]};
                            minHeap.push(toPush);
                        }
                    }
                    else
                    {
                        generatedNodes.insert(v);
                        distances[v] = distance;
                        previous[v] = u;
                        HeapContent<NodeValue, DistanceValue> toPush = {v, distances[v]};
                        minHeap.push(toPush);
                    }
                }
            }
         }

         for(auto mapIt = previous.find(target); mapIt->first != source; mapIt = previous.find(mapIt->second))
         {
            toReturn.nodes.push_back(mapIt->first);
            toReturn.distances.push_back(distances[mapIt->first]);
         }

         toReturn.nodes.push_back(source);

         std::reverse(toReturn.nodes.begin(), toReturn.nodes.end());
         std::reverse(toReturn.distances.begin(), toReturn.distances.end());

         return toReturn;
    }

    template <typename T_node, typename T_edge>
    std::vector<typename graph<T_node, T_edge>::node_list_iterator> dijkstra_nodes(graph<T_node, T_edge>& g, typename graph<T_node, T_edge>::node_list_iterator source, typename graph<T_node, T_edge>::node_list_iterator sink, const std::map<T_edge, int>& edgeCosts)
    {
        std::vector<typename graph<T_node, T_edge>::node_list_iterator> toReturn;
        std::priority_queue<HeapContent<T_node, int>, std::vector<HeapContent<T_node, int>>, std::greater<HeapContent<T_node, int>>> minHeap;
        std::map<T_node, int> distances;
        std::set<T_node> seen;
        std::map<T_node, T_node> previous;

        int zeroCost(0);
        int infiniteCost(-1);

        // Inserting the source into heap and graph
        for(auto nodeIt = g.node_begin(); nodeIt != g.node_end(); ++nodeIt)
            distances[*nodeIt] = infiniteCost;
        distances[*source] = zeroCost;
        previous[*source] = *source;

        HeapContent<T_node, int> source_heap = {*source, distances[*source]};
        minHeap.push(source_heap);

        while(!minHeap.empty())
        {

           HeapContent<T_node, int> top = minHeap.top();
           auto uIt = g.node_find(top.id);

           minHeap.pop();

           if(seen.count(*uIt))
               continue;
           seen.insert(*uIt);
           if(uIt == sink)
               break;

           for(auto u_vIt = uIt->leftmost_output(); u_vIt != g.edge_end(); ++u_vIt)
           {
               auto vIt = u_vIt->sink();
               int cost_u_v = edgeCosts.at(*u_vIt);
               int distance = distances[*uIt] + cost_u_v;
               if(distances[*vIt] == infiniteCost || distance < distances[*vIt])
               {
                   distances[*vIt] = distance;
                   previous[*vIt] = *uIt;
                   HeapContent<T_node, int> toPush = {*vIt, distances[*vIt]};
                   minHeap.push(toPush);
               }

           }
        }

        for(auto mapIt = previous.find(*sink); mapIt->first != *source; mapIt = previous.find(mapIt->second))
           toReturn.push_back(g.node_find(mapIt->first));

        toReturn.push_back(source);

        std::reverse(toReturn.begin(), toReturn.end());

        return toReturn;
    }


    template <typename T_node, typename T_edge>
    std::vector<typename graph<T_node, T_edge>::edge_list_iterator> dijkstra(graph<T_node, T_edge>& g, typename graph<T_node, T_edge>::node_list_iterator source, typename graph<T_node, T_edge>::node_list_iterator sink, const std::map<T_edge, int>& edgeCosts)
    {
        std::vector<typename graph<T_node, T_edge>::node_list_iterator> buildList = dijkstra_nodes(g, source, sink, edgeCosts);
        std::vector<typename graph<T_node, T_edge>::edge_list_iterator> toReturn;

        unsigned n = buildList.size();
        for(unsigned i = 0; i < n - 1; ++i)
        {
            auto nodeIt = buildList[i];
            for(auto edgeIt = nodeIt->leftmost_output(); edgeIt != g.edge_end(); ++edgeIt)
            {
                auto nextNodeIt = edgeIt->sink();
                if(*nextNodeIt == *buildList[i+1])
                {
                    toReturn.push_back(edgeIt);
                    break;
                }
            }

        }
        return toReturn;
    }

}

#endif // DIJKSTRA_HPP_
