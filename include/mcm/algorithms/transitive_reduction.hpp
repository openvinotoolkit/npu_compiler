#ifndef TRANSITIVE_REDUCTION_HPP_
#define TRANSITIVE_REDUCTION_HPP_

#include <list>
#include <map>
#include <unordered_set>
#include <vector>

#include "include/mcm/graph/graph.hpp"
#include "include/mcm/algorithms/topological_sort.hpp"
#include "include/mcm/compiler/compilation_profiler.hpp"

#include <cstdio>


namespace mv
{
    template <typename NodeIterator>
    struct OpItComparatorTemplate
    {
        bool operator()(NodeIterator lhs, NodeIterator rhs) const
        {
            return (*lhs) < (*rhs);
        }
    };

    template <typename T_node, typename T_edge, typename NodeItComp>
    bool allChildrenAtSameLevel_(const graph<T_node, T_edge>& g,
        typename graph<T_node, T_edge>::node_list_iterator node,
        const std::map<typename graph<T_node, T_edge>::node_list_iterator,
          size_t, NodeItComp>& level_map) {
      auto e = node->leftmost_output();
      if (e == g.edge_end()) { return true; }

      auto citr = level_map.find(e->sink());
      assert(citr != level_map.end());
      size_t first_child_level = citr->second;
      for (++e; e != g.edge_end(); ++e) {
        citr = level_map.find(e->sink());
        if (citr->second != first_child_level) { return false; }
      }
      return true;
    }


    // NOTE: This graph non member function works only on DAGs
    template <typename T_node, typename T_edge, typename EdgeItComp, typename NodeItComp>
    void transitiveReduction_(graph<T_node, T_edge>& g,
                typename graph<T_node, T_edge>::node_list_iterator root,
                const std::set<typename graph<T_node, T_edge>::edge_list_iterator, EdgeItComp>& filteredEdges,
                std::set<typename graph<T_node, T_edge>::node_list_iterator, NodeItComp>& processedNodes) {


        // Collecting the set of neighbours, as edges
        // NOTE: Can't use unordered map because node_list_iterator needs to be hashable (requirement too strict)
        std::map<typename graph<T_node, T_edge>::node_list_iterator,
                typename graph<T_node, T_edge>::edge_list_iterator,
                OpItComparatorTemplate<typename graph<T_node, T_edge>::node_list_iterator>> root_adj, toEliminate;

        for(auto e = root->leftmost_output(); e != g.edge_end(); ++e)
            root_adj[e->sink()] = e;

        // Starting a DFS from each neighbour v
        // If a node u is reachable from v and it's also a neighbour of the root
        // Eliminate the edge between root and u

        size_t total_edges_traversed=0UL;
        size_t max_edges_per_traversal=0UL;
        size_t eliminated=0UL;
        for(auto e = root->leftmost_output(); e != g.edge_end(); ++e)
        {
            // Must skip first edge (itself)
            typename graph<T_node, T_edge>::edge_dfs_iterator edge_dfs(e);
            ++edge_dfs;
            size_t edges_traversed = 0UL;
            for (; edge_dfs != g.edge_end(); ++edge_dfs)
            {
              ++edges_traversed;
                auto u = edge_dfs->sink();
                auto it = root_adj.find(u);
                if(it != root_adj.end())
                {
                    auto it2 = filteredEdges.find(it->second);
                    if(it2 != filteredEdges.end())
                        continue;
                    toEliminate[u] = it->second;
                    eliminated++;
                }
            }
            if (edges_traversed > max_edges_per_traversal) {
              max_edges_per_traversal = edges_traversed;
            }
            total_edges_traversed += edges_traversed;
        }
       
        assert(mv::isDAG(g));
        for(auto edgeToEliminatePair : toEliminate)
            g.edge_erase(edgeToEliminatePair.second);

        processedNodes.insert(root);
        for(auto e = root->leftmost_output(); e != g.edge_end(); ++e)
        {
            auto v = e->sink();
            if (processedNodes.find(v) == processedNodes.end())
                transitiveReduction_(g, v, filteredEdges, processedNodes);
        }

    }


    template<typename T_node, typename T_edge>
    bool hasPathWithAtleastOneNode_(const graph<T_node, T_edge>& g,
        typename graph<T_node, T_edge>::node_list_iterator src,
        typename graph<T_node, T_edge>::node_list_iterator sink) {
      typedef typename graph<T_node, T_edge>::node_list_iterator node_iterator;

      std::unordered_set<size_t> visited;
      std::list<node_iterator> bfs_list;

      for (auto e=src->leftmost_output(); e!=g.edge_end(); ++e) {
        if (e->sink()->getID() == sink->getID()) { continue; }
        bfs_list.push_back(e->sink());
        visited.insert(e->sink()->getID());
      }

      while (!bfs_list.empty()) {
        node_iterator curr_node = bfs_list.front();
        bfs_list.pop_front();
        if (curr_node->getID() == sink->getID()) { return true; }
        for (auto e=curr_node->leftmost_output(); e!=g.edge_end(); ++e) {
          size_t nid = e->sink()->getID();
          if (visited.find(nid) == visited.end()) {
            bfs_list.push_back(e->sink());
            visited.insert(nid);
          }
        }
      }
      return false;
    }


    template <typename T_node, typename T_edge, typename NodeItComp>
    void computeNodeLevels_(const graph<T_node, T_edge>& g,
        std::map<typename graph<T_node, T_edge>::node_list_iterator, size_t,
                  NodeItComp>& level_map) {
      typedef typename graph<T_node, T_edge>::node_list_iterator
          node_iterator;

      std::map<node_iterator, size_t,
          OpItComparatorTemplate<node_iterator> > in_degree_map;

      std::list<node_iterator> zero_in_degree_list[2UL];

      for (node_iterator nitr=g.node_begin(); nitr!=g.node_end(); ++nitr) {
        // compute the in-degree//
        in_degree_map[nitr] = nitr->inputs_size();
        if (!(nitr->inputs_size())) {
          zero_in_degree_list[0].push_back(nitr);
        }
      }


      assert(!zero_in_degree_list[0].empty());
      size_t current_level = 0UL, levelled_nodes = 0UL;
      size_t node_count = in_degree_map.size();



      while (levelled_nodes < node_count) {
        bool curr_parity = ((current_level%2UL) != 0);
        bool next_parity = !curr_parity;
        zero_in_degree_list[next_parity].clear();

        for (auto zitr=(zero_in_degree_list[curr_parity]).begin();
              zitr!=zero_in_degree_list[curr_parity].end(); ++zitr) {
          level_map[*zitr] = current_level;
          levelled_nodes++;

          // now reduce the in-degree of all outgoing nodes //
          for (auto e=(*zitr)->leftmost_output(); e != g.edge_end(); ++e) {
            node_iterator sink_node = e->sink();
            auto in_degree_sink_itr = in_degree_map.find(sink_node);
            assert(in_degree_sink_itr != in_degree_map.end());
            assert(in_degree_sink_itr->second > 0UL);
            (in_degree_sink_itr->second)--;

            if (!(in_degree_sink_itr->second)) {
              zero_in_degree_list[next_parity].push_back(sink_node);
            }
          }
        }
        ++current_level;
      }
    }


    template <typename T_node, typename T_edge>
    size_t graphEdgeCount(const graph<T_node, T_edge>& g) {
      size_t diedge_count = 0UL;
      for (auto node = g.node_begin(); node != g.node_end(); ++node) {
        size_t degree = 0UL;
        for (auto e = node->leftmost_output(); e != g.edge_end(); ++e) {
          degree++;
        }
        diedge_count += degree;
      }
      return diedge_count;
    }


    // NOTE: This graph non member function works only on DAGs
    template <typename T_node, typename T_edge, typename EdgeItComparator, typename NodeItComparator>
    void transitiveReductionOld(graph<T_node, T_edge>& g,
                const std::set<typename graph<T_node, T_edge>::edge_list_iterator, EdgeItComparator>&
                filteredEdges = std::set<typename graph<T_node, T_edge>::edge_list_iterator, EdgeItComparator>())
    {
        typedef typename graph<T_node, T_edge>::edge_list_iterator
            edge_iterator;

        MV_PROFILED_FUNCTION(MV_PROFILE_ALGO)
        size_t input_edges = graphEdgeCount(g);
        std::map<typename graph<T_node, T_edge>::node_list_iterator, size_t, 
            NodeItComparator> nodeLevels;
        std::map<size_t, std::list<edge_iterator> > ordered_edges;


        //STEP-1: label each node with its level in the DAG //
        computeNodeLevels_(g, nodeLevels);
        //STEP-1.1: order the edges based on the level of the source//

        for (auto e=g.edge_begin(); e!=g.edge_end(); ++e) {
          auto src_itr = nodeLevels.find(e->source());
          assert(src_itr != nodeLevels.end());
          ordered_edges[ src_itr->second ].push_back(e);
        }

        size_t eliminated = 0UL;
        for (auto litr=ordered_edges.begin(); litr!=ordered_edges.end();
              ++litr) {
          for (auto eitr=litr->second.begin(); eitr!=litr->second.end();
                ++eitr) {
            edge_iterator e = *eitr;
            auto src_itr = nodeLevels.find(e->source());
            auto sink_itr = nodeLevels.find(e->sink());
            assert(src_itr != nodeLevels.end());
            assert(sink_itr != nodeLevels.end());
            assert(sink_itr->second > src_itr->second);

            // edge (u,w) can be eliminated if there is a path between 'u' and
            // 'w' of at least one node between them.
            size_t level_diff = sink_itr->second - src_itr->second;
            if ((level_diff > 1UL) &&
                  hasPathWithAtleastOneNode_(g, e->source(), e->sink())) {
              // eliminate the edge //
              if (filteredEdges.find(*eitr) == filteredEdges.end()) {
                g.edge_erase(*eitr);
                eliminated++;
              }
            }
          }
        }

    }

//
// DAG_Transitive_Reducer: given a DAG G=(V,E) computes a sub DAG G*=(V,E*)
// defined as follows: 
// E* = {(u,v) | (u,v) \in E and  \noexists |path(u,v)| > 1 }
//
// NOTE: NodeItCompType and EdgeItCompType are weak ordering on the nodes and
// edges associated with corresponding iterators.
template<typename DAGType, typename EdgeItCompType, typename NodeItCompType>
class DAG_Transitive_Reducer {

  public:
  //////////////////////////////////////////////////////////////////////////////
    typedef DAGType dag_t;
    typedef NodeItCompType node_comparator_t;
    typedef EdgeItCompType edge_comparator_t;
    typedef typename dag_t::node_list_iterator node_iterator_t;
    typedef typename dag_t::edge_list_iterator edge_iterator_t;
    //TODO(vamsikku): parameterize on the allocator //
    typedef std::map<node_iterator_t, size_t, node_comparator_t> level_map_t;
    typedef std::unordered_map<size_t, size_t> level_node_count_map_t;
    typedef std::set<edge_iterator_t, edge_comparator_t> filter_edge_set_t;
  //////////////////////////////////////////////////////////////////////////////

    DAG_Transitive_Reducer(dag_t& dag) : dag_(dag), level_map_(),
      input_edge_count_(0UL), eliminated_edge_count_(0UL){}


    void dump_reduce_info() const {
      printf("[TransitiveReduction] input=%lu eliminated=%lu\n",
          input_edge_count_, eliminated_edge_count_);
    }

    void dump_graph() const {
      static size_t graph_id = 0UL;
      char buf[1024];

      sprintf(buf, "./input_dag_%lu.txt", ++graph_id);
      FILE *fptr = fopen(buf, "w");
      if (!fptr) {
        throw std::string("unable to open file\n");
      }

      dag_t &g = dag_;
      fprintf(fptr, "nodes: %lu\n", g.node_size());
      // nodes //
      for (node_iterator_t nitr=g.node_begin(); nitr!=g.node_end(); ++nitr) {
        fprintf(fptr, "%lu\n", nitr->getID());
      }

      // edges //
      fprintf(fptr, "edges: %lu\n", g.edge_size());
      for (edge_iterator_t eitr=g.edge_begin(); eitr!=g.edge_end(); ++eitr) {
        fprintf(fptr, "%lu %lu\n", eitr->source()->getID(),
              eitr->sink()->getID());
      }
      fclose(fptr);
    }


    bool reduce(const filter_edge_set_t &filtered_edges=filter_edge_set_t()) {
      MV_PROFILED_FUNCTION(MV_PROFILE_ALGO)

      dag_t &g = dag_;
      input_edge_count_ = g.edge_size();
      eliminated_edge_count_ = 0UL;

      //STEP-1: label each node with its level in the DAG //
      bool is_dag = compute_level_map();
      if (!is_dag) { return false; }

      //STEP-1.1: order the edges based on the level of the source//
      std::map<size_t, std::list<edge_iterator_t> > ordered_edges;
      for (auto e=g.edge_begin(); e!=g.edge_end(); ++e) {
        auto src_itr = level_map_.find(e->source());
        assert(src_itr != level_map_.end());
        ordered_edges[ src_itr->second ].push_back(e);
      }

      //STEP-1.2: compute 

      size_t curr_level=0;
      for (auto litr=ordered_edges.begin(); litr!=ordered_edges.end();
            ++litr) {
        for (auto eitr=litr->second.begin(); eitr!=litr->second.end();
              ++eitr) {
          edge_iterator_t e = *eitr;
          auto src_itr = level_map_.find(e->source());
          auto sink_itr = level_map_.find(e->sink());
          assert(src_itr != level_map_.end());
          assert(sink_itr != level_map_.end());
          assert(sink_itr->second > src_itr->second);

          // edge (u,w) can be eliminated if there is a path between 'u' and
          // 'w' of at least one node between them.
          size_t level_diff = sink_itr->second - src_itr->second;
          if ((level_diff > 1UL) &&
                has_path_with_atleast_one_node( e->source(), e->sink())) {
            // eliminate the edge //
            if (filtered_edges.find(*eitr) == filtered_edges.end()) {
              g.edge_erase(*eitr);
              ++eliminated_edge_count_;
            }
          }
        }
      }
      return true;
    } // bool reduce //

  private:

    bool has_path_with_atleast_one_node(node_iterator_t src,
          node_iterator_t sink) const {

      const dag_t& g = dag_;
      std::unordered_set<size_t> visited;
      std::list<node_iterator_t> bfs_list;

      size_t sink_level;
      auto level_itr = level_map_.find(sink);
      if (level_itr == level_map_.end()) { 
        throw "missing level entry";
      }
      sink_level = level_itr->second;


      //TODO(vamsikku): while doing DFS don't add nodes which exceed the
      //level of the sink into the bfs_list //
      for (auto e=src->leftmost_output(); e!=g.edge_end(); ++e) {
        if (e->sink()->getID() == sink->getID()) { continue; }

        node_iterator_t node_itr = e->sink();
        auto level_itr = level_map_.find(node_itr);
        if (level_itr == level_map_.end()) { 
          throw "missing level entry";
        }
        if ((level_itr->second) > sink_level) { continue; }


        bfs_list.push_back(e->sink());
        visited.insert(e->sink()->getID());
      }

      while (!bfs_list.empty()) {
        node_iterator_t curr_node = bfs_list.front();
        bfs_list.pop_front();
        if (curr_node->getID() == sink->getID()) { return true; }
        for (auto e=curr_node->leftmost_output(); e!=g.edge_end(); ++e) {

          node_iterator_t node_itr = e->sink();
          auto level_itr = level_map_.find(node_itr);
          if (level_itr == level_map_.end()) { throw "missing level entry"; }
          if ((level_itr->second) > sink_level) { continue; }

          size_t nid = e->sink()->getID();
          if (visited.find(nid) == visited.end()) {
            bfs_list.push_back(e->sink());
            visited.insert(nid);
          }
        }
      }
      return false;
    }

    bool compute_level_map() {
      dag_t& g = dag_;
      std::map<node_iterator_t, size_t, node_comparator_t> in_degree_map;
      std::list<node_iterator_t> zero_in_degree_list[2UL];

      level_map_.clear();

      for (node_iterator_t nitr=g.node_begin(); nitr!=g.node_end(); ++nitr) {
        // compute the in-degree//
        in_degree_map[nitr] = nitr->inputs_size();
        if (!(nitr->inputs_size())) {
          zero_in_degree_list[0].push_back(nitr);
        }
      }


      size_t current_level = 0UL, levelled_nodes = 0UL;
      size_t node_count = in_degree_map.size();

      while (levelled_nodes < node_count) {
        bool curr_parity = ((current_level%2UL) != 0);
        bool next_parity = !curr_parity;

        if (zero_in_degree_list[curr_parity].empty()) { return false; }
        zero_in_degree_list[next_parity].clear();

        for (auto zitr=(zero_in_degree_list[curr_parity]).begin();
              zitr!=zero_in_degree_list[curr_parity].end(); ++zitr) {
          level_map_[*zitr] = current_level;
          levelled_nodes++;

          // now reduce the in-degree of all outgoing nodes //
          for (auto e=(*zitr)->leftmost_output(); e != g.edge_end(); ++e) {
            node_iterator_t sink_node = e->sink();
            auto in_degree_sink_itr = in_degree_map.find(sink_node);
            assert(in_degree_sink_itr != in_degree_map.end());
            assert(in_degree_sink_itr->second > 0UL);
            (in_degree_sink_itr->second)--;

            if (!(in_degree_sink_itr->second)) {
              zero_in_degree_list[next_parity].push_back(sink_node);
            }
          }
        }
        ++current_level;
      }

      return (levelled_nodes == node_count);
    }

  private:
    dag_t &dag_;
    level_map_t level_map_;
    size_t input_edge_count_;
    size_t eliminated_edge_count_;
}; // class DAG_Transitive_Reducer //

#define MV_MCM_USE_NEW_TRANS_REDUCTION

template <typename T_node, typename T_edge, typename EdgeItComparator,
         typename NodeItComparator>
void transitiveReduction(graph<T_node, T_edge>& g,
    const std::set<typename graph<T_node, T_edge>::edge_list_iterator,
      EdgeItComparator>& filteredEdges = std::set<typename graph<T_node,
        T_edge>::edge_list_iterator, EdgeItComparator>()) 
{
  MV_PROFILED_FUNCTION(MV_PROFILE_ALGO)
#ifdef MV_MCM_USE_NEW_TRANS_REDUCTION
  {
    typedef graph<T_node, T_edge> dag_t;
    typedef DAG_Transitive_Reducer<dag_t, EdgeItComparator, NodeItComparator>
       transitive_reducer_t;

    transitive_reducer_t reducer(g);
    reducer.reduce(filteredEdges);
    reducer.dump_reduce_info();
  }
#else
    transitiveReductionOld<T_node, T_edge, EdgeItComparator,
      NodeItComparator>(g, filteredEdges);
#endif

}


} // namespace mv //

#endif
