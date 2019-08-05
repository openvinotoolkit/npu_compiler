#include "limits"
#include "tuple"

#include "include/mcm/pass/graphOptimizations/StrategyManager.hpp"
//#include "include/mcm/pass/graphOptimizations/descartes_product.hpp"
#include "include/mcm/pass/graphOptimizations/StrategyRegistry.hpp"
#include "include/mcm/base/element.hpp"
#include "include/mcm/algorithms/dijkstra.hpp"

namespace mv {
namespace graphOptimizer {

using namespace std;

StrategyManager::StrategyManager(OpModel& model,mv::Element& passDesc) :
        model_(model),passDesc_(passDesc)
{

}

void StrategyManager::updateValuesFromJSON()
{
    auto graphOptimizerConfig = passDesc_.get<mv::Element>("graphOptimizerConfig");


    auto globalConfigs = graphOptimizerConfig.get<vector<mv::Element>>("globalConfigs");
    auto globalStrategies = graphOptimizerConfig.get<vector<mv::Element>>("globalStrategies");
    auto layerStrategySets  = graphOptimizerConfig.get<vector<mv::Element>>("layerStrategies");

    for( auto globalConfig : globalConfigs)
    {
        auto configName = globalConfig.getName();
        auto configValue = globalConfig.get("value");
        globalConfig_[configName] = configValue;
    }

    for( auto globalStrategy : globalStrategies)
    {
        auto strategyName = globalStrategy.getName();
        auto strategyValue = globalStrategy.get("value");
        globalStrategies_[strategyName] = strategyValue;
    }

    for( auto layerStrategySet : layerStrategySets)
    {
        auto layerName = layerStrategySet.getName();
        auto strategySets = layerStrategySet.get<vector<mv::Element>>("strategies");

        for(auto strategySet : strategySets)
        {
            auto strategySetName = strategySet.getName();
//            auto strategies = strategySet.get<vector<string>>("value");

            auto strategyValue = strategySet.get("value");
            layerStrategies_[layerName][strategySetName] = strategyValue;
//            if(strategiesType == typeid(vector<string>))
//            {
//                auto strategies = strategySet.get<vector<string>>("value");
//                for( auto strategy : strategies)
//                {
//                    layerStrategies_[layerName][strategySetName].insert(strategy);
//                }
//            }
//            else
//            {
//                layerStrategies_[layerName][strategySetName].insert(strategySet.get("value"));
//            }
        }
    }
}

void StrategyManager::updateDefaultValues()
{
    //TODO:: solve the "multiple registry" problem
    auto& globalConfigRegistry = mv::graphOptimizer::GlobalConfigRegistry::instance();
    auto& globalStrategyRegistry = mv::graphOptimizer::GlobalStrategyRegistry::instance();
    auto& layerStrategyRegistry = mv::graphOptimizer::LayerStrategyRegistry::instance();

    for (const auto& globalConfigName : globalConfigRegistry.list())
    {
       if(globalConfig_.find(globalConfigName) == globalConfig_.end() )
       {
           auto configVal = globalConfigRegistry.find(globalConfigName)->getAttr();
           globalConfig_[globalConfigName] = configVal;
       }
    }

    for (const auto& globalStrategyName : globalStrategyRegistry.list() )
    {
        if(globalStrategies_.find(globalStrategyName) == globalStrategies_.end())
        {
            auto strategyVal = globalStrategyRegistry.find(globalStrategyName)->getAttr();
            globalStrategies_[globalStrategyName] = strategyVal;
        }
    }

    for(auto& layer : layerStrategyRegistry.list())
    {
        auto strategySet = layerStrategyRegistry.find(layer)->getStrategySet();
        auto recordedStrategySet = layerStrategies_.find(layer);
        if(recordedStrategySet == layerStrategies_.end())
        {
            layerStrategies_[layer] = strategySet;

        }
        else
        {
            for(const auto& strategy : strategySet)
            {
                if( recordedStrategySet->second.find(strategy.first) == recordedStrategySet->second.end())
                {
                    layerStrategies_[layer][strategy.first] = strategy.second;
                }
            }
        }
    }

}

void StrategyManager::printStrategy()
{
    cout <<"########## Final Strategy Config" << endl;

    cout <<"Global Configs" << endl;
    for( const auto elem : globalConfig_)
    {
        cout <<"\t"<<elem.first << " : " << elem.second.toString() << endl;
    }

    cout <<"Global Strategies" << endl;
    for( const auto elem : globalStrategies_)
    {
        cout <<"\t"<<elem.first << " : " << elem.second.toString() << endl;
    }

    cout <<"LayerWise Strategies" << endl;
    for( const auto layer : layerStrategies_)
    {
        cout <<"\t"<<layer.first <<endl;

        for (const auto strategySet : layer.second )
        {
            cout <<"\t"<<strategySet.first << " : " << strategySet.second.toString() << endl;
//            cout <<"\t " << strategySet.first << "[ ";
//            for( const auto strategy : strategySet.second)
//            {
//                cout << strategy.toString() <<" ";
//            }
//            cout << "]" << endl;
        }
    }
}


//TODO:: error if the strategy is not there...
Attribute& StrategyManager::getStrategy(mv::Op op,string strategy)
{
    auto layerEntry = layerStrategies_.find(op.getName());

    if(layerEntry == layerStrategies_.end())
    {
        layerEntry = layerStrategies_.find(op.getOpType());
    }

    if(layerEntry == layerStrategies_.end())
        throw LogicError(*this, "could not find strategy entry for " + op.getName());

    auto& layerCfg = layerEntry->second;

    auto strategyEntry = layerCfg.find(strategy);
    if(strategyEntry == layerCfg.end())
    {
        strategyEntry = globalStrategies_.find(strategy);
    }

//    cout<< "for op: " << op.getName() << " str: " << strategy << " got: " << strategyEntry->second.toString() << endl;
    return strategyEntry->second;
}

void StrategyManager::writeDot(mv::graph<std::tuple<mv::Op&,StrategySet,int>,double>& optimizationGraph, bool skipInf)
{
    ofstream ostream;
    string outputFile = dotFileLocation;

    ostream.open(outputFile,ios::trunc | ios::out);

    ostream << "digraph G {\n\tgraph [splines=spline]\n";

    for(auto node = optimizationGraph.node_begin(); node != optimizationGraph.node_end(); ++ node)
    {
        std::string nodeDef = "\t\"" + get<1>(*node)["name"].get<string>()  + "_" + to_string((long long unsigned)(void*)&(*node)) + "\" [shape=box,";
//        nodeDef += " label=\"" + (*node)["name"].toString() + "\\n";
        //TODO:: using an object's address to uniquely identify it is a baaaaaaaaad idea. Come up with something normal
        //basically, in our graph, we can have multiple nodes with the same name, but cannot have that in the dotfile
        nodeDef += " label=<<TABLE BORDER=\"0\" CELLPADDING=\"0\" CELLSPACING=\"0\"> \
                    <TR><TD ALIGN=\"CENTER\" COLSPAN=\"2\"><FONT POINT-SIZE=\"10.0\"><B>"
                    + get<1>(*node)["name"].get<string>() + "_" + to_string((long long unsigned)(void*)&(*node))
                    + "</B></FONT></TD></TR>";
        for(const auto strategy : get<1>(*node))
        {
//            nodeDef += strategy.first + ": " + strategy.second.toString() + "\\n";
            nodeDef += "<TR><TD ALIGN=\"LEFT\"><FONT POINT-SIZE=\"11.0\">"
                        + strategy.first
                        + ": </FONT></TD> <TD ALIGN=\"RIGHT\"><FONT POINT-SIZE=\"11.0\">"
                        + strategy.second.toString() + "</FONT></TD></TR>";
        }

//        nodeDef += "\"";
        nodeDef += "</TABLE>>";

        ostream << nodeDef << "];\n";
    }

    for(auto edge = optimizationGraph.edge_begin(); edge != optimizationGraph.edge_end(); ++edge)
    {
//        if( skipInf and ( (*edge) == numeric_limits<double>::infinity()))
skipInf=false;
        if( skipInf and ( (*edge) == 999999.999))
            continue;
        //TODO:: using an object's address to uniquely identify it is a baaaaaaaaad idea. Come up with something normal
        std::string edgeDef = "\t\""
                            + get<1>(*(edge->source()))["name"].get<string>() + "_" + to_string((long long unsigned)(void*)&(*(edge->source())))
                            + "\" -> \""
                            + get<1>(*(edge->sink()))["name"].get<string>() + "_" + to_string((long long unsigned)(void*)&(*(edge->sink())))
                            + "\"";

        edgeDef += " [penwidth=2.0, label=<<TABLE BORDER=\"0\" CELLPADDING=\"0\" \
                    CELLSPACING=\"0\"><TR><TD ALIGN=\"CENTER\" COLSPAN=\"2\"> \
                    <FONT POINT-SIZE=\"14.0\"><B>" \
                    + std::to_string(*edge)
                    + "</B></FONT></TD></TR>";

        edgeDef += "</TABLE>>];";

        ostream << edgeDef << "\n";
    }

    ostream << "}\n";
    ostream.close();
}

void StrategyManager::saveStrategy(std::vector<graph<std::tuple<mv::Op&,StrategySet,int>,double>::edge_list_iterator> cPathEdges)
{
/* 
    cout << "Saving streaming Strategy to Compilation Descriptor" << endl;
    cout << "    Critical Path info: length = " << cPathEdges.size() << endl ;
    for (int showPath=0; showPath<cPathEdges.size(); showPath++)
        cout << "    edge_"<< showPath << " cost is " << *(cPathEdges[showPath]) << endl ;
*/

    auto globalParams = model_.getGlobalConfigParams();
    auto strategyList = globalParams->get<std::vector<mv::Element>>("streaming_strategy");

    //determine if node already has strategy from JSON text, do not override text specification
    std::map<std::string, bool> hasSpec;

    for (auto s : strategyList)
    {
        std::string nodeName = s.get<std::string>("name_filter");
        auto splitList = s.get<std::vector<mv::Element>>("splits");
        for (int i = 0; i < splitList.size(); i++)
        {
            if ((splitList[i].hasAttr("C"))||(splitList[i].hasAttr("H"))||(splitList[i].hasAttr("W"))||(splitList[i].hasAttr("K")))
                hasSpec.insert(std::pair<std::string, bool>(nodeName, true));
            else
                hasSpec.insert(std::pair<std::string, bool>(nodeName, false));
        }
    }
            
    //save streaming strategy into compilation descriptor
    auto copyElement = strategyList[0];
    auto copyName = copyElement.get<std::string>("name_filter");
    auto copySplits =  copyElement.get<std::vector<mv::Element>>("splits");
    for (int i=copySplits.size(); i<4; i++)
        copySplits.push_back(copySplits[0]);    // 4 element vector for streaming strategies c,h,w,k
    for (int savePath=1; savePath<cPathEdges.size(); savePath++)
    {
        auto parentNode = cPathEdges[savePath]->leftmost_parent()->sink();
        auto parentStrategySet = get<1>(*parentNode) ;
        mv:Shape newStrategy = parentStrategySet["streaming"];
        std::string newName = parentStrategySet["name"] ;
        if ( hasSpec.find(newName) == hasSpec.end())
        { 
            copyElement.set("name_filter",newName);
            copySplits[0].set<int>("C", newStrategy[0]);
            copySplits[1].set<int>("H", newStrategy[1]);
            copySplits[2].set<int>("W", newStrategy[2]);
            copySplits[3].set<int>("K", newStrategy[3]);
            copyElement.set("splits",copySplits);
            strategyList.push_back(copyElement);
        }
    }
    globalParams->set("streaming_strategy", strategyList);

    //test updated compilation descriptor
    auto globalParams2 = model_.getGlobalConfigParams();
    auto strategyList2 = globalParams2->get<std::vector<mv::Element>>("streaming_strategy");

    for (auto s : strategyList2)
    {
        std::string nodeName = s.get<std::string>("name_filter");
        std::cout <<" Streaming strategy (from compilation descriptor) for node " << s.get<std::string>("name_filter") <<  std::endl ;
        auto splitList = s.get<std::vector<mv::Element>>("splits");
        for (int i = 0; i < splitList.size(); i++)
        {
            if (splitList[i].hasAttr("C"))
            {
                std::cout << "     C : " << splitList[i].get<int>("C") << std::endl;
            }
            else if (splitList[i].hasAttr("H"))
            {
                std::cout << "     H : " << splitList[i].get<int>("H") << std::endl;
            }
            else if (splitList[i].hasAttr("W"))
            {
                std::cout << "     W : " << splitList[i].get<int>("W") << std::endl;
            }
            else if (splitList[i].hasAttr("K"))
            {
                std::cout << "     K : " << splitList[i].get<int>("K") << std::endl;
            }
        }
    }
}

void StrategyManager::linearDijkstra(mv::Data::OpListIterator opBegin)
{
    struct costEdgeIteratorComp
    {
        bool operator()(const graph<std::tuple<mv::Op&,StrategySet,int>,double>::edge_list_iterator lhs, const graph<std::tuple<mv::Op&,StrategySet,int>,double>::edge_list_iterator rhs) const
        {
            return (*lhs) < (*rhs);
        }
    };

    struct costNodeIteratorComp
    {
        bool operator()(const graph<std::tuple<mv::Op&,StrategySet,int>,double>::node_list_iterator lhs, const graph<std::tuple<mv::Op&,StrategySet,int>,double>::node_list_iterator rhs) const
        {
            int x = get<2>(*lhs) ;
            int y = get<2>(*rhs) ;
            return (x < y) ;
        }
    };

    mv::graph<std::tuple<mv::Op&,StrategySet,int>,double> optimizationGraph;
    vector<vector<StrategySet>> nodeStrategies;
    vector<mv::graph<std::tuple<mv::Op&,StrategySet,int>,double>::node_list_iterator> old_nodes,new_nodes;

    old_nodes.clear();

    map<typename graph<std::tuple<mv::Op&,StrategySet,int>,double>::edge_list_iterator, double, costEdgeIteratorComp> edgeCostMap;
    int opCtr = 0;
    int optionCtr = 0;
    
    for(auto op = opBegin; op != model_.opEnd(); ++op)
    {
        if(op->getOpType() == "ConstantInt")
            continue;
        vector<StrategySet> nodeStrategy;

        opCtr++;

        generateStrategySetForLayer(*op,nodeStrategy);
        new_nodes.clear();
        for(auto strategy : nodeStrategy)
        {
            new_nodes.push_back(optimizationGraph.node_insert(std::tuple<mv::Op&,StrategySet,int>(*op,strategy,optionCtr++)));
        }

        for(const auto oldNode : old_nodes)
            for(const auto newNode : new_nodes)
            {
//                cout<< "In linearDykstra: inserting edge to optGraph from " << get<1>(*oldNode)["name"].get<string>()  << "_"<< to_string((long long unsigned)(void*)&(*oldNode)) << " to " << get<1>(*newNode)["name"].get<string>()  << "_"<< to_string((long long unsigned)(void*)&(*newNode)) << endl;
                double edgeCost = transitionCost( get<0>(*oldNode),
                                                  get<0>(*newNode),
                                                  get<1>(*oldNode),
                                                  get<1>(*newNode));
                int edgeCostInt = edgeCost ;
                auto newEdge = optimizationGraph.edge_insert(oldNode,newNode,edgeCost);
                edgeCostMap.insert(std::pair<mv::graph<std::tuple<mv::Op&,StrategySet,int>,double>::edge_list_iterator, double>(newEdge, edgeCost));
 //               cout<< "        cost = " << edgeCost << endl;
            }
        old_nodes.swap(new_nodes);

        nodeStrategies.push_back(nodeStrategy);
    }

    writeDot(optimizationGraph,true);
    costEdgeIteratorComp costEdgeCompare;

    auto sinkNodeIt = optimizationGraph.node_begin() ;
    for (int ii=0; ii<optimizationGraph.node_size()-1; ii++) ++sinkNodeIt;

    auto criticalPathEdges = dijkstra<std::tuple<mv::Op&,StrategySet,int>,double,costNodeIteratorComp,costEdgeIteratorComp, double>(optimizationGraph,optimizationGraph.node_begin(),sinkNodeIt,edgeCostMap);
    
    saveStrategy(criticalPathEdges);
}



void StrategyManager::recursiveDijkstra(mv::Data::OpListIterator opBegin)
{

    cout << "calling extract subgraphs on the data graph" << endl ;
    std::unordered_set<std::string> recursedNodes;
    //Each vector of edges corresponds to a linear portion of the graph
    vector<vector<graph<std::tuple<mv::Op&,StrategySet,int>,double>::edge_list_iterator>> criticalPaths = recursiveCriticalPath(opBegin, recursedNodes);

}

vector<StrategyManager::CriticalEdges> StrategyManager::recursiveCriticalPath(typename graph<mv::Op, mv::DataFlow>::node_list_iterator modelSource, 
                                                                                std::unordered_set<std::string>& recursedNodes){
    struct costEdgeIteratorComp
    {
        bool operator()(const graph<std::tuple<mv::Op&,StrategySet,int>,double>::edge_list_iterator lhs, const graph<std::tuple<mv::Op&,StrategySet,int>,double>::edge_list_iterator rhs) const
        {
            return (*lhs) < (*rhs);
        }
    };

    struct costNodeIteratorComp
    {
        bool operator()(const graph<std::tuple<mv::Op&,StrategySet,int>,double>::node_list_iterator lhs, const graph<std::tuple<mv::Op&,StrategySet,int>,double>::node_list_iterator rhs) const
        {
            int x = get<2>(*lhs) ;
            int y = get<2>(*rhs) ;
            return (x < y) ;
        }
    };

    vector<vector<CriticalEdges>> allLinearCriticalPaths;
    vector<CriticalEdges> linearCriticalPaths;
    vector<vector<StrategyManager::CriticalEdges>> sinkRetVecs;
    vector<mv::graph<std::tuple<mv::Op&,StrategySet,int>,double>> allOptimizationGraphs;
    mv::graph<mv::Op, mv::DataFlow> g;
    auto next_modelSource = modelSource;

    //BASE CASE - end of graph, or previously recursed on this source
    std::string opName = (*modelSource).getName();
    if(modelSource->leftmost_child() == g.node_end() || recursedNodes.find(opName) != recursedNodes.end() || (*modelSource).getOpType() == "ConstantInt"){
        return vector<CriticalEdges>();
    }
    int childCtr = 0;
    recursedNodes.insert(opName); //haven't seen this source before, save it
    //RECURSIVE CASE - iterate over the children of source build linear subgraphs, kick off recursion if another pivot node hit
    for(auto model_edge  = modelSource->leftmost_output(); model_edge !=  g.edge_end(); ++model_edge)
    {
        if((*(model_edge->sink())).getOpType() == "ConstantInt")
            continue;
        childCtr++;
         //Create the subgraph to pass to linear dijkstra, build iteratively until hit next pivot node
        mv::graph<std::tuple<mv::Op&,StrategySet,int>,double> optimizationGraph;
        vector<vector<StrategySet>> nodeStrategies;
        vector<mv::graph<std::tuple<mv::Op&,StrategySet,int>,double>::node_list_iterator> old_nodes, new_nodes, first_nodes, last_nodes;

        old_nodes.clear();

        map<typename graph<std::tuple<mv::Op&,StrategySet,int>,double>::edge_list_iterator, double, costEdgeIteratorComp> edgeCostMap;
        int opCtr = 0;
        int optionCtr = 0;

        //Add modelSource (start pivot) to the graph
        vector<StrategySet> nodeStrategy;
        opCtr++;
        generateStrategySetForLayer(*modelSource,nodeStrategy);
        new_nodes.clear();
        for(auto strategy : nodeStrategy)
        {
            new_nodes.push_back(optimizationGraph.node_insert(std::tuple<mv::Op&,StrategySet,int>(*modelSource,strategy,optionCtr++)));
        }
        first_nodes = new_nodes;
        for(const auto oldNode : old_nodes)
            for(const auto newNode : new_nodes)
            {
                double edgeCost = transitionCost( get<0>(*oldNode), get<0>(*newNode), get<1>(*oldNode), get<1>(*newNode));
                int edgeCostInt = edgeCost ;
                auto newEdge = optimizationGraph.edge_insert(oldNode,newNode,edgeCost);
                edgeCostMap.insert(std::pair<mv::graph<std::tuple<mv::Op&,StrategySet,int>,double>::edge_list_iterator, double>(newEdge, edgeCost));
            }
        old_nodes.swap(new_nodes);
        nodeStrategies.push_back(nodeStrategy);

        auto model_child = model_edge->sink();
        auto model_parent = model_edge->source();


        //ITERATE through linear section, building the optimization subgraph, while (model_child is not a pivot node)
        while ( model_child->children_size() == 1 && ( model_child->parents_size() == 1 || ( model_child->parents_size() == 2 && 
                ( (*model_child->leftmost_input()->source()).getOpType() == "ConstantInt"  || 
                (*model_child->rightmost_input()->source()).getOpType() == "ConstantInt" ) ))) 
        {
            vector<StrategySet> nodeStrategy;
            opCtr++;

            generateStrategySetForLayer(*model_child,nodeStrategy);
            new_nodes.clear();
            for(auto strategy : nodeStrategy)
            {
                new_nodes.push_back(optimizationGraph.node_insert(std::tuple<mv::Op&,StrategySet,int>(*model_child,strategy,optionCtr++)));
            }

            for(const auto oldNode : old_nodes)
                for(const auto newNode : new_nodes)
                {
                    double edgeCost = transitionCost( get<0>(*oldNode), get<0>(*newNode), get<1>(*oldNode), get<1>(*newNode));
                    int edgeCostInt = edgeCost ;
                    auto newEdge = optimizationGraph.edge_insert(oldNode,newNode,edgeCost);
                    edgeCostMap.insert(std::pair<mv::graph<std::tuple<mv::Op&,StrategySet,int>,double>::edge_list_iterator, double>(newEdge, edgeCost));
                }
            old_nodes.swap(new_nodes);

            nodeStrategies.push_back(nodeStrategy);

            //move to next child node in this linear section
            model_child = model_child->leftmost_output()->sink();
        }
        //Iteration ends when next pivot node is found, include this pivot node in the graph
        nodeStrategy.clear();
        opCtr++;
        generateStrategySetForLayer(*model_child,nodeStrategy);
        new_nodes.clear();
        for(auto strategy : nodeStrategy)
        {
            new_nodes.push_back(optimizationGraph.node_insert(std::tuple<mv::Op&,StrategySet,int>(*model_child,strategy,optionCtr++)));
        }
        for(const auto oldNode : old_nodes)
            for(const auto newNode : new_nodes)
            {
                double edgeCost = transitionCost( get<0>(*oldNode), get<0>(*newNode), get<1>(*oldNode), get<1>(*newNode));
                int edgeCostInt = edgeCost ;
                auto newEdge = optimizationGraph.edge_insert(oldNode,newNode,edgeCost);
                edgeCostMap.insert(std::pair<mv::graph<std::tuple<mv::Op&,StrategySet,int>,double>::edge_list_iterator, double>(newEdge, edgeCost));
            }
        old_nodes.swap(new_nodes);
        nodeStrategies.push_back(nodeStrategy);
        last_nodes = old_nodes;

/*
    Next call dijkstra on the each linear subgraph found from the modelSource on this recursive pass
 */
	allOptimizationGraphs.push_back(optimizationGraph);
        writeDot(optimizationGraph,true);
        costEdgeIteratorComp costEdgeCompare;
        auto sinkNodeIt = optimizationGraph.node_begin() ;
        for (int ii=0; ii<optimizationGraph.node_size()-1; ii++) ++sinkNodeIt;

        //Critical Path is node iter source, node iter sink, vector of edge iters critical path, double edge cost sum
        //cout << "Found starting and endings nodes: " << first_nodes.size() << ", " << last_nodes.size() << endl;
        for(auto startingNode : first_nodes){
            for( auto endingNode : last_nodes)
            {
                //CriticalEdges criticalPathEdges = dijkstra<std::tuple<mv::Op&,StrategySet,int>,double,costNodeIteratorComp,costEdgeIteratorComp, double>(optimizationGraph,startingNode,endingNode,edgeCostMap);
                linearCriticalPaths.emplace_back(dijkstra<std::tuple<mv::Op&,StrategySet,int>,double,costNodeIteratorComp,costEdgeIteratorComp, double>(optimizationGraph,startingNode,endingNode,edgeCostMap));
            }
        }

        allLinearCriticalPaths.push_back(linearCriticalPaths);

        //cout << "finished calling dijkstra on strategy optimization graph" << endl;
        next_modelSource = model_child;
        if((*next_modelSource).getOpType() != "ConstantInt"){
            vector<CriticalEdges> temp = recursiveCriticalPath(next_modelSource, recursedNodes); 
            if(!temp.empty())
                sinkRetVecs.push_back(temp);
        }
    }
    //cout << "Rolling up the recursion" << endl;
    //If the childRetVecs is empty, just pick the best of criticalPaths (linear graph)
    if(sinkRetVecs.empty()){
        //cout << "Found linear graph - choosing best of " << allLinearCriticalPaths[0].size() << " dijkstra output from input to output" << endl;
        CriticalEdges finalCriticalPath;// = allLinearCriticalPaths[0].front();
        double max = 999999.999;

        for(auto criticalPath : allLinearCriticalPaths[0])
        {
            double cost = 0;
            //cout << "Critical Path has " << criticalPath.size() << " edges" << endl;
            for (int showPath=0; showPath < criticalPath.size(); showPath++){
                cost += *criticalPath[showPath];
            }
            //cout << "Found cost of " << cost << endl;
            if( cost < max)
            {
                finalCriticalPath = criticalPath;
                max = cost;
                //cout << "Updated best cost to " << max << endl;
            }
        }
        saveStrategy(finalCriticalPath);
        vector<CriticalEdges> ret;
        ret.push_back(finalCriticalPath);
        return ret;
    }
    //Linear section no branches, use found source strategy to lock the sink for criticalPaths and pick the best with this constraint (linear part of parallel graph)
     else if(childCtr == 1){
     	vector<CriticalEdges> finalCriticalPaths = sinkRetVecs.back();
     	//Consider only potential paths which match strategy previously chosen for this sections sink
     	CriticalEdges previousSection = sinkRetVecs.back().back();
     	vector<CriticalEdges> thisSection = allLinearCriticalPaths.at(0); //only one linear part in this section
        CriticalEdges finalCriticalPath; 
        double max = 999999.999;

        //check first that this is a CP we should consider (does the sink match the strategy already decided for that node)
        auto prevSource = previousSection.front()->source();
        auto prevStrategy = get<1>(*prevSource);

        for(auto criticalPath : thisSection)
        {
        	double cost = 0;
            //cout << "Critical Path has " << criticalPath.size() << " edges" << endl;
            for (int showPath=0; showPath < criticalPath.size(); showPath++){
                cost += *criticalPath[showPath];
            }
            //TODO check that we can compare StrategySet in this way -> if strategy is a match we want this to execute
            if(prevStrategy == get<1>(*criticalPath.front()->source()) && cost < max)
            {
                finalCriticalPath = criticalPath;
                max = cost;
            }
        }

        saveStrategy(finalCriticalPath);
        finalCriticalPaths.push_back(finalCriticalPath);
        return finalCriticalPaths;
    }
    //Parallel branches between this source and sink, do addition to pick the best
    else{
    	
        vector<CriticalEdges> finalCriticalPaths = sinkRetVecs.back();
       /* CriticalEdges childCP = sinkRetVecs[0].back();
        CriticalEdges finalCriticalPath = allLinearCriticalPaths[0].front();
        double max = 999999.999;

        //check first that this is a CP we should consider (does the sink match the strategy already decided for that node)
        auto childSource = childCP.source;
        auto childSourceStrategy = get<1>(*childSource);

        vector<CriticalPath> tempBestStrats;

        for(auto& criticalPath : allLinearCriticalPaths[0])
        {
            if(childSourceStrategy != get<1>(*criticalPath.sink)) continue; //if the sink doesn't match what we already decided, skip this path
            auto cost = criticalPath.sumCost;
            for(int i = 1; i < childCtr; i++){ //for each other parallel branch
                //consider the strategy that starts with the same 
                for(auto& cp : allLinearCriticalPaths[childCtr])
                    if(get<1>(*criticalPath.source) == get<1>(*cp.source)){ //must come from matching source
                        tempBestStrats.push_back(cp);
                        cost += cp.sumCost;
                        continue;
                    }
            }

            //TODO check that we can compare StrategySet in this way -> if strategy is a match we want this to execute
            if(cost < max)
            {
                finalCriticalPaths.push_back(criticalPath);
                for(int i = 1; i < childCtr; i++){
                    finalCriticalPaths.push_back(tempBestStrats[i]);
                }
                max = cost;
            }
        }

        for(auto criticalPath : finalCriticalPaths){
        	saveStrategy(criticalPath);
        }
*/
        return finalCriticalPaths;
    }

}
void StrategyManager::generateStrategySetForLayer(mv::Op& op,vector<StrategySet>& strategyVec)
{
    //TODO:: error
    cout<<"ERROR generateStrategySetForLayer" << endl;
}

double StrategyManager::transitionCost(Op& parentOp,Op& childOp,StrategySet& parent,StrategySet& child)
{
    cout<<"ERROR transitionCost" << endl;
    return -1;
}
}
}