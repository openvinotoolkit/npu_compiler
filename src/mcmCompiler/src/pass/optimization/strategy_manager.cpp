#include "limits"
#include "tuple"
#include "chrono"
#include <utility>

#include "include/mcm/pass/graphOptimizations/StrategyManager.hpp"
#include "include/mcm/pass/graphOptimizations/StrategyRegistry.hpp"
#include "include/mcm/base/element.hpp"
#include "include/mcm/algorithms/dijkstra.hpp"
#include "include/mcm/utils/env_loader.hpp"

namespace mv {
namespace graphOptimizer {

using namespace std;

std::atomic<int> MetaEdge::unique_ctr(0);
std::atomic<int> MetaGraph::unique_ctr(0);
std::atomic<int> StrategyManager::unique_ctr(0);

StrategyManager::StrategyManager(OpModel& model,mv::Element& passDesc) :
        model_(model),passDesc_(passDesc)
{
    topologicalModel_ = model_.topologicalSort();
}

//TODO:: error if the strategy is not there...
Attribute& StrategyManager::getStrategy(mv::Op op,string strategy)
{
    auto op_name = op.getName();
    auto software = op.hasAttr("softwareExecuted") && op.get<bool>("softwareExecuted");
    if (!(op.hasTypeTrait("optimizable")) || software)
    {
        log(Logger::MessageType::Debug, "StrategyManager: using Default strategy for " + op_name + " op");
        op_name = "Default";
    }
    auto layerEntry = layerStrategies_.find(op_name);

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

    return strategyEntry->second;
}

void  StrategyManager::setGlobalStrategy(string& name, Attribute& strategy)
{
    globalStrategies_[name]= strategy;
}

void StrategyManager::setGlobalConfig(string& name,Attribute& config)
{
    globalConfig_[name] = config;
}

const Attribute& StrategyManager::getGlobalConfig(const string& name) const
{
    auto it = globalConfig_.find(name);
    if(it == globalConfig_.end())
        throw ArgumentError(*this, "name", name, "Undefined attribute");
    return it->second;
}

const Attribute& StrategyManager::getGlobalStrategy(const string& name) const
{
    auto it = globalStrategies_.find(name);
    if(it == globalStrategies_.end())
        throw ArgumentError(*this, "name", name, "Undefined attribute");
    return it->second;
}

const StrategyManager::StrategySet& StrategyManager::getLayerStrategySet(const string& name) const
{
    auto it = layerStrategies_.find(name);
    if(it == layerStrategies_.end())
        throw ArgumentError(*this, "name", name, "Undefined attribute");
    return it->second;
}

bool StrategyManager::hasAttr(const GlobalSetting& map,const string& name) const
{
    return map.find(name) != map.end();
}

bool StrategyManager::hasAttr(const LayerStrategySet& map,const string& name) const
{
    return map.find(name) != map.end();
}

std::string StrategyManager::getLogID() const
{
    return "GraphOptimizer-StrategyManager";
}

void StrategyManager::updateValuesFromJSON()
{
    auto graphOptimizerConfig = passDesc_.get<mv::Element>("graphOptimizerConfig");
    auto globalConfigs = graphOptimizerConfig.get<vector<mv::Element>>("globalConfigs");
    auto globalStrategies = graphOptimizerConfig.get<vector<mv::Element>>("globalStrategies");
    auto layerStrategySets  = graphOptimizerConfig.get<vector<mv::Element>>("layerStrategies");

    for( auto globalConfig : globalConfigs)
        globalConfig_[globalConfig.getName()] = globalConfig.get("value");
    
    globalConfig_["referenceDevice"] = model_.getGlobalConfigParam("referenceDevice");
    globalConfig_["totalClusters"] = model_.getGlobalConfigParam("Number_of_Clusters");
    globalConfig_["clusterMemory"] = (int)model_.getGlobalConfigParam("totalCmx").get<unsigned>();
    globalConfig_["dpuPerCluster"] = 
        model_.getGlobalConfigParam("Number_of_DPUs").get<int>() / model_.getGlobalConfigParam("Number_of_Clusters").get<int>();

    for( auto globalStrategy : globalStrategies)
    {
        auto strategyName = globalStrategy.getName();
         if (model_.hasGlobalConfigParam(strategyName))
            globalStrategies_[strategyName] = model_.getGlobalConfigParam(strategyName);
        else
            globalStrategies_[strategyName] = globalStrategy.get("value");
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

mv::Element convertToStreamingElement(mv::Element element, mv::Shape strategy , std::string name)
{
    element.set("name_filter",name);

    std::vector<mv::Element> copySplits;
    for(int i=0;i<5;i++)
        copySplits.emplace_back("");

    copySplits[0].set<int>("W", strategy[0]);
    copySplits[1].set<int>("H", strategy[1]);
    copySplits[2].set<int>("C", strategy[2]);
    copySplits[3].set<int>("K", strategy[3]);
    copySplits[4].set<int>("N", strategy[4]);
    element.set("splits",copySplits);

    return element;
}

std::vector<mv::Element> StrategyManager::convertStreamingStrategyToElement(CriticalPathNodes &strategiesToConvert, std::shared_ptr<mv::Element> compDesc)
{

    std::vector<mv::Element> streamingStrategyList;

    if(compDesc->hasAttr("streaming_strategy"))
    { 
        streamingStrategyList = compDesc->get<std::vector<mv::Element>>("streaming_strategy");
        //determine if node already has streaming strategy from JSON text, do not override text specification
        std::vector<std::string> hasSpec;
        for (auto s : streamingStrategyList)
        {
            std::string nodeName = s.get<std::string>("name_filter");
            auto splitList = s.get<std::vector<mv::Element>>("splits");
            for (unsigned i = 0; i < splitList.size(); i++)
            {
                if ((splitList[i].hasAttr("C"))||(splitList[i].hasAttr("H"))||(splitList[i].hasAttr("W"))||(splitList[i].hasAttr("K"))||(splitList[i].hasAttr("N")))
                    hasSpec.push_back(nodeName);
            }
        }
        
        //cast streaming strategy into Element
        mv::Element copyElement("");
        if(streamingStrategyList.size() != 0)
            copyElement = streamingStrategyList[0];

        for (auto elem : strategiesToConvert)
        {
            auto& strategy = *elem;
            mv::Shape newStrategy = strategy["streaming"];
            std::string newName = strategy["name"] ;
            if ( std::find(hasSpec.begin(), hasSpec.end(), newName) == hasSpec.end())
            {
                streamingStrategyList.push_back(convertToStreamingElement(copyElement,newStrategy,newName));
            }
        }
        return streamingStrategyList;
    }
    else
    {
        for(auto elem : strategiesToConvert)
        {
            auto& strategy = *elem;
            mv::Shape newStrategy = strategy["streaming"];
            std::string newName = strategy["name"];
            
            mv::Element copyElement(""); 
            streamingStrategyList.emplace_back(std::move(convertToStreamingElement(copyElement,newStrategy,newName)));            
        }
        return streamingStrategyList;
    }
}

mv::Element convertToClusteringElement(std::string strategy , std::string name)
{
    mv::Element element("");
    element.set("name_filter",name);
    element.set("strategy",strategy);

    return element;
}

std::vector<mv::Element> StrategyManager::convertClusteringStrategyToElement(CriticalPathNodes &strategiesToConvert,
                                                                                 std::shared_ptr<mv::Element> compDesc)
{
    std::vector<mv::Element> clusteringStrategyList;

    if(compDesc->hasAttr("split_strategy"))
    { 
        clusteringStrategyList = compDesc->get<std::vector<mv::Element>>("split_strategy");
        //determine if node already has clustering strategy from JSON text, do not override text specification
        std::vector<std::string> hasClusterSpec ;
        for (auto s : clusteringStrategyList)
        {
            std::string nodeName = s.get<std::string>("name_filter");
            std::string strategyName = s.get<std::string>("strategy");
            if ((strategyName=="SplitOverH") ||
                (strategyName=="SplitOverK") ||
                (strategyName=="SplitOverHOverlapped") ||
                (strategyName=="HKSwitch"))
            {
                hasClusterSpec.push_back(nodeName);
            }
        }

        //save clustering strategy into compilation descriptor
        for (auto elem : strategiesToConvert)
        {
            auto& strategy = *elem;
            std::string newStrategy = strategy["clustering"].get<std::string>();
            std::string newName = strategy["name"].get<std::string>();
            auto op = model_.getOp(newName);
            if(op->getOpType() == "Concat")
                newStrategy = std::string("Clustering");
            if ( std::find(hasClusterSpec.begin(), hasClusterSpec.end(), newName) == hasClusterSpec.end())
            {
                clusteringStrategyList.push_back(convertToClusteringElement(newStrategy,newName));
            }
        }
        return clusteringStrategyList;
    }
    else
    {
        for (auto elem : strategiesToConvert)
        {
            auto& strategy = *elem;
            std::string newStrategy = strategy["clustering"];
            std::string newName = strategy["name"];
    
            clusteringStrategyList.push_back(convertToClusteringElement(newStrategy,newName));
        }
        return clusteringStrategyList;
    }
}

std::vector<mv::Element> StrategyManager::convertLocationStrategyToElement(CriticalPathNodes &strategiesToConvert)
{
    mv::Element copyLElement("");
    std::vector<mv::Element> locationStrategyList;

    for(auto elem : strategiesToConvert)
    {
        auto& strategy = *elem;
        auto spilling = strategy["spilling"].get<bool>();
        auto opName   = strategy["name"].get<string>();

        std::string DDRLocation = "DDR";
        std::string CMXLocation = "CMX";
        
        //todo::don't search the whole model for this
        auto op = model_.getOp(opName);
        if(op->getOpType() == "Output")
            continue;

        if(spilling)
            copyLElement.set("mem_location",DDRLocation);
        else
            copyLElement.set("mem_location",CMXLocation);
        copyLElement.set("name_filter", opName);
    
        locationStrategyList.push_back(copyLElement);
    }

    return locationStrategyList;
}

std::vector<mv::Element> StrategyManager::convertSparsityStrategyToElement(CriticalPathNodes &strategiesToConvert){
    log(Logger::MessageType::Debug, "GraphOptimizer: Converting Sparsity Strategies to Element");

    mv::Element copyLElement("");
    std::vector<mv::Element> sparsityStrategyList;

    for(auto elem: strategiesToConvert)
    {
        auto& strategy = *elem;
        auto inputActivationSparsity = strategy["inputSparsity"].get<bool>();
        auto outputActivationSparsity = strategy["outputSparsity"].get<bool>();
        auto weightsSparsity = strategy["weightsSparsity"].get<bool>();
        auto opName   = strategy["name"].get<string>();

        auto op = model_.getOp(opName);

        copyLElement.set("inputActivationSparsity",inputActivationSparsity);
        copyLElement.set("outputActivationSparsity",outputActivationSparsity);
        copyLElement.set("weightsSparsity",weightsSparsity);
        copyLElement.set("name_filter", opName);

        sparsityStrategyList.push_back(copyLElement);
    }

    return sparsityStrategyList;
}

std::vector<mv::Element> StrategyManager::convertPipeliningStrategyToElement(CriticalPathNodes &strategiesToConvert)
{
    log(Logger::MessageType::Debug, "GraphOptimizer: Converting Pipelining Strategies to Element");

    mv::Element copyLElement("");
    std::vector<mv::Element> pipeliningStrategyList;

    for(auto elem: strategiesToConvert)
    {
        auto& strategy = *elem;
        auto pipelining = strategy["pipelined"].get<bool>();
        auto opName   = strategy["name"].get<string>();
        std::string pipelineStrategy = "None";

        if(pipelining)
        {
            auto streaming = strategy["streaming"].get<mv::Shape>();
        
            if(streaming["K"] > 1)
                pipelineStrategy = "PipelineWeights";
            else if(streaming["H"] > 1)
                pipelineStrategy = "PipelineActivations";
        }
        
     copyLElement.set("pipelining", pipelineStrategy);
        copyLElement.set("name_filter", opName);

        pipeliningStrategyList.push_back(copyLElement);
    }

    return pipeliningStrategyList;
}

void StrategyManager::saveMetaStrategy(CriticalPathNodes& criticalPathNodes)
{
    struct {
        bool operator() (OptimizationGraph::node_list_iterator a,OptimizationGraph::node_list_iterator b) const
        {
            auto left = *a;
            auto right = *b;

            return left["name"].get<string>().compare(right["name"].get<string>()) < 0;
        }
    }strategyNameComparator;

    sort(criticalPathNodes.begin(),criticalPathNodes.end(),strategyNameComparator);
    const bool enableSaveStrategyToDescriptor = true;
    const bool enableSaveStrategyToJsonFile = true;

    // Give pipelined attribute
    for(auto elem : criticalPathNodes)
    {
        auto& strategy = *elem;
        auto opName = strategy["name"].get<std::string>();

        auto op = model_.getOp(opName);

        auto software = op->hasAttr("softwareExecuted") && op->get<bool>("softwareExecuted");
        strategy["pipelined"] = false;
        if ((op->hasTypeTrait("executable")) && !software && op->getOpType() != "Input")
        {
            // Determine if final strategy choice enabled pipelining for this op
            auto parentOp = model_.getSourceOp(op->getInputTensor(0));
            auto parentSpill = true;
            for(auto parentElem : criticalPathNodes)
            {
                auto& parentStrategy = *parentElem;
                if(parentOp->getName() == parentStrategy["name"].get<std::string>())
                {
                    parentSpill = parentStrategy["spilling"].get<bool>();
                    break;
                }
            }
            if(isPipeliningPossible(*op, strategy, parentSpill))
            {
                strategy["pipelined"] = true;
            }
        }
    }

    auto globalParams = model_.getGlobalConfigParams();

    std::vector<mv::Element> streamingStrategyElements = convertStreamingStrategyToElement(criticalPathNodes, globalParams);
    std::vector<mv::Element> multiClusterStrategyElements = convertClusteringStrategyToElement(criticalPathNodes, globalParams);
    std::vector<mv::Element> locationStrategyElements = convertLocationStrategyToElement(criticalPathNodes);
    std::vector<mv::Element> sparsityStrategyElements = convertSparsityStrategyToElement(criticalPathNodes);
    std::vector<mv::Element> pipeliningStrategyElements = convertPipeliningStrategyToElement(criticalPathNodes);

    if (enableSaveStrategyToDescriptor)
    {
        log(Logger::MessageType::Debug, "GraphOptimizer: Saving Strategy to Compilation Descriptor");
        auto compDesc = model_.getGlobalConfigParams();
        compDesc->set("streaming_strategy", streamingStrategyElements);
        compDesc->set("split_strategy", multiClusterStrategyElements);
        compDesc->set("sparsity_strategy", sparsityStrategyElements);
        compDesc->set("pipelining_strategy", pipeliningStrategyElements);
    }

    if (enableSaveStrategyToJsonFile)
    {
        log(Logger::MessageType::Debug, "GraphOptimizer: Saving Strategy to JSON file");
        std::ofstream jsonOutputFile ;
        jsonOutputFile.open(jsonOutFileName, std::ios::out );
        if (!(jsonOutputFile.is_open()))
            log(Logger::MessageType::Debug, "GraphOptimizer: Could not open output file " + jsonOutFileName);

        auto currentTime= chrono::system_clock::to_time_t(chrono::system_clock::now());
        std::string timeStamp(ctime(&currentTime));
        if (!timeStamp.empty() && timeStamp[timeStamp.length()-1] == '\n')
            timeStamp.erase(timeStamp.length()-1);

        mv::Element SSA("Streaming strategies generated by mcmCompiler "+timeStamp);
        mv::Element CSA("Clustering strategies generated by mcmCompiler "+timeStamp);
        mv::Element LSA("Tensor placement strategies generated by mcmCompiler "+timeStamp);
        mv::Element SpSA("Sparsity strategies generated by mcmCompiler "+timeStamp);
        mv::Element PSA("Pipelining strategies generated by mcmCompiler "+timeStamp);
        SSA.set("streaming_strategy",streamingStrategyElements);
        CSA.set("split_strategy",multiClusterStrategyElements);
        LSA.set("tensor_placement_override",locationStrategyElements);
        SpSA.set("sparsity_strategy",sparsityStrategyElements);
        PSA.set("pipelining_strategy",pipeliningStrategyElements);
        auto jsonSStrategy = SSA.toJSON(true);
        auto jsonCStrategy = CSA.toJSON(true);
        auto jsonLStrategy = LSA.toJSON(true);
        auto jsonSpStrategy = SpSA.toJSON(true);
        auto jsonPStrategy = PSA.toJSON(true);
        jsonOutputFile << jsonSStrategy.stringifyPretty() << "," << std::endl;
        jsonOutputFile << jsonCStrategy.stringifyPretty() << "," << std::endl;
        jsonOutputFile << jsonLStrategy.stringifyPretty()  << "," << std::endl;
        jsonOutputFile << jsonSpStrategy.stringifyPretty()  << "," << std::endl;
        jsonOutputFile << jsonPStrategy.stringifyPretty() << std::endl;

        jsonOutputFile.close();
    }

    // attach optimal tensor location (CMX or DDR) attribute to tensor
    for(auto elem : criticalPathNodes)
    {
        auto& strategy = *elem;
        auto spilling = strategy["spilling"].get<bool>();
        auto streamShape = strategy["streaming"].get<mv::Shape>();
        auto opName   = strategy["name"].get<std::string>();

        auto op = model_.getOp(opName);
        if(op->getOpType() == "Output")
            continue;

        auto outTensor = op->getOutputTensor(0);
        auto executable = op->hasTypeTrait("executable") ? true : false;

        bool isStreaming = ((streamShape["W"] * streamShape["H"] * streamShape["C"] 
                                            * streamShape["K"] * streamShape["B"]) > 1) ? true : false;
        if((spilling && executable) || isStreaming) // TODO remove this isStreaming check
            outTensor->set<mv::Tensor::MemoryLocation>("Location",mv::Tensor::MemoryLocation::DDR);
        else
            outTensor->set<mv::Tensor::MemoryLocation>("Location",mv::Tensor::MemoryLocation::NNCMX);

        log(Logger::MessageType::Debug, "GraphOptimizer: Output tensor location (from tensor attribute) for node " + op->getName() + " is " + outTensor->get("Location").toString());
    }
}

void StrategyManager::initLayerStrategySets()
{
    for(auto opIt = model_.opBegin(); opIt != model_.opEnd() ; ++ opIt)
    {
        const auto& opType = opIt->getOpType();
        //todo:: have a generic trait marker for "Constant" operations at opDef ( among other generic traits todo's)
        if ((opType != "Constant") &&
            (opType != "ConstantInt") &&
            (opType != "ConstantDataElement") &&
            (opType != "WeightsTable") &&
            (opType != "SparsityMap"))
        {
            auto nodeStrategies = make_shared<vector<StrategySet>>(0);
            generateStrategySetForLayer(*opIt,*nodeStrategies);

            opIt->set<shared_ptr<vector<StrategySet>>>("StrategySet",nodeStrategies);
        }
    }

    return;
}

bool StrategyManager::isLinearGraph(mv::Data::OpListIterator opBegin,
                                        mv::Data::OpListIterator opEnd,
                                        vector<mv::Data::OpListIterator> children)
{
    if(children.size() > 1)
        return false;

    auto modelEnd = model_.opEnd();

    for(mv::Data::OpDFSIterator dfsIterator(children[0]); dfsIterator != opEnd; ++dfsIterator)
    {
        //we will need to check for linearity between 2 nodes; If we get to OpEnd troughout the DFS iteration
        //then the subGraph builder did some urecoverable logic error
        if(dfsIterator == modelEnd)
        {
            mv::LogicError(*this,"Logic error: recursive graphOptimizer got pivots " +
                            opBegin->getName() + " to " + opEnd->getName());
        }

        //While iterating DFS, if we get multiple child ops, then we have a pivot node
        if(dfsIterator.childrenSize() > 1)
        {
            return false;
        }
    }

    return true;
}

int StrategyManager::countInputLayers(mv::Data::OpListIterator op){
    int inputs = 0;
    for(auto inputOp = op.leftmostParent(); inputOp != model_.opEnd(); ++inputOp)
    {
        auto inputType = inputOp->getOpType();
        if ((inputType == "Constant") ||
        (inputType == "ConstantInt") ||
        (inputType == "ConstantDataElement") ||
        (inputType == "WeightsTable") ||
        (inputType == "SparsityMap"))
            continue;
        inputs++;
    }
    return inputs;
}

// Function to find the lowest common child (LCA) of opBegin
// 1. Use topological sort of the graph
// 2. Count opening/closing of branches (like a parser for matching parentheses for example...)
// 3. If (current count) - (inputs to next node) < 0, found "bottleneck" of graph i.e. end of parallel branches
// 4. At each step current count gets set to (current count) + (outputs from node) - (inputs to node)
mv::Data::OpListIterator StrategyManager::LCA(mv::Data::OpListIterator opBegin, mv::Data::OpListIterator opEnd)
{
    // cout << "Finding LCA between " << opBegin->getName() << " and " << opEnd->getName() << endl;
    int preceedingBranches = 0;
    int followingBranches = 0;

    bool atStart = false;

    for(auto node : topologicalModel_){
        if(node == opBegin) { // TODO more efficient way to jump to correct place
            atStart = true; 
            followingBranches = node.outputsSize() - 1;
            continue;
        }
        if(!atStart) continue;
        if(node == opEnd) return opEnd;

        auto opType = node->getOpType();
        // Skip weights/constants nodes (have already been added to convs/software layers at this point)
        if ((opType == "Constant") ||
            (opType == "ConstantInt") ||
            (opType == "ConstantDataElement") ||
            (opType == "WeightsTable") ||
            (opType == "SparsityMap") ||
            (opType == "Input") ||
            (opType == "Output") )
            continue;

        // Get number of input and output to node
        int input = countInputLayers(node);
        int output = node.childrenSize();

        // This node can't be a pivot if it has just 1 input and output, skip it
        if(input == 1 && output == 1)
            continue;

        preceedingBranches = followingBranches;
        followingBranches += (output - input);

        // Note: To capture the case of pivot being the end of one parallel section, and 
        // the start of the next. First we test considering just input (did parallel end here)
        if((preceedingBranches - input) < 0)
        {
            return node;
        }
    }
    return opEnd;
}

// Note: A non-exclusive node is one which we pass through when following different parallel paths,
// but is not also the lowest common child (lcsa) of those parallel paths
std::vector<mv::Data::OpListIterator> StrategyManager::getNonExclusiveNodes(mv::Data::OpListIterator opBegin, 
                                                                                mv::Data::OpListIterator opEnd)
{
    std::vector<mv::Data::OpListIterator> nonExclusiveNodes;
    std::vector<mv::Data::OpListIterator> dfsStartNodes;

    set<std::string> nodesSeen;
    set<std::string> nodesAdded;
    set<std::string> dfsNodesFound;

    // cout << "Getting non-exclusive nodes from " << opBegin->getName() << " to " << opEnd->getName() << endl;

    for(auto child = opBegin.leftmostChild(); child != model_.opEnd(); ++child)
        dfsStartNodes.push_back(child);

    size_t index = 0;
    while(index < dfsStartNodes.size())
    {
        mv::Data::OpDFSIterator it(dfsStartNodes[index]);

        for( ;it != opEnd; ++it ){
            // Add other children not on this dfs path to dfsStartNodes
            if(it.childrenSize() > 1)
            {
                auto childIt = it.leftmostChild();
                ++childIt; // skip the path we're already on
                while(childIt != model_.opEnd()){
                    if(dfsNodesFound.find(childIt->getName()) == dfsNodesFound.end()){
                        dfsStartNodes.push_back(childIt);
                    }
                    dfsNodesFound.insert(childIt->getName());
                    ++childIt;
                }
            }
            // TODO more efficient way than comparing strings?
            if((nodesSeen.find(it->getName()) != nodesSeen.end()) &&
                countInputLayers(it) != 1)
                {
                    if(nodesAdded.find(it->getName()) == nodesAdded.end())
                    {
                        nonExclusiveNodes.push_back(it);
                        nodesAdded.insert(it->getName());
                    }
                }
            else
            {
                nodesSeen.insert(it->getName());
            }
        }
        index++;
    }

    return nonExclusiveNodes;
}

// Special handling for non-exclusive branches. These have dependencies that we can't represent yet.
// 1. For each non-exclusive node, remove all but one input branch (poss immprovement, branch ending in dpu task if exists)
// 2. Reconnect directly to the LCSA. All nodes affected (non-exclusive node, it's input node, and the lcsa) 
//    are forced to have clustering strategy in GO.
// 3. After solving graph these will need to be reconnected to the correct nodes before the pass ends
//    So remember the flows added/removed appropriately
void StrategyManager::handleNonExclusiveSubgraphs(std::vector<mv::Data::OpListIterator> nonExclusiveNodes, mv::Data::OpListIterator lcsa)
{
    // cout << " Non-Exclusive nodes are : ";
    // for(auto node : nonExclusiveNodes){
    //     cout << node->getName() << ", ";
    // }
    // cout << endl;
    for(auto node : nonExclusiveNodes){
        // Remove all but one input edge and mark source of that edge as clustering required
        auto input = node.leftmostParent();
        ++input;
        std::vector<mv::Data::FlowSiblingIterator> flowsToRemove;
        std::vector<mv::Data::OpListIterator> opsToLink;
        for(; input != model_.opEnd(); ++input ){
            auto inputType = input->getOpType();
            if ((inputType == "Constant") ||
            (inputType == "ConstantInt") ||
            (inputType == "ConstantDataElement") ||
            (inputType == "WeightsTable") ||
            (inputType == "SparsityMap"))
                continue;
            input->set<bool>("forceClustering", true);
            opsToLink.push_back(input);
            // Find the edge between input and node
            // cout << "   Removing edge: " << input->getName() << " --> " << node->getName() << endl;
            flowsToRemove.push_back(input.leftmostOutput());
        }
        for(auto flow : flowsToRemove){
            auto sink = flow.sink();
            auto source = flow.source();
            size_t inputIdx = 0;
            for(size_t idx = 0; idx < sink.parentsSize(); idx++){
                if(model_.getSourceOp(sink->getInputTensor(idx)) == source)
                    inputIdx = idx;
            }
            removedFlows_.push_back(make_tuple(flow.source(), (size_t) 0, flow.sink(), inputIdx));
            model_.undefineFlow(flow);
        }
        for(auto op : opsToLink){
            auto flow = model_.defineFlow(op->getOutputTensor(0), lcsa, lcsa.parentsSize());
            addedFlows_.push_back(flow);
        }
        // cout << "  Set forceClustering: " << node->getName() << endl;
        node->set<bool>("forceClustering", true);
    }
    // cout << "Set forceClustering: " << lcsa->getName() << endl;
    lcsa->set<bool>("forceClustering", true);
}

shared_ptr<vector<StrategyManager::SubGraph>> StrategyManager::extractSubgraphs(mv::Data::OpListIterator opBegin,
                                                                                    mv::Data::OpListIterator opEnd,
                                                                                    vector<mv::Data::OpListIterator> children)
{
    auto sGraphs = make_shared<vector<SubGraph>>();
    auto travelingNode = opBegin;
    auto travelingChildren = children;

    while(travelingNode != opEnd)
    {
        if(travelingChildren.size() == 1)
        {
            mv::Data::OpDFSIterator it(travelingChildren[0]);
            for( ;(it.childrenSize() == 1) && (it != opEnd); ++it );

            sGraphs->push_back( SubGraph(travelingNode,it,{travelingChildren[0]}));

            // cout<<"Traveled linear section " << travelingNode->getName() << " -> " << it->getName() << endl;
            travelingNode = it;
            travelingChildren.clear();

            for(auto child = travelingNode.leftmostChild(); child != model_.opEnd(); ++child)
            {
                travelingChildren.push_back(child);
            }
        }
        else
        {
            auto lcsa = LCA(travelingNode, opEnd);
            // cout << "Found branching out section " << travelingNode->getName() << "->" << lcsa->getName() << endl;
// Original Notes from Istvan:
//             once we have the LCSA (lowest common SINGLE ancestor), we need to check for exclusivity of the branches.
//             we will do this via DFS-ing each child branch of the branching node, with the ending contition being the lcsa.
//             if we found a dfs path that exclusive (i.e. only this path touches the nodes), then it means we have a "good" subgraph
//             if we found branches with non-exclusive nodes, then they in summary will compose a subGraph.
//             The "special" scenario will arise, when there are no exclusive branches. This needs to go to the "special handilng"
//             TODO:: for now assume all child branches are exclusive, and just add them. Need to implement check to
//                    see if a path trough a specific child is exclusive or not, and group them until they become exclusive
//             TODO:: implement special case handling. If we cannot group children until they become exclusive, then need
//                    start removing edges until they do

            // Note: Temporarily disable complex graph handling when the leaky relu workaround is enabled
            auto globalParams = model_.getGlobalConfigParams();
            bool PPEAccuracy = globalParams->hasAttr("PPEAccuracy") ? globalParams->get<bool>("PPEAccuracy") : false;
            if(!PPEAccuracy){
                std::vector<mv::Data::OpListIterator> nonExclusiveNodes = getNonExclusiveNodes(travelingNode, lcsa);
                if(!nonExclusiveNodes.empty()){
                    handleNonExclusiveSubgraphs(nonExclusiveNodes, lcsa);
                }
            }
        
            for(auto child = travelingNode.leftmostChild(); child != model_.opEnd(); ++child)
                sGraphs->push_back( SubGraph(travelingNode,lcsa,{child}));

            travelingNode = lcsa;

            travelingChildren.clear();
            for(auto child = travelingNode.leftmostChild(); child != model_.opEnd(); ++child)
            {
                travelingChildren.push_back(child);
            }
        }
    }
    return sGraphs;
}

std::shared_ptr<MetaGraph> StrategyManager::linearGraphSolver(mv::Data::OpDFSIterator opBegin,
                                                                mv::Data::OpDFSIterator opEnd,
                                                                mv::Data::OpDFSIterator firstChild)
{
    // cout << "Solving Linear Section " << opBegin->getName() << " -> " << opEnd->getName() << " via " << firstChild->getName() << endl;
    auto linearMeta = make_shared<MetaGraph>();
    auto modelEnd = model_.opEnd();

    auto cost = [this](Op& parentOp,Op& childOp,StrategySet& a,StrategySet& b) ->double
            {return this->transitionCost(parentOp,childOp,a,b); };

    //do first pivot out of the loop
    {
        auto nodeStrategies = opBegin->get<shared_ptr<vector<StrategySet>>>("StrategySet");
        linearMeta->addNewLevel(*opBegin,nodeStrategies,cost);
    }

    for(auto dfsIterator = firstChild; dfsIterator != opEnd; ++dfsIterator)
    {
        //we will need to check for linearity between 2 nodes; If we get to OpEnd troughout the DFS iteration
        //then the subGraph builder did some urecoverable logic error
        if(dfsIterator == modelEnd)
        {
            mv::LogicError(*this,"Logic error: recursive graphOptimizer got pivots " +
                            opBegin->getName() + " to " + opEnd->getName());
        }

        auto nodeStrategies = dfsIterator->get<shared_ptr<vector<StrategySet>>>("StrategySet");
        linearMeta->addNewLevel(*dfsIterator,nodeStrategies,cost);
    }

    //do last pivot out of the loop
    {
        auto nodeStrategies = opEnd->get<shared_ptr<vector<StrategySet>>>("StrategySet");
        linearMeta->addNewLevel(*opEnd,nodeStrategies,cost);
    }

    linearMeta->solve();

    return linearMeta;
}

std::shared_ptr<MetaGraph> StrategyManager::recursiveGraphSolver(mv::Data::OpListIterator opBegin,
                                                                    mv::Data::OpListIterator opEnd,
                                                                    vector<mv::Data::OpListIterator> children)
{
    if(isLinearGraph(opBegin,opEnd,children))
    {
        return linearGraphSolver(opBegin,opEnd,children[0]);
    }
    else
    {
        auto subGraphs = extractSubgraphs(opBegin,opEnd,children);

        vector<std::shared_ptr<MetaGraph>> childMetas;
        auto masterMeta = make_shared<MetaGraph>();

        if (subGraphs != nullptr) {
            for( auto sGraph : *subGraphs )
            {
                auto& sGraphStart = get<0>(sGraph);
                auto& sGraphEnd   = get<1>(sGraph);
                auto& sGraphChildren = get<2>(sGraph);

                auto meta = recursiveGraphSolver(sGraphStart,sGraphEnd,sGraphChildren);
                //TODO Complexgraph solver here if no children
                childMetas.push_back(meta);
            }
        }

        for(const auto& meta : childMetas)
        {
            if(createStrategyDots)
                meta->write(dotFileLocation,true);
            masterMeta->fuseMeta(meta);
        }

        masterMeta->solve();
        //todo:: implement sanity check function, to verify the metaGraph

        return masterMeta;
    }
}

void StrategyManager::graphParameterOptimizations()
{
    initLayerStrategySets();

    auto startingNode = model_.getInput();
    auto endingNode = model_.getOutput();
    vector<mv::Data::OpListIterator> children;

    for( auto child = startingNode.leftmostChild(); child != model_.opEnd(); ++child)
        children.push_back(child);

    auto generalizedLinearMeta = recursiveGraphSolver(startingNode,endingNode,children);
    if(createStrategyDots)
        generalizedLinearMeta->write(dotFileLocation,true);

    auto finalMetaGraph = make_shared<MetaGraph>();
    finalMetaGraph->fuseMeta(generalizedLinearMeta);
    finalMetaGraph->solve();

    auto bestPath = finalMetaGraph->getLowestCriticalPathExtended();
    // Revert changes to the op model now that critical path is calculated
    for(auto flow : addedFlows_){
        model_.undefineFlow(flow);
    }
    for(auto flow : removedFlows_){
        model_.defineFlow(get<0>(flow), get<1>(flow), get<2>(flow), get<3>(flow));
    }
    saveMetaStrategy(*bestPath->nodes);

    if(createStrategyDots)
        finalMetaGraph->write(dotFileLocation,true);

}

bool StrategyManager::isPipeliningPossible(mv::Op& op, StrategySet& /*strategy*/, bool /*parentSpilling*/)
{
    throw mv::ArgumentError("StrategyManager", "isPipeliningPossible", op.toString(), "Unable to determine pipelining");
}
void StrategyManager::generateStrategySetForLayer(mv::Op& op,vector<StrategySet>& /*strategyVec*/)
{
    throw mv::ArgumentError("StrategyManager", "generateStrategySetForLayer", op.toString(), "No strategy for this layer");
}

double StrategyManager::transitionCost(Op&, Op& childOp, StrategySet&, StrategySet&)
{
    log(mv::Logger::MessageType::Warning, "No transition cost asssociated with this op:  " + childOp.toString());
    return -1;
}
}
}
