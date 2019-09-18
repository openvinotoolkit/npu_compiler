#include "include/mcm/pass/pass_registry.hpp"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/target/keembay/lemon_graph_scheduler.hpp"
#include <iostream>
#include "include/mcm/compiler/compilation_profiler.hpp"

static void maxTopologicalCutAndPartialSerialisationPass(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);
static void markLastNodeForMaxTopologicalCutFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor& target, mv::Element&, mv::Element&);

namespace mv
{

    namespace pass
    {

        MV_REGISTER_PASS(MarkLastNodeForMaxTopologicalCut)
        .setFunc(markLastNodeForMaxTopologicalCutFcn)
        .setDescription(
            "Perform the max topological cut algorithm and partial serialisation (if required) to schedule the DAG."
        );

        MV_REGISTER_PASS(MaxTopologicalCutAndPartialSerialisation)
        .setFunc(maxTopologicalCutAndPartialSerialisationPass)
        .setDescription(
            "Perform the max topological cut algorithm and partial serialisation (if required) to schedule the DAG."
        );
    }
}
void markLastNodeForMaxTopologicalCutFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor& target, mv::Element&, mv::Element&)
{

    mv::ControlModel cm(model);
    mv::OpModel om(model);
    auto sinkNode = cm.switchContext(om.getOutput());
    sinkNode.leftmostParent()->set<bool>("lastOpKoala", true);
}


void maxTopologicalCutAndPartialSerialisationPass(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor& target, mv::Element&, mv::Element& compOutput)
{

    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)

    int networkMemoryRequirement;
    double percentageMemory; 
    mv::LemonGraphScheduler flowGraph;
    bool memoryHack = false;

    auto returnedParams = model.getGlobalConfigParams();
    memoryHack = returnedParams->get<bool>("MemoryHack");

    /*Convert to MCM graph to Lemon graph*/
    flowGraph.convertMcMGraphToLemonGraph(pass, model);

    /*Calculate max topological cut and get the cut edges*/
    auto maxTopologicalCut = flowGraph.calculateMaxTopologicalCut(pass, model);
    
    compOutput.set<uint64_t>("maxTopologicalCut", maxTopologicalCut.first);
    mv::DataModel dm(model);
    auto outflow = dm.getOutputFlow();
    outflow->set<uint64_t>("MaxTopologicalCutValue", maxTopologicalCut.first);
    
    double cmxMemory = returnedParams->get<unsigned>("cmx");

    networkMemoryRequirement = maxTopologicalCut.first / 1024;
    percentageMemory = (maxTopologicalCut.first / cmxMemory) * 100.00;

    pass.log(mv::Logger::MessageType::Info, "The network requires " + std::to_string(networkMemoryRequirement) + " kB of available CMX memory " + std::to_string(percentageMemory) + "%");

    /*Repeat partial serialisation until max topological cut is less than CMX memory*/
    while (maxTopologicalCut.first > cmxMemory)
    {
        flowGraph.performPartialSerialisation(pass, maxTopologicalCut.second);
        maxTopologicalCut = flowGraph.calculateMaxTopologicalCut(pass, model);
        networkMemoryRequirement = maxTopologicalCut.first / 1024;
        percentageMemory = (maxTopologicalCut.first / cmxMemory) * 100;
        pass.log(mv::Logger::MessageType::Info, "The network requires " + std::to_string(networkMemoryRequirement) + " kB of available CMX memory " + std::to_string(percentageMemory) + "%");
    }

    /*Add the partial serialisaion edges to the mcmGraph*/
    flowGraph.insertpartialSerialisationEdgesInMcmGraph(model);

    /*Check again if the peak memory is less than CMX as we did not insert all the specific PS edges*/

    //--------------------------------------------------
    mv::LemonGraphScheduler flowGraph_extraCheck;

    /*Convert to MCM graph to KOALA graph*/
    flowGraph_extraCheck.convertMcMGraphToLemonGraph(pass, model);

    /*Calculate max topological cut and get the cut edges*/
    maxTopologicalCut = flowGraph_extraCheck.calculateMaxTopologicalCut(pass, model);

    pass.log(mv::Logger::MessageType::Info, "After PS the network now requires " + std::to_string(networkMemoryRequirement) + " kB of available CMX memory " + std::to_string(percentageMemory) + "%");
    
    if(maxTopologicalCut.first > cmxMemory)
        throw std::runtime_error("The maximum peak memory requirment of the graph exceeds CMX after PS, logic is broken!!!");
    
    pass.log(mv::Logger::MessageType::Info, "Exiting max cut pass ");

}
