#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/op/op.hpp"
#include "include/mcm/algorithms/transitive_reduction.hpp"
#include "include/mcm/algorithms/critical_path.hpp"
#include "include/mcm/algorithms/path_exists.hpp"
#include "include/mcm/algorithms/scheduling_sort.hpp"

mv::ControlModel::ControlModel(ComputationModel &other) :
ComputationModel(other)
{

}

mv::ControlModel::~ControlModel()
{

}

mv::Control::OpListIterator mv::ControlModel::switchContext(Data::OpListIterator other)
{
    return opsGraph_->get_second_iterator(other);
}

mv::Control::OpListIterator mv::ControlModel::getFirst()
{
   conjoined_graph<Op, DataFlow, ControlFlow>::first_graph::node_list_iterator it = *input_;
   return opsGraph_->get_second_iterator(it);
}

mv::Control::OpListIterator mv::ControlModel::getLast()
{
   conjoined_graph<Op, DataFlow, ControlFlow>::first_graph::node_list_iterator it = *output_;
   return opsGraph_->get_second_iterator(it);
}

mv::Control::OpListIterator mv::ControlModel::opBegin()
{
    return controlGraph_.node_begin();
}

mv::Control::OpListIterator mv::ControlModel::opEnd()
{
    return *controlOpEnd_;
}

mv::Control::FlowListIterator mv::ControlModel::getInput()
{
    return switchContext(*input_).leftmostOutput();
}

mv::Control::FlowListIterator mv::ControlModel::getOutput()
{
    return switchContext(*output_).leftmostInput();
}

mv::Control::FlowListIterator mv::ControlModel::flowBegin()
{
    return controlGraph_.edge_begin();
}

mv::Control::FlowListIterator mv::ControlModel::flowEnd()
{
    return *controlFlowEnd_;
}

void mv::ControlModel::addGroupElement(Control::OpListIterator element, GroupIterator group)
{
    if (!isValid(element))
        throw ArgumentError(*this, "newElement:iterator", "invalid", "Invalid iterator passed while including op to a group");
    if (!isValid(group))
        throw ArgumentError(*this, "group:iterator", "invalid", "Invalid iterator passed while including op to a group");

    group->include(element);
}

void mv::ControlModel::addGroupElement(Control::FlowListIterator element, GroupIterator group)
{
    if (!isValid(element))
        throw ArgumentError(*this, "newElement:iterator", "invalid", "Invalid iterator passed while including control flow to a group");
    if (!isValid(group))
        throw ArgumentError(*this, "group:iterator", "invalid", "Invalid iterator passed while including control flow to a group");

    group->include(element);
}

void mv::ControlModel::removeGroupElement(Control::OpListIterator element, GroupIterator group)
{
    if (!isValid(element))
        throw ArgumentError(*this, "newElement:iterator", "invalid", "Invalid iterator passed while excluding op from a group");
    if (!isValid(group))
        throw ArgumentError(*this, "group:iterator", "invalid", "Invalid iterator passed while excluding op from a group");
    group->exclude(element);
}

void mv::ControlModel::removeGroupElement(Control::FlowListIterator element, GroupIterator group)
{
    if (!isValid(element))
        throw ArgumentError(*this, "newElement:iterator", "invalid", "Invalid iterator passed while excluding control flow from a group");
    if (!isValid(group))
        throw ArgumentError(*this, "group:iterator", "invalid", "Invalid iterator passed while excluding control flow from a group");
    group->exclude(element);
}

mv::Control::StageIterator mv::ControlModel::addStage()
{   
    
    auto it = stages_->emplace(stages_->size(), std::make_shared<Stage>(*this, stages_->size()));
    return it.first;

}

mv::Control::StageIterator mv::ControlModel::getStage(std::size_t stageIdx)
{
    return stages_->find(stageIdx);
}

void mv::ControlModel::removeStage(Control::StageIterator stage)
{

    if (!isValid(stage))
        throw ArgumentError(*this, "stage", "invalid", "Invalid stage iterator passed for stage deletion");
    
    stage->clear();
    stages_->erase(stage->getIdx());

}

void mv::ControlModel::addToStage(Control::StageIterator stage, Control::OpListIterator op)
{

    if (!isValid(stage))
        throw ArgumentError(*this, "stage", "invalid", "Invalid stage iterator passed during appending an op to a stage");

    if (!isValid(op))
        throw ArgumentError(*this, "op", "invalid", "Invalid op iterator passed during appending an op to a stage");

    stage->include(op);

}

void mv::ControlModel::addToStage(Control::StageIterator stage, Data::OpListIterator op)
{
    addToStage(stage, switchContext(op));
}

void mv::ControlModel::removeFromStage(Control::OpListIterator op)
{

    if (!isValid(op))
        throw ArgumentError(*this, "stage", "invalid", "Invalid op iterator passed during removing an op from a stage");

    if (!op->hasAttr("stage"))
        throw ArgumentError(*this, "op", "invalid", "Attempt of removing an unassigned op from a stage");

    auto stage = getStage(op->get<std::size_t>("stage"));
    stage->exclude(op);

}

std::size_t mv::ControlModel::stageSize() const
{
    return stages_->size();
}

mv::Control::StageIterator mv::ControlModel::stageBegin()
{
    return stages_->begin();
}

mv::Control::StageIterator mv::ControlModel::stageEnd()
{
    return stages_->end();
}

mv::Control::FlowListIterator mv::ControlModel::defineFlow(Control::OpListIterator sourceOp, Control::OpListIterator sinkOp)
{

    if (!isValid(sourceOp))
        return flowEnd();

    if (!isValid(sinkOp))
        return flowEnd();

    Control::FlowListIterator flow = controlGraph_.edge_insert(sourceOp, sinkOp, ControlFlow(*this, sourceOp, sinkOp));

    if (flow != *controlFlowEnd_)
    {
        controlFlows_->emplace(flow->getName(), flow);
        log(Logger::MessageType::Info, "Defined " + flow->toString());
        return flow;
    }
    else
    {
        log(Logger::MessageType::Error, "Unable to define new control flow between " + 
            sourceOp->getName() + " and " + sinkOp->getName());
    }

    return flowEnd();

} 

mv::Control::FlowListIterator mv::ControlModel::defineFlow(Data::OpListIterator sourceOp, Data::OpListIterator sinkOp)
{
   return defineFlow(switchContext(sourceOp), switchContext(sinkOp));
}

std::vector<mv::Control::OpListIterator> mv::ControlModel::topologicalSort()
{
    // Necessary for correct iterator casting
    auto topologicalSortResult = mv::topologicalSort(controlGraph_);
    std::vector<mv::Control::OpListIterator> toReturn(topologicalSortResult.begin(), topologicalSortResult.end());
    return toReturn;
}

std::vector<mv::Control::OpListIterator> mv::ControlModel::schedulingSort()
{
    // Necessary for correct iterator casting
    auto schedulingSortResult = mv::schedulingSort(controlGraph_, getFirst());
    std::vector<mv::Control::OpListIterator> toReturn(schedulingSortResult.begin(), schedulingSortResult.end());
    return toReturn;
}

struct OpItComparator
{
    bool operator()(mv::graph<mv::Op, mv::ControlFlow>::node_list_iterator lhs, mv::graph<mv::Op, mv::ControlFlow>::node_list_iterator rhs) const
    {
        return (*lhs) < (*rhs);
    }
};

struct EdgeItComparator
{
    bool operator()(mv::graph<mv::Op, mv::ControlFlow>::edge_list_iterator lhs, mv::graph<mv::Op, mv::ControlFlow>::edge_list_iterator rhs) const
    {
        return (*lhs) < (*rhs);
    }
};

std::vector<mv::Control::FlowListIterator> mv::ControlModel::criticalPath(Control::OpListIterator sourceOp, Control::OpListIterator sinkOp, const std::string& nodeAttribute, const std::string& edgeAttribute)
{
    std::map<mv::graph<mv::Op, mv::ControlFlow>::node_list_iterator, unsigned, OpItComparator> nodeCosts;
    std::map<mv::graph<mv::Op, mv::ControlFlow>::edge_list_iterator, unsigned, EdgeItComparator> edgeCosts;

    if(nodeAttribute != "")
        for(auto opIt = opBegin(); opIt != opEnd(); ++opIt)
            if(opIt->hasAttr(nodeAttribute))
                nodeCosts[opIt] = opIt->get<unsigned>(nodeAttribute);

    if(edgeAttribute != "")
        for(auto edgeIt = flowBegin(); edgeIt != flowEnd(); ++edgeIt)
            if(edgeIt->hasAttr(edgeAttribute))
                edgeCosts[edgeIt] = edgeIt->get<unsigned>(edgeAttribute);

    auto toReturnToBeCasted = mv::critical_path<Op, ControlFlow, OpItComparator, EdgeItComparator>(controlGraph_, sourceOp, sinkOp, nodeCosts, edgeCosts);
    std::vector<mv::Control::FlowListIterator> toReturn(toReturnToBeCasted.begin(), toReturnToBeCasted.end());
    return toReturn;
}

std::vector<mv::Control::FlowListIterator> mv::ControlModel::criticalPath(Data::OpListIterator sourceOp, Data::OpListIterator sinkOp, const std::string& nodeAttribute, const std::string& edgeAttribute)
{
   return criticalPath(switchContext(sourceOp), switchContext(sinkOp), nodeAttribute, edgeAttribute);
}

void mv::ControlModel::transitiveReduction(const std::string& edgeAttribute)
{
    std::set<mv::graph<mv::Op, mv::ControlFlow>::edge_list_iterator, EdgeItComparator> toSave;
    if(edgeAttribute != "")
        for(auto edgeIt = flowBegin(); edgeIt != flowEnd(); ++edgeIt)
            if(edgeIt->hasAttr(edgeAttribute))
                toSave.insert(edgeIt);
    mv::transitiveReduction<Op, ControlFlow, EdgeItComparator, OpItComparator>(controlGraph_, toSave);
}

bool mv::ControlModel::isDag()
{
    return mv::isDAG(controlGraph_);
}

void mv::ControlModel::undefineFlow(Control::FlowListIterator flow)
{

    if (!ComputationModel::isValid(flow))
        throw ArgumentError(*this, "flow", "invalid", "An invalid flow iterator passed for flow deletion");

    controlFlows_->erase(flow->getName());
    controlGraph_.edge_erase(flow);

}

std::string mv::ControlModel::getLogID() const
{
    return "ControlModel:" + name_;
}

bool mv::ControlModel::checkControlFlow(mv::Control::OpListIterator source, mv::Control::OpListIterator sink)
{
    // Extra check to verify we are not adding a self-edge
    if(source == sink)
        return true;

    // Extra check to enforce constraint: the target of a control flow cannot be non executable
    if((!sink->hasTypeTrait("executable")) && (sink->getOpType() != "Output"))
        return true;

    bool found = false;
    for(auto childOp = source.leftmostChild(); childOp != opEnd(); ++childOp)
    {
        if(childOp == sink)
        {
            found = true;
            break;
        }
    }

    return found;
}

bool mv::ControlModel::pathExists(Control::OpListIterator source, Control::OpListIterator target)
{
    return mv::pathExists(controlGraph_, source, target);
}



bool mv::ControlModel::checkControlFlow(mv::Data::OpListIterator source, mv::Data::OpListIterator sink)
{
    return checkControlFlow(switchContext(source), switchContext(sink));
}


