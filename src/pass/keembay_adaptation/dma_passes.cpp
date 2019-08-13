#include "include/mcm/pass/pass_registry.hpp"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/utils/custom_strings.hpp"
#include "include/mcm/utils/warning_manager.hpp"


static void AddDPUTasksWeightsDMATasksFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element& passDesc, mv::json::Object&);
static void AddUPATasksExtraInputsDMATasksFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element& passDesc, mv::json::Object&);

namespace mv
{
    namespace pass
    {
        MV_REGISTER_PASS(AddDPUTasksWeightsDMATasks)
            .setFunc(AddDPUTasksWeightsDMATasksFcn)
            .setDescription(
               "Add DMA Tasks for DPU Tasks weights");

        MV_REGISTER_PASS(AddUPATasksExtraInputsDMATasks)
            .setFunc(AddUPATasksExtraInputsDMATasksFcn)
            .setDescription(
               "Add DMA Tasks for UPA Tasks extra inputs");
    }
}

// NOTE: This is not checked using allocators for the simple reason that they are not assigned
// to tensors yet.
bool isTensorInNNCMX(mv::Data::TensorIterator tensor, mv::BaseOpModel& opModel)
{
    auto sourceOp = opModel.getSourceOp(tensor);
    std::string opType(sourceOp->getOpType());
    if(opType == "DPUTask")
        return true;
    else if(opType == "DMATask")
    {
        auto direction = sourceOp->get<mv::DmaDirection>("direction");
        if(direction == mv::DmaDirectionEnum::DDR2NNCMX)
            return true;
        else if(direction == mv::DmaDirectionEnum::UPACMX2NNCMX)
            return true;
    }
    return false;
}

bool isTensorInUPACMX(mv::Data::TensorIterator tensor, mv::BaseOpModel& opModel)
{
    auto sourceOp = opModel.getSourceOp(tensor);
    std::string opType(sourceOp->getOpType());
    if(opType == "DPUTask")
        return true;
    else if(opType == "DMATask")
    {
        auto direction = sourceOp->get<mv::DmaDirection>("direction");
        if(direction == mv::DmaDirectionEnum::DDR2UPACMX)
            return true;
        else if(direction == mv::DmaDirectionEnum::NNCMX2UPACMX)
            return true;
    }
    return false;
}

// Pass role: Add DMA Tasks for weights tensors input of DPUTasks (if needed).
void AddDPUTasksWeightsDMATasksFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor& target, mv::Element& passDesc, mv::json::Object&)
{
    UNUSED(pass);
    UNUSED(target);
    mv::OpModel om(model);
    mv::DataModel dm(model);

    for(auto opIt = om.opBegin(); opIt != om.opEnd(); ++opIt)
    {
        std::string opType = opIt->getOpType();
        if (opType == "DPUTask")
        {
            auto opId = opIt->get<unsigned>("opId");
            unsigned n = opIt->inputSlots();
            for(unsigned i = 0; i < n; ++i)
            {
                auto inputTensor = opIt->getInputTensor(i);
                mv::QuantizationParams quantParams = {{},{},{},{}};
                if(inputTensor->hasAttr("quantParams"))
                    quantParams = inputTensor->get<mv::QuantizationParams>("quantParams");
                auto inputOp = om.getSourceOp(inputTensor);
                if(!isTensorInNNCMX(inputTensor, om))
                {
                    auto flows = inputTensor->get<std::set<std::string>>("flows");

                    mv::Data::TensorIterator inputTensorDma = om.dMATask(inputTensor, mv::DmaDirectionEnum::DDR2NNCMX, quantParams, mv::createDMATaskDDR2NNCMXName(inputOp->getName()));
                    auto inputTensorDmaOp = om.getSourceOp(inputTensorDma);
                    inputTensorDmaOp->set<unsigned>("opId", opId);

                    for(auto flowStr: flows)
                    {
                        auto backupFlow = dm.getDataFlow(flowStr);
                        auto idx = backupFlow->get<std::size_t>("sinkInput");
                        auto sink = backupFlow.sink();
                        om.undefineFlow(backupFlow);
                        sink->setInputTensor(inputTensorDma, idx, false);
                        om.defineFlow(inputTensorDmaOp, 0, sink, idx);
                    }
                }
            }
        }
    }
}

// Pass role: Add DMA Tasks for input tensors input of UPATasks (if needed).
void AddUPATasksExtraInputsDMATasksFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor& target, mv::Element& passDesc, mv::json::Object&)
{
    UNUSED(pass);
    UNUSED(target);
    mv::OpModel om(model);
    mv::DataModel dm(model);

    for(auto opIt = om.opBegin(); opIt != om.opEnd(); ++opIt)
    {
        std::string opType = opIt->getOpType();
        if (opType == "UPATask")
        {
            std::string taskOp = opIt->get<std::string>("taskOp");
            if(taskOp == "Dummy")
                continue;
            auto opId = opIt->get<unsigned>("opId");
            unsigned n = opIt->inputSlots();
            for(unsigned i = 0; i < n; ++i)
            {
                auto inputTensor = opIt->getInputTensor(i);
                mv::QuantizationParams quantParams = {{},{},{},{}};
                if(inputTensor->hasAttr("quantParams"))
                    quantParams = inputTensor->get<mv::QuantizationParams>("quantParams");
                auto inputOp = om.getSourceOp(inputTensor);
                if(!isTensorInUPACMX(inputTensor, om))
                {
                    auto flows = inputTensor->get<std::set<std::string>>("flows");

                    mv::Data::TensorIterator inputTensorDma = om.dMATask(inputTensor, mv::DmaDirectionEnum::DDR2UPACMX, quantParams, mv::createDMATaskDDR2UPACMXName(inputOp->getName()));
                    auto inputTensorDmaOp = om.getSourceOp(inputTensorDma);
                    inputTensorDmaOp->set<unsigned>("opId", opId);

                    for(auto flowStr: flows)
                    {
                        auto backupFlow = dm.getDataFlow(flowStr);
                        auto idx = backupFlow->get<std::size_t>("sinkInput");
                        auto sink = backupFlow.sink();
                        om.undefineFlow(backupFlow);
                        sink->setInputTensor(inputTensorDma, idx, false);
                        om.defineFlow(inputTensorDmaOp, 0, sink, idx);
                    }
                }
            }
        }
    }
}
