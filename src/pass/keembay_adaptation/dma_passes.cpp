#include "include/mcm/pass/pass_registry.hpp"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/utils/custom_strings.hpp"
#include "include/mcm/utils/warning_manager.hpp"

static void AddDPUTasksWeightsDMATasksFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element& passDesc, mv::Element&);
static void AddUPATasksExtraInputsDMATasksFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element& passDesc, mv::Element&);
static void addFinalDMATaskFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);
static void ensureSplitStrategiesForSpilling(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);
static std::vector<mv::Data::OpListIterator> findSinkLayers(mv::DataModel &dataModel, const mv::Data::TensorIterator& tensor);

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


        MV_REGISTER_PASS(EnsureSplitStrategiesForSpilling)
            .setFunc(ensureSplitStrategiesForSpilling)
            .setDescription(
               "Ensures Split Strategies still valid after Spilling cases");
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
void AddDPUTasksWeightsDMATasksFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor& target, mv::Element& passDesc, mv::Element &)
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
                auto inputOp = om.getSourceOp(inputTensor);
                if(!isTensorInNNCMX(inputTensor, om))
                {
                    auto flows = inputTensor->get<std::set<std::string>>("flows");

                    mv::Data::TensorIterator inputTensorDma = om.dMATask(inputTensor, mv::DmaDirectionEnum::DDR2NNCMX, mv::createDMATaskDDR2NNCMXName(inputOp->getName()));
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
void AddUPATasksExtraInputsDMATasksFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor& target, mv::Element& passDesc, mv::Element &)
{

    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
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
                auto inputOp = om.getSourceOp(inputTensor);
                if(isTensorInNNCMX(inputTensor, om) || isTensorInUPACMX(inputTensor, om))
                {
                    auto flows = inputTensor->get<std::set<std::string>>("flows");

                    mv::Data::TensorIterator inputTensorDma;
                    if(isTensorInNNCMX(inputTensor, om))
                        inputTensorDma = om.dMATask(inputTensor, mv::DmaDirectionEnum::NNCMX2DDR, mv::createDMATaskNNCMX2DDRName(inputOp->getName()));
                    else if (isTensorInUPACMX(inputTensor, om))
                        inputTensorDma = om.dMATask(inputTensor, mv::DmaDirectionEnum::UPACMX2DDR, mv::createDMATaskUPACMX2DDRName(inputOp->getName()));
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

// Pass role: Splitting Strategies propagation algorithm may create an incompatibility
void ensureSplitStrategiesForSpilling(const mv::pass::PassEntry& , mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{

    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    mv::OpModel om(model);
    mv::DataModel dm(model);
    std::vector<std::pair<std::string, std::string>>incompatibleStrategies =
    {
        {"SplitOverHOverlapped", "Clustering"},
        {"SplitOverHOverlapped", "SplitOverK"},
        {"SplitOverH", "Clustering"},
        {"SplitOverH", "SplitOverK"},
        {"SplitOverK", "SplitOverH"},
        {"Clustering", "SplitOverH"},
        {"SplitOverK", "HKSwitch"},
        {"Clustering", "HKSwitch"}
    };
    auto globalParams = model.getGlobalConfigParams();
    unsigned numClusters = globalParams->get<int>("Number_of_Clusters");

    if (numClusters > 1)
    {
        for(auto opIt = om.opBegin(); opIt != om.opEnd(); ++opIt)
        {
            std::string opType = opIt->getOpType();
            if (opType == "DMATask")
            {
                if (opIt->get<mv::DmaDirection>("direction") == mv::DmaDirectionEnum::DDR2NNCMX &&
                    !opIt->getOutputTensor(0)->isPopulated())
                {
                    std::vector<mv::Data::OpListIterator> sinkOperators = findSinkLayers(dm, opIt->getOutputTensor(0));
                    auto opStrategy = sinkOperators[0]->get<std::string>("splitStrategy");
                    for (auto restrictedCombination:incompatibleStrategies)
                    {
                        std::pair<std::string, std::string> possibleCombination(opIt->getOutputTensor(0)->get<std::string>("splitStrategy"), opStrategy);
                        if (possibleCombination == restrictedCombination)
                        {
                            opIt->getOutputTensor(0)->set<std::string>("splitStrategy", opStrategy);
                            opIt->getInputTensor(0)->set<std::string>("splitStrategy", opStrategy);
                        }
                    }
                }
            }
        }
    }
    return;

}

static std::vector<mv::Data::OpListIterator> findSinkLayers(mv::DataModel &dataModel, const mv::Data::TensorIterator &tensor)
{
    std::vector<mv::Data::OpListIterator> sinkOperations;
    auto flowsNames = (tensor)->get<std::set<std::string>>("flows");
    for(auto flowName : flowsNames)
    {
        auto df = dataModel.getDataFlow(flowName);
        sinkOperations.push_back(df.sink());
    }
    return sinkOperations;
}
