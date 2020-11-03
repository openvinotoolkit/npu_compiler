#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/pass/pass_utils.hpp"


static void kmbQuantizeConversionFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);
static void configureOutputPrecisionFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);

namespace mv
{

    namespace pass
    {

        MV_REGISTER_PASS(KMBQuantizeConversion)
        .setFunc(kmbQuantizeConversionFcn)
        .setDescription(
            "This pass inserts Quantize conversion layers between DPUTask-to-UPATask transitions (& vice-versa)."
        );

        MV_REGISTER_PASS(ConfigureOutputPrecision)
        .setFunc(configureOutputPrecisionFcn)
        .setDescription(
            "This pass inserts Quantize conversion layers in order to guarantee the appropriate precision."
        );
    }

}

void addQuantizationLayers(mv::OpModel & om, std::vector<mv::Data::OpListIterator>& tasks, const mv::DType& dtypeNeededInInput)
{
    for(auto& task : tasks)
    {
        if (task->hasAttr("taskOp") && task->get<std::string>("taskOp") == "Quantize")
            continue;

        auto inputFlow = task.leftmostInput();
        auto outputDType = task->getOutputTensor(0)->getDType();
        std::size_t id = 0;
        while(inputFlow != om.flowEnd())
        {
            auto tensor = inputFlow->getTensor();
            auto tensorDType = tensor->getDType();
            auto tensorQuantParams = tensor->getQuantParams();

            // NOTE: Maybe here a check for mixed precision should be added
            if(!tensor->isPopulated() && tensorDType != dtypeNeededInInput)
            {
                //if the previous Op is "Align" need to place it after the quantize
                auto previousOpIt = om.getSourceOp(tensor);
                bool alignCase = false;
                if (previousOpIt->getOpType() == "Align")
                {
                    tensor = previousOpIt->getInputTensor()[0];
                    alignCase = true;
                }
                auto quantize = om.uPATaskQuantize("Quantize" + task->getName() + std::to_string(id), {tensor});
                quantize->setDType(outputDType);
                quantize->setQuantParams(tensorQuantParams);
                if (tensor->hasAttr("splitStrategy"))
                    quantize->set<std::string>("splitStrategy", tensor->get<std::string>("splitStrategy"));
                auto quantizeOp = om.getSourceOp(quantize);
                quantizeOp->set<unsigned>("opId", task->get<unsigned>("opId"));

                if (alignCase)
                {
                    auto backup = previousOpIt.leftmostInput();
                    auto slot = backup->get<size_t>("sinkInput");
                    ++inputFlow;
                    om.undefineFlow(backup);
                    previousOpIt->setInputTensor(quantize, slot, false);
                    previousOpIt->getOutputTensor(0)->setDType(outputDType);
                    om.defineFlow(quantize, previousOpIt, slot);
                }
                else
                {
                    auto backup = inputFlow;
                    auto slot = backup->get<size_t>("sinkInput");
                    ++inputFlow;
                    om.undefineFlow(backup);
                    task->setInputTensor(quantize, slot, false);
                    om.defineFlow(quantize, task, slot);
                }


                id++;
            }
            else
                ++inputFlow;
        }
    }
}

void addSliceQuantizationLayer(mv::OpModel & om, std::vector<mv::Data::OpListIterator>& slices, const mv::DType& dtypeNeededInInput)
{
    std::vector <mv::Data::TensorIterator> sliceInputs;
    std::map <std::string, std::vector<mv::Data::OpListIterator>> sliceLeafs;
    std::map <std::string, std::vector<mv::Data::FlowListIterator>> sliceFlows;
    std::vector<mv::Data::OpListIterator> slicesToRemove;

    for (auto& slice : slices)
    {
        auto it = std::find (sliceInputs.begin(), sliceInputs.end(), slice->getInputTensor()[0]);
        if (it != sliceInputs.end())
        {
            sliceLeafs[slice->getInputTensor()[0]->getName()].push_back(slice);
            slicesToRemove.push_back(slice);
        }
        else
        {
            sliceInputs.push_back(slice->getInputTensor()[0]);
            auto previousOpIt = om.getSourceOp(slice->getInputTensor(0));
            for (auto sinkFlow = previousOpIt.leftmostOutput(); sinkFlow != om.flowEnd(); ++sinkFlow)
            {
                if (sinkFlow.sink()->getOpType() == "Slice")
                    sliceFlows[slice->getInputTensor()[0]->getName()].push_back(sinkFlow);
            }
        }
    }

    for (auto slice: slicesToRemove)
    {
        slices.erase(std::remove(slices.begin(), slices.end(), slice), slices.end());
    }

    for(auto& slice : slices)
    {
        auto inputFlow = slice.leftmostInput();
        auto outputDType = slice->getOutputTensor(0)->getDType();
        std::size_t id = 0;
        while(inputFlow != om.flowEnd())
        {
            auto tensor = inputFlow->getTensor();
            auto tensorDType = tensor->getDType();

            // NOTE: Maybe here a check for mixed precision should be added
            if(!tensor->isPopulated() && tensorDType != dtypeNeededInInput)
            {
                //before adding UPATask, check if any of the other outputs of the tensor has already been quantized
                auto previousOpIt = om.getSourceOp(tensor);
                mv::Data::TensorIterator quantize;
                bool alreadyQuantized = false;
                for (auto sinkFlow = previousOpIt.leftmostOutput(); sinkFlow != om.flowEnd(); ++sinkFlow)
                {
                    auto task = sinkFlow.sink();
                    if (task->getOpType() == "UPATask" && task->hasAttr("taskOp") && task->get<std::string>("taskOp") == "Quantize")
                    {
                        quantize = task->getOutputTensor()[0];
                        alreadyQuantized = true;
                        break;
                    }

                }

                if (!alreadyQuantized)
                {
                    auto quantParams = tensor->getQuantParams();
                    quantize = om.uPATaskQuantize("Quantize" + slice->getName() + std::to_string(id), {tensor});
                    quantize->setDType(outputDType);
                    quantize->setQuantParams(quantParams);
                    if (tensor->hasAttr("splitStrategy"))
                        quantize->set<std::string>("splitStrategy", tensor->get<std::string>("splitStrategy"));
                    auto quantizeOp = om.getSourceOp(quantize);
                    quantizeOp->set<unsigned>("opId", slice->get<unsigned>("opId"));
                }
                auto backup = inputFlow;
                auto slot = backup->get<size_t>("sinkInput");
                ++inputFlow;
                for (auto flow:sliceFlows[tensor->getName()])
                {
                    om.undefineFlow(flow);
                }
                slice->setInputTensor(quantize, slot, false);
                om.defineFlow(quantize, slice, slot);
                for (auto op:sliceLeafs[tensor->getName()])
                {
                    op->setInputTensor(quantize, slot, false);
                    om.defineFlow(quantize, op, slot);
                }
                id++;
            }
            else
                ++inputFlow;
        }
    }
}

static void kmbQuantizeConversionFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)

    mv::OpModel om(model);
    mv::DataModel dm(model);

    auto dpuTasks = om.getOps("DPUTask");
    std::vector<mv::Data::OpListIterator> dpuTasksFP16;
    std::vector<std::string> dpuTasksFP16Names;
    for (auto& dpuTask : dpuTasks)
    {
        if (dpuTask->hasAttr("floatPrecision") && dpuTask->get<bool>("floatPrecision"))
        {
            dpuTasksFP16.push_back(dpuTask);
            dpuTasksFP16Names.push_back(dpuTask->getName());
        }
    }

    for (auto& dpuTaskFP16 : dpuTasksFP16)
        dpuTasks.erase(std::remove(dpuTasks.begin(), dpuTasks.end(), dpuTaskFP16), dpuTasks.end());

    auto upaTasks = om.getOps("UPATask");

    // NOTE: At this moment in the model, all the concats are implicit
    auto implicitConcats = om.getOps("ImplicitConcat");
    auto implicitJoins = om.getOps("ImplicitJoin");
    // NOTE: For now only operations with U8/DPU Tasks are streamed
    auto slices = om.getOps("Slice");
    std::vector<mv::Data::OpListIterator> slicesFP16 = {};
    for (auto& slice: slices)
    {
        std::vector<mv::Data::OpListIterator> executable_ops;
        std::queue<mv::Data::OpListIterator> op_itr_bfs;
        op_itr_bfs.push(slice);
        // BFS the non-executable subtree to find other slice and executable leafs
        while (!op_itr_bfs.empty()) {
            auto current_op_itr = op_itr_bfs.front();
            for(auto outputFlow = current_op_itr.leftmostOutput();
                outputFlow != om.flowEnd(); ++outputFlow) {
                if (outputFlow.sink()->hasTypeTrait("executable")) {
                    executable_ops.push_back(outputFlow.sink());
                } else if (outputFlow.sink()->getOpType() != "Slice") {
                    op_itr_bfs.push(outputFlow.sink());
                }
            }
            op_itr_bfs.pop();
        }
        for (auto op : executable_ops)
            if (std::find(dpuTasksFP16Names.begin(), dpuTasksFP16Names.end(),
                    op->getName()) != dpuTasksFP16Names.end())
                slicesFP16.push_back(slice);
    }

    for (auto& sliceFP16 : slicesFP16)
        slices.erase(std::remove(slices.begin(), slices.end(), sliceFP16), slices.end());

    auto U8 = mv::DType("UInt8");
    auto FP16 = mv::DType("Float16");

    addQuantizationLayers(om, upaTasks, FP16);
    addQuantizationLayers(om, dpuTasksFP16, FP16);
    addQuantizationLayers(om, dpuTasks, U8);
    addSliceQuantizationLayer(om, slices, U8);
    addSliceQuantizationLayer(om, slicesFP16, FP16);
    // NOTE: Concat have the extra requirement that output tensor and input tensor have to match their DType, so
    // we split them in two vectors
    std::vector<mv::Data::OpListIterator> implicitConcatsU8;
    std::vector<mv::Data::OpListIterator> implicitConcatsFP16;

    for(auto& implicitConcat: implicitConcats)
    {
        auto outputDType = implicitConcat->getOutputTensor(0)->getDType();
        if(outputDType == U8)
            implicitConcatsU8.push_back(implicitConcat);
        else if(outputDType == FP16)
            implicitConcatsFP16.push_back(implicitConcat);
    }

    addQuantizationLayers(om, implicitConcatsU8, U8);
    addQuantizationLayers(om, implicitConcatsFP16, FP16);

    std::vector<mv::Data::OpListIterator> implicitJoinU8;
    std::vector<mv::Data::OpListIterator> implicitJoinFP16;

    for(auto& implicitJoin: implicitJoins)
    {
        auto outputDType = implicitJoin->getOutputTensor(0)->getDType();
        if(outputDType == U8)
            implicitJoinU8.push_back(implicitJoin);
        else if(outputDType == FP16)
            implicitJoinFP16.push_back(implicitJoin);
    }

    addQuantizationLayers(om, implicitJoinU8, U8);
    addQuantizationLayers(om, implicitJoinFP16, FP16);
}

static void configureOutputPrecisionFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{
    mv::OpModel om(model);

    auto multipleOutputs = [&om](const mv::Data::OpListIterator& outputOp) {
        return om.getSourceOp(outputOp->getInputTensor(0))->getOpType() == "ImplicitUnion";
    };

    auto requiresQuantize = [](const mv::DType& type1, const mv::DType& type2) {
        return (type1 == mv::DType("Float16") && type2 == mv::DType("UInt8")) ||
               (type1 == mv::DType("UInt8") && type2 == mv::DType("Float16"));
    };

    auto supportedConversion = [](const mv::DType& type1, const mv::DType& type2) {
        return (type1 == mv::DType("Float16") && type2 == mv::DType("Float32")) ||
               (type1 == mv::DType("Float32") && type2 == mv::DType("Float16")) ||
               (type1 == mv::DType("Float16") && type2 == mv::DType("Int32")) ||
               (type1 == mv::DType("Int32") && type2 == mv::DType("Float16")) ||
               (type1 == mv::DType("UInt8") && type2 == mv::DType("Float32")) ||
               (type1 == mv::DType("Float32") && type2 == mv::DType("UInt8")) ||
               (type1 == mv::DType("Int32") && type2 == mv::DType("UInt8"));
    };

    auto processOutput = [&](mv::Data::OpListIterator& outputOp) {
        auto inputTensor = outputOp->getInputTensor(0);

        auto inputPrecision  = inputTensor->getDType();
        auto targetPrecision = outputOp->get<mv::DType>("precision");
        if (targetPrecision == mv::DType("Default") || targetPrecision == inputPrecision)
            return;

        mv::Data::TensorIterator tensor;
        if (requiresQuantize(inputPrecision, targetPrecision))
            tensor = om.uPATaskQuantize(outputOp->getName() + "_quantize", {inputTensor});
        else if (supportedConversion(inputPrecision, targetPrecision))
            tensor = om.uPATaskConversion(outputOp->getName() + "_conversion", {inputTensor}, targetPrecision);
        else
            throw std::runtime_error("Unsupported output conversion: " +
                  inputPrecision.toString() + " to " + targetPrecision.toString());

        tensor->setQuantParams(inputTensor->getQuantParams());
        tensor->setDType(targetPrecision);
        if (outputOp->outputSlots() > 0)
            outputOp->getOutputTensor(0)->setDType(targetPrecision);

        if (inputTensor->hasAttr("splitStrategy"))
            tensor->set<std::string>("splitStrategy", inputTensor->get<std::string>("splitStrategy"));
        tensor->set<mv::Tensor::MemoryLocation>("Location", mv::Tensor::MemoryLocation::OUTPUT);
        inputTensor->set<mv::Tensor::MemoryLocation>("Location", mv::Tensor::MemoryLocation::DDR);
        while (om.getSourceOp(inputTensor)->isImplicit())
        {
            inputTensor = om.getSourceOp(inputTensor)->getInputTensor(0);
            inputTensor->set<mv::Tensor::MemoryLocation>("Location", mv::Tensor::MemoryLocation::DDR);
        }

        om.getSourceOp(tensor)->set<unsigned>("opId", outputOp->get<unsigned>("opId") - 1);
        om.undefineFlow(outputOp.leftmostInput());
        outputOp->setInputTensor(tensor, 0, false);
        om.defineFlow(tensor, outputOp, 0);
    };

    for (auto outputOp : om.getOps("Output"))
    {
        if (multipleOutputs(outputOp))
        {
            auto implicitUnionOp = om.getSourceOp(outputOp->getInputTensor(0));
            for (auto& implicitOutput : implicitUnionOp->getInputTensor())
            {
                auto implicitOutputOp = om.getSourceOp(implicitOutput);
                processOutput(implicitOutputOp);
            }
        }
        else
        {
            processOutput(outputOp);
        }
    }
}
