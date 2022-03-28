//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/pass/pass_utils.hpp"
#include "include/mcm/pass/pass_dtype_utils.hpp"
#include "mcm/utils/custom_math.hpp"

static void kmbQuantizeConversionFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);
static void configureIOPrecisionFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);
static void deQuantizeU8ConstToFP16ConstFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);

namespace mv
{

    namespace pass
    {

        MV_REGISTER_PASS(DeQuantizeU8ConstToFP16Const)
        .setFunc(deQuantizeU8ConstToFP16ConstFcn)
        .setDescription(
        "This pass de-quantize U8 ConstantInt ops to FP16 ConstantInt when the following op is UPATask."
        );

        MV_REGISTER_PASS(KMBQuantizeConversion)
        .setFunc(kmbQuantizeConversionFcn)
        .setDescription(
            "This pass inserts Quantize conversion layers between DPUTask-to-UPATask transitions (& vice-versa)."
        );

        MV_REGISTER_PASS(ConfigureIOPrecision)
        .setFunc(configureIOPrecisionFcn)
        .setDescription(
            "This pass inserts Conversion layers in order to guarantee the appropriate input / output precision."
        );
    }

}

void addQuantizationLayers(mv::OpModel & om, std::vector<mv::Data::OpListIterator>& tasks, const mv::DType& dtypeNeededInInput)
{
    for(auto& task : tasks)
    {
        if (task->hasAttr("taskOp"))
        {
            auto taskOp = task->get<std::string>("taskOp");
            if (taskOp == "Quantize" ||
                taskOp == "Conversion")
            {
                // Skip inserting Quantization operation for exisitng Quantization tasks and
                // Conversion operations which can handle quantization on their own
                continue;
            }

            if (task->hasAttr("forceU8") && task->get<bool>("forceU8") && dtypeNeededInInput == mv::DType("Float16"))
                continue;
        }

        auto inputFlow = task.leftmostInput();
        auto outputDType = task->getOutputTensor(0)->getDType();
        std::size_t id = 0;
        while(inputFlow != om.flowEnd())
        {
            auto tensor = inputFlow->getTensor();
            auto tensorDType = tensor->getDType();
            auto tensorQuantParams = tensor->getQuantParams();
                
            /// For upa gather op, index input need stay Int32 according to Runtime
            if(task->hasAttr("taskOp") && task->get<std::string>("taskOp") == "Gather" 
                    && tensorDType.toString() == "Int32"){
                ++inputFlow;
                continue;
            }
            
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

                // avoid to add redundant Quantize
                mv::Data::TensorIterator quantize;
                mv::DataModel dm(om);
                bool findExistQuantize = false;
                auto childOps = mv::findSinkLayers(dm, tensor);
                for(auto& op: childOps)
                {
                    if((op->getOpType() == "UPATask") && (op->get<std::string>("taskOp") == "Quantize"))
                    {
                        quantize = op->getOutputTensor(0);
                        findExistQuantize = true;
                        break;
                    }
                }

                if(!findExistQuantize)
                {
                    quantize = om.uPATaskQuantize("Quantize" + task->getName() + std::to_string(id), {tensor});
                    quantize->setDType(dtypeNeededInInput);
                    quantize->setQuantParams(tensorQuantParams);
                }

                auto quantOp = om.getSourceOp(quantize);
                auto sourceOp = om.getSourceOp(tensor);

                if (tensor->hasAttr("splitStrategy"))
                    quantize->set<std::string>("splitStrategy", tensor->get<std::string>("splitStrategy"));
                else if (sourceOp->hasAttr("splitStrategy"))
                    quantOp->set<std::string>("splitStrategy", sourceOp->get<std::string>("splitStrategy"));
                quantOp->set<unsigned>("opId", task->get<unsigned>("opId"));

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
        }

        auto previousOpIt = om.getSourceOp(slice->getInputTensor(0));
        for (auto sinkFlow = previousOpIt.leftmostOutput(); sinkFlow != om.flowEnd(); ++sinkFlow)
        {
            if (sinkFlow.sink()->getName() == slice->getName())
            {
                sliceFlows[slice->getInputTensor()[0]->getName()].push_back(sinkFlow);
                break;
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

                    auto quantOp = om.getSourceOp(quantize);
                    auto sourceOp = om.getSourceOp(tensor);

                    if (tensor->hasAttr("splitStrategy"))
                        quantize->set<std::string>("splitStrategy", tensor->get<std::string>("splitStrategy"));
                    else if (sourceOp->hasAttr("splitStrategy"))
                        quantOp->set<std::string>("splitStrategy", sourceOp->get<std::string>("splitStrategy"));

                    quantOp->set<unsigned>("opId", slice->get<unsigned>("opId"));
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

void addMultiOutputQuantizationLayers(mv::OpModel & om, const mv::pass::PassEntry& pass) {
    auto outputOps = om.getOps("ImplicitOutput");
    if (outputOps.size() < 2)
        return;
    for (auto& outputOp : outputOps) {
        auto parentOp = om.getSourceOp(outputOp->getInputTensor(0));
        if (!(parentOp->hasAttr("taskOp") &&
              parentOp->get<std::string>("taskOp") == "Eltwise" &&
              parentOp->hasAttr("softwareExecuted") &&
              parentOp->get<bool>("softwareExecuted") &&
              outputOp->hasAttr("precision") && outputOp->get<mv::DType>("precision") == mv::DType("Float16")))
            // if output precision is not Float16, a conversion op will be added, no need for quantization.
            continue;
        unsigned outputFlowSize = 0;
        mv::Data::FlowListIterator flowToRemove(parentOp.leftmostOutput());
        for(auto outputFlow = parentOp.leftmostOutput(); outputFlow != om.flowEnd(); ++outputFlow) {
            if (outputFlow.sink()->getOpType() == "ImplicitOutput")
                flowToRemove = outputFlow;
            ++outputFlowSize;
        }
        if (outputFlowSize < 2)
            continue;

        pass.log(mv::Logger::MessageType::Warning, "Handle multiple outputs with quantization for " + outputOp->getName());

        auto quantize = om.uPATaskQuantize("Quantize" + parentOp->getName(), {parentOp->getOutputTensor(0)});
        quantize->setDType(mv::DType("UInt8"));
        quantize->setQuantParams({{128},{2.0 / 255.0},{-1.0},{1.0}});
        auto quantizeOp = om.getSourceOp(quantize);
        quantizeOp->set<unsigned>("opId", parentOp->get<unsigned>("opId"));
        if (parentOp->hasAttr("splitStrategy"))
            quantizeOp->set<std::string>("splitStrategy", parentOp->get<std::string>("splitStrategy"));

        outputOp->setInputTensor(quantize, 0, false);
        om.defineFlow(quantize, outputOp, 0);

        om.undefineFlow(flowToRemove);
    }
}

// Removes pairs of Quantize ops where the initial and final tensors have the same data type
// e.g. [DTYPE1] -> Quantize -> [DTYPE2] -> Quantize -> [DTYPE1]
void cleanRedundantConversions(mv::ComputationModel& model)
{
    mv::OpModel om(model);
    mv::DataModel dm(model);

    const auto upaTasks = om.getOps("UPATask");
    std::unordered_map<std::string, mv::Data::OpListIterator> quantizeOps;
    for (auto& upaTask : upaTasks)
    {
        if (upaTask->get<std::string>("taskOp") == "Quantize")
            quantizeOps.insert({upaTask->getName(), upaTask});
    }

    auto quantizeOpIt = quantizeOps.begin();
    while (quantizeOpIt != quantizeOps.end())
    {
        const auto quantizeOp  = quantizeOpIt->second;
        const auto inputTensor = quantizeOp->getInputTensor(0);
        const auto consumerOps = mv::findSinkLayers(dm, quantizeOp->getOutputTensor(0));
        if (!std::all_of(consumerOps.cbegin(), consumerOps.cend(), [&inputTensor](const mv::Data::OpListIterator& op) {
                return op->hasAttr("taskOp") && op->get<std::string>("taskOp") == "Quantize" &&
                       op->getOutputTensor(0)->getDType() == inputTensor->getDType();
            }))
        {
            ++quantizeOpIt;
            continue;
        }

        for (auto& consumerOp : consumerOps)
        {
            quantizeOps.erase(consumerOp->getName());
            mv::linkNewOperationsRemove(quantizeOp, quantizeOp->getOutputTensor(0), om, consumerOp);
        }

        quantizeOpIt = quantizeOps.erase(quantizeOpIt);
        mv::linkNewOperationsRemove(om.getSourceOp(inputTensor), inputTensor, om, quantizeOp);
    }
}

static void kmbQuantizeConversionFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor& td, mv::Element&, mv::Element&)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    if (td.getTarget() == mv::Target::ma3720)
        return; //for now anything is allowed

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
                } else if (outputFlow.sink()->getOpType() == "ImplicitOutput") {
                    executable_ops.push_back(outputFlow.sink());
                } else if (outputFlow.sink()->getOpType() != "Slice") {
                    op_itr_bfs.push(outputFlow.sink());
                }
            }
            op_itr_bfs.pop();
        }
        for (auto op : executable_ops)
            if (std::find(dpuTasksFP16Names.begin(), dpuTasksFP16Names.end(),
                    op->getName()) != dpuTasksFP16Names.end() ||
                op->getOpType() == "UPATask" ||
                (op->getOpType() == "Output" && op->getInputTensor(0)->getDType() == mv::DType("Float16")) ||
                (op->getOpType() == "ImplicitOutput" && op->getOutputTensor(0)->getDType() == mv::DType("Float16")))
                slicesFP16.push_back(slice);
    }

    for (auto& sliceFP16 : slicesFP16)
    {
        slices.erase(std::remove(slices.begin(), slices.end(), sliceFP16), slices.end());
        auto outputFlow = sliceFP16.leftmostOutput();
        if(outputFlow.sink()->getOpType() == "UPATask")
        {
            auto outputTensor = sliceFP16->getOutputTensor(0);
            outputTensor->setDType(mv::DType("Float16"));
        }
    }

    auto U8 = mv::DType("UInt8");
    auto FP16 = mv::DType("Float16");

    // handle the multi-output cases where some of the outputs feed into following operations.
    // for SuperResolution enabling
    addMultiOutputQuantizationLayers(om, pass);

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

    cleanRedundantConversions(model);
}

void propagateLocationToParents(mv::OpModel& om, const mv::Data::OpListIterator& opIt, const mv::Tensor::MemoryLocation& location) {
    for (auto& inputTensor : opIt->getInputTensor()) {
        auto parentOp = om.getSourceOp(inputTensor);
        if (inputTensor->hasAttr("noPropagate") && inputTensor->get<bool>("noPropagate"))
            return;
        inputTensor->set<mv::Tensor::MemoryLocation>("Location", location);
        if (parentOp->isImplicit())
            propagateLocationToParents(om, parentOp, location);
    }
}

bool isConversionSupported(const mv::DType& from, const mv::DType& to)
{
    return mv::supportedConversions.find(std::make_pair(from, to)) != mv::supportedConversions.end();
}

static void configureIOPrecisionFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{
    mv::OpModel om(model);
    mv::DataModel dm(model);

    auto requiresQuantize = [](const mv::DType& type1, const mv::DType& type2) {
        return (type1 == mv::DType("Float16") && type2 == mv::DType("UInt8")) ||
               (type1 == mv::DType("UInt8") && type2 == mv::DType("Float16"));
    };

    auto processInput = [&](mv::Data::OpListIterator& inputOp) {
        auto tensor = inputOp->getOutputTensor(0);
        const auto quantParams = tensor->getQuantParams();
        const auto inputPrecision  = inputOp->get<mv::DType>("dType");
        const auto targetPrecision = tensor->getDType();
        if (inputPrecision == tensor->getDType())
            return;

        // Save child ops and their flows to link to converted ops
        std::vector<mv::Data::FlowListIterator> flows;
        std::vector<mv::Data::OpListIterator> opsToLink;
        std::vector<std::size_t> inputSlots;
        for (auto sinkFlow = inputOp.leftmostOutput(); sinkFlow != om.flowEnd(); ++sinkFlow)
        {
            flows.push_back(sinkFlow);
            opsToLink.push_back(sinkFlow.sink());
            inputSlots.push_back(sinkFlow->get<std::size_t>("sinkInput"));
        }

        // Check if input is consumed by Conversion op(s) and a direct conversion is supported
        if (std::all_of(opsToLink.cbegin(), opsToLink.cend(), [&opsToLink](const mv::Data::OpListIterator& childOp) {
                return childOp->getOpType() == "Conversion" &&
                       childOp->getOutputTensor(0)->getDType() == opsToLink[0]->getOutputTensor(0)->getDType();
            }))
        {
            const auto childOutputDType = opsToLink[0]->getOutputTensor(0)->getDType();
            if (isConversionSupported(inputPrecision, childOutputDType))
            {
                tensor->setDType(inputPrecision);
                return;
            }
        }

        // Add conversion layer
        tensor->setDType(inputPrecision);
        mv::Data::TensorIterator convertedTensor;
        if (isConversionSupported(inputPrecision, targetPrecision))
            convertedTensor = om.uPATaskConversion(inputOp->getName() + "_conversion", {tensor}, targetPrecision);
        else
            throw std::runtime_error("Unsupported input conversion: " +
                  inputPrecision.toString() + " to " + targetPrecision.toString());

        mv::propagateUpThroughOps(om, dm, inputOp, {}, {{"dType", inputPrecision}});

        // Set quantization parameters
        convertedTensor->setQuantParams(quantParams);
        if (tensor->isFloatingPointType())
            tensor->setQuantParams(mv::QuantizationParams::initial());
        if (convertedTensor->isFloatingPointType())
            convertedTensor->setQuantParams(mv::QuantizationParams::initial());

        // Set operation and tensor specific attributes
        auto conversionOp = om.getSourceOp(convertedTensor);
        if (tensor->hasAttr("scaleShiftFused") && tensor->get<bool>("scaleShiftFused"))
        {
            // When ScaleShift is fused into the first Convolution, FakeQuantize is altered so that the
            // input range is [0-255] and the output range is adapted based on the ScaleShift parameters
            // This neutral input range is meant to be used to quantize the data while the tensor only
            // has the quantization parameters from the output range
            conversionOp->set<double>("scale", 1.0);
            conversionOp->set<int64_t>("bias", 0);
        }
        conversionOp->set<unsigned>("opId", inputOp->get<unsigned>("opId"));
        if (inputOp->hasAttr("splitStrategy"))
            conversionOp->set<std::string>("splitStrategy", inputOp->get<std::string>("splitStrategy"));
        convertedTensor->set<mv::Tensor::MemoryLocation>("Location", mv::Tensor::MemoryLocation::DDR);

        // Undefine previous flows between input op and consumers
        for (auto& flow : flows)
            om.undefineFlow(flow);
        
        // Link converted tensor to consumers
        for (unsigned i = 0; i < opsToLink.size(); ++i)
        {
            opsToLink[i]->setInputTensor(convertedTensor, inputSlots[i], false);
            om.defineFlow(convertedTensor, opsToLink[i], inputSlots[i]);
        }
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
        else if (isConversionSupported(inputPrecision, targetPrecision))
            tensor = om.uPATaskConversion(outputOp->getName() + "_conversion", {inputTensor}, targetPrecision);
        else
            throw std::runtime_error("Unsupported output conversion: " +
                  inputPrecision.toString() + " to " + targetPrecision.toString());

        tensor->setQuantParams(inputTensor->getQuantParams());
        tensor->setDType(targetPrecision);
        if (outputOp->outputSlots() > 0)
            outputOp->getOutputTensor(0)->setDType(targetPrecision);

        auto quantOp = om.getSourceOp(tensor);
        auto sourceOp = om.getSourceOp(inputTensor);

        if (inputTensor->hasAttr("splitStrategy"))
            tensor->set<std::string>("splitStrategy", inputTensor->get<std::string>("splitStrategy"));
        else if (sourceOp->hasAttr("splitStrategy"))
            quantOp->set<std::string>("splitStrategy", sourceOp->get<std::string>("splitStrategy"));

        tensor->set<mv::Tensor::MemoryLocation>("Location", mv::Tensor::MemoryLocation::OUTPUT);
        propagateLocationToParents(om, outputOp, mv::Tensor::MemoryLocation::DDR);

        quantOp->set<unsigned>("opId", outputOp->get<unsigned>("opId") - 1);
        om.undefineFlow(outputOp.leftmostInput());
        outputOp->setInputTensor(tensor, 0, false);
        om.defineFlow(tensor, outputOp, 0);

        // WA: Conversion layer added before output will have its name
        // overwritten in runtime model by the name of the network output. This results in
        // Conversion layer having the same name for input and output tensors.
        const std::string outName =
            outputOp->hasAttr("networkOutputName") ?
            outputOp->get<std::string>("networkOutputName"):
            outputOp->getName();

        inputTensor->setName(tensor->getName());
        tensor->setName(outName);
    };

    auto inputOps = om.getOps("ImplicitInput");
    if (inputOps.empty())
        inputOps = om.getOps("Input");
    for (auto inputOp : inputOps)
        processInput(inputOp);

    auto outputOps = om.getOps("ImplicitOutput");
    if (outputOps.empty())
        outputOps = om.getOps("Output");
    for (auto outputOp : outputOps)
        processOutput(outputOp);
}


// Replace ConstantInt ops of U8 dtype together with its following Quantize ops with new ConstantInt ops of FP16 dtype.
// This is used to enable the topology in super-resolution,
// where the ConstantInt is converted to FP16 by a Quantize and then feeds into a UPATask (e.g. eltwise_add)
static void deQuantizeU8ConstToFP16ConstFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)

    mv::OpModel om(model);
    mv::DataModel dm(model);

    auto u8ConstOps = om.getOps("ConstantInt");
    for (auto& opIt : u8ConstOps){
        if (opIt->outputSlots() != 1)
            continue;
        auto outputTensor = opIt->getOutputTensor(0);
        auto nextOp = mv::findSinkLayers(dm, outputTensor)[0];
        if ((outputTensor->getDType() != mv::DType("UInt8")) || (nextOp->getOpType() != "UPATask") || (!outputTensor->hasAttr("quantParams")))
            continue;

        auto dequantFP16Weights = dequantizeWeightsToFP16(outputTensor, nextOp, om);
        nextOp->setInputTensor(dequantFP16Weights, 1, false);
        om.defineFlow(dequantFP16Weights, nextOp, 1);
        om.removeOp(opIt);
    }
}
