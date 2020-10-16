#include "include/mcm/pass/pass_utils.hpp"

std::vector<std::pair<mv::Data::OpListIterator,size_t>> mv::getOutputDataFlow(mv::OpModel& om, mv::Data::OpListIterator &opIt, bool deleteOp)
{
    std::vector<std::pair<mv::Data::OpListIterator,size_t>> toReturn;
    auto outputTensor = opIt->getOutputTensor()[0];

    for(auto output = opIt.leftmostOutput(); output != om.flowEnd(); ++output)
    {
        auto consumer = output.sink();
        std::size_t slot = 0;
        for (std::size_t input_idx = 0; input_idx < consumer->getInputTensor().size(); input_idx++)
            if (consumer->getInputTensor()[input_idx]->getName() == outputTensor->getName())
                slot = input_idx;
        toReturn.push_back(std::make_pair(consumer, slot));
    }

    if(deleteOp)
    {
        auto backup = opIt;
        ++opIt;
        om.removeOp(backup);
    }

    return toReturn;
}

void mv::setOutputDataFlow(mv::OpModel& om, mv::Data::TensorIterator &dpuTaskOutputTensor, const std::vector<std::pair<mv::Data::OpListIterator,size_t>>& outDataFlows)
{
    for(auto& flowPair: outDataFlows)
    {
        flowPair.first->setInputTensor(dpuTaskOutputTensor, flowPair.second, false);
        om.defineFlow(dpuTaskOutputTensor, flowPair.first, flowPair.second);
    }
}

std::vector<mv::Control::OpListIterator> mv::getOutputControlFlow(mv::ControlModel& cm, mv::Control::OpListIterator opIt)
{
    std::vector<mv::Control::OpListIterator> toReturn;

    for(auto outputFlow = opIt.leftmostChild(); outputFlow != cm.opEnd(); ++outputFlow)
        toReturn.push_back(outputFlow);
    return toReturn;
}

std::vector<mv::Control::OpListIterator> mv::getInputControlFlow(mv::ControlModel& cm, mv::Control::OpListIterator opIt)
{
    std::vector<mv::Control::OpListIterator> toReturn;

    for(auto outputFlow = opIt.leftmostParent(); outputFlow != cm.opEnd(); ++outputFlow)
        toReturn.push_back(outputFlow);
    return toReturn;
}

void mv::setInputControlFlow(mv::ControlModel& cm, mv::Control::OpListIterator op, const std::vector<mv::Control::OpListIterator>& inputControlFlows)
{
    for(auto& inputOp: inputControlFlows)
        cm.defineFlow(inputOp, op);
}

void mv::setOutputControlFlow(mv::ControlModel& cm, mv::Control::OpListIterator op, const std::vector<mv::Control::OpListIterator>& outputControlFlows)
{
    for(auto& outputOp: outputControlFlows)
        cm.defineFlow(op, outputOp);
}

mv::Data::OpListIterator mv::linkNewOperationsRemove(mv::Data::OpListIterator parentOpIt,
                                                 mv::Data::TensorIterator sourceTensor, mv::OpModel & om, mv::Data::OpListIterator opIt)
{
    //Important: do not change the order of this ops
    std::vector<mv::Data::OpListIterator> opsToLink;
    std::vector<std::size_t> inputSlots;
    for (auto sinkFlow = opIt.leftmostOutput(); sinkFlow != om.flowEnd(); ++sinkFlow)
    {
        opsToLink.push_back(sinkFlow.sink());
        inputSlots.push_back(sinkFlow->get<std::size_t>("sinkInput"));
    }

    auto paramOp = opIt.leftmostParent();
    while(paramOp != om.opEnd())
    {
        if (paramOp->getOutputTensor(0) != sourceTensor && (paramOp->getOpType() == "Constant" || paramOp->getOpType() == "ConstantInt"
            || paramOp->getOpType() == "ConstantDataElement"))
        {
            auto backUp = paramOp;
            ++paramOp;
            om.removeOp(backUp);
        }
        else
            ++paramOp;
    }

    om.removeOp(opIt);
    opIt = parentOpIt;

    if(sourceTensor == om.tensorEnd())
        sourceTensor = parentOpIt->getOutputTensor(0);

    for (unsigned j = 0; j < opsToLink.size(); ++j)
    {
        opsToLink[j]->setInputTensor(sourceTensor, inputSlots[j], false);
        om.defineFlow(sourceTensor, opsToLink[j], inputSlots[j]);
    }

    return opIt;
}
mv::Data::OpListIterator mv::linkNewOperationsReplacement(mv::Data::OpListIterator parentOpIt,
                                                      mv::Data::TensorIterator sourceTensor, mv::OpModel & om, mv::Data::OpListIterator opIt)
{
    //Important: do not change the order of this ops
    std::vector<mv::Data::OpListIterator> opsToLink;
    std::vector<std::size_t> inputSlots;
    for (auto sinkFlow = opIt.leftmostOutput(); sinkFlow != om.flowEnd(); ++sinkFlow)
    {
        opsToLink.push_back(sinkFlow.sink());
        inputSlots.push_back(sinkFlow->get<std::size_t>("sinkInput"));
    }

    auto paramOp = opIt.leftmostParent();
    while(paramOp != om.opEnd())
    {
        if (paramOp->getOutputTensor(0) != sourceTensor && (paramOp->getOpType() == "Constant" || paramOp->getOpType() == "ConstantInt"
            || paramOp->getOpType() == "ConstantDataElement"))
        {
            auto backUp = paramOp;
            ++paramOp;
            om.removeOp(backUp);
        }
        else
            ++paramOp;
    }

    om.removeOp(opIt);
    opIt = parentOpIt;

    if(sourceTensor == om.tensorEnd())
        sourceTensor = parentOpIt->getOutputTensor(0);

    for (unsigned j = 0; j < opsToLink.size(); ++j)
    {
        opsToLink[j]->setInputTensor(sourceTensor, inputSlots[j], false);
        om.defineFlow(sourceTensor, opsToLink[j], inputSlots[j]);
    }

    return opIt;
}


mv::Data::OpListIterator mv::linkNewMultipleOperationsReplacement(mv::Data::OpListIterator parentOpIt,
                                                      std::vector<mv::Data::TensorIterator> sourceTensors, mv::OpModel & om, mv::Data::OpListIterator opIt)
{
    
    //Important: do not change the order of this ops
    std::vector<mv::Data::OpListIterator> opsToLink;
    std::vector<std::size_t> inputSlots;
    //consumers
    for (auto sinkFlow = opIt.leftmostOutput(); sinkFlow != om.flowEnd(); ++sinkFlow)
    {                
        opsToLink.push_back(sinkFlow.sink());
        inputSlots.push_back(sinkFlow->get<std::size_t>("sinkInput"));
    }

    auto paramOp = opIt.leftmostParent();
    for (auto sourceTensorIt = sourceTensors.begin(); sourceTensorIt != sourceTensors.end(); sourceTensorIt++)
    {
        while(paramOp != om.opEnd())
        {
            if (paramOp->getOutputTensor(0) != *sourceTensorIt && (paramOp->getOpType() == "Constant" || paramOp->getOpType() == "ConstantInt"
                || paramOp->getOpType() == "ConstantDataElement"))
            {
                auto backUp = paramOp;
                ++paramOp;
                om.removeOp(backUp);
            }
            else
                ++paramOp;
        }
    }

    om.removeOp(opIt);
    opIt = parentOpIt;
    
    for (auto sourceTensorIt = sourceTensors.begin(); sourceTensorIt != sourceTensors.end(); sourceTensorIt++)
    {
        if(*sourceTensorIt == om.tensorEnd())
            *sourceTensorIt = parentOpIt->getOutputTensor(0);

        for (unsigned j = 0; j < opsToLink.size(); ++j)
        {
            opsToLink[j]->setInputTensor(*sourceTensorIt, inputSlots[j], false);
            om.defineFlow(*sourceTensorIt, opsToLink[j], inputSlots[j]);
        }
    }

    return opIt;
}

mv::Data::OpListIterator mv::linkNewOperationsReplacementRemoveFlows(mv::Data::OpListIterator childOpIt,
                                                      mv::Data::TensorIterator sourceTensor, mv::OpModel & om, mv::Data::OpListIterator opIt)
{
    std::vector<mv::Data::OpListIterator> opsToLink;
    std::vector<std::size_t> inputSlots;
    std::vector<mv::Data::FlowSiblingIterator> flowsToRemove;
    //consumers
    for (mv::Data::FlowSiblingIterator sinkFlow(opIt.leftmostOutput()); sinkFlow != om.flowEnd(); ++sinkFlow)
    {
        opsToLink.push_back(sinkFlow.sink());
        inputSlots.push_back(sinkFlow->get<std::size_t>("sinkInput"));
        flowsToRemove.push_back(sinkFlow);
    }
    
    for (unsigned flowIdx = 0; flowIdx < flowsToRemove.size(); flowIdx++)
    {
        om.undefineFlow(flowsToRemove[flowIdx]);
    }
    om.removeOp(opIt);
    opIt = childOpIt;
    
    //create links
    for(unsigned op = 0 ; op < opsToLink.size(); ++op)
    {
        opsToLink[op]->setInputTensor(sourceTensor, inputSlots[op], false);
        om.defineFlow(sourceTensor, opsToLink[op], inputSlots[op]);
    }
    return opIt;
}

void calcZeroPointAndScalePerTensor(double outputMax,  double outputMin, double& outScale, int64_t& outZp)
{
    outScale = (outputMax - outputMin)/255;
    if (outputMin >= 0.0)
        outZp = 0;
    else if (outputMax <= 0.0)
        outZp = 255;
    else if ((outputMin < 0.0) && (outputMax > 0.0))
    {
        auto max_diff = (outputMax/(std::abs(outputMin) + outputMax)) * 255;
        outZp = std::ceil(255 - max_diff);
    }
}

void updateInfMinMaxPerTensor(mv::Data::TensorIterator tensor)
{
    auto& tensorQuantization = tensor->get<mv::QuantizationParams>("quantParams");

    //Note: if input Tensor has min, max of infs...we need to compute them
    if (tensorQuantization.infinitelimits())
    {
        //Quantization equation Real = scale(Quantized - zeroPoint)
        double maximumFloat = tensorQuantization.getScale()[0] * (255 - tensorQuantization.getZeroPoint()[0]);
        double minimumFloat = -tensorQuantization.getZeroPoint()[0] * tensorQuantization.getScale()[0];
        if (minimumFloat == -0)
            minimumFloat = 0;

        mv::QuantizationParams newTensorQuantization(tensorQuantization.getZeroPoint(),
                                                    tensorQuantization.getScale(),{minimumFloat},{maximumFloat});
        tensor->set<mv::QuantizationParams>("quantParams", newTensorQuantization);
    }
}

void updateInfMinMaxPerChannel(mv::Data::TensorIterator tensor)
{
    auto& tensorQuantization = tensor->get<mv::QuantizationParams>("quantParams");

    //Note: Do not care if populated or unpopulated....batch = 1
    if (tensorQuantization.infinitelimits())
    {
        std::vector <double> maximums, minimums;
        double maximumFloat, minimumFloat;
        for (uint32_t channel = 0; channel < tensor->getShape()[mv::KERNEL_OUTPUT_CHANNELS]; channel++)
        {
            //Quantization equation Real = scale(Quantized - zeroPoint)
            maximumFloat = tensorQuantization.getScale()[channel] * (255 - tensorQuantization.getZeroPoint()[0]);
            minimumFloat = -tensorQuantization.getZeroPoint()[0] * tensorQuantization.getScale()[channel];
            if (minimumFloat == -0)
                minimumFloat = 0;
            maximums.push_back(maximumFloat);
            minimums.push_back(minimumFloat);
        }
        mv::QuantizationParams newTensorQuantization(tensorQuantization.getZeroPoint(),
                                                    tensorQuantization.getScale(),minimums, maximums);
        tensor->set<mv::QuantizationParams>("quantParams", newTensorQuantization);
    }
}

//template <class T>
//std::vector<T> extendToK(size_t size, std::vector<T> value, std::string tensorName)
//{
//    if (value.size() == 1)
//        return mv::utils::generateSequence<T>(size, static_cast<T>(value[0]) , 0);

//    // We enter in this case if and only if we specified multi channel scales and
//    // the tensor has been aligned
//    if (value.size() < size)
//    {
//        auto toReturn = mv::utils::generateSequence<T>(size, static_cast<T>(0) , 0);
//        for(unsigned i = 0; i < value.size(); ++i)
//            toReturn[i] = value[i];
//        return toReturn;
//    }

//    if (value.size() == size)
//        return value;

//    throw mv::ArgumentError("QuantizationPass", "extendToK", "parameters for " + tensorName + " dimensions doesn't match size of output_channels or 1",
//                std::to_string(value.size()));
//}

std::vector<double> extendToK(size_t size, std::vector<double> value, std::string tensorName)
{
    if (value.size() == 1)
        return mv::utils::generateSequence<double>(size, static_cast<double>(value[0]) , 0);

    // We enter in this case if and only if we specified multi channel scales and
    // the tensor has been aligned
    if (value.size() < size)
    {
        auto toReturn = mv::utils::generateSequence<double>(size, static_cast<double>(0) , 0);
        for(unsigned i = 0; i < value.size(); ++i)
            toReturn[i] = value[i];
        return toReturn;
    }

    if (value.size() == size)
        return value;

    throw mv::ArgumentError("QuantizationPass", "extendToK", "parameters for " + tensorName + " dimensions doesn't match size of output_channels or 1",
                std::to_string(value.size()));
}

std::vector<int64_t> extendToK(size_t size, std::vector<int64_t> value, std::string tensorName)
{
    if (value.size() == 1)
        return mv::utils::generateSequence<int64_t>(size, static_cast<int64_t>(value[0]) , 0);

    // We enter in this case if and only if we specified multi channel scales and
    // the tensor has been aligned
    if (value.size() < size)
    {
        auto toReturn = mv::utils::generateSequence<int64_t>(size, static_cast<int64_t>(0) , 0);
        for(unsigned i = 0; i < value.size(); ++i)
            toReturn[i] = value[i];
        return toReturn;
    }

    if (value.size() == size)
        return value;

    throw mv::ArgumentError("QuantizationPass", "extendToK", "parameters for " + tensorName + " dimensions doesn't match size of output_channels or 1",
                std::to_string(value.size()));
}

std::vector<mv::Data::OpListIterator> mv::findSinkLayers(mv::DataModel &dataModel, const mv::Data::TensorIterator &tensor)
{
    std::vector<mv::Data::OpListIterator> sinkOperations;
    if ((tensor)->hasAttr("flows"))
    {
        auto flowsNames = (tensor)->get<std::set<std::string>>("flows");
        for(auto flowName : flowsNames)
        {
            auto df = dataModel.getDataFlow(flowName);
            sinkOperations.push_back(df.sink());
        }
    }
    return sinkOperations;
}

bool mv::checkA0SOHSparsityBug(mv::Data::FlowListIterator flow, std::string referenceDevice)
{
    if (referenceDevice != "A0")
        return false;
    auto sink = flow.sink();
    auto tensor = flow->getTensor();

    if(!tensor->isPopulated())
    {
        if(sink->hasAttr("splitStrategy"))
        {
            std::string splitStrategy = sink->get<std::string>("splitStrategy");

            if(splitStrategy == "SplitOverH" &&
               sink->getOpType() == "DPUTask" &&
               sink->get<std::string>("taskOp") == "Conv" &&
               (
                sink->get<std::array<unsigned short, 2>>("kSize")[1] > 1))

                return true;

        }

    }
    return false;
}


bool mv::isVectorsEqual(const std::vector<double>& left, const std::vector<double>& right) {
    if(left.size() != right.size()) {
        return false;
    }

    for (int i = 0; i < left.size(); i++) {
        //if (fabs(left[i] - right[i]) > std::numeric_limits<float>::epsilon()) {
        if (fabs(left[i] - right[i]) > (1.0e-4) ) {
            return  false;
        }
    }
    return true;
}

bool mv::isEqualScale(const mv::QuantizationParams& left, const mv::QuantizationParams& right) {
    //in keembay the two eltwises can have different zero point
    bool isScaleEqual = isVectorsEqual(left.getScale(), right.getScale());
    return isScaleEqual;
}

bool mv::isEqual(const mv::QuantizationParams& left, const mv::QuantizationParams& right) {
    bool isZpEqual = left.getZeroPoint() == right.getZeroPoint();
    bool isMinEqual = isVectorsEqual(left.getMin(), right.getMin());
    bool isMaxEqual = isVectorsEqual(left.getMax(), right.getMax());
    bool isScaleEqual = isVectorsEqual(left.getScale(), right.getScale());
    return isZpEqual && isMinEqual && isMaxEqual && isScaleEqual;
}
