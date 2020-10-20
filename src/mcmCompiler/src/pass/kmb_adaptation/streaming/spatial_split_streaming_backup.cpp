//backup file that saves - no duplicates of slices in nested streaming and no control flow for first layer of streaming
#include "math.h"
#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/utils/custom_strings.hpp"
#include "include/mcm/base/exception/runtime_error.hpp"
#include "include/mcm/tensor/tiling.hpp"

static void streamingOperationsFcn(const mv::pass::PassEntry& pass,
                                        mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&,
                                        mv::Element&);

static void streamBinaryDataWeightsFcn(const mv::pass::PassEntry&,
                                        mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&,
                                        mv::Element&);

namespace mv
{
    namespace pass
    {
        MV_REGISTER_PASS(StreamingOperations)
        .setFunc(streamingOperationsFcn)
        .setDescription(
                "Generates New Ops according to Streaming Strategies that the graph provides");

        MV_REGISTER_PASS(StreamBinaryDataWeights)
        .setFunc(streamBinaryDataWeightsFcn)
        .setDescription(
            "The StreamOverK on Costant Operastions creates Constant + Slice, which is new smaller/fused Constants"
        );
    }
}

mv::Data::OpListIterator operationsReplacement(mv::Data::OpListIterator parentOpIt, mv::Data::TensorIterator sourceTensor, mv::OpModel om, mv::Data::OpListIterator opIt)
{
    //Important: do not change the order of this ops
    std::vector<mv::Data::OpListIterator> opsToLink;
    std::vector<std::size_t> inputSlots;
    for (mv::Data::FlowSiblingIterator sinkFlow(opIt.leftmostOutput()); sinkFlow != om.flowEnd(); ++sinkFlow)
    {
        opsToLink.push_back(sinkFlow.sink());
        inputSlots.push_back(sinkFlow->get<std::size_t>("sinkInput"));
    }

    while(opIt.parentsSize() > 1)
    {
        auto paramOp = opIt.leftmostParent();
        ++paramOp;
        om.removeOp(paramOp);
    }

    om.removeOp(opIt);
    opIt = parentOpIt;

    for (unsigned j = 0; j < opsToLink.size(); ++j)
    {
        //no need to trigger a cascade, we know what we are doing
        opsToLink[j]->setInputTensor(sourceTensor, inputSlots[j], false);
        om.defineFlow(sourceTensor, opsToLink[j], inputSlots[j]);
    }

    return opIt;
}

struct opStreamingSplitDef
{
    std::string axis ;
    size_t numSplits ;
};

mv::Data::TensorIterator solveWeightsTiling(mv::ComputationModel& model, mv::Data::OpListIterator op, mv::Tiling& tiling, std::map<std::string, std::vector<opStreamingSplitDef>>& thisGraphStrategy, std::unordered_map<std::string, bool> &createSlicesPerStream, std::map<std::pair<std::string, unsigned>, mv::Data::TensorIterator> &name_firstStream_sliceOp);
mv::Data::TensorIterator solveSpatialTiling(mv::ComputationModel& model, mv::Data::OpListIterator op, mv::Tiling& tiling, std::map<std::string, std::vector<opStreamingSplitDef>>& thisGraphStrategy, std::unordered_map<std::string, bool> &createSlicesPerStream, std::map<std::pair<std::string, unsigned>, mv::Data::TensorIterator> &name_firstStream_sliceOp);

std::map<std::string, std::function<mv::Data::TensorIterator(mv::OpModel&, mv::Data::OpListIterator, mv::Tiling&, std::map<std::string, std::vector<opStreamingSplitDef>>&, std::unordered_map<std::string, bool>& createSlicesPerStream, std::map<std::pair<std::string, unsigned>, mv::Data::TensorIterator> &name_firstStream_sliceOp)>> streamSplit =
{
//    {"W",solveSpatialTiling},
    {"H",solveSpatialTiling},
    {"K",solveWeightsTiling} //NOTE::Only Convolution is supported for SoK now
};


std::function<mv::Data::TensorIterator(mv::OpModel&, mv::Data::OpListIterator, mv::Tiling&, std::map<std::string, std::vector<opStreamingSplitDef>> &, std::unordered_map<std::string, bool>& createSlicesPerStream, std::map<std::pair<std::string, unsigned>, mv::Data::TensorIterator> &name_firstStream_sliceOp)> convSpatialTiling = solveSpatialTiling;
std::function<mv::Data::TensorIterator(mv::OpModel&, mv::Data::OpListIterator, mv::Tiling&, std::map<std::string, std::vector<opStreamingSplitDef>> &, std::unordered_map<std::string, bool>& createSlicesPerStream, std::map<std::pair<std::string, unsigned>, mv::Data::TensorIterator> &name_firstStream_sliceOp)> convOutChannelTiling = solveWeightsTiling;

static void setStreamingStrategy(const mv::pass::PassEntry &pass, mv::ComputationModel &model, std::map<std::string, std::vector<opStreamingSplitDef>> &thisGraphStrategy)
{
    // get ops to split and number of splits from descriptor
    auto globalParams = model.getGlobalConfigParams();
    if (!globalParams->hasAttr("streaming_strategy"))
    {
        pass.log(mv::Logger::MessageType::Info, "No custom streaming strategy provided");
        return;
    }
    auto strategyList = globalParams->get<std::vector<mv::Element>>("streaming_strategy");
    // each s refers to the name of an op, from the JSON strategy list
    for (auto layerNameStrategy : strategyList)
    {
        std::vector<opStreamingSplitDef> opxSplits;
        bool nodeHasSplit = false;
        std::string nodeName = layerNameStrategy.get<std::string>("name_filter");
        auto splitList = layerNameStrategy.get<std::vector<mv::Element>>("splits");
        for (std::size_t i = 0; i < splitList.size(); i++)
        {
            opStreamingSplitDef opxSplitx;
            if (splitList[i].hasAttr("H"))
            {
                if (splitList[i].get<int>("H") > 1)
                {
                    opxSplitx.axis = "H";
                    opxSplitx.numSplits = splitList[i].get<int>("H");
                    opxSplits.push_back(opxSplitx);
                    nodeHasSplit = true;
                    pass.log(mv::Logger::MessageType::Debug, "Streaming for node: " + nodeName + " has stream H = " + opxSplitx.numSplits);
                }
            }
            //NOTE:: Streaming over width, channels are not used
//            else if (splitList[i].hasAttr("W"))
//            {
//                if (splitList[i].get<int>("W")>1)
//                {
//                    opxSplitx.axis = "W";
//                    opxSplitx.numSplits = splitList[i].get<int>("W");
//                    opxSplits.push_back(opxSplitx);
//                    nodeHasSplit=true;
//                }
//            }
//            else if (splitList[i].hasAttr("C"))
//            {
//                if (splitList[i].get<int>("C")>1)
//                {
//                    opxSplitx.axis = "C";
//                    opxSplitx.numSplits = splitList[i].get<int>("C");
//                    opxSplits.push_back(opxSplitx);
//                    nodeHasSplit=true;
//                }
//            }
            if (splitList[i].hasAttr("K"))
            {
                if (splitList[i].get<int>("K") > 1)
                {
                    opxSplitx.axis = "K";
                    opxSplitx.numSplits = splitList[i].get<int>("K");
                    opxSplits.push_back(opxSplitx);
                    nodeHasSplit = true;
                    pass.log(mv::Logger::MessageType::Debug, "Streaming for node: " + nodeName + " has stream K = " + splitList[i].get<int>("K"));
                }
            }
        }
        if (nodeHasSplit)
            thisGraphStrategy.insert(std::pair<std::string, std::vector<opStreamingSplitDef>>(nodeName, opxSplits));
    }
}

void storeExistingSlice(std::string opName, unsigned streamId, mv::Data::TensorIterator slice,
    std::map<std::pair<std::string, unsigned>, mv::Data::TensorIterator>& name_firstStream_sliceOp)
{
    std::pair<std::string, unsigned> keyPair;
    keyPair.first = opName;
    keyPair.second = streamId;
    name_firstStream_sliceOp[keyPair] = slice;
}

mv::Data::TensorIterator solveWeightsTiling(mv::ComputationModel& model, mv::Data::OpListIterator op,
    mv::Tiling& tiling, std::map<std::string, std::vector<opStreamingSplitDef> > &thisGraphStrategy,
    std::unordered_map<std::string, bool>& createSlicesPerStream,
    std::map<std::pair<std::string, unsigned>, mv::Data::TensorIterator> &name_firstStream_sliceOp)
{
    mv::OpModel om(model);
    mv::DataModel dm(model);
    mv::ControlModel cm(model);

    //solve SOW/H location
    //TODO:: stop hardcoding index....
    auto inputTensor = op->getInputTensor(0);
    auto inputTensor2Conv = inputTensor ;
    auto kernelTensor = op->getInputTensor(1);
    auto outputTensor = op->getOutputTensor(0);
    bool nestedLayerStreaming = false;


    auto attrsToCopy = op->getAttrs({"stride", "padding", "shape", "bias"});

    auto quantParams = inputTensor->getQuantParams();
    auto opId = op->get<unsigned>("opId");
    auto number_of_splits = tiling.childTiles().size();
    auto axisToSplit =  mv::Shape::getAxis(tiling.getAxis());
    auto childTiles = tiling.childTiles();
    std::vector<mv::Data::TensorIterator> slices(number_of_splits);
    std::vector<mv::Data::TensorIterator> convs(number_of_splits);
    std::vector<mv::Data::TensorIterator> final_outputs(number_of_splits);
    size_t biasStartIndex = 0;
    size_t biasEndIndex = 0;
    std::string splitStrategy = op->get<std::string>("splitStrategy");

    for (unsigned split = 0; split < number_of_splits; split++)
    {
        mv::Data::TensorIterator slice;
        bool foundOpName = false;
        //NOTE:THIS OP NAME DOES NOT EXIST SO NO NESTED...GO AND BUILD SLICE
        auto foundIt = createSlicesPerStream.find(op->getName());
        if (foundIt != createSlicesPerStream.end())
            foundOpName = true;
        if ((!foundOpName) || (foundOpName && foundIt->second))
        {
            if (kernelTensor->hasAttr("quantParams"))
            {
                slice = om.slice(kernelTensor->getName() + inputTensor->getName() + "_sliceK_" + std::to_string(split),
                                kernelTensor,
                                childTiles[split].getStartCoord(),
                                childTiles[split].getSize());
                slice->setQuantParams(quantParams);
            }
            else
            {
                slice = om.slice(kernelTensor->getName() + "_sliceK_" + std::to_string(split),
                                kernelTensor,
                                childTiles[split].getStartCoord(),
                                childTiles[split].getSize());
            }
            storeExistingSlice(kernelTensor->getName(), split, slice, name_firstStream_sliceOp);
            om.getSourceOp(slice)->set<unsigned>("opId", opId);
        }
        else
        {
            std::pair<std::string, unsigned> keyPair;
            keyPair.first = kernelTensor->getName();
            keyPair.second = split;
            slice = name_firstStream_sliceOp[keyPair];
        }
        std::string streamingOpName = op->getName() + "_split_" + std::to_string(split);
        auto outputQuantParams = op->getOutputTensor(0)->getQuantParams();
        auto conv = om.conv(streamingOpName,
                            inputTensor,
                            slice,
                            op->get("stride"),
                            op->get("padding"),
                            op->get<unsigned>("dilationFactor"),
                            op->get<unsigned>("group"));
        conv->setDType(op->getOutputTensor(0)->getDType());
        conv->setQuantParams(quantParams);
        //NOTE: Nested streaming case KH
        if (thisGraphStrategy[op->getName()].size() > 1)
        {
            nestedLayerStreaming = true;
            thisGraphStrategy[streamingOpName].insert(thisGraphStrategy[streamingOpName].begin(), thisGraphStrategy[op->getName()].begin() + 1,
                    thisGraphStrategy[op->getName()].end());
            if (split == 0)
                createSlicesPerStream[streamingOpName] = true;
            else
                createSlicesPerStream[streamingOpName] = false;
        }

        if (op->hasAttr("bias"))
        {
            auto tileSize = childTiles[split].getSize()[axisToSplit];
            biasStartIndex = biasEndIndex;
            biasEndIndex = biasStartIndex + tileSize;
            auto biasTensorName = op->get<std::string>("bias");
            auto originalBiasTensor = dm.getTensor(biasTensorName);
            auto oiginalBiasData = originalBiasTensor->getData();
            if ( biasEndIndex > oiginalBiasData.size())
                biasEndIndex = oiginalBiasData.size();
            std::vector<mv::DataElement>::const_iterator biasFirst = oiginalBiasData.begin() + biasStartIndex;
            std::vector<mv::DataElement>::const_iterator biasLast = oiginalBiasData.begin() + biasEndIndex;
            std::vector<mv::DataElement> subBiasData(biasFirst, biasLast);
            std::string newBiasTensorName = mv::createBiasName(op->getName() + "_split_" + std::to_string(split));
            mv::Data::TensorIterator biasTensor;
            mv::Data::TensorIterator biasTensorX;
            if (originalBiasTensor->hasAttr("quantParams"))
            {
                auto biasAttrQPs = originalBiasTensor->get("quantParams");
                biasTensorX = dm.defineTensor(mv::Tensor(newBiasTensorName, {tileSize}, originalBiasTensor->getDType(), originalBiasTensor->getOrder(), subBiasData, biasAttrQPs ));
            }
            else
                biasTensorX = dm.defineTensor(mv::Tensor(newBiasTensorName, {tileSize}, originalBiasTensor->getDType(), originalBiasTensor->getOrder(), subBiasData));
            om.addAttr(om.getSourceOp(conv), "bias", biasTensorX->getName());
        }
        auto newOp = om.getSourceOp(conv);

        newOp->set<bool>("splitted",true);//TODO::temporary hack. To remove once the iteration conditions are updated
        newOp->setAttrs(attrsToCopy);

        slices[split] = slice;
        convs[split] = conv;
        bool enableSerialStreaming = true;
        if ((split>0)&&(enableSerialStreaming))
            cm.defineFlow(om.getSourceOp(convs[split-1]), om.getSourceOp(convs[split]));
    }

    kernelTensor->set<mv::Tensor::MemoryLocation>("Location", mv::Tensor::MemoryLocation::BLOB);
    // decide on the location of the I/O Tensors of the conv;
    // basically, for each operation, if we are the last inside the recursive splitting schema, then we can make the
    // assumption that we are fitting into CMX. The check is assumed to be made by the scheduler. This pass only implements
    // the respective schedule inside the graph.
    // If we are not the last split, we will basically, inherit the location our parent inputTensor;

    for(unsigned split = 0 ; split < number_of_splits; split++)
    {
        mv::Tensor::MemoryLocation inputLocation;
        mv::Tensor::MemoryLocation outputLocation;
        if(nestedLayerStreaming)
        {
            inputLocation.relocate(inputTensor->get<mv::Tensor::MemoryLocation>("Location"));
            outputLocation.relocate(outputTensor->get<mv::Tensor::MemoryLocation>("Location"));
        }
        else
        {
            inputLocation.relocate(mv::Tensor::MemoryLocation::NNCMX);
            outputLocation.relocate(mv::Tensor::MemoryLocation::NNCMX);
            inputLocation.force();
            outputLocation.force();
        }
        slices[split]->set<mv::Tensor::MemoryLocation>("Location", inputLocation);
        convs[split]->set<mv::Tensor::MemoryLocation>("Location", outputLocation);
    }

    for(unsigned split = 0; split < number_of_splits; split++)
    {
        mv::Data::TensorIterator out;
        if(childTiles[split].childTiles().size() > 1)
        {
            out = (streamSplit[childTiles[split].getAxis()])(om,om.getSourceOp(convs[split]),childTiles[split], thisGraphStrategy, createSlicesPerStream, name_firstStream_sliceOp);
            om.removeOp( om.getSourceOp(convs[split]));
        }
        else
            out = convs[split];
        final_outputs[split] = out;
    }

    auto outputQuantParams = op->getOutputTensor(0)->getQuantParams();
    auto concat = om.concat(op->getName() + "concat_",
                    final_outputs,
                    "C");
    concat->setDType(op->getOutputTensor(0)->getDType());
    concat->setQuantParams(outputQuantParams);
    om.getSourceOp(concat)->set<unsigned>("opId", opId);
    om.getSourceOp(concat)->set<std::string>("splitStrategy", splitStrategy);

    concat->set<mv::Tensor::MemoryLocation>("Location",outputTensor->get<mv::Tensor::MemoryLocation>("Location"));

    return concat;
}

mv::Data::TensorIterator solveSpatialTiling(mv::ComputationModel& model, mv::Data::OpListIterator op, mv::Tiling& tiling, std::map<std::string,
                std::vector<opStreamingSplitDef> > &thisGraphStrategy, std::unordered_map<std::string, bool>& createSlicesPerStream, std::map<std::pair<std::string, unsigned>, mv::Data::TensorIterator> &name_firstStream_sliceOp)
{
    mv::OpModel om(model);
    mv::ControlModel cm(model);
    bool nestedLayerStreaming = true;
    auto outputTensor = op->getOutputTensor("output");
    auto opId = op->get<unsigned>("opId");
    std::string splitStrategy = op->get<std::string>("splitStrategy");
    auto number_of_splits = tiling.childTiles().size();
    auto axisToSplit =  mv::Shape::getAxis(tiling.getAxis());
    auto childTiles = tiling.childTiles();

    // NOTE: In the streaming case, we can't just blindly copy everything like we
    // do in the DPUTask conversion case. We have to overwrite shape, padding, etc.
    auto attrsToCopy = op->getAttrs({"stride", "padding", "shape"});

    std::vector<mv::Shape> spatial_indexes(number_of_splits);
    std::vector<std::vector<mv::Data::TensorIterator>> slices(number_of_splits);
    std::vector<mv::Data::TensorIterator> convs(number_of_splits);
    std::vector<mv::Data::TensorIterator> final_outputs(number_of_splits);
    std::array<unsigned short, 2> kernelStride;
    if (op->hasAttr("stride"))
        kernelStride = op->get<std::array<unsigned short, 2>>("stride");
    else
        kernelStride = {1,1};//fake stride

    //NOTE: assuming order of paddings: left,right,top,bottom
    std::array<unsigned short, 4> padding;
    if (op->hasAttr("padding"))
        padding = op->get<std::array<unsigned short, 4>>("padding");
    else
        padding = {0, 0, 0, 0};

    auto startPad = padding;
    auto endPad = padding;
    auto middlePad = padding;
    auto currentPad = padding;

//    if (axisToSplit == mv::Shape::getAxis("W"))
//    {
//        startPad[1] = 0;
//        endPad[0] = 0;
//        middlePad[0] = 0;
//        middlePad[1] = 0;
//    }
    if (axisToSplit == mv::Shape::getAxis("H"))
    {
        startPad[3] = 0;
        endPad[2] = 0;
        middlePad[2] = 0;
        middlePad[3] = 0;
    }

    for (unsigned split = 0; split < number_of_splits; split++)
    {
        if (split == 0)
            currentPad = startPad;
        else if (split == (number_of_splits -1))
            currentPad = endPad;
        else
            currentPad = middlePad;

        mv::Data::TensorIterator newTensor;
        std::string opType = op->getOpType();
        std::string streamingOpName = op->getName() + "_split_" + std::to_string(split);
        if (opType == "MaxPool" || opType == "Conv" || opType == "DepthwiseConv")
        {
            auto inputTensor = op->getInputTensor(0);
            //NOTE: NESTED STREAM NEEDS SLICE OPS ONLY FOR THE FIRST PART AND RE-USE
            mv::Data::TensorIterator slice;
            auto foundIt = createSlicesPerStream.find(op->getName());
            bool foundOpName = false;
            //NOTE:THIS OP NAME DOES NOT EXIST SO NO NESTED...GO AND BUILD SLICE
            if (foundIt != createSlicesPerStream.end())
                foundOpName = true;
            if ((!foundOpName) || (foundOpName && foundIt->second))
            {
                auto quantParams = inputTensor->getQuantParams();
                slice = om.slice(op->getName() + "_sliceH_" + std::to_string(split),
                                inputTensor,
                                childTiles[split].getStartCoord(),
                                childTiles[split].getSize());
                slice->setQuantParams(quantParams);
                storeExistingSlice(inputTensor->getName(), split, slice, name_firstStream_sliceOp);
                om.getSourceOp(slice)->set<unsigned>("opId", opId);
            }
            else
            {
                std::pair<std::string, unsigned> keyPair;
                keyPair.first = inputTensor->getName();
                keyPair.second = split;
                slice = name_firstStream_sliceOp[keyPair];
            }
            auto quantParams = op->getOutputTensor(0)->getQuantParams();
            if (opType == "MaxPool")
                newTensor = om.maxPool(streamingOpName,
                                slice,
                                op->get<std::array<unsigned short, 2UL>>("kSize"),
                                kernelStride,
                                currentPad,
                                op->get<const bool>("exclude_pad"));

            if (opType == "DepthwiseConv")
                newTensor = om.depthwiseConv(streamingOpName,
                                slice,
                                op->getInputTensor(1),
                                kernelStride,
                                currentPad,
                                op->get<unsigned>("dilationFactor"));

            if (opType == "Conv")
                newTensor = om.conv(streamingOpName,
                                slice,
                                op->getInputTensor(1),
                                kernelStride,
                                currentPad,
                                op->get<unsigned>("dilationFactor"),
                                op->get<unsigned>("group"));
            newTensor->setDType(op->getOutputTensor(0)->getDType());
            newTensor->setQuantParams(quantParams);
            slices[split].push_back(slice);
        }
        else if (opType == "Eltwise")
        {
            auto inputSlots = op->inputSlots();
            auto eltwiseType = op->get<std::string>("eltwiseType");
            auto originalDType = op->getOutputTensor(0)->getDType();
            for (auto i = 0; i < inputSlots; i++)
            {
                auto inputTensor = op->getInputTensor(i);
                auto quantParams = inputTensor->getQuantParams();

                auto slice = om.slice(op->getName() + "_sliceH_" + std::to_string(split) + "_" + std::to_string(i),
                                inputTensor,
                                childTiles[split].getStartCoord(),
                                childTiles[split].getSize());
                slice->setQuantParams(quantParams);
                om.getSourceOp(slice)->set<unsigned>("opId", opId);
                slices[split].push_back(slice);
            }

            auto quantParams = op->getOutputTensor(0)->getQuantParams();
            newTensor = om.eltwise(op->getName() + "_split_" + std::to_string(split), slices[split], eltwiseType);
            newTensor->setDType(originalDType);
            newTensor->setQuantParams(quantParams);
        }

        //NOTE: Nested streaming case
        if (thisGraphStrategy[op->getName()].size() > 1)
        {
            nestedLayerStreaming = true;
            thisGraphStrategy[streamingOpName].insert(thisGraphStrategy[streamingOpName].begin(), thisGraphStrategy[op->getName()].begin() + 1,
                    thisGraphStrategy[op->getName()].end());
            //NOTE:: If we go HK streaming...
            if (split == 0)
                createSlicesPerStream[streamingOpName] = true;
            else
                createSlicesPerStream[streamingOpName] = false;
        }
        auto newOp = om.getSourceOp(newTensor);

        newOp->setAttrs(attrsToCopy);
        newOp->set<bool>("splitted", true);//TODO::temporary hack. To remove once the iteration conditions are updated

        convs[split] = newTensor;

        bool enableSerialStreaming = true;
        if ((split > 0) && enableSerialStreaming)
            cm.defineFlow(om.getSourceOp(convs[split-1]), om.getSourceOp(convs[split]));
    }

    // decide on the location of the I/O Tensors of the conv;
    // basically, for each operation, if we are the last inside the recursive splitting schema, then we can make the
    // assumption that we are fitting into CMX. The check is assumed to be made by the scheduler. This pass only implements
    // the respective schedule inside the graph.
    // If we are not the last split, we will basically, inherit the location our parent inputTensor;
    for (unsigned split = 0 ; split < number_of_splits; split++)
    {
        auto numInputs = slices[split].size();
        std::vector<mv::Tensor::MemoryLocation> inputLocation(numInputs);
        mv::Tensor::MemoryLocation outputLocation;
        if (nestedLayerStreaming)
        {
            for (std::size_t i = 0; i < numInputs; i++)
            {
                auto inputTensor = op->getInputTensor(i);
                inputLocation[i].relocate(inputTensor->get<mv::Tensor::MemoryLocation>("Location"));
            }
            outputLocation.relocate(outputTensor->get<mv::Tensor::MemoryLocation>("Location"));
        }
        else
        {
            for (std::size_t i = 0; i < numInputs; i++)
            {
                inputLocation[i].relocate(mv::Tensor::MemoryLocation::NNCMX);
                inputLocation[i].force();
            }
            outputLocation.relocate(mv::Tensor::MemoryLocation::NNCMX);
            outputLocation.force();
        }
        for (std::size_t i = 0; i < numInputs; i++)
            slices[split][i]->set<mv::Tensor::MemoryLocation>("Location", inputLocation[i]);
        convs[split]->set<mv::Tensor::MemoryLocation>("Location", outputLocation);
    }


    for (unsigned split = 0; split < number_of_splits; split++)
    {
        mv::Data::TensorIterator out;
        if (childTiles[split].childTiles().size() > 1)
        {
            out = (streamSplit[childTiles[split].getAxis()])(om, om.getSourceOp(convs[split]), childTiles[split], thisGraphStrategy, createSlicesPerStream, name_firstStream_sliceOp);
            om.removeOp(om.getSourceOp(convs[split]));
        }
        else
            out = convs[split];
        final_outputs[split] = out;
    }
    std::vector<mv::Shape> final_outputs_deb(number_of_splits);
    for (std::size_t i=0; i < number_of_splits; ++i)
        final_outputs_deb[i] = final_outputs[i]->getShape();

    auto quantParams = op->getOutputTensor(0)->getQuantParams();
    auto concat = om.concat(op->getName() + "concat_",
                    final_outputs,
                    tiling.getAxis());
    concat->setDType(op->getOutputTensor(0)->getDType());
    concat->setQuantParams(quantParams);
    om.getSourceOp(concat)->set<unsigned>("opId", opId);
    om.getSourceOp(concat)->set<std::string>("splitStrategy", splitStrategy);
    concat->set<mv::Tensor::MemoryLocation>("Location", outputTensor->get<mv::Tensor::MemoryLocation>("Location"));
    return concat;
}

void streamingOperationsFcn(const mv::pass::PassEntry& pass,
                                mv::ComputationModel& model,
                                mv::TargetDescriptor&,
                                mv::Element&,
                                mv::Element&)
{

    mv::OpModel om(model);
    std::map<std::string, std::vector<opStreamingSplitDef>> thisGraphStrategy;
    setStreamingStrategy(pass, model, thisGraphStrategy);
    std::vector<opStreamingSplitDef> thisOpStrategy;

    //NOTE: NESTED STREAMING MEANS 2 LEVELS OF STREAMING, eg. HK, Stream Over H will stream
    //the input Tensor of the Op and then for every new Op have to stream it over K, which
    //means the weights will be repeated for the second level of streaming, this is why need
    //the data structures below...to create only one pair of nested slices
    std::unordered_map<std::string, bool> createSlicesPerStream = {};
    std::map<std::pair<std::string, unsigned>, mv::Data::TensorIterator> name_firstStream_sliceOp;

    for (auto layerNameStrategy: thisGraphStrategy)
    {
        std::string nodeName = layerNameStrategy.first;
        //NOTE: Graph optimizer will never do that but needs to be her for manual Scheduling
        if (!om.checkOp(nodeName))
        {
            pass.log(mv::Logger::MessageType::Error, nodeName + " is not present in model, skipping streaming");
            continue;
        }
        auto opIt =  om.getOp(nodeName);
        thisOpStrategy = thisGraphStrategy[nodeName];
        std::string opType = opIt->getOpType();

        if ((opType == "Conv" || opType == "DepthwiseConv" ||  (opType == "MaxPool") || (opType == "Eltwise")))
        {
            int numberOfSplits = thisOpStrategy[0].numSplits ;
            std::string axisToSplit = thisOpStrategy[0].axis ;
            mv::Tiling masterTile(axisToSplit, numberOfSplits);
            mv::Shape masterSize;

            if (axisToSplit == "K")
            {
                masterTile.setSize(opIt->getInputTensor(1)->getShape());
                masterTile.generateWeightsTiling();
            }
            else
            {
                masterTile.setSize(opIt->getInputTensor(0)->getShape());
                masterTile.generateSpatialTiling(opIt);
            }

            auto sourceTensor = opIt->getInputTensor(0);
            auto parentOpIt = om.getSourceOp(sourceTensor);
            auto result = (streamSplit[masterTile.getAxis()])(om, opIt, masterTile,
                               thisGraphStrategy, createSlicesPerStream, name_firstStream_sliceOp);

            // reconnect children to subgraph
            std::vector<mv::Data::OpListIterator> opsToLink;
            std::vector<std::size_t> inputSlots;
            for (mv::Data::FlowSiblingIterator sinkFlow(opIt.leftmostOutput()); sinkFlow != om.flowEnd(); ++sinkFlow)
            {
                opsToLink.push_back(sinkFlow.sink());
                inputSlots.push_back(sinkFlow->get<std::size_t>("sinkInput"));
            }
            om.removeOp(opIt);
            for (unsigned j = 0; j < opsToLink.size(); ++j)
            {
                opsToLink[j]->setInputTensor(result, inputSlots[j], false);
                om.defineFlow(result, opsToLink[j], inputSlots[j]);
            }
        }
    }
}


static void streamBinaryDataWeightsFcn(const mv::pass::PassEntry& ,
                                        mv::ComputationModel& model,
                                        mv::TargetDescriptor& ,
                                        mv::Element& ,
                                        mv::Element &)
{
    //Need to duplicate the consts to number equal to streams, cause of the binary_data
    mv::OpModel om(model);

    std::set <std::string> removeConstantsSet;
    for(auto opIterator = om.opBegin(); opIterator != om.opEnd(); ++opIterator)
    {
        std::string opType = opIterator->getOpType();
        std::vector<mv::Data::TensorIterator> toSort;

        if (opType == "Slice" && opIterator->getInputTensor(0)->isPopulated())
        {
            auto inTensorSlice = opIterator->getInputTensor(0);
            removeConstantsSet.insert(om.getSourceOp(inTensorSlice)->getName());
            auto outTensorSlice = opIterator->getOutputTensor(0);
            auto parentOpIt = om.getSourceOp(opIterator->getInputTensor(0));
            auto shape = outTensorSlice->getShape();
            auto quantParams = outTensorSlice->getQuantParams();

            auto newConstant = om.constantDataElement(opIterator->getName() + "_weights",
                                                      outTensorSlice->getData(), shape,
                                                      outTensorSlice->getDType(), outTensorSlice->getOrder());
            newConstant->setQuantParams(quantParams);
            newConstant->set<mv::Tensor::MemoryLocation>("Location", mv::Tensor::MemoryLocation::BLOB);
            auto constantOp = om.getSourceOp(newConstant);
            if(opIterator->hasAttr("opId"))
            {
                unsigned currentOpId = opIterator->get<unsigned>("opId");
                constantOp->set<unsigned>("opId", currentOpId);
            }
            opIterator = operationsReplacement(parentOpIt, newConstant, om, opIterator);
        }
    }
    for (auto& opName:removeConstantsSet)
        om.removeOp(om.getOp(opName));
}
