#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/tensor/quantization_params.hpp"
#include "include/mcm/utils/custom_strings.hpp"
#include "include/mcm/pass/pass_utils.hpp"
#include <math.h>

static void generateSparsityMapsPopulatedTensorsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);
static void generateSparsityMapsUnpopulatedTensorsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);

namespace mv
{
    namespace pass
    {

        MV_REGISTER_PASS(GenerateSparsityMapsPopulatedTensors)
        .setFunc(generateSparsityMapsPopulatedTensorsFcn)
        .setDescription(
            "Generates sparsity maps for populated tensors."
        );

        MV_REGISTER_PASS(GenerateSparsityMapsUnpopulatedTensors)
        .setFunc(generateSparsityMapsUnpopulatedTensorsFcn)
        .setDescription(
            "Generates sparsity maps for unpopulated tensors."
        );
    }
}

mv::Data::TensorIterator createFakeSparsityMap(mv::OpModel om, mv::Data::OpListIterator dpuTaskOp, const std::string& sparsityMapName, const mv::Shape& sparsityShape, const std::vector<int64_t>& sparsityMapData)
{
    auto sparsityMap = om.constantInt(sparsityMapData, sparsityShape, mv::DType("UInt8"), mv::Order("NCHW"), {{},{},{},{}},sparsityMapName);
    om.getSourceOp(sparsityMap)->set<unsigned>("opId", dpuTaskOp->get<unsigned>("opId"));
    unsigned newSize = dpuTaskOp->addInputTensor(sparsityMap);
    om.defineFlow(sparsityMap, dpuTaskOp, newSize - 1);

    return sparsityMap;
}

uint16_t getWindowSize(uint16_t kx, uint16_t sx)
{
    //Find max mpe where if calc window <= 32
    //return window size for the max mpe
    uint16_t windowSize, maxMpeWindowSize = 64;
    int mpe = 1;

    //mpe in [1,2,4,8,16]
    while(mpe <= 16)
    {
        if (sx <= kx)
            windowSize = kx + sx * (mpe - 1);
        else
            windowSize = kx * mpe;

        if (windowSize <= 32)
            maxMpeWindowSize = windowSize;

        mpe *= 2;
    }

    return maxMpeWindowSize;
}

std::vector<int8_t> createBitPattern(uint16_t kernelW, uint16_t kernelH, uint16_t windowsSize, uint16_t inputChannels)
{
    std::vector<int8_t> bitpattern;
    bitpattern.reserve(windowsSize*kernelH*inputChannels);
    for(size_t c = 0; c < inputChannels; c++)
        for(size_t y = 0; y < kernelH; y++)
            for(size_t x = 0; x < windowsSize; x++)
                if (x < kernelW)
                    bitpattern.emplace_back(1);
                else
                    bitpattern.emplace_back(0);
    return bitpattern;
}

// The sparsity maps relative to populated tensors have to be generated BEFORE the dma passes.
// As they have to be DMAed into CMX.
static void generateSparsityMapsPopulatedTensorsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element& passDesc, mv::Element&)
{

    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    mv::OpModel om(model);
    mv::DataModel dm(model);

    for(auto dpuTask = om.opBegin(); dpuTask != om.opEnd(); ++dpuTask)
    {
        if(dpuTask->getOpType() == "DPUTask")
        {
            bool weightsSparsity = dpuTask->hasAttr("weightsSparsity") ? dpuTask->get<bool>("weightsSparsity") : false;
            std::string taskOp = dpuTask->get<std::string>("taskOp");
            pass.log(mv::Logger::MessageType::Debug, " taskOp "  + dpuTask->get<std::string>("taskOp"));
            bool isChannelMajorConv = taskOp == "ChannelMajorConvolution";
            bool isPooling = taskOp == "MaxPool";
            bool isDepthWiseConv = taskOp == "DepthwiseConv";

            // NOTE by Marco: Checking for Elementwise is necessary, as it is the only operation
            // that does not support neither Real Sparsity or Fake Sparsity (aka Activation Window)
            // being it the hackiest operation ever
            bool isElementWise = taskOp == "Add" || taskOp == "Subtract" || taskOp == "Multiply";

            //for max pooling and deptwise convolution and channel-major convolution we need to generate sparsity data
            //even if those layers does not support sparsity.
            if (isPooling || isDepthWiseConv || isChannelMajorConv)
            {
                uint16_t kernelW, kernelH;

                auto strides = dpuTask->get<std::array<unsigned short, 2>>("stride");
                auto inputChannels = dpuTask->getInputTensor(0)->getShape()[mv::IO_CHANNEL_DIMENSION];
                auto outputChannels = dpuTask->getOutputTensor(0)->getShape()[mv::IO_CHANNEL_DIMENSION];

                // Using the check in this way, instead of on the operation type
                // makes this pass work on both aligned and unaligned convolutions.
                if (dpuTask->hasAttr("kSize"))
                {
                    auto kernelShape = dpuTask->get<std::array<unsigned short, 2>>("kSize");
                    kernelW = kernelShape[0];
                    kernelH = kernelShape[1];
                }
                else
                {
                    auto weightsShape = dpuTask->getInputTensor(1)->getShape();
                    kernelW = weightsShape[mv::KERNEL_WIDTH];
                    kernelH = weightsShape[mv::KERNEL_HEIGHT];
                }

                auto windowsSize = getWindowSize(kernelW, strides[0]);

                pass.log(mv::Logger::MessageType::Debug, "windowSize " + std::to_string(windowsSize));
                pass.log(mv::Logger::MessageType::Debug, "OutputChannels " + std::to_string(outputChannels));

                std::vector<int8_t> bitpattern;
                std::vector<uint8_t> perChannelSparsity;
                std::vector<std::size_t> ndims(4);
                if (isPooling || isDepthWiseConv)
                {
                    bitpattern = std::move(createBitPattern(kernelW, kernelH, windowsSize, 1));
                    perChannelSparsity.resize(static_cast<std::size_t>(std::ceil(bitpattern.size() / 128.0)) * 16);//allocate once
                    ndims = {16, static_cast<std::size_t>(std::ceil(bitpattern.size() / 128.0)), 1, inputChannels};
                }
                else //isChannelMajorConvolution
                {
                    bitpattern = std::move(createBitPattern(kernelW, kernelH, windowsSize, inputChannels));
                    auto windowSparsitySize = static_cast<std::size_t>(std::ceil(windowsSize/8.0)); //how many bytes we need per window
                    auto NumberOfRowsSparistyBytes = static_cast<std::size_t>(std::ceil((kernelH * inputChannels * windowSparsitySize) / 16.0 ));
                    perChannelSparsity.resize(NumberOfRowsSparistyBytes * 16);//allocate once
                    ndims = {16, NumberOfRowsSparistyBytes, 1, outputChannels};
                }

                int channelLenght = bitpattern.size();

                //populate sparsity
                pass.log(mv::Logger::MessageType::Debug, " perChannelSize = " + std::to_string(perChannelSparsity.size()) );
                for (size_t i = 0; i < bitpattern.size(); i++)
                {
                    //pass.log(mv::Logger::MessageType::Debug, " i = " + std::to_string(i));
                    //pass.log(mv::Logger::MessageType::Debug, " perChannelIDx = " + std::to_string(((i/128)*16 + (i%128)/8)) );
                    perChannelSparsity.at((i/128)*16 + (i%128)/8) |= bitpattern[i] << (i%8); //use at to check boundaries - just in case
                }

                //Create Tensor with Sparsity Data
                mv::Shape sparsityShape(ndims);
                std::vector<int64_t> data(sparsityShape.totalSize(), 0);
                //mv::Tensor sparsityTensor("backup", sparsityShape, mv::DType("UInt8"), mv::Order("WHCN"), data);
                auto sparsityTensor = mv::Tensor(dpuTask->getName() + "_sparse_dw", sparsityShape, mv::DType("UInt8"), mv::Order("NCHW"), data);

                for(unsigned kx = 0; kx < sparsityShape[0]; ++kx)
                    for(unsigned ky = 0; ky < sparsityShape[1]; ++ky)
                        for(unsigned ic = 0; ic < sparsityShape[2]; ++ic)
                            for(unsigned oc = 0; oc < sparsityShape[3]; ++oc)
                                sparsityTensor.at({kx, ky, ic, oc}) = static_cast<int64_t>(perChannelSparsity[ky*sparsityShape[0] + kx]);

                std::string opName = dpuTask->getName();

                auto fakeSparsityMap = createFakeSparsityMap(om, dpuTask, mv::createFakeSparsityMapName(opName), sparsityShape, sparsityTensor.getIntData());
                fakeSparsityMap->set<int>("channelLength", channelLenght);

                dpuTask->set<bool>("fakeSparsity", true);
                dpuTask->set<size_t>("fakeSparsityIndex", dpuTask->inputSlots()-1);
            }
            else if(weightsSparsity && !isElementWise)
            {
                //Here only in the case of ZMajorConvolution
                auto weightsTensor = dpuTask->getInputTensor(1);
                weightsTensor->setOrder(mv::Order("NHWC"));
                if(weightsTensor->setSparse())
                    dm.defineTensor(weightsTensor->getSparsityMap());
            }
        }
    }
}

bool checkA0SOHSparsityBug(mv::Data::FlowListIterator flow)
{
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
               (sink->get<std::array<unsigned short, 2>>("kSize")[0] > 1 ||
                sink->get<std::array<unsigned short, 2>>("kSize")[1] > 1))

                return true;
        }
    }
    return false;
}



// Result of chat with Alessandro:
// An activation tensor can be sparse if and only if
// 1) It is an output of a DPUTask. (ODU populates storage element and sparsity map).
// 2) It is the input of a ZMajorConv (Only layer that supports IDU)

// An activation tensor MUST be sparse if it's:
// 1) SplitOverH
// 2) Involved in a ZMajorConvolution with kernel > 1 (HW bug)

// In the future, these two conditions could change. We have to sync with runtime.

// Eltwise, being the hackiest operation ever, potentially can support sparsity input, sharing the IDU with ZMajorConv, but the runtime currently doesn't allow it.
// The inner loop of this function must iterate on tensors, not data flows
static void generateSparsityMapsUnpopulatedTensorsFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element& passDesc, mv::Element&)
{

    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    mv::DataModel dm(model);

    for(auto tensor = dm.tensorBegin(); tensor != dm.tensorEnd(); ++tensor)
    {
        // Probably an inner tensor, skip
        if(!tensor->hasAttr("flows"))
            continue;

        // Populated tensor, skip
        if(tensor->isPopulated())
            continue;

        auto flows = tensor->get<std::set<std::string>>("flows");

        bool tensorSparsifiable = true;
        bool tensorNeedsSparsity = false;
        bool inputActivationSparsity = false;
        bool outputActivationSparsity = false;

        // Conditions for activation sparsity
        // 1) The source of the activation tensor must be a DPUTask (Otherwise there's no ODU to write SM, SE)
        // 2) The sink of the activation tensor must be a ZMajor Convolution, the only operation
        //    with an IDU capable of handling sparsity data
        // 3) Runtime doesn't support SOK and activation sparsity neither in input nor output
        for(auto& flowStr: flows)
        {
            auto flow = dm.getDataFlow(flowStr);
            if(checkA0SOHSparsityBug(flow))
            {
                tensorNeedsSparsity = true;
                break;
            }
        }
        for(auto& flowStr: flows)
        {
            auto flow = dm.getDataFlow(flowStr);
            auto source = flow.source();
            auto sink = flow.sink();
            outputActivationSparsity |= source->hasAttr("outputActivationSparsity") ? source->get<bool>("outputActivationSparsity") : false;
            inputActivationSparsity |= sink->hasAttr("inputActivationSparsity") ? sink->get<bool>("inputActivationSparsity") : false;
            if(source->getOpType() != "DPUTask" ||
               source->get<std::string>("splitStrategy") == "SplitOverK" ||
               source->get<std::string>("taskOp") == "DepthwiseConv" || //TESTING ONLY!
               sink->getOpType() != "DPUTask" ||
               sink->get<std::string>("taskOp") != "Conv" ||
               sink->get<std::string>("splitStrategy") == "SplitOverK")
            {
                tensorSparsifiable = false;
                break;
            }
        }

        if(tensorNeedsSparsity && !tensorSparsifiable)
            throw std::runtime_error("Wrong strategy generated: tensor " + tensor->getName() + " needs sparsity but it can't be sparsified");
        if((tensorSparsifiable && inputActivationSparsity && outputActivationSparsity) || tensorNeedsSparsity)
            tensor->setSparse();
    }
}
