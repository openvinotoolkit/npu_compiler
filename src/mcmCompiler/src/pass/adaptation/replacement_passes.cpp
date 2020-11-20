#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/utils/custom_math.hpp"
#include "include/mcm/utils/warning_manager.hpp"
#include "include/mcm/pass/pass_utils.hpp"
#include "include/mcm/base/exception/logic_error.hpp"
#include <cmath>
#include <functional>
#include <string>

const size_t FULLY_CONNECTED_KERNEL = 1;

void fullyConnectedAsConv2DFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model);
static void handleEltWiseDifferentScales(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);
void averageAsDepthWiseFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model);
void interpAsAvgPoolingFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model);
void interpAsDepthConvFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model);
void flattenAsReshapeFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model);
void topKAsArgMaxFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model);
static void replacementOpsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);
void scaleAsDepthwiseFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model);
void replaceLargeKernelsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model);
void replaceLargeStridesFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model);
void replaceAsymmetricStridesFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model);
void replacePoolReshapePatternFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model);
void replaceConcatOfPopulatedTensorsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model);
void reorgYoloAsConvConcatFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model);
void replaceExpReduceSumMultipyFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model);
void insertPermuteBeforeDetFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model);
void replacePermuteAsReshape(const mv::pass::PassEntry& pass, mv::ComputationModel& model);
static void detectEltWiseUpaInputs(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);
static void markCMCompatibleConvsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);
static void addPermuteToNonCMConvPathsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);

namespace mv
{

    namespace pass
    {

        MV_REGISTER_PASS(ReplacementOps)
        .setFunc(replacementOpsFcn)
        .setDescription(
            "Replaces Operations"
        );

        MV_REGISTER_PASS(EltwiseToSWEltwise)
        .setFunc(handleEltWiseDifferentScales)
        .setDescription(
            "Replaces Eltwise with SW Layer Eltwise in case scales of inputs are different"
        );

        MV_REGISTER_PASS(MarkEltWiseUpaInputs)
        .setFunc(detectEltWiseUpaInputs)
        .setDescription(
            "Detect if Eltwise has Upa inputs and in case of Upa inputs avoid HKSwitch for possible strategy"
        );

        MV_REGISTER_PASS(MarkCMCompatibleConvs)
        .setFunc(markCMCompatibleConvsFcn)
        .setDescription(
            "Mark Convs that support CMConv, & ops on Input that require C-Major input tensor"
        );

        MV_REGISTER_PASS(AddPermuteToNonCMConvPaths)
        .setFunc(addPermuteToNonCMConvPathsFcn)
        .setDescription(
            "For C-Major mode, input paths to Z-Major ops need input tensor permuted from C-Major to Z-Major"
        );
    }

}

void replacementOpsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model,
                       mv::TargetDescriptor&, mv::Element&, mv::Element&)
{
    fullyConnectedAsConv2DFcn(pass, model);
    replacePoolReshapePatternFcn(pass, model);
    replaceExpReduceSumMultipyFcn(pass, model);
    replaceLargeKernelsFcn(pass, model);
    replaceLargeStridesFcn(pass, model);
    replaceAsymmetricStridesFcn(pass, model);
    topKAsArgMaxFcn(pass, model);
    //interpAsAvgPoolingFcn(pass, model); for now we are using SW layer
    interpAsDepthConvFcn(pass, model);
    averageAsDepthWiseFcn(pass, model);
    scaleAsDepthwiseFcn(pass, model);
    flattenAsReshapeFcn(pass, model);
    replaceConcatOfPopulatedTensorsFcn(pass, model);
    reorgYoloAsConvConcatFcn(pass, model);
    insertPermuteBeforeDetFcn(pass, model);
    replacePermuteAsReshape(pass, model);
}

void insertPermuteBeforeDetFcn(const mv::pass::PassEntry&, mv::ComputationModel& model)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)

    mv::OpModel om(model);
    mv::DataModel dm(model);

    auto detectionOps = om.getOps("DetectionOutput");

    for (auto& opIt : detectionOps)
    {
        auto confData = opIt->getInputTensor(1);
        auto parent = om.getSourceOp(confData);

        std::vector<mv::Data::OpListIterator> opsToLink;
        std::vector<std::size_t> inputSlots;
        std::vector<mv::Data::FlowSiblingIterator> flowsToRemove;

        auto sourceFlowStart = parent.leftmostOutput();

        for (mv::Data::FlowSiblingIterator sinkFlow(sourceFlowStart); sinkFlow != om.flowEnd(); ++sinkFlow)
        {
            opsToLink.push_back(sinkFlow.sink());
            inputSlots.push_back(sinkFlow->get<std::size_t>("sinkInput"));
            flowsToRemove.push_back(sinkFlow);
        }

        for (unsigned flowIdx = 0; flowIdx < flowsToRemove.size(); flowIdx++)
        {
            om.undefineFlow(flowsToRemove[flowIdx]);
        }

        double inf = std::numeric_limits<double>::infinity();
        uint64_t numClasses = opIt->get<int64_t>("num_classes");
        auto totalSize = confData->getShape().totalSize();
        mv::Shape newShape({numClasses, totalSize / numClasses, 1, 1});
        mv::Data::TensorIterator reshapeBeforePermuteData = om.reshape("reshapeBeforePermute", confData, newShape);
        reshapeBeforePermuteData->setQuantParams({{0}, {1}, {-inf}, {inf}});

        std::string newOrder = "NCWH";
        mv::Data::TensorIterator transposedData = om.permute("new_permute", reshapeBeforePermuteData, mv::Order(newOrder));
        transposedData->setQuantParams({{0}, {1}, {-inf}, {inf}});

        mv::Data::TensorIterator reshapeAfterPermuteData = om.reshape("reshapeAfterPermute", transposedData, confData->getShape());
        reshapeAfterPermuteData->setQuantParams({{0}, {1}, {-inf}, {inf}});

        for(unsigned op = 0 ; op < opsToLink.size(); ++op)
        {
            opsToLink[op]->setInputTensor(reshapeAfterPermuteData, inputSlots[op], false);
            om.defineFlow(reshapeAfterPermuteData, opsToLink[op], inputSlots[op]);
        }

        auto reshapeBeforeOp = om.getSourceOp(reshapeBeforePermuteData);
        auto permuteOp = om.getSourceOp(transposedData);
        auto reshapeAfterOp = om.getSourceOp(reshapeAfterPermuteData);
        if(parent->hasAttr("opId"))
        {
            unsigned currentOpId = parent->get<unsigned>("opId");
            reshapeBeforeOp->set<unsigned>("opId", currentOpId);
            permuteOp->set<unsigned>("opId", currentOpId);
            reshapeAfterOp->set<unsigned>("opId", currentOpId);
        }
        auto outputMemoryLocation = parent->getOutputTensor(0)->get<mv::Tensor::MemoryLocation>("Location");
        reshapeBeforePermuteData->set<mv::Tensor::MemoryLocation>("Location", outputMemoryLocation);
        transposedData->set<mv::Tensor::MemoryLocation>("Location", outputMemoryLocation);
        reshapeAfterPermuteData->set<mv::Tensor::MemoryLocation>("Location", outputMemoryLocation);
    }
}

void replacePermuteAsReshape(const mv::pass::PassEntry&, mv::ComputationModel& model)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    using namespace mv;

    OpModel om(model);
    auto permuteOps = om.getOps("Permute");

    for (auto& opIt : permuteOps)
    {
        auto inputShape = opIt->getInputTensor(0)->getShape();
        auto outputShape = opIt->getOutputTensor(0)->getShape();

        std::vector<size_t> inputRealShape, outputRealShape;
        for (int i = 0; i < 4; i++)
        {
            if(inputShape[i] != 1)
            {
                inputRealShape.push_back(inputShape[i]);
            }
            if(outputShape[i] != 1)
            {
                outputRealShape.push_back(outputShape[i]);
            }
        }

        if(inputRealShape.size() == outputRealShape.size())
        {
            bool match = true;
            for(unsigned i = 0; i < inputRealShape.size(); i++)
            {
                match &= (inputRealShape[i] == outputRealShape[i]);
            }

            if(match)
            {
                // do permute when 2 dimensions' size equal.
                std::set<size_t> noRepeatShape(inputRealShape.begin(), inputRealShape.end());
                match &= (noRepeatShape.size()==inputRealShape.size());
                // do permute when next step is reshape
                mv::DataModel dm(model);
                auto nextOp = mv::findSinkLayers(dm, opIt->getOutputTensor(mv::IO_TENSOR_OUTPUT))[0];
                if (nextOp->getOpType() == "Reshape") {
                    match = false;
                }
            }
            if(match)
            {
                auto outputMemoryLocation = opIt->getOutputTensor(0)->get<mv::Tensor::MemoryLocation>("Location");
                auto sourceTensor = opIt->getInputTensor(0);
                auto parentOpIt = om.getSourceOp(sourceTensor);
                auto outputTensorType = opIt->getOutputTensor(0)->getDType();
                auto outputTensorQuantizationParams = opIt->getOutputTensor(0)->getQuantParams();
                auto reshape = om.reshape(opIt->getName() + "_reshape", sourceTensor, outputShape);
                reshape->setDType(outputTensorType);
                reshape->setQuantParams(outputTensorQuantizationParams);
                auto reshapeOp = om.getSourceOp(reshape);

                if(opIt->hasAttr("opId"))
                {
                    unsigned currentOpId = opIt->get<unsigned>("opId");
                    reshapeOp->set<unsigned>("opId", currentOpId);
                }

                linkNewOperationsReplacement(parentOpIt, reshape, om, opIt);
                reshape->set<mv::Tensor::MemoryLocation>("Location", outputMemoryLocation);
            }
        }
    }
}

void fullyConnectedAsConv2DFcn(const mv::pass::PassEntry&, mv::ComputationModel& model)
{

    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    using namespace mv;

    OpModel om(model);

    auto fullyConnectedOps = om.getOps("FullyConnected");

    for (auto& opIt : fullyConnectedOps)
    {
        auto outputMemoryLocation = opIt->getOutputTensor(0)->get<mv::Tensor::MemoryLocation>("Location");
        auto sourceTensor = opIt->getInputTensor(0);
        auto parentOpIt = om.getSourceOp(sourceTensor);
        auto weightsData = opIt->getInputTensor(1)->getData();
        auto inputShape = sourceTensor->getShape();
        mv::QuantizationParams weightsTensorQuantizationParams = {{},{},{},{}};
        mv::QuantizationParams outputTensorQuantizationParams = {{},{},{},{}};

        if (opIt->getInputTensor(1)->isQuantized())
        {
            weightsTensorQuantizationParams = opIt->getInputTensor(1)->getQuantParams();
            outputTensorQuantizationParams  = opIt->getOutputTensor(0)->getQuantParams();
        }

        auto weights = om.constantDataElement(opIt->getName() + "_weights",
                                              weightsData,
                                              {FULLY_CONNECTED_KERNEL, FULLY_CONNECTED_KERNEL, inputShape[mv::IO_CHANNEL_DIMENSION],
                                              opIt->getInputTensor(1)->getShape()[mv::IO_HEIGHT_DIMENSION]},
                                              opIt->getInputTensor(1)->getDType(),
                                              mv::Order::getZMajorID(4));
        weights->setQuantParams(weightsTensorQuantizationParams);
        auto outputTensorType = opIt->getOutputTensor(0)->get<mv::DType>("dType");

        auto conv2D = om.conv(opIt->getName() + "_2DConv", sourceTensor, weights, {1, 1}, {0, 0, 0, 0}, 1, 1);
        conv2D->setDType(outputTensorType);
        conv2D->setQuantParams(outputTensorQuantizationParams);
        if (opIt->hasAttr("bias"))
        {
            auto biasTensorName = opIt->get<std::string>("bias");
            om.addAttr(om.getSourceOp(conv2D), "bias", biasTensorName);
        }

        auto convOp = om.getSourceOp(conv2D);
        auto weightsOp = om.getSourceOp(weights);

        if(opIt->hasAttr("opId"))
        {
            unsigned currentOpId = opIt->get<unsigned>("opId");
            weightsOp->set<unsigned>("opId", currentOpId);
            convOp->set<unsigned>("opId", currentOpId);
        }

        linkNewOperationsReplacement(parentOpIt, conv2D, om, opIt);
        conv2D->set<mv::Tensor::MemoryLocation>("Location", outputMemoryLocation);
    }
}

//NOTE: This pass will handle cases that we have Convs -> Eltwise for testing ResNet first of all....
//General solution dequantize the input Tensors of these special Elwise, even with sw de-quantize
void handleEltWiseDifferentScales(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)

    mv::OpModel om(model);

    auto eltWiseOps = om.getOps("Eltwise");

    for (auto& opIt : eltWiseOps)
    {
        auto eltwiseType = opIt->get<std::string>("eltwiseType");
        if(eltwiseType == "Multiply" || eltwiseType == "Divide")
            continue;

        auto firstEltwiseInputTensor = opIt->getInputTensor(0);
        auto secondEltwiseInputTensor = opIt->getInputTensor(1);

        mv::QuantizationParams firstEltwiseInputTensorQuantizationParams = firstEltwiseInputTensor->getQuantParams();
        mv::QuantizationParams secondEltwiseInputTensorQuantizationParams = secondEltwiseInputTensor->getQuantParams();

        auto scale1 = firstEltwiseInputTensorQuantizationParams.getScale();
        auto scale2 = secondEltwiseInputTensorQuantizationParams.getScale();

        auto size = scale1.size();
        auto size2 = scale2.size();

        if (size != size2 && (size == 0 || size2 == 0)) {
            // one of inputs does not have quant params
            opIt->set<bool>("softwareExecuted", true);
            return;
        }

        std::vector <double> scaleDifference(size), absRelativeErrorScale(size), relativeErrorScale(size);
        std::transform(scale1.begin(),
                       scale1.end(),
                       scale2.begin(), scaleDifference.begin(), std::minus<double>());

        double (*fabs)(double) = &std::abs;
        std::transform(scaleDifference.begin(), scaleDifference.end(),
                       scale1.begin(), relativeErrorScale.begin(),
                       std::divides<double>());
        std::transform(relativeErrorScale.begin(),relativeErrorScale.end(),
                       absRelativeErrorScale.begin(), fabs);
        for (auto it = absRelativeErrorScale.begin(); it != absRelativeErrorScale.end(); it++)
        {
            if (*it > 0.01)
                opIt->set<bool>("softwareExecuted", true);
        }
    }
}

static void markCMCompatibleConvsFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)

    mv::OpModel om(model);

    // Skip pass if C-Major isn't enabled
    if (!(om.getGlobalConfigParams()->get<bool>("enable_channel_major_conv")))
        return;

    //set the CM Conv flag to 'False'. But if there are any convs exist that support CMConv, then set the flag to true
    om.getGlobalConfigParams()->set<bool>("enable_channel_major_conv", false);

    // Mark CMConv-compatible convolutions
    auto convs = om.getOps("Conv");
    for(auto& conv : convs)
    {
        auto supports_CMConv = (*conv).supportsCMConv();
        (*conv).set<bool>("supportsCM", supports_CMConv);
        if(supports_CMConv)
            om.getGlobalConfigParams()->set<bool>("enable_channel_major_conv", true);
    }
}

static void addPermuteToNonCMConvPathsFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)

    mv::OpModel om(model);

    // Skip pass if C-Major isn't enabled
    if (!(om.getGlobalConfigParams()->get<bool>("enable_channel_major_conv")))
        return;

    // Change Input op to C-Major
    auto inputOp = om.getOps("Input")[0];
    inputOp->set<mv::Order>("order", mv::Order("NCHW"));
    inputOp->getOutputTensor(0)->setOrder(mv::Order("NCHW"));

    // Change input tensor to C-Major
    auto ops = om.getOps();
    for(auto& opIt : ops)
        if (opIt->hasAttr("CMinput") && opIt->get<bool>("CMinput"))
            for (auto& input : opIt->getInputTensor())
                input->setOrder(mv::Order("NCHW"));

    // Change CMConv input to C-Major
    auto convs = om.getOps("Conv");
    for(auto& conv : convs)
        if (conv->supportsCMConv())
            conv->getInputTensor(0)->setOrder(mv::Order("NCHW"));

    // Check for any Z-Major ops needing permute
    auto permute_needed = false;
    for (auto sinkFlow = inputOp.leftmostOutput(); sinkFlow != om.flowEnd(); ++sinkFlow)
    {
        if (!(sinkFlow.sink()->hasAttr("CMinput") && sinkFlow.sink()->get<bool>("CMinput")))
            permute_needed = true;
    }

    if (!(permute_needed))
        return;

    // Add permute between input & Z-Major ops
    auto outputTensor = inputOp->getOutputTensor(0);

    std::vector<mv::Data::OpListIterator> opsToLink;
    std::vector<std::size_t> inputSlots;
    std::vector<mv::Data::FlowSiblingIterator> flowsToRemove;

    for (auto sinkFlow = inputOp.leftmostOutput(); sinkFlow != om.flowEnd(); ++sinkFlow)
    {
        if (!(sinkFlow.sink()->hasAttr("CMinput") && sinkFlow.sink()->get<bool>("CMinput")))
        {
            opsToLink.push_back(sinkFlow.sink());
            inputSlots.push_back(sinkFlow->get<std::size_t>("sinkInput"));
            flowsToRemove.push_back(sinkFlow);
        }
    }

    for (unsigned flowIdx = 0; flowIdx < flowsToRemove.size(); flowIdx++)
    {
        om.undefineFlow(flowsToRemove[flowIdx]);
    }

    auto transposedData = om.permute(inputOp->getName() + "_permute", outputTensor, mv::Order("NCHW"));
    transposedData->setQuantParams(outputTensor->getQuantParams());

    for(unsigned op = 0 ; op < opsToLink.size(); ++op)
    {
        opsToLink[op]->setInputTensor(transposedData, inputSlots[op], false);
        om.defineFlow(transposedData, opsToLink[op], inputSlots[op]);
    }

    auto permuteOp = om.getSourceOp(transposedData);
    if(inputOp->hasAttr("opId"))
    {
        unsigned currentOpId = inputOp->get<unsigned>("opId");
        permuteOp->set<unsigned>("opId", currentOpId);
    }

    transposedData->setOrder(mv::Order("NHWC"));
    permuteOp->set<bool>("ZMoutput", true);
}

void interpAsAvgPoolingFcn(const mv::pass::PassEntry&, mv::ComputationModel& model)
{

    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)

    mv::OpModel om(model);
    mv::DataModel dm(model);

    auto interpOps = om.getOps("Interp");

    for (auto& opIt : interpOps)
    {
        auto sourceTensor = opIt->getInputTensor(0);
        auto inputShape = sourceTensor->getShape();
        auto outputTensor = opIt->getOutputTensor(0);
        auto outputShape = outputTensor->getShape();

        auto inWidth = inputShape[mv::IO_WIDTH_DIMENSION];
        auto inHeight = inputShape[mv::IO_HEIGHT_DIMENSION];
        auto outWidth = outputShape[mv::IO_WIDTH_DIMENSION];
        auto outHeight = outputShape[mv::IO_HEIGHT_DIMENSION];
        if (inWidth > outWidth && inHeight > outHeight &&
            (inHeight % outHeight == 0) && (inWidth % outWidth == 0) &&
            (inHeight / outHeight) == inWidth / outWidth)
        {
            auto outputMemoryLocation = opIt->getOutputTensor(0)->get<mv::Tensor::MemoryLocation>("Location");
            unsigned short factor = inHeight / outHeight;
            auto parentOpIt = om.getSourceOp(sourceTensor);

            std::array<unsigned short, 2> kSize = {factor, factor};
            std::array<unsigned short, 2> stride = {factor, factor};
            auto name = opIt->getName();

            auto quantParams = mv::QuantizationParams::empty();
            if (sourceTensor->isQuantized())
                quantParams = outputTensor->getQuantParams();

            auto avgPool = om.averagePool(name + "_AvgPool", sourceTensor, kSize, stride, {0,0,0,0}, false);
            avgPool->setQuantParams(quantParams);

            auto avgOp = om.getSourceOp(avgPool);

            if (opIt->hasAttr("opId"))
            {
                unsigned currentOpId = opIt->get<unsigned>("opId");
                avgOp->set<unsigned>("opId", currentOpId);
            }
            avgOp->set<mv::Tensor::MemoryLocation>("Location", outputMemoryLocation);
            linkNewOperationsReplacement(parentOpIt, avgPool, om, opIt);
        }
    }
}

void interpAsDepthConvFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model)
{

    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)

    mv::OpModel om(model);
    mv::DataModel dm(model);

    auto interpOps = om.getOps("Interp");

    for (auto& opIt : interpOps)
    {
        auto inputShape = opIt->getInputTensor(0)->getShape();
        auto outputShape = opIt->getOutputTensor(0)->getShape();
        auto outQuantParams  = opIt->getOutputTensor(0)->get<mv::QuantizationParams>("quantParams");

        auto parentOpIt = om.getSourceOp(opIt->getInputTensor(0));

        auto sourceTensor = parentOpIt->getOutputTensor(0);
        auto inQuantParams = sourceTensor->get<mv::QuantizationParams>("quantParams");

        if ((inputShape == outputShape) && !(isEqual(inQuantParams, outQuantParams) || outQuantParams.isNeutral()))
        {

            parentOpIt = om.getSourceOp(opIt->getInputTensor(0));
            auto outputMemoryLocation = opIt->getOutputTensor(0)->get<mv::Tensor::MemoryLocation>("Location");

            pass.log(mv::Logger::MessageType::Debug, "Replacing with DW requanitze");

            //FIND THE APPROPRIATE FLOW
            mv::Data::TensorIterator weights;
            std::vector<int64_t> zp = { 0 };
            std::vector<double> min = { 1 };
            std::vector<double> max = { 1 };
            std::vector<double> scale = { 1 };
            mv::QuantizationParams weightsQuantParams(zp, scale, min, max);
            int64_t weightsValue = 1;
            std::vector<int64_t> weightsData(sourceTensor->getShape()[mv::IO_CHANNEL_DIMENSION], weightsValue);
            weights = om.constantInt("",
                                     weightsData,
                                     {1, 1, sourceTensor->getShape()[mv::IO_CHANNEL_DIMENSION], 1},
                                     mv::DType("UInt8"),
                                     mv::Order(mv::Order::getRowMajorID(4)));
            weights->setQuantParams(weightsQuantParams);
            auto reQuantizeDepthwise = om.depthwiseConv(opIt->getName() + "_DepthwiseRequantize",
                                                        sourceTensor, weights, {1,1}, {0, 0, 0, 0}, 1);
            reQuantizeDepthwise->setQuantParams({outQuantParams.getZeroPoint(),outQuantParams.getScale(),{},{}});
            reQuantizeDepthwise->setDType(mv::DType("UInt8"));
            auto reQuantizeDepthwiseOp = om.getSourceOp(reQuantizeDepthwise);
            auto weightsOp = om.getSourceOp(weights);
            reQuantizeDepthwiseOp->set<unsigned>("opId", opIt->get<unsigned>("opId"));
            weightsOp->set<unsigned>("opId", opIt->get<unsigned>("opId"));
            linkNewOperationsReplacement(parentOpIt, reQuantizeDepthwise, om, opIt);
            reQuantizeDepthwise->set<mv::Tensor::MemoryLocation>("Location", outputMemoryLocation);

        }
    }
}

void reorgYoloAsConvConcatFcn(const mv::pass::PassEntry&, mv::ComputationModel& model)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)

    mv::OpModel om(model);

    auto reorgYoloOps = om.getOps("ReorgYolo");

    for (auto &opIt : reorgYoloOps)
    {
        auto outputMemoryLocation = opIt->getOutputTensor(0)->get<mv::Tensor::MemoryLocation>("Location");
        auto sourceTensor = opIt->getInputTensor(0);
        auto parentOpIt = om.getSourceOp(sourceTensor);
        auto inputShape = sourceTensor->getShape();
        unsigned short stride = opIt->get<unsigned>("stride");
        unsigned C = inputShape[2];
        mv::QuantizationParams weightsTensorQuantizationParams = {{0}, {1.}, {}, {}};
        mv::QuantizationParams outputTensorQuantizationParams = {{}, {}, {}, {}};

        if (opIt->getInputTensor(0)->isQuantized())
        {
            outputTensorQuantizationParams = opIt->getOutputTensor(0)->get<mv::QuantizationParams>("quantParams");
        }

        std::vector<mv::Data::TensorIterator> convOutputs;

        int kernelStep = stride * stride;
        for (unsigned rowIdx = 0; rowIdx < stride; rowIdx++)
        {
            for (unsigned colIdx = 0; colIdx < stride; colIdx++)
            {
                int kernelOffset = colIdx + stride * rowIdx;
                mv::Data::TensorIterator weight;
                if (sourceTensor->isFloatingPointType())
                {
                    std::vector<double> weightData(C * stride * stride, 0.);
                    for (unsigned cIdx = 0; cIdx < C; cIdx++)
                    {
                        weightData[kernelOffset + cIdx * kernelStep] = 1.;
                    }
                    weight = om.constant("", weightData, {stride, stride, C, 1},
                                         mv::DType("Float64"),
                                         mv::Order(mv::Order::getColMajorID(4)));
                }
                else
                {
                    std::vector<int64_t> weightData(C * stride * stride, 0);
                    for (unsigned cIdx = 0; cIdx < C; cIdx++)
                    {
                        weightData[kernelOffset + cIdx * kernelStep] = 1ll;
                    }
                    weight = om.constantInt("", weightData, {stride, stride, C, 1},
                                            sourceTensor->getDType(),
                                            mv::Order(mv::Order::getColMajorID(4)));
                }
                weight->setQuantParams(weightsTensorQuantizationParams);
                auto gridConv = om.depthwiseConv(opIt->getName() + "_DepthwiseConvGrid" + ":_" + std::to_string(rowIdx) + "_" + std::to_string(colIdx) + "_",
                                                sourceTensor, weight,
                                                {stride, stride},
                                                {0, 0, 0, 0}, 1);
                gridConv->setQuantParams(outputTensorQuantizationParams);
                auto convOp = om.getSourceOp(gridConv);
                auto weightOp = om.getSourceOp(weight);
                if (opIt->hasAttr("opId"))
                {
                    unsigned currentOpId = opIt->get<unsigned>("opId");
                    convOp->set<unsigned>("opId", currentOpId);
                    weightOp->set<unsigned>("opId", currentOpId);
                }
                convOutputs.push_back(gridConv);
            }
        }
        auto concat = om.concat(opIt->getName() + "_Concat", convOutputs, "C");
        concat->setQuantParams(outputTensorQuantizationParams);
        auto concatOp = om.getSourceOp(concat);
        if (opIt->hasAttr("opId"))
        {
            unsigned currentOpId = opIt->get<unsigned>("opId");
            concatOp->set<unsigned>("opId", currentOpId);
        }
        linkNewOperationsReplacement(parentOpIt, concat, om, opIt);
        concat->set<mv::Tensor::MemoryLocation>("Location", outputMemoryLocation);
    }
}

void scaleAsDepthwiseFcn(const mv::pass::PassEntry&, mv::ComputationModel& model)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)

    mv::OpModel om(model);

    auto scaleOps = om.getOps("Scale");

    for (auto& opIt : scaleOps)
    {
        auto outputMemoryLocation = opIt->getOutputTensor(0)->get<mv::Tensor::MemoryLocation>("Location");
        auto sourceTensor = opIt->getInputTensor(0);
        auto parentOpIt = om.getSourceOp(sourceTensor);
        auto weightsData = opIt->getInputTensor(1)->getData();
        auto inputShape = sourceTensor->getShape();
        auto weightsTensorQuantizationParams = opIt->getInputTensor(1)->getQuantParams();
        auto outputTensorQuantizationParams = opIt->getOutputTensor(0)->getQuantParams();
        auto outputTensorType = opIt->getOutputTensor(0)->getDType();

        if (parentOpIt->getOpType() == "Conv")
            continue;

        auto finalDType = opIt->getInputTensor(1)->getDType();
        if (sourceTensor->getDType() == mv::DType("UInt8")) {
            finalDType = sourceTensor->getDType();
        }

        auto weights = om.constantDataElement(opIt->getName() + "_weights",
                                              weightsData,
                                              {FULLY_CONNECTED_KERNEL, FULLY_CONNECTED_KERNEL, inputShape[mv::IO_CHANNEL_DIMENSION], 1},
                                              finalDType,
                                              mv::Order::getZMajorID(4));
        weights->setQuantParams(weightsTensorQuantizationParams);

        auto conv2D = om.depthwiseConv(opIt->getName() + "_DepthwiseConv", sourceTensor, weights, {1, 1}, {0, 0, 0, 0}, 1);
        conv2D->setDType(outputTensorType);
        conv2D->setQuantParams(outputTensorQuantizationParams);

        if (opIt->hasAttr("bias"))
        {
            auto biasTensorName = opIt->get<std::string>("bias");
            om.addAttr(om.getSourceOp(conv2D), "bias", biasTensorName);
        }

        auto convOp = om.getSourceOp(conv2D);
        auto weightsOp = om.getSourceOp(weights);

        if(opIt->hasAttr("opId"))
        {
            unsigned currentOpId = opIt->get<unsigned>("opId");
            weightsOp->set<unsigned>("opId", currentOpId);
            convOp->set<unsigned>("opId", currentOpId);
        }

        linkNewOperationsReplacement(parentOpIt, conv2D, om, opIt);
        conv2D->set<mv::Tensor::MemoryLocation>("Location", outputMemoryLocation);
        convOp->set<bool>("isScaleShift", true);
    }
}

void averageAsDepthWiseFcn(const mv::pass::PassEntry&, mv::ComputationModel& model)
{

    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)

    mv::OpModel om(model);
    mv::DataModel dm(model);

    auto averagePoolOps = om.getOps("AveragePool");

    for (auto& opIt : averagePoolOps)
    {
        auto sourceTensor = opIt->getInputTensor(0);
        auto outputMemoryLocation = opIt->getOutputTensor(0)->get<mv::Tensor::MemoryLocation>("Location");

        auto parentOpIt = om.getSourceOp(sourceTensor);

        auto inputShape = sourceTensor->getShape();
        std::array<unsigned short, 2> kSize = opIt->get<std::array<unsigned short, 2>>("kSize");
        std::array<unsigned short, 2> stride = opIt->get<std::array<unsigned short, 2>>("stride");
        std::array<unsigned short, 4> padding = opIt->get<std::array<unsigned short, 4>>("padding");

        unsigned int total_shape = 1 * inputShape[mv::IO_CHANNEL_DIMENSION] * kSize[1] * kSize[0];
        double scaleValue = 1/double(kSize[0] * kSize[1]);

        unsigned short channel_multiplier = 1;

        auto name = opIt->getName();
        mv::Data::TensorIterator weights;
        std::vector<int64_t> zp = { 0 };
        std::vector<double> min = { 1 };
        std::vector<double> max = { 1 };

        std::vector<double> scale(1, scaleValue);
        mv::QuantizationParams weightsQuantParams(zp, scale, min, max);
        double inf = std::numeric_limits<double>::infinity();
        mv::QuantizationParams neutralWeightsQuantParams = {{0},{1.0},{-inf},{inf}};

        if (sourceTensor->isFloatingPointType())
        {
            double weightsValue = scaleValue;
            std::vector<double> weightsData(total_shape, weightsValue);
            //NOTE: For FP, weights quant params not used - put divisor in weights directly instead of scale
            weights = om.constant("",
                                  weightsData,
                                  {kSize[0], kSize[1], inputShape[mv::IO_CHANNEL_DIMENSION], channel_multiplier},
                                  mv::DType("Float64"),
                                  mv::Order(mv::Order::getColMajorID(4)));
            weights->setQuantParams(neutralWeightsQuantParams);
        }
        else
        {
            std::vector<int64_t> weightsData(total_shape, 1ll);
            // If the input model is quantized, then the replacement pass needs to create
            // quantization params for the weights parameter of the depthwise convolution.
            weights = om.constantInt("",
                                     weightsData,
                                     {kSize[0], kSize[1], inputShape[mv::IO_CHANNEL_DIMENSION], channel_multiplier},
                                     mv::DType("UInt8"),
                                     mv::Order(mv::Order::getColMajorID(4)));
            weights->setQuantParams(weightsQuantParams);
        }

        auto quantParams = mv::QuantizationParams::empty();
        if (sourceTensor->isQuantized())
            quantParams = opIt->getOutputTensor(0)->getQuantParams();

        //Check the last argument name!!!
        mv::Data::TensorIterator depthwise_conv = om.depthwiseConv(name + "_DepthwiseConv", sourceTensor, weights, stride, padding, 1);
        depthwise_conv->setQuantParams(quantParams);

        auto depthwiseConvOp = om.getSourceOp(depthwise_conv);
        auto weightsOp = om.getSourceOp(weights);

        if(opIt->hasAttr("opId"))
        {
            unsigned currentOpId = opIt->get<unsigned>("opId");
            weightsOp->set<unsigned>("opId", currentOpId);
            depthwiseConvOp->set<unsigned>("opId", currentOpId);
        }
        depthwise_conv->set<mv::Tensor::MemoryLocation>("Location", outputMemoryLocation);
        linkNewOperationsReplacement(parentOpIt, depthwise_conv, om, opIt);
    }
}

void topKAsArgMaxFcn(const mv::pass::PassEntry&, mv::ComputationModel& model)
{

    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    using namespace mv;

    OpModel om(model);

    auto topKOps = om.getOps("TopK");

    for (auto& opIt : topKOps)
    {
        //Check if first output has no data flows
        //check if mode and sort are different from what is supported by argmax
        auto firstoutput = opIt->getOutputTensor(0);
        auto attrs = opIt->getAttrs();

        if(firstoutput->hasAttr("flows") ||
           (attrs.at("mode").get<std::string>().compare("max") != 0) ||
           (attrs.at("sort").get<std::string>().compare("index") != 0))
            throw ArgumentError("topKAsArgMaxFcn", "flows", "", "cannot convert topK to argMax");// TODO replace with continue when we add topK support

        auto outputMemoryLocation = opIt->getOutputTensor(1)->get<mv::Tensor::MemoryLocation>("Location");

        auto sourceTensor = opIt->getInputTensor(0);
        auto parentOpIt = om.getSourceOp(sourceTensor);
        auto inputShape = sourceTensor->getShape();

        auto out_max_val = 0; //only support this for this conversion
        auto top_k = attrs.at("top_k").get<int64_t>();
        auto axis = attrs.at("axis").get<int64_t>();

        auto quantParams = firstoutput->getQuantParams();
        auto dType = firstoutput->getDType();

        auto argmax = om.argmax(opIt->getName() + "_argmax", sourceTensor, out_max_val, top_k, axis);
        argmax->setQuantParams(quantParams);
        argmax->setDType(dType);

        auto argmaxOp = om.getSourceOp(argmax);

        if(opIt->hasAttr("opId"))
        {
            unsigned currentOpId = opIt->get<unsigned>("opId");
            argmaxOp->set<unsigned>("opId", currentOpId);
        }
        linkNewOperationsReplacement(parentOpIt, argmax, om, opIt);
        argmax->set<mv::Tensor::MemoryLocation>("Location", outputMemoryLocation);
    }
}

void flattenAsReshapeFcn(const mv::pass::PassEntry&, mv::ComputationModel& model)
{

    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    using namespace mv;

    OpModel om(model);

    auto flattenOps = om.getOps("Flatten");

    for (auto& opIt : flattenOps)
    {
        auto outputMemoryLocation = opIt->getOutputTensor(0)->get<mv::Tensor::MemoryLocation>("Location");

        auto sourceTensor = opIt->getInputTensor(0);
        auto parentOpIt = om.getSourceOp(sourceTensor);
        auto inputShape = sourceTensor->getShape();

        auto outputTensorType = opIt->getOutputTensor(0)->getDType();
        auto outputShape = opIt->getOutputTensor(0)->getShape();
        auto outputOrder = opIt->getOutputTensor(0)->getOrder();

        auto reshape = om.reshape(opIt->getName() + "_reshape", sourceTensor, outputShape);
        reshape->setDType(outputTensorType);

        auto reshapeOp = om.getSourceOp(reshape);

        if(opIt->hasAttr("opId"))
        {
            unsigned currentOpId = opIt->get<unsigned>("opId");
            reshapeOp->set<unsigned>("opId", currentOpId);
        }
        linkNewOperationsReplacement(parentOpIt, reshape, om, opIt);
        reshape->set<mv::Tensor::MemoryLocation>("Location", outputMemoryLocation);
    }
}

std::vector<std::pair<unsigned short, unsigned short>> getFactorsList(unsigned short n)
{
    std::vector<std::pair<unsigned short, unsigned short>> factors;
    for(int i = 2; i <= sqrt(n); i++)
    {
        if(n % i == 0)
        {
            // factors.push_back(std::make_pair(i, n/i)); // smaller, larger
            factors.push_back(std::make_pair(n/i, i)); // larger, smaller
        }
    }
    return factors;
}

std::pair<unsigned short, unsigned short> getFactors(unsigned short kernelSize)
{
    std::vector<std::pair<unsigned short, unsigned short>> allFactors = getFactorsList(kernelSize);
    std::pair<unsigned short, unsigned short> factors;
    if (allFactors.empty()) // Original kernel size IS PRIME
    {
        // Use the factors for kernel size - 1, this is guaranteed to have factors, as it will be an even number > 11
        allFactors = getFactorsList(kernelSize - 1);
        factors = allFactors.back(); // Get the most equal factors

        // These factors are for ksize - 1, so increase smaller factor by 1
        if( factors.first == factors.second)
            factors.first++;
        else
            factors.second++;


        if ( (factors.first * factors.second > (kernelSize + factors.first/2) ) ||
             (factors.first * factors.second > (kernelSize + factors.second/2) ) )
        {
            // Use the factors for kernel size + 1,  an even number as we added 1 to a prime number, first number greater than the prime number
            allFactors = getFactorsList(kernelSize + 1);
            factors = allFactors.back(); // Get the most equal factors
        }
    }
    else // Original kernel size NOT PRIME
    {
        factors = allFactors.back(); // Get the most equal factors
    }
    return factors;
}

unsigned short getPad(std::pair<unsigned short, unsigned short> factors, size_t inputShape, size_t outputShape)
{
    double updatedOutput = outputShape * factors.second;
    unsigned short pad = updatedOutput*factors.first -  inputShape;

    return pad;
}

mv::Data::TensorIterator createPartialDepthwise(mv::OpModel & om, mv::Data::OpListIterator opIt, mv::Data::TensorIterator sourceTensor,
                                                std::string name, double scaleValue, std::array<unsigned short, 2> newKernel,
                                                std::array<unsigned short, 4> padding, bool quantRequired)
{
    auto inputShape = sourceTensor->getShape();

    // Calculate strides based on new kernel sizes
    std::array<unsigned short, 2> stride = {newKernel[0], newKernel[1]};

    unsigned int total_shape = 1 * inputShape[mv::IO_CHANNEL_DIMENSION] * newKernel[0] * newKernel[1];

    unsigned short channel_multiplier = 1;

    mv::Data::TensorIterator weights;
    std::vector<int64_t> zp = { 0 };
    std::vector<double> min = { 1 };
    std::vector<double> max = { 1 };
    // Both depthwise will take 1/original_kernel_size as a scale (if was 1 dw would be kernel^2)
    // Note: For non-prime kernels, could take scale of each exactly replacing ap/dw,
    // but using original kernel for scale improves observed accuracy
    std::vector<double> scale(1, scaleValue);
    mv::QuantizationParams weightsQuantParams(zp, scale, min, max);

    // Create weights tensor
    if (sourceTensor->isFloatingPointType())
    {
        double weightsValue = scaleValue;
        std::vector<double> weightsData(total_shape, weightsValue);
        //NOTE: For FP, weights quant params not used - put divisor in weights directly instead of scale
        weights = om.constant("", weightsData,
                              {newKernel[0], newKernel[1], inputShape[mv::IO_CHANNEL_DIMENSION], channel_multiplier},
                              mv::DType("Float64"), mv::Order(mv::Order::getColMajorID(4)));
    }
    else
    {
        std::vector<int64_t> weightsData(total_shape, 1ll);
        // If the input model is quantized, then the replacement pass needs to create
        // quantization params for the weights parameter of the depthwise convolution.
        weights = om.constantInt("", weightsData,
                                 {newKernel[0], newKernel[1], inputShape[mv::IO_CHANNEL_DIMENSION], channel_multiplier},
                                 sourceTensor->getDType(), mv::Order(mv::Order::getColMajorID(4)));
        weights->setQuantParams(weightsQuantParams);
    }

    double inf = std::numeric_limits<double>::infinity();
    mv::QuantizationParams quantParams({{0}, {1}, {-inf}, {inf}});
    if (sourceTensor->isQuantized() && quantRequired)
        quantParams = opIt->getOutputTensor(0)->getQuantParams();

    // Create depthwise conv (default dilation factor)
    auto depthwise_conv = om.depthwiseConv(name, sourceTensor, weights, stride, padding, 1);
    depthwise_conv->setQuantParams(quantParams);

    // Add depthwise conv to op model
    auto depthwiseOp = om.getSourceOp(depthwise_conv);
    auto weightsOp = om.getSourceOp(weights);

    if(opIt->hasAttr("opId"))
    {
        unsigned currentOpId = opIt->get<unsigned>("opId");
        weightsOp->set<unsigned>("opId", currentOpId);
        depthwiseOp->set<unsigned>("opId", currentOpId);
    }
    auto outputMemoryLocation = opIt->getOutputTensor(0)->get<mv::Tensor::MemoryLocation>("Location");
    depthwise_conv->set<mv::Tensor::MemoryLocation>("Location", outputMemoryLocation);

    return depthwise_conv;
}

bool matchPattern(const std::vector<std::string>& pattern, mv::Data::OpListIterator it, mv::ComputationModel& model) {
    mv::OpModel om(model);
    auto opIt = it;

    for (auto& layer : pattern) {
        if (opIt->getOpType() != layer) {
            return false;
        }

        opIt = om.getSourceOp(opIt->getInputTensor(0));
    }

    return true;
}

bool canReplaceAveragePool(mv::Data::OpListIterator first, mv::Data::OpListIterator second, mv::OpModel& om) {
    auto first_attrs  = first->getAttrs({"opId"});
    auto second_attrs = second->getAttrs({"opId"});

    auto first_scale  = first->getOutputTensor(0)->getQuantParams().getScale();
    auto second_scale = second->getOutputTensor(0)->getQuantParams().getScale();

    auto first_dtype  = first->getOutputTensor(0)->getDType();
    auto second_dtype = second->getOutputTensor(0)->getDType();

    if (!(first_scale == second_scale &&
          first_attrs.at("stride") == second_attrs.at("stride") &&
          first_attrs.at("padding") == first_attrs.at("padding") &&
          first_attrs.at("exclude_pad") == second_attrs.at("exclude_pad") &&
          first_dtype == second_dtype))
        return false;

    auto first_kernel = first_attrs["kSize"].get<std::array<unsigned short, 2>>();
    auto second_kernel = second_attrs["kSize"].get<std::array<unsigned short, 2>>();

    auto reshape_dims = om.getSourceOp(second->getInputTensor(0))->get<mv::Shape>("shape");

    // This condition handles these cases
    // nx1 -> reshape -> reshape -> nx1
    // nx1 -> reshape -> 1xn
    return (first_kernel[0] == second_kernel[1] && first_kernel[1] == 1 && second_kernel[0] == 1) ||
           (first_kernel == second_kernel && first_kernel[0] == 1 && reshape_dims[0] == 1);
}

void replacePoolReshapePatternFcn(const mv::pass::PassEntry& , mv::ComputationModel& model) {
    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    mv::OpModel om(model);
    // Note: Pattern is reversed. First AveragePool in vector is the last AveragePool in graph
    //const std::vector<std::string> pattern = {"AveragePool", "Reshape", "Reshape", "AveragePool", "Reshape"};
    const std::vector<std::string> pattern = {"Reshape", "AveragePool", "Reshape", "Reshape", "AveragePool"};
    auto ops = om.getOps(*pattern.begin());

    auto can_replace_pool = [&pattern, &om](mv::Data::OpListIterator opIt) -> bool {
        std::vector<mv::Data::OpListIterator> poolOps;

        auto it = opIt;
        for (size_t i = 0; i < pattern.size(); ++i) {
            if (it->getOpType() == "AveragePool") {
                poolOps.push_back(it);
            }

            it = om.getSourceOp(it->getInputTensor(0));
        }
        assert(poolOps.size() == 2);

        return canReplaceAveragePool(poolOps[1], poolOps[0], om);
    };

    for (auto& opIt : ops)
    {
        if (!opIt) {
            continue;
        }

        if (matchPattern(pattern, opIt, model) && can_replace_pool(opIt)) {
            auto poolingOp = om.getSourceOp(opIt->getInputTensor(0));
            auto kernel = std::max(poolingOp->get<std::array<unsigned short, 2>>("kSize")[0], poolingOp->get<std::array<unsigned short, 2>>("kSize")[1]);
            std::array<unsigned short, 2> kSize = {kernel, kernel};
            auto op_attrs =  poolingOp->getAttrs({"kSize"});
            auto dType = poolingOp->getOutputTensor(0)->getDType();
            auto quantParams = poolingOp->getOutputTensor(0)->getQuantParams();

            auto it = om.getSourceOp(opIt->getInputTensor(0));

            for (size_t i = 0; i < pattern.size() - 1; ++i) {
                auto parentOpIt = om.getSourceOp( it->getInputTensor(0));
                auto sourceTensor = parentOpIt->getOutputTensor(0);
                it = linkNewOperationsRemove(parentOpIt, sourceTensor, om, it);
            }

            auto ap = om.averagePool("",
                                     it->getOutputTensor(0),
                                     kSize,
                                     op_attrs.at("stride"),
                                     op_attrs.at("padding"),
                                     op_attrs.at("exclude_pad"));
            ap->setDType(dType);
            ap->setQuantParams(quantParams);

            if(opIt->hasAttr("opId")) {
                unsigned currentOpId = opIt->get<unsigned>("opId");
                ap->set<unsigned>("opId", currentOpId);
                om.getSourceOp(ap)->set<unsigned>("opId", currentOpId);
            }

            linkNewOperationsReplacement(it, ap, om, opIt);
        }
    }
}

// Check for average pooling layers with kernels bigger than supported by hardware (11x11)
// and replace with equivalent two average pooling (approx equiv in case of prime kernel i.e. 13x13)
// Example: 13x13 kernel is replaced with 2 depthwise convolutions, each 4x4 kernel, stride 4, scale 1/13
// Example: 14x14 kernel is replaced with 1 depthwise 7x7 kernel, stride 7, scale 1/14 followed by
// depthwise 2x2 kernel, stride 2, scale 1/14
void replaceLargeKernelsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)

    mv::OpModel om(model);
    mv::DataModel dm(model);
    std::vector<std::string> opList = {"AveragePool", "MaxPool"};
    std::unordered_map<std::string, std::vector<mv::Data::OpListIterator>> operations = om.getOpsOfTypes(opList);
    std::vector <mv::Data::OpListIterator> ops;
    ops.reserve(operations["AveragePool"].size() + operations["MaxPool"].size() );
    ops.insert(ops.end(), operations["AveragePool"].begin(), operations["AveragePool"].end());
    ops.insert(ops.end(), operations["MaxPool"].begin(), operations["MaxPool"].end());

    for (auto opIt : ops)
    {
        std::array<unsigned short, 2> kSize;
        if (opIt->hasAttr("kSize"))
            kSize = opIt->get<std::array<unsigned short, 2>>("kSize");
        else
        {
            kSize[mv::KERNEL_HEIGHT] = opIt->getInputTensor(mv::IO_TENSOR_WEIGHTS_SET)->getShape()[mv::KERNEL_HEIGHT];
            kSize[mv::KERNEL_WIDTH] = opIt->getInputTensor(mv::IO_TENSOR_WEIGHTS_SET)->getShape()[mv::KERNEL_WIDTH];
        }
        if(kSize[mv::KERNEL_WIDTH] <= mv::MAX_KERNEL && kSize[mv::KERNEL_HEIGHT] <= mv::MAX_KERNEL) // can do as single depthwise, skip
            continue;//skip for this avgPool

        //figure out the bigger kernel dimension width or height when having an asymmetric kernel
        auto kernelSize = kSize[mv::KERNEL_WIDTH];
        auto largeDim = mv::KERNEL_WIDTH;
        auto asymmetricCase = false;
        auto asymmetricBothKernelsLarge = false;

        if((kSize[mv::KERNEL_WIDTH] != kSize[mv::KERNEL_HEIGHT]) && (kSize[mv::KERNEL_WIDTH] >  mv::MAX_KERNEL || kSize[mv::KERNEL_HEIGHT] >  mv::MAX_KERNEL))
        {
            if (kSize[mv::KERNEL_WIDTH] >  mv::MAX_KERNEL && kSize[mv::KERNEL_HEIGHT] >  mv::MAX_KERNEL)
                asymmetricBothKernelsLarge = true;

            // deal with asymetric kernels when one dim is larger than MAX_KERNEL
            asymmetricCase = true;
            if (kSize[mv::KERNEL_WIDTH] <  kSize[mv::KERNEL_HEIGHT])
            {
                kernelSize = kSize[mv::KERNEL_HEIGHT];
                largeDim = mv::KERNEL_HEIGHT;
            }
        }

        pass.log(mv::Logger::MessageType::Debug, "largeKernel " +  std::to_string(kernelSize) + " kernelDim " + std::to_string(largeDim));
        auto name = opIt->getName();
        auto sourceTensor = opIt->getInputTensor(0);

        auto parentOpIt = om.getSourceOp(sourceTensor);

        auto inputShape = sourceTensor->getShape();
        auto outputShape = opIt->getOutputTensor()[0]->getShape();

        std::vector<std::pair<unsigned short, unsigned short>> allFactors;
        std::pair<unsigned short, unsigned short> factors;
        std::pair<unsigned short, unsigned short> factorsDim2;

        // If average pool kernel size is greater than 11, we will turn it in to multiple depthwise convs here
        // Note: Kernel sizes should be chosen so the output tensor of the second depthwise
        // is the correct size for the network. The division scale of the weights will be used to improve accuracy.

        factors = getFactors(kernelSize);
        pass.log(mv::Logger::MessageType::Debug, "kernel " +  std::to_string(kernelSize) + " , factor1=" + std::to_string(factors.first)+ " , factor2=" + std::to_string(factors.second));
        if (factors.first >  mv::MAX_KERNEL || factors.second >  mv::MAX_KERNEL)
        {
            //unable to split into appropriate size
            throw std::runtime_error(std::string(__FUNCTION__).append(" ERROR: factors are larger the MAX_KERNEL 11"));
        }

        if (asymmetricBothKernelsLarge)
        {
            pass.log(mv::Logger::MessageType::Debug, "largeKernel " +  std::to_string(kernelSize) + " kernelDim " + std::to_string(mv::KERNEL_HEIGHT - largeDim));
            factorsDim2 = getFactors(kSize[mv::KERNEL_HEIGHT - largeDim]);//toggling between the two kernel sizes
            pass.log(mv::Logger::MessageType::Debug, "kernel " +  std::to_string(kSize[mv::KERNEL_HEIGHT - largeDim]) + " , factor1=" + std::to_string(factorsDim2.first)+ " , factor2=" + std::to_string(factorsDim2.second));

            if (factorsDim2.first >  mv::MAX_KERNEL || factorsDim2.second >  mv::MAX_KERNEL)
            {
                //unable to split into appropriate size
                throw std::runtime_error(std::string(__FUNCTION__).append(" ERROR: factors are larger the MAX_KERNEL 11"));
            }
        }
        //cascading supported ops
        //first op kernel [factor1.first, factor2.first] - newKernel
        //sequenced op kernel [factor1.second, factor2.second] - newKernel_1
        // Padding quantity relationship is (input size + pad) / k = output size, padding config is TRUE, FALSE
        std::array<unsigned short, 4> padding = {0, 0, 0, 0};
        std::array<unsigned short, 2> newKernel, newKernel_1 = {1,1};
        std::pair<bool, bool> producers_quantized(true, true);
        auto sinkOps = findSinkLayers(dm, opIt->getOutputTensor(0));
        if (sinkOps[0]->isUPA()){
            producers_quantized.first = true;
            producers_quantized.second = false;
        }
        else if (sinkOps[0]->getOpType() == "Output"){
            if (sinkOps[0]->hasAttr("precision")
                && sinkOps[0]->get<mv::DType>("precision") == mv::DType("Float16"))
            {
                producers_quantized.first = true;
                producers_quantized.second = false;
            }
        }

        newKernel[largeDim] = factors.first;//first was the large dimension
        newKernel_1[largeDim] = factors.second;
        if (asymmetricCase)
        {
            if (asymmetricBothKernelsLarge)
            {
                newKernel[mv::KERNEL_HEIGHT - largeDim] = factorsDim2.first;
                newKernel_1[mv::KERNEL_HEIGHT - largeDim] = factorsDim2.second;

                //factors multiplication > kernel, we need padding
                padding[mv::PADDING_RIGHT] = (newKernel[largeDim] * newKernel_1[largeDim] > kSize[largeDim]) ? 1 : 0;
                padding[mv::PADDING_BOT] = (newKernel[1 - largeDim] * newKernel_1[1 - largeDim] > kSize[mv::KERNEL_HEIGHT - largeDim]) ? 1 : 0;
                if (largeDim == mv::KERNEL_WIDTH)
                {
                    /*mv::KERNEL_WIDTH -> compute padding done already*/
                }
                else
                {
                    //change the padding on the other dimensions as largeDim was not on the width dimension - PADD_RIGHT
                    padding[mv::PADDING_RIGHT] = padding[mv::PADDING_BOT];
                    padding[mv::PADDING_BOT] = padding[mv::PADDING_RIGHT];
                }
            }
            else
            {
                newKernel[mv::KERNEL_HEIGHT - largeDim] = kSize[mv::KERNEL_HEIGHT - largeDim];
                newKernel_1[mv::KERNEL_HEIGHT - largeDim] = 1; //the 1-largeDim was not factorized, the multiplication kSize*1 covers the second depthwise

                //factors multiplication > kernel, we need padding
                padding[mv::PADDING_RIGHT] = (newKernel[largeDim] * newKernel_1[largeDim] > kSize[largeDim]) ? 1 : 0;
                padding[mv::PADDING_BOT] = 0;
                if (largeDim == mv::KERNEL_WIDTH)
                {
                    /*mv::KERNEL_WIDTH -> compute padding done already*/
                }
                else
                {
                    //change the padding on the other dimensions as largeDim was not on the width dimension - PADD_RIGHT
                    padding[mv::PADDING_RIGHT] = padding[mv::PADDING_BOT];
                    padding[mv::PADDING_BOT] = padding[mv::PADDING_RIGHT];
                }
            }
        }
        else
        {
            newKernel[mv::KERNEL_HEIGHT - largeDim] = factors.first;//largeDim has the same kernel size as 1-largeDim
            newKernel_1[mv::KERNEL_HEIGHT - largeDim] = factors.second;
            padding[mv::PADDING_RIGHT] = padding[mv::PADDING_BOT] = (newKernel[largeDim] * newKernel_1[largeDim] > kSize[largeDim]) ? 1 : 0;
        }
        double firstRescale = 1.0 / (newKernel[0] * newKernel[1]);
        double secondRescale = static_cast<double>((newKernel[0] * newKernel[1])) / (kSize[0] * kSize[1]);
        mv::Data::TensorIterator op0;
        if (opIt->getOpType() == "AveragePool")
            op0 = createPartialDepthwise(om, opIt, sourceTensor,
                                         name + "_DepthwiseConv0",
                                         firstRescale, newKernel, padding, producers_quantized.first);
        else if (opIt->getOpType()== "MaxPool")
        {
            auto dType = opIt->getInputTensor(mv::IO_TENSOR_INPUT)->getDType();
            auto quantParams = opIt->getOutputTensor(0)->getQuantParams();
            op0 = om.maxPool(opIt->getName() + "_MaxPool0",
                             sourceTensor,
                             newKernel,
                             newKernel,
                             padding,
                             true); //exclude pad
            op0->setDType(dType);
            op0->setQuantParams(quantParams);
            if(opIt->hasAttr("opId"))
            {
                unsigned currentOpId = opIt->get<unsigned>("opId");
                op0->set<unsigned>("opId", currentOpId);
                om.getSourceOp(op0)->set<unsigned>("opId", currentOpId);
            }
        }
        else
            throw std::runtime_error( "Error: Op= " + opIt->getName() + " compiler doesn't support large kernel for a " + opIt->getOpType() );

        linkNewOperationsReplacement(parentOpIt, op0, om, opIt);

        // Remove old flow, remember to it to put next depthwise into model in correct place
        std::vector<mv::Data::OpListIterator> opsToLink;
        std::vector<std::size_t> inputSlots;
        std::vector<mv::Data::FlowSiblingIterator> flowsToRemove;

        auto input_op0 = om.getSourceOp(op0);
        auto sourceFlowStart = input_op0.leftmostOutput();

        if (asymmetricCase)
        {
            input_op0->set<unsigned>("asymmetricKernel", 1-largeDim);//record dimension we need workload to stride over.
        }

        for (mv::Data::FlowSiblingIterator sinkFlow(sourceFlowStart); sinkFlow != om.flowEnd(); ++sinkFlow)
        {
            opsToLink.push_back(sinkFlow.sink());
            inputSlots.push_back(sinkFlow->get<std::size_t>("sinkInput"));
            flowsToRemove.push_back(sinkFlow);
        }
        // Remove old flow before creating new dw
        for (unsigned flowIdx = 0; flowIdx < flowsToRemove.size(); flowIdx++)
        {
            om.undefineFlow(flowsToRemove[flowIdx]);
        }

        pass.log(mv::Logger::MessageType::Debug, "newKernel " +  std::to_string(newKernel[0]) + " , " + std::to_string(newKernel[1]));

        mv::Data::TensorIterator op1;
        if (input_op0->getOpType() == "DepthwiseConv" || input_op0->getOpType() == "AveragePool" )
            op1 = createPartialDepthwise(om, input_op0, op0,
                                         name + "_DepthwiseConv1", secondRescale, newKernel_1, {0,0,0,0}, producers_quantized.second);
        else if (input_op0->getOpType() == "MaxPool")
        {
            auto dType = input_op0->getInputTensor(mv::IO_TENSOR_INPUT)->getDType();
            auto quantParams = op0->getQuantParams();
            op1 = om.maxPool(op0->getName() + "_MaxPool1",
                             op0,
                             newKernel_1,
                             newKernel_1,
                             padding,
                             true); //exclude pad
            op1->setDType(dType);
            op1->setQuantParams(quantParams);
            if(input_op0->hasAttr("opId"))
            {
                unsigned currentOpId = input_op0->get<unsigned>("opId");
                op1->set<unsigned>("opId", currentOpId);
                om.getSourceOp(op1)->set<unsigned>("opId", currentOpId);
            }
        }
        else
            throw std::runtime_error( "Error: Op= " + input_op0->getName() + " compiler doesn't support large kernel for a " + input_op0->getOpType() );

        // Now generate the second depthwise conv

        for(unsigned op = 0 ; op < opsToLink.size(); ++op)
        {
            opsToLink[op]->setInputTensor(op1, inputSlots[op], false);
            om.defineFlow(op1, opsToLink[op], inputSlots[op]);
        }

    } // end for
}

// Replace concat of populated inputs with single populated input
void replaceConcatOfPopulatedTensorsFcn(const mv::pass::PassEntry&, mv::ComputationModel& model)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)

    mv::OpModel om(model);
    auto concats = om.getOps("Concat");
    for(auto& concat : concats)
    {
        auto all_inputs_are_populated = true;
        auto inputs = concat->getInputTensor();
        for (auto& input : inputs)
            if (!input->isPopulated())
                all_inputs_are_populated = false;

        if (!all_inputs_are_populated)
            continue;

        // Check for supported params
        auto axis = concat->get<std::string>("axis");
        if (axis != "W")
            throw std::runtime_error("Compiler doesn't support manual concat of axis != W");

        // Manually concat populated tensors
        auto newShape = concat->getOutputTensor()[0]->getShape();
        std::vector<double> newData(newShape.totalSize());
        auto concat_offset = 0;
        for (unsigned i=0; i<inputs.size(); i++)
        {
            auto inputData = inputs[i]->getDoubleData();
            auto inputShape = inputs[i]->getShape();
            for (unsigned H=0; H<inputShape[1]; H++)
            {
                auto input_start = (H)*inputShape[0];
                auto input_length = inputShape[0];
                auto new_start = concat_offset + H*newShape[0];
                for(unsigned j=0; j<input_length; j++)
                {
                    newData[new_start + j] = inputData[input_start + j];
                }
            }
            concat_offset += inputShape[0];
        }

        // Replace existing Concat'd tensors with a single tensor
        std::vector<std::string> ops_to_remove;
        for (unsigned i=0; i<concat->getInputTensor().size(); i++)
            ops_to_remove.push_back(om.getSourceOp(concat->getInputTensor(i))->getName());

        auto order = concat->getInputTensor(0)->getOrder();
        auto quantParams = concat->getOutputTensor(0)->getQuantParams();
        auto newKernel = om.constant("", newData, newShape, concat->getOutputTensor(0)->getDType(), order);
        newKernel->setQuantParams(quantParams);
        auto newKernelOp = om.getSourceOp(newKernel);
        newKernelOp->set<unsigned>("opId", concat->get<unsigned>("opId"));
        auto flows = mv::getOutputDataFlow(om, concat);
        mv::setOutputDataFlow(om, newKernel, flows);

        for (auto& op_name : ops_to_remove)
            om.removeOp(om.getOp(op_name));
    }
}
//pass slicing horizontally by the provided width also slicing vertically by the provided height
//original operation  is performed per small partitions, 
//result concatenated per each line (horizontally) and at the end concatening the  1 column of results vertically
mv::Data::OpListIterator  splitOperationSlicingFixedWidthHeight ( mv::ComputationModel& model, mv::Data::OpListIterator operation, size_t widthSlice, size_t heightSlice, mv::Data::OpListIterator nextOpIt)
{
    mv::OpModel om(model);
    mv::DataModel dm(model);
    auto inputTensor = operation->getInputTensor(mv::IO_TENSOR_INPUT);
    unsigned initialOpId = operation->get<unsigned>("opId");
    auto inputShape = inputTensor->getShape();
    auto width = inputShape[mv::IO_WIDTH_DIMENSION];
    auto height = inputShape[mv::IO_HEIGHT_DIMENSION];
    std::array<unsigned short, 2> stride = operation->get<std::array<unsigned short, 2>>("stride");
    std::array<unsigned short, 2> newStride = {1,1};
    std::array<unsigned short, 2> kSize;
    if ( operation->hasAttr("kSize") )
        kSize = operation->get<std::array<unsigned short, 2>>("kSize");
    else
        kSize = { static_cast<unsigned short>(operation->getInputTensor(mv::IO_TENSOR_WEIGHTS_SET)->getShape()[mv::KERNEL_WIDTH]),
                  static_cast<unsigned short>(operation->getInputTensor(mv::IO_TENSOR_WEIGHTS_SET)->getShape()[mv::KERNEL_HEIGHT]) };

    std::array<unsigned short, 4> initialPadding = operation->get<std::array<unsigned short, 4>>("padding");
    std::array<unsigned short, 4> padding = initialPadding;
    std::size_t branchWidth, branchHeight;
    mv::Shape beginInputShape, branchInputSize; //coordinates to slice
    //if strides bigger than supported stride, replace the operation with big strides by the chunks of operations with strides of a batch

    branchWidth = stride[mv::STRIDE_HORIZONTAL];//no overlap
    branchHeight = stride[mv::STRIDE_VERTICAL];//no overlap
    branchInputSize = {branchWidth, branchHeight, inputTensor->getShape()[mv::IO_CHANNEL_DIMENSION], inputTensor->getShape()[mv::IO_BATCH_DIMENSION]};
    padding = {0, 0, 0, 0};
    //sliced op iteratively and the vector
    mv::Data::TensorIterator op;
    mv::Data::TensorIterator opConcatHorizontal;
    std::vector <mv::Data::TensorIterator> opsHorizontal;

    //concat iteratively on the line vertically on the width axis and the vector of Horizontally Concats to be concatenated Vertically
    mv::Data::TensorIterator opConcat;
    std::vector <mv::Data::TensorIterator> opsSlicesConcatHorizontally;
    const unsigned int hslices = width/widthSlice;
    const unsigned int vslices = height/heightSlice;

    auto outputQuantParams = operation->getOutputTensor(0)->getQuantParams();
    auto dType = operation->getInputTensor(mv::IO_TENSOR_INPUT)->getDType();

    unsigned int i = 0; //counts the slices on the 0x axis
    unsigned int j = 0; //counts the slices on the 0y axis
    do {//slicing on the vertical axis , agnostic whether we need to slice vertically that's why [do .. while] is chosen to execute at least once the code if we not slice on oy axis 
        do {//slicing on the horizontal axis , agnostic whether we need to slice horizontally that's why [do .. while] is chosen to execute at least once the code if we not slice on ox axis
            //start new tile from the boundaries, with origin at the strides
            beginInputShape = { (unsigned long)( (i)*widthSlice),
                                (unsigned long)( (j)*heightSlice),
                                0, 0};
            //window of the slice equal to the stride as the stride becomes 1x1 (if stride equal (MAX+1)x(MAX+1))so we have one operation per slicing with strides dimension
            //if tensor is sliced on a dimension, kernel is the window size, if not, the actual dimension does not change
            branchWidth = (hslices > 1) ? stride[mv::STRIDE_HORIZONTAL] : width;
            branchHeight = (vslices > 1) ? stride[mv::STRIDE_VERTICAL] : height;
            //when slicing on a dimension the stride becomes 1
            model.log(mv::Logger::MessageType::Debug, "newStride hor=" + std::to_string(newStride[mv::STRIDE_HORIZONTAL])+ " , newStride vert=" + std::to_string(newStride[mv::STRIDE_VERTICAL]));
            branchInputSize = {branchWidth,
                               branchHeight,
                               inputTensor->getShape()[mv::IO_CHANNEL_DIMENSION],
                               inputTensor->getShape()[mv::IO_BATCH_DIMENSION]};

            padding = { static_cast<unsigned short>((i == 0)       ? initialPadding[mv::PADDING_LEFT]  : 0),
                        static_cast<unsigned short>((i == hslices) ? initialPadding[mv::PADDING_RIGHT] : 0),
                        static_cast<unsigned short>((j == 0)       ? initialPadding[mv::PADDING_TOP]   : 0),
                        static_cast<unsigned short>((j == vslices) ? initialPadding[mv::PADDING_BOT]   : 0) };
            std::string sliceName ("Slice_Input_l" + std::to_string(i) + "c" + std::to_string(j));
            auto quantParams = inputTensor->getQuantParams();
            auto sliceInput = om.slice("Slice" + operation->getName() +  sliceName,
                                       inputTensor,
                                       beginInputShape,
                                       branchInputSize);
            sliceInput->setQuantParams(quantParams);
            auto sliceInputOp = om.getSourceOp(sliceInput);
            sliceInputOp->set<unsigned>("opId", initialOpId);
            auto parentOpIt = om.getSourceOp(inputTensor);

            if (operation->getOpType() == "AveragePool")
            {
                op = om.averagePool(operation->getName() + sliceName,
                                    sliceInput,
                                    kSize,
                                    newStride,
                                    padding,
                                    true); // exclude pad
            }
            else if (operation->getOpType() == "DepthwiseConv")
            {
                op  = om.depthwiseConv(operation->getName() + sliceName,
                                       sliceInput,
                                       operation->getInputTensor(mv::IO_TENSOR_WEIGHTS_SET),
                                       newStride,
                                       padding,
                                       operation->get<unsigned>("dilationFactor"));
                om.getSourceOp(op)->set<bool>("DWWithReplaceLargeStrides", true);
            }
            else if (operation->getOpType()== "Conv")
            {
                op = om.conv(operation->getName() + sliceName,
                             sliceInput,
                             operation->getInputTensor(mv::IO_TENSOR_WEIGHTS_SET),
                             newStride,
                             padding,
                             operation->get<unsigned>("dilationFactor"),
                             1); // no group dilation
            }
            else if (operation->getOpType()== "MaxPool")
            {
                op = om.maxPool(operation->getName() + sliceName,
                                sliceInput,
                                kSize,
                                newStride,
                                padding,
                                true); // exclude pad
            }
            op->setDType(dType);
            op->setQuantParams(outputQuantParams);
            op->set<unsigned>("opId", initialOpId);
            auto opSlice = om.getSourceOp(op);
            opSlice->set<unsigned>("opId", initialOpId);
            opsHorizontal.push_back(op);

            i++;
        } while (i < hslices);
        if (opsHorizontal.size() == 1)
            opConcatHorizontal = mv::Data::TensorIterator(*opsHorizontal.begin());
        else {
            opConcatHorizontal = om.concat(operation->getName() + "_concat_l" + std::to_string(j),
                                           opsHorizontal,
                                           "W");
            opConcatHorizontal->setDType(dType);
            opConcatHorizontal->setQuantParams(outputQuantParams);
        }
        opsHorizontal.clear();

        opConcatHorizontal->set<unsigned>("opId", initialOpId);
        auto opConcatHorizontalSlice = om.getSourceOp(opConcatHorizontal);
        opConcatHorizontalSlice->set<unsigned>("opId", initialOpId);
        opsSlicesConcatHorizontally.push_back(opConcatHorizontal);
        i = 0;//reiterate the horizontal slices for the next vertical slice
        j++;
    } while( j < vslices); //i,j non zero means we need to slice
    if (opsSlicesConcatHorizontally.size() == 1)
        opConcat = mv::Data::TensorIterator(*opsSlicesConcatHorizontally.begin());
    else {
        opConcat = om.concat(operation->getName() + "concat_full",
                             opsSlicesConcatHorizontally,
                             "H");
        opConcat->setDType(dType);
        opConcat->setQuantParams(outputQuantParams);
    }
    opConcat->set<unsigned>("opId", initialOpId);
    auto opConcatSlice = om.getSourceOp(opConcat);
    opConcatSlice->set<unsigned>("opId", initialOpId);
    //recircuit the graph flow
    operation = linkNewOperationsReplacementRemoveFlows(nextOpIt, opConcat, om, operation);
    return operation;
}

void replaceLargeStridesFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)

    mv::OpModel om(model);
    mv::DataModel dm(model);

    for (auto opIt = om.getInput(); opIt != om.opEnd(); ++opIt)
    {
        //zm ops except eltwise
        if (opIt->getOpType() == "Conv" || opIt->getOpType() == "DepthwiseConv" || opIt->getOpType() == "MaxPool" || opIt->getOpType() == "AveragePool")
        {
            std::array<unsigned short, 2> stride = opIt->get<std::array<unsigned short, 2>>("stride");
            if( (stride[mv::STRIDE_HORIZONTAL] <= mv::MAX_STRIDE) && (stride[mv::STRIDE_VERTICAL] <= mv::MAX_STRIDE) ) // can do as single operation in DPU, skip
                continue;

            pass.log(mv::Logger::MessageType::Debug, "stride hor=" + std::to_string(stride[mv::STRIDE_HORIZONTAL])+ " , stride vert=" + std::to_string(stride[mv::STRIDE_VERTICAL]));
            auto nextOp = mv::findSinkLayers(dm, opIt->getOutputTensor(mv::IO_TENSOR_OUTPUT))[0];
            //stride supported not slicing, stride not supported slicing with slices dimensions of stride
            opIt = splitOperationSlicingFixedWidthHeight (om,
                                                          opIt,
                                                          stride[mv::STRIDE_HORIZONTAL],
                                                          stride[mv::STRIDE_VERTICAL],
                                                          nextOp);
        }
    }
}

void replaceAsymmetricStridesFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)

    mv::OpModel om(model);
    mv::DataModel dm(model);
    std::vector<std::string> opList = {"AveragePool", "MaxPool", "DepthwiseConv"};
    std::unordered_map<std::string, std::vector<mv::Data::OpListIterator>> operations = om.getOpsOfTypes(opList);
    std::vector <mv::Data::OpListIterator> ops;
    ops.reserve(operations["AveragePool"].size() + operations["MaxPool"].size() + operations["DepthwiseConv"].size() );
    ops.insert(ops.end(), operations["AveragePool"].begin(), operations["AveragePool"].end());
    ops.insert(ops.end(), operations["MaxPool"].begin(), operations["MaxPool"].end());
    ops.insert(ops.end(), operations["DepthwiseConv"].begin(), operations["DepthwiseConv"].end());

    for (auto opIt : ops)
    {
        std::array<unsigned short, 2> stride = opIt->get<std::array<unsigned short, 2>>("stride");
        if( stride[mv::STRIDE_HORIZONTAL] == stride[mv::STRIDE_VERTICAL] ) // symmetric
            continue;
        // if stride vertical equals 1 and input horizontal equals kernel horizontal size, no need to split.
        // but replace strides (s_w,1) with (s_w,s_w) s_w>=2 when height=1
        // e.g kernel=[8, 1], stride = [8, 1], input = [4000, 1], stride replaced with [8, 8]
        // avoid cutting in too many workloads and obtaining a bad performance
        if (stride[mv::STRIDE_VERTICAL] == 1)
        {
            bool verticalMatch = false;
            auto inputTensorShape = opIt->getInputTensor(0)->getShape();
            if (opIt->getOpType() == "DepthwiseConv")
            {
                auto kSize = opIt->getInputTensor(1)->getShape();
                if (kSize[mv::IO_HEIGHT_DIMENSION] == inputTensorShape[mv::IO_HEIGHT_DIMENSION])
                    verticalMatch = true;
            }
            else if (opIt->getOpType() == "AveragePool" || opIt->getOpType() == "MaxPool")
            {
                auto kSize = opIt->get<std::array<unsigned short, 2>>("kSize");
                if (kSize[mv::IO_HEIGHT_DIMENSION] == inputTensorShape[mv::IO_HEIGHT_DIMENSION])
                    verticalMatch = true;
            }
            if (verticalMatch)  // operate only one time on vertical.
            {
                stride[mv::STRIDE_VERTICAL] = stride[mv::STRIDE_HORIZONTAL];
                opIt->set("stride", stride);
                continue;
            }
        }
        pass.log(mv::Logger::MessageType::Debug, "stride hor=" + std::to_string(stride[mv::STRIDE_HORIZONTAL])+ " , stride vert=" + std::to_string(stride[mv::STRIDE_VERTICAL]));
        auto nextOp = mv::findSinkLayers(dm, opIt->getOutputTensor(mv::IO_TENSOR_OUTPUT))[0];
        //stride supported not slicing, stride not supported slicing with slices dimensions of stride
        opIt = splitOperationSlicingFixedWidthHeight (om,
                                                    opIt,
                                                    stride[mv::STRIDE_HORIZONTAL],
                                                    stride[mv::STRIDE_VERTICAL],
                                                    nextOp);
    }
}

bool matchPattern(const std::vector<std::string>& pattern, mv::Data::OpListIterator it, mv::Data::OpListIterator& lastIt, mv::ComputationModel& model) {
    mv::OpModel om(model);
    auto opIt = it;

    for (auto& layer : pattern) {
        if (opIt->getOpType() != layer) {
            return false;
        }

        lastIt = opIt;
        opIt = om.getSourceOp(opIt->getInputTensor(0));
    }
    
    return true;
}

void replaceExpReduceSumMultipyFcn(const mv::pass::PassEntry& /*pass*/, mv::ComputationModel& model)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    using namespace mv;

    OpModel om(model);

    const std::vector<std::string> pattern = {"Reciprocal", "Reshape", "Scale", "AveragePool", "Reshape", "Exp"};
    auto ops = om.getOps("Eltwise");

    for (auto& opIt : ops)
    {
        if (!opIt) {
            continue;
        }

        if (opIt->get<std::string>("eltwiseType") == "Multiply")
        {
            auto itLeft = om.getSourceOp(opIt->getInputTensor(0));
            auto itRight = om.getSourceOp(opIt->getInputTensor(1));
            mv::Data::OpListIterator scaleIt, dataIt, lastOp;

            if (itLeft->getOpType() == "Reciprocal" && matchPattern(pattern, itLeft, lastOp, model))
            {
                scaleIt = itLeft;
                dataIt = itRight;
            }
            else if (itRight->getOpType() == "Reciprocal" && matchPattern(pattern, itRight, lastOp, model))
            {
                scaleIt = itRight;
                dataIt = itLeft;
            }
            else {
                return;
            }

            if (dataIt->getName() == lastOp->getName())
            {
                for (size_t i = 0; i < pattern.size() - 2; ++i) {
                    auto parentOpIt = om.getSourceOp( scaleIt->getInputTensor(0));
                    auto sourceTensor = parentOpIt->getOutputTensor(0);
                    scaleIt = linkNewOperationsRemove(parentOpIt, sourceTensor, om, scaleIt);
                }

                om.removeOp(scaleIt);
                opIt = linkNewOperationsRemove(dataIt, dataIt->getOutputTensor(0), om, opIt);

                auto scoreMap = opIt->getInputTensor(0);
                auto sm = om.softmax("", scoreMap, "C");

                if(opIt->hasAttr("opId")) {
                    unsigned currentOpId = opIt->get<unsigned>("opId");
                    sm->set<unsigned>("opId", currentOpId);
                    om.getSourceOp(sm)->set<unsigned>("opId", currentOpId);
                }

                linkNewOperationsReplacement(om.getSourceOp(scoreMap), sm, om, opIt);  
            }
        }
    }
}

void detectEltWiseUpaInputs(const mv::pass::PassEntry& /*pass*/, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)

    mv::OpModel om(model);

    auto eltWiseOps = om.getOps("Eltwise");

    for (auto& opIt : eltWiseOps)
    {
        auto firstEltwiseParent = om.getSourceOp(opIt->getInputTensor(0));
        auto secondEltwiseParent = om.getSourceOp(opIt->getInputTensor(1));

        if(!(firstEltwiseParent->isHardwarizable() && secondEltwiseParent->isHardwarizable()))
        {
            opIt->set<bool>("upaInputs", true);
        }
    }
}
