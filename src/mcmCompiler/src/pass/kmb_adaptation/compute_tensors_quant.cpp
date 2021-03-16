﻿#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/pass/pass_quantization.hpp"
#include "include/mcm/pass/pass_utils.hpp"
#include "include/mcm/tensor/quantization_params.hpp"
#include "mcm/utils/custom_math.hpp"
#include <numeric>
#include <cmath>

static void computeTensorsQuantParams(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);
static void postTrainingQuantize(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);
static void alignConcatScales(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);
static void placeReQuantizeDepthwiseBefore(mv::OpModel & om, mv::Data::OpListIterator concat, mv::Data::TensorIterator inputTensor, std::size_t index, double &weightScale, double &alignedScale, int64_t &alignedZeroPoint);
//static void compensateDepthWiseAfter(mv::OpModel om, mv::Data::OpListIterator nextOp, mv::Data::OpListIterator concat);
//static std::vector<mv::Data::OpListIterator> findNextConcat(mv::DataModel &dataModel, const mv::Data::TensorIterator &tensor);

namespace mv
{

    namespace pass
    {

        MV_REGISTER_PASS(ComputeTensorsQuantParams)
        .setFunc(computeTensorsQuantParams)
        .setDescription(
            "This pass computes the appropriate quantize params extends and prepares them for serialization."
        );

        MV_REGISTER_PASS(PostTrainingQuantize)
        .setFunc(postTrainingQuantize)
        .setDescription(
            "The pass will estimate output tensor quantization param where quantization is needed."
        );
    }
}

void postTrainingQuantize(const mv::pass::PassEntry& pass, mv::ComputationModel& model,
                       mv::TargetDescriptor& td, mv::Element& e0, mv::Element& e1)
{
    alignConcatScales(pass, model, td, e0, e1);
}

void placeReQuantizeDepthwiseBefore(mv::OpModel & om, mv::Data::OpListIterator concat, mv::Data::TensorIterator inputTensor, std::size_t index, double &weightScale, double &alignedScale, int64_t &alignedZeroPoint)
{
    //FIND THE APPROPRIATE FLOW
    auto inputFlow = concat.leftmostInput();
    while(inputFlow != om.flowEnd())
    {
        auto tensor = inputFlow->getTensor();
        if (tensor->getName() == inputTensor->getName())
        {
            break;
        }
        ++inputFlow;
    }
    mv::Data::TensorIterator weights;
    std::vector<int64_t> zp = { 0 };
    std::vector<double> min = { 1 };
    std::vector<double> max = { 1 };

    std::vector<double> scale(1, weightScale);
    mv::QuantizationParams weightsQuantParams(zp, scale, min, max);
    int64_t weightsValue = 1;
    std::vector<int64_t> weightsData(inputTensor->getShape()[mv::IO_CHANNEL_DIMENSION], weightsValue);
    weights = om.constantInt("",
                        weightsData,
                        {1, 1, inputTensor->getShape()[mv::IO_CHANNEL_DIMENSION], 1},
                        mv::DType("UInt8"),
                        mv::Order(mv::Order::getRowMajorID(4)));
    weights->setQuantParams(weightsQuantParams);
    auto reQuantizeDepthwise = om.depthwiseConv(concat->getName() + inputTensor->getName() + "Depthwise" + std::to_string(index),
                        inputTensor, weights, {1,1}, {0, 0, 0, 0}, 1);
    reQuantizeDepthwise->setDType(mv::DType("UInt8"));
    reQuantizeDepthwise->setQuantParams({{alignedZeroPoint}, {alignedScale}, {}, {}});
    reQuantizeDepthwise->set<double>("oldScale", inputTensor->get<double>("oldScale"));
    auto reQuantizeDepthwiseOp = om.getSourceOp(reQuantizeDepthwise);
    auto weightsOp = om.getSourceOp(weights);
    reQuantizeDepthwiseOp->set<unsigned>("opId", concat->get<unsigned>("opId"));
    weightsOp->set<unsigned>("opId", concat->get<unsigned>("opId"));
    om.undefineFlow(inputFlow);
    concat->setInputTensor(reQuantizeDepthwise, index, false);
    om.defineFlow(reQuantizeDepthwise, concat, index);
}

//void compensateDepthWiseAfter(mv::OpModel om, mv::Data::OpListIterator nextOp, mv::Data::OpListIterator concat)
//{
//    auto inputFlow = nextOp.leftmostInput();
//    mv::Data::TensorIterator weights;
//    std::vector<int64_t> zp = { 0 }; // why always assume ZP 0?? it is in this case but doesnt have to be always
//    std::vector<double> min = { 1 };
//    std::vector<double> max = { 1 };

//    std::vector<double> scale(concat->getOutputTensor()[0]->getShape()[mv::IO_CHANNEL_DIMENSION], 1 );
//    double masterScale = concat->getInputTensor()[0]->get<mv::QuantizationParams>("quantParams").getScale()[0];
//    int64_t weightsValue = 1;
//    std::vector<int64_t> weightsData(concat->getOutputTensor()[0]->getShape()[mv::IO_CHANNEL_DIMENSION], weightsValue);
//    std::size_t sumChannels = 0;
//    for (std::size_t i = 0; i < concat->getInputTensor().size(); i++)
//    {
//        auto oldScale = concat->getInputTensor()[i]->get<double>("oldScale");
//        for (std::size_t outputChannel = sumChannels;  outputChannel < sumChannels + concat->getInputTensor()[i]->getShape()[mv::IO_CHANNEL_DIMENSION];
//             outputChannel++)
//        {
//            scale[outputChannel] = oldScale/masterScale;
//        }
//        sumChannels +=concat->getInputTensor()[i]->getShape()[mv::IO_CHANNEL_DIMENSION];
//    }

//    mv::QuantizationParams weightsQuantParams(zp, scale, min, max);
//    auto inputTensor = concat->getOutputTensor()[0];
//    weights = om.constantInt(weightsData,
//                        {1, 1, inputTensor->getShape()[mv::IO_CHANNEL_DIMENSION], 1},
//                        mv::DType("UInt8"),
//                        mv::Order(mv::Order::getRowMajorID(4)),
//                        weightsQuantParams);

//    auto compensateDepthWise = om.depthwiseConv(inputTensor, weights, {1,1}, {0, 0, 0, 0},
//                1, mv::DType("UInt8"), {{concat->getOutputTensor(0)->get<mv::QuantizationParams>("quantParams").getZeroPoint()[0]},{masterScale},{},{}}, nextOp->getName() + "compensateDepthwise");
//    auto compensateDepthWiseOp = om.getSourceOp(compensateDepthWise);
//    auto weightsOp = om.getSourceOp(weights);
//    compensateDepthWiseOp->set<unsigned>("opId", nextOp->get<unsigned>("opId"));
//    weightsOp->set<unsigned>("opId", nextOp->get<unsigned>("opId"));
//    om.undefineFlow(inputFlow);
//    nextOp->setInputTensor(compensateDepthWise, 0, false);
//    om.defineFlow(compensateDepthWise, nextOp, 0);
//}

static std::vector<mv::Data::OpListIterator> findNextConcat(mv::DataModel &dataModel, const mv::Data::TensorIterator &tensor)
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

static void markCompensatedConcats(std::vector<mv::Data::OpListIterator> &concats)
{
    //NOTE: Mark the concats that need compensation
    for(auto& concatIt : concats)
    {
        auto tempQuant =  concatIt->getInputTensor(0)->get<mv::QuantizationParams>("quantParams");
        auto tempScale = concatIt->getInputTensor(0)->get<mv::QuantizationParams>("quantParams").getScale()[0];
        concatIt->set<bool>("compensateNeed", false);
        concatIt->getInputTensor(0)->set<double>("oldScale", tempScale);

        for (std::size_t i = 1; i < concatIt->getInputTensor().size(); i++)
        {
            auto concatInputQuantParams = concatIt->getInputTensor(i)->get<mv::QuantizationParams>("quantParams");
            //NOTE: Activation tensors need to support only one value
            if (std::abs(tempScale - concatInputQuantParams.getScale()[0])/tempScale >
                    0.01)
            {
                auto oldScale = concatInputQuantParams.getScale()[0];
                concatIt->getInputTensor(i)->set<double>("oldScale", oldScale);
                concatIt->set<bool>("compensateNeed", true);
            }
            else
                continue;
        }
    }
}

static mv::QuantizationParams computeAlignedQuantParams(mv::Data::OpListIterator &concatIt)
{
    std::vector<double> minInputFloats, maxInputFloats;

    //NOTE: Compute the min/max of every tensor that goes in the Concat
    for (std::size_t i = 0; i < concatIt->getInputTensor().size(); i++)
    {
        //Note: if input Tensor has min, max of infs...we need to compute them
        updateInfMinMaxPerTensor(concatIt->getInputTensor(i));

        auto& inputQuantization = concatIt->getInputTensor(i)->get<mv::QuantizationParams>("quantParams");

        minInputFloats.push_back(inputQuantization.getMin()[0]);
        maxInputFloats.push_back(inputQuantization.getMax()[0]);
    }

    double minConcatScale = *std::min_element(minInputFloats.begin(), minInputFloats.end());
    double maxConcatScale = *std::max_element(maxInputFloats.begin(), maxInputFloats.end());

    double masterScale = 1.0;
    int64_t zeroPoint = 0;
    calcZeroPointAndScalePerTensor(
        maxConcatScale,
        minConcatScale,
        256,
        mv::getDType(mv::Precision::U8),
        masterScale,
        zeroPoint);

    return {{zeroPoint},{masterScale},{minConcatScale},{maxConcatScale}};
}

void alignConcatScales(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    mv::OpModel om(model);
    mv::DataModel dm(model);

    auto concats = om.getOps("Concat");

    // NOTE: For concats that go to concats the solution need to be recursive
    // Covering 2 recursion rounds now

    // Trim out float concats
    auto new_end = std::remove_if(concats.begin(), concats.end(),
                [](const mv::Data::OpListIterator op)
                {
                    for (auto tensor : op->getInputTensor())
                        if (!(tensor->getDType().isDoubleType() ||
                            tensor->getDType() == mv::DType("Float16")))
                            return false;
                    return true;
                });
    concats.erase(new_end, concats.end());

    std::vector<mv::Data::OpListIterator> childConcats;
    for(auto& concatIt : concats)
    {
        auto nextOp = findNextConcat(dm, concatIt->getOutputTensor()[0])[0];
        if (nextOp->getOpType() == "Concat")
            childConcats.push_back(nextOp);
    }

    for (auto& childConcatIt : childConcats)
        concats.erase(std::remove(concats.begin(), concats.end(), childConcatIt), concats.end());

    markCompensatedConcats(concats);
    for(auto& concatIt : concats)
    {
        if (concatIt->get<bool>("compensateNeed"))
        {
            mv::QuantizationParams masterQuant = computeAlignedQuantParams(concatIt);
            for (std::size_t i = 0; i < concatIt->getInputTensor().size(); i++)
            {
                if (std::abs(masterQuant.getScale()[0] - concatIt->getInputTensor(i)->get<mv::QuantizationParams>("quantParams").getScale()[0])/masterQuant.getScale()[0] <= 0.01)
                    continue;
                double weightScale = 1.0f;
               
                placeReQuantizeDepthwiseBefore(om, concatIt, concatIt->getInputTensor(i), i, weightScale, masterQuant.getScale()[0], masterQuant.getZeroPoint()[0]);
            }
            concatIt->getOutputTensor(0)->setQuantParams(masterQuant);
        }
    }
    markCompensatedConcats(childConcats);
    for(auto& concatIt : childConcats)
    {
        if (concatIt->get<bool>("compensateNeed"))
        {
            mv::QuantizationParams masterQuant = computeAlignedQuantParams(concatIt);
            for (std::size_t i = 0; i < concatIt->getInputTensor().size(); i++)
            {
                if (masterQuant.getScale()[0] == concatIt->getInputTensor()[i]->get<mv::QuantizationParams>("quantParams").getScale()[0])
                    continue;
                double weightScale = 1.0f;
                placeReQuantizeDepthwiseBefore(om, concatIt, concatIt->getInputTensor(i), i, weightScale, masterQuant.getScale()[0], masterQuant.getZeroPoint()[0]);
            }
            concatIt->getOutputTensor(0)->setQuantParams(masterQuant);
        }
    }
}

void computeQuantMultShift(
    std::vector<float>& scale,
    std::vector<unsigned>& shift,
    std::vector<unsigned>& mult)
{
    //TODO need to handle 16bits case - per Alessandro bias need to be converted to int32
    auto bits = 15;
    auto scaleSize = scale.size();
    int exponent;
    double mantissa;

    {
        for (size_t i = 0; i < scaleSize; i++)
        {
            mantissa = std::frexp(scale[i], &exponent);
            shift[i] = bits - exponent;
            mult[i] = (mantissa * pow(2, bits));
        }
    }
}

void updatePWLQuantParams(mv::Data::OpListIterator& op,
    std::vector<float>& inputScale) {

    if(!op->hasAttr("pwlQuantParams"))
        return;

    auto pwlQuant = op->get<mv::QuantizationParams>("pwlQuantParams");

    auto reQuantScale = std::vector<float>(inputScale.begin(), inputScale.end());
    auto quantSize = reQuantScale.size();

    pwlQuant.setScale(
        extendToK(quantSize, pwlQuant.getScale(), op->getName()));
    pwlQuant.setZeroPoint(
        extendToK(quantSize, pwlQuant.getZeroPoint(), op->getName()));

    // (input_scale * weight_scale) / output_scale
    std::transform(
        reQuantScale.begin(),
        reQuantScale.end(),
        pwlQuant.getScale().begin(),
        reQuantScale.begin(),
        std::divides<float>());

    std::vector<unsigned> reQuantShift(quantSize, 0);
    std::vector<unsigned> reQuantMult(quantSize, 1);

    computeQuantMultShift(reQuantScale, reQuantShift, reQuantMult);

    op->get<mv::QuantizationParams>("pwlQuantParams").quantize(reQuantShift, reQuantMult);
}

void computeTensorsQuantParams(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor& td, mv::Element&, mv::Element&)
{

    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    mv::OpModel om(model);
    mv::DataModel dm(model);

    auto operations = om.topologicalSort();
    for(auto op = operations.begin(); op != operations.end(); ++op)
    {
        auto opIt = *op;
        if(opIt->getOpType() != "DPUTask")
            continue;
        std::string taskOp = opIt->get<std::string>("taskOp");

        bool isEltwise = taskOp == "Eltwise";
        bool isEltwiseMult = false;
        bool isEltwiseAddSub = false;
        if(isEltwise)
        {
            auto eltwiseType = opIt->get<std::string>("eltwiseType");
            if(eltwiseType == "Add" || eltwiseType == "Subtract" || eltwiseType == "And")
                isEltwiseAddSub = true;
            if(eltwiseType == "Multiply")
                isEltwiseMult = true;
        }
        bool isConv = (taskOp == "Conv" || taskOp == "DepthwiseConv" || taskOp == "ChannelMajorConvolution");
        if (isConv || taskOp == "MaxPool" || isEltwiseMult || isEltwiseAddSub)
        {
            auto output = opIt->getOutputTensor(0);
            auto input = opIt->getInputTensor(0);
            bool floatScaleTable = false;

            if(td.generalTargetConfigs().floatScaleTable)
            {
                auto inputDType = opIt->getInputTensor(0)->getDType();
                floatScaleTable = inputDType == mv::DType("Float16") || inputDType == mv::DType("BFloat16");
            }
            auto outputChannels = output->getShape()[mv::IO_CHANNEL_DIMENSION];
            outputChannels = mv::round_up(outputChannels, 16);

            std::vector<unsigned> shift(outputChannels, 0);
            std::vector<unsigned> mult(outputChannels, 0);

            if (output->isQuantized() && input->isQuantized())
            {
                // Quantization for Gemmlowp output
                // S1 = weight scale
                // S2 = input activation scale
                // S3 = output activation scale
                // m  = (S1 * S2)/S3, scale for MAC output
                // zeroPointScaled = output zero point scaled to MAC output precision
                // biasScaled = bias scaled to MAC output precision

                auto& inputQuantization = input->get<mv::QuantizationParams>("quantParams");
                //inputQuantization.extendParamsToOutputChannelSize(outputChannels);

                auto scale = extendToK(outputChannels, inputQuantization.getScale(), input->getName());
                std::vector<float> S2(scale.begin(), scale.end());

                std::vector <float> S3(outputChannels, 1);
                std::vector <float> floatScale(outputChannels, std::pow(2, -16));
                std::vector <int64_t> zeroPoint(outputChannels, 0);
                bool outputOfAccWithBias = true;
                mv::QuantizationParams &outputQuantization = output->get<mv::QuantizationParams>("quantParams");
                if (!(output->hasAttr("dType") && output->getDType() == mv::DType("Int32")))
                {
                    //NOTE: Here I compute all the quantization parameters like they should be
                    //in order to dequantize the output of the DPU TASK, as Int32
                    //32 bit is not a correct statement, practically it is Int33 as
                    //the output of the accumulator+bias is an int33 number, but in
                    //graphfile and everywhere will be noted as Int32 for better exposing to the user
                    outputOfAccWithBias = false;
                    scale = extendToK(outputChannels, outputQuantization.getScale(), output->getName());
                    S3 = {scale.begin(), scale.end()};

                    auto zeroPointU =  extendToK(outputChannels, outputQuantization.getZeroPoint(), output->getName());
                    zeroPoint = {zeroPointU.begin(), zeroPointU.end()};
                }
                if (opIt->hasAttr("mixedToFloat") && opIt->get<bool>("mixedToFloat"))
                    S3 = {floatScale.begin(), floatScale.end()};

                bool isPooling = taskOp == "MaxPool";
                //Workaround for HW bug #227
                if (isPooling)
                {
                    auto inZP = extendToK(outputChannels, inputQuantization.getZeroPoint(), input->getName());
                    std::vector<int64_t> inputZeroPoint(inZP.begin(), inZP.end());
                    std::transform(zeroPoint.begin(), zeroPoint.end(), inputZeroPoint.begin(), zeroPoint.begin(), std::minus<int32_t>());
                }

                auto m = S2;

                if ((opIt->hasAttr("hasWeights") && opIt->get<bool>("hasWeights")) || isEltwiseMult)
                {
                    auto weights = opIt->getInputTensor(1);
                    if (weights->hasAttr("quantParams"))
                    {
                        auto& weightsQuantization = weights->get<mv::QuantizationParams>("quantParams");
                        scale = extendToK(outputChannels, weightsQuantization.getScale(), weights->getName());
                        std::vector<float> S1(scale.begin(), scale.end());
                        //S1*S2
                        std::transform(m.begin(), m.end(), S1.begin(), m.begin(), std::multiplies<float>());
                        if (output->hasAttr("dType") && output->get<mv::DType>("dType") == mv::DType("Int32"))
                        {
                            std::vector<double> output_scale;
                            output_scale = inputQuantization.getScale();
                            std::transform(output_scale.begin(), output_scale.end(),
                                        weightsQuantization.getScale().begin(), output_scale.begin(), std::multiplies<double>());
                            outputQuantization.setScale(output_scale);
                        }
                    }
                    else
                    {
                        if (output->hasAttr("dType") && output->get<mv::DType>("dType") == mv::DType("Int32"))
                        {
                            outputQuantization.setScale(inputQuantization.getScale());
                        }
                    }
                }
                else if (isEltwiseAddSub) //Add Subtract
                {
                    auto input2 = opIt->getInputTensor(1);
                    auto& input2Quantization = input2->get<mv::QuantizationParams>("quantParams");
                    auto input1Scale = extendToK(outputChannels, inputQuantization.getScale(), input->getName());
                    auto input2Scale = extendToK(outputChannels, input2Quantization.getScale(), input2->getName());
                    std::vector<unsigned> shift(outputChannels, 0);
                    std::vector<unsigned> mult(outputChannels, 0);
                    if (std::fabs(input1Scale[0] - input2Scale[0]) > std::numeric_limits<double>::epsilon())
                    {
                        if (td.generalTargetConfigs().allowMultipleInputScales)
                        {
                            //one of the scales will be 1 after adjustment
                            std::vector<unsigned> scale1Shift(outputChannels, 14);
                            std::vector<unsigned> scale1Mult(outputChannels, 16384);

                            //Scaling is per-tensor (not per channel) => all values are the same
                            if (input1Scale[0] >= input2Scale[0])
                            {
                                    std::vector <float> scale2Adjusted(outputChannels);

                                    std::transform(input2Scale.begin(), input2Scale.end(), input1Scale.begin(), scale2Adjusted.begin(), std::divides<double>());
                                    computeQuantMultShift(scale2Adjusted, shift, mult);
                                    input2->set<std::vector<unsigned>>("preAdjustedShift", input2Quantization.getShift());
                                    input2->set<std::vector<unsigned>>("preAdjustedMult", input2Quantization.getMult());
                                    input2Quantization.quantize(shift, mult);

                                    input->set<std::vector<unsigned>>("preAdjustedShift", inputQuantization.getShift());
                                    input->set<std::vector<unsigned>>("preAdjustedMult", inputQuantization.getMult());
                                    inputQuantization.quantize(scale1Shift, scale1Mult);
                                    //m = S2 Already done above
                            }
                            else
                            {
                                    std::vector <float> scale1Adjusted(outputChannels);
                                    std::transform(input1Scale.begin(), input1Scale.end(), input2Scale.begin(), scale1Adjusted.begin(), std::divides<double>());
                                    computeQuantMultShift(scale1Adjusted, shift, mult);
                                    //save old values for weightsTable calculations
                                    input->set<std::vector<unsigned>>("preAdjustedShift", inputQuantization.getShift());
                                    input->set<std::vector<unsigned>>("preAdjustedMult", inputQuantization.getMult());
                                    inputQuantization.quantize(shift, mult);

                                    input2->set<std::vector<unsigned>>("preAdjustedShift", input2Quantization.getShift());
                                    input2->set<std::vector<unsigned>>("preAdjustedMult", input2Quantization.getMult());
                                    input2Quantization.quantize(scale1Shift, scale1Mult);
                                    auto scaleExtended = extendToK(outputChannels, input2Scale, input2->getName());
                                    std::vector<float> scaleFloat(scaleExtended.begin(), scaleExtended.end());
                                    m = scaleFloat;
                            }

                        }
                        else
                        {
                            auto size = input1Scale.size();
                            std::vector <double> scaleDifference(size), absRelativeErrorScale(size), relativeErrorScale(size);
                            std::transform(input1Scale.begin(), input1Scale.end(), input2Scale.begin(), scaleDifference.begin(), std::minus<double>());

                            double (*fabs)(double) = &std::fabs;
                            std::transform(scaleDifference.begin(), scaleDifference.end(), input1Scale.begin(), relativeErrorScale.begin(), std::divides<double>());
                            std::transform(relativeErrorScale.begin(),relativeErrorScale.end(), absRelativeErrorScale.begin(), fabs);
                            for (auto it = absRelativeErrorScale.begin(); it != absRelativeErrorScale.end(); it++)
                            {
                                if (*it > 0.01)
                                    throw mv::RuntimeError(om, opIt->getName() + ": The relative difference in the input scales is > 1%. This is not supported for Eltwise operation."
                                                    + std::to_string(input1Scale[0]) + " " + std::to_string(input2Scale[0]));
                            }
                        }
                    }
                }

                updatePWLQuantParams(opIt, m);

                // m / S3
                std::transform(m.begin(), m.end(), S3.begin(), m.begin(), std::divides<float>());
                if (floatScaleTable)
                    opIt->set<std::vector<float>>("floatScale", m);

                if (outputOfAccWithBias)
                {
                    for (size_t i = 0; i < m.size(); i++)
                    {
                        shift[i] = 0;
                        mult[i] = 1;
                    }
                }
                else
                    computeQuantMultShift(m, shift, mult);

                outputQuantization.quantize(shift, mult);

                // Custom PWL table, supports only Leaky Relu at this point
                if (opIt->hasAttr("postOpTypes"))
                {
                    signed postShift = 0;
                    // TODO: post_shift of 4 is just the current computed value for the sole
                    // case of leaky relu done in PWL, need to provide a path from target descritptor
                    // to pass logic to parametrize each custom PWL table entries.
                    // We may just as well have two custom PWL tables during inference with diff params
                    std::unordered_map<std::string, int> dpuPwlScale = {
                        {"LeakyRelu", 4},
                        {"Mish", 0},
                    };
                    //"FLEXARB"
                    auto postOps = opIt->get<std::vector<std::string>>("postOpTypes");
                    auto ppeIterator = std::find_if(postOps.begin(), postOps.end(), mv::ControlModel::isDpuPwl);

                    if (ppeIterator != postOps.end()) {
                        postShift = dpuPwlScale[*ppeIterator];
                    }

                    if (ppeIterator != postOps.end() && *ppeIterator == "Mish") {
                        const auto outQuantParams = output->get<mv::QuantizationParams>("quantParams");
                        const std::map<int32_t, int> MISH_SCALES = {
                            {388125, 3},
                            {355313, 1},
                            {189063, 1},
                            {307500, 1},
                            {254844, 1},
                        };
                        int32_t max_quant = std::round(outQuantParams.getMax().at(0) * 10000.f);
                        if (MISH_SCALES.count(max_quant) > 0) {
                            postShift = MISH_SCALES.at(max_quant);
                        } else {
                            throw std::runtime_error("compute_tensor_quant: Couldn't find max_quant: " + std::to_string(max_quant));
                        }
                    }

                    mv::QuantizationParams postQuantization = {
                            {outputQuantization.getZeroPoint()},
                            {outputQuantization.getScale()},
                            {outputQuantization.getMin()},
                            {outputQuantization.getMax()},
                            shift,
                            mult,
                            postShift
                        };
                    output->set<mv::QuantizationParams>("quantParams", postQuantization);
                }
            }
        }
    }
}

