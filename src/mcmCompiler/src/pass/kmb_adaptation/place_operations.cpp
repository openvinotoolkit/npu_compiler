#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/pass/pass_utils.hpp"
#include "include/mcm/utils/custom_strings.hpp"
#include "include/mcm/tensor/quantization_params.hpp"
#include "mcm/utils/custom_math.hpp"
#include <numeric>
#include <cmath>

void placeEltwiseDequantize(mv::OpModel & om, mv::Data::OpListIterator task);
static void placementOfOps(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);
void placeNeutralMaxPoolBefore(const mv::pass::PassEntry &pass, mv::ComputationModel &model, mv::TargetDescriptor &, mv::Element &, mv::Element &);

namespace mv
{

    namespace pass
    {

        MV_REGISTER_PASS(PlaceNeutralMaxPoolBefore)
        .setFunc(placeNeutralMaxPoolBefore)
        .setDescription(
            "This pass handles a specific case in yoloV3/Unet/Deblur, when an UPA Op goes into a concat."
        );

        MV_REGISTER_PASS(PlacementOfOps)
        .setFunc(placementOfOps)
        .setDescription(
            "This pass handles the DPU's output Tensor Data Type."
        );
    }
}

void placeNeutralMaxPoolBefore(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{
    mv::OpModel om(model);
    auto concats = om.getOps("Concat");

    for (auto& concatOp : concats)
    {
        auto numInputs = concatOp->getInputTensor().size();
        unsigned short numUPAOps = 0;

        for (size_t i = 0; i < numInputs; i++)
        {
            auto sourceOp = om.getSourceOp(concatOp->getInputTensor(i));
            if (sourceOp->isUPA() || sourceOp->getOpType() == "UPATask")
                numUPAOps++;
        }
        if (numUPAOps == 0 || numUPAOps == numInputs) //no mixed upa + dpu
            continue;

        for (size_t i = 0; i < numInputs; i++)
        {
            auto sourceOp = om.getSourceOp(concatOp->getInputTensor(i));
            if (sourceOp->isUPA() || sourceOp->getOpType() == "UPATask")
            {
                auto inputFlow = concatOp.leftmostInput();
                auto outputTensor = sourceOp->getOutputTensor(0);
                auto neutralMaxPool = om.maxPool(concatOp->getName() + "MaxPool", outputTensor, {1,1}, {1,1}, {0, 0, 0, 0}, false);
                neutralMaxPool->setDType(mv::DType("UInt8"));
                neutralMaxPool->setQuantParams(outputTensor->getQuantParams());
                auto maxPoolOp = om.getSourceOp(neutralMaxPool);
                maxPoolOp->set<unsigned>("opId", sourceOp->get<unsigned>("opId"));
                while(inputFlow != om.flowEnd())
                {
                    auto tensor = inputFlow->getTensor();
                    if (tensor->getName() == outputTensor->getName())
                    {
                        auto slot = inputFlow->get<size_t>("sinkInput");
                        om.undefineFlow(inputFlow);
                        concatOp->setInputTensor(neutralMaxPool, slot, false);
                        om.defineFlow(neutralMaxPool, concatOp, slot);
                        break;
                    }
                    ++inputFlow;
                }
            }
        }
    }

}

void placeEltwiseDequantize(mv::OpModel & om, mv::Data::OpListIterator task)
{
    auto quantParams = task->getInputTensor(0)->getQuantParams();
    auto neutralCopy = om.copy(task->getName() + "Neutral", task->getInputTensor(0));
    neutralCopy->setDType(mv::DType("UInt8"));
    neutralCopy->setQuantParams(quantParams);
    auto neutralCopyOp = om.getSourceOp(neutralCopy);
    neutralCopyOp->set<unsigned>("opId", task->get<unsigned>("opId"));

    std::vector<mv::Data::TensorIterator> andInputs = {task->getInputTensor(0), neutralCopy};
    auto placeEltwiseDequantize = om.eltwise(task->getName() + "AND_Conversion", andInputs, "And");
    placeEltwiseDequantize->setQuantParams({{0}, {1.0f}, {}, {}});
    auto placeEltwiseDequantizeOp = om.getSourceOp(placeEltwiseDequantize);

    placeEltwiseDequantizeOp->getInputTensor(0)->setDType(mv::DType("UInt8"));
    placeEltwiseDequantizeOp->getInputTensor(1)->setDType(mv::DType("UInt8"));
    placeEltwiseDequantizeOp->getOutputTensor(0)->setDType(mv::DType("Float16"));
    placeEltwiseDequantizeOp->set<bool>("mixedToFloat", true);

    placeEltwiseDequantizeOp->set<unsigned>("opId", task->get<unsigned>("opId"));
    task->setInputTensor(placeEltwiseDequantize, 0, false);
    om.defineFlow(placeEltwiseDequantize, task, 0);
}

void placementOfOps(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)

    mv::OpModel om(model);
    mv::DataModel dm(model);

    auto convOps = om.getOps("Conv");
    auto convDepthwiseOps = om.getOps("DepthwiseConv");
    convOps.insert(convOps.end(),convDepthwiseOps.begin(),convDepthwiseOps.end());

    for (auto& opIt : convOps)
    {
        if (opIt->hasAttr("placeConversionToFloat"))
        {
            if (opIt->get<bool>("placeConversionToFloat"))
            {
                auto previousOpIt = om.getSourceOp(opIt->getInputTensor(0));
                std::vector<double> inputScale = opIt->getInputTensor(0)->get<mv::QuantizationParams>("quantParams").getScale();
                bool isHSwish = false;
                if (previousOpIt->getOpType() == "HSwish") {
                    isHSwish = true;
                    placeEltwiseDequantize(om, previousOpIt);
                    auto oldInputFlow = previousOpIt.leftmostInput();
                    while(oldInputFlow != om.flowEnd()) {
                        om.undefineFlow(oldInputFlow);
                        ++oldInputFlow;
                    }
                    // do not set FP16 for HSwish since it is UPA Taks and precision will be set later
                } else {
                    placeEltwiseDequantize(om, opIt);
                }
                //NOTE: For now take for granted that the next guy is a convolution
                opIt->set<bool>("floatPrecision", true);
                opIt->getOutputTensor(0)->setDType(mv::DType("Float16"));
                //NOTE: Do not care of the data type of input but it will be float so all
                //inputs and outputs need to be converted to float, populated need de-quantize!!!
                for(std::size_t i = 0; i < opIt->inputSlots(); ++i)
                    opIt->getInputTensor(i)->setDType(mv::DType("Float16"));
                opIt->getOutputTensor(0)->setDType(mv::DType("Float16"));
//                opIt->setDType(mv::DType("Float16"));
                bool hasBias = opIt->hasAttr("bias");

                // If this conv was a tiled conv, pass the conversion to the adds as well
                if(opIt->hasAttr("partitionedKernelToAdd"))
                {
                    if(opIt->get<bool>("partitionedKernelToAdd"))
                    {
                        auto partitionAdd = opIt.leftmostOutput().sink();
                        partitionAdd->getOutputTensor(0)->setDType(mv::DType("Float16"));
                    }
                }

                if (opIt->hasWeights())
                {
                    mv::Data::TensorIterator weightsTensor =  opIt->getInputTensor(1);
                    auto dequantFP16Weights = dequantizeWeightsToFP16(weightsTensor, opIt, om);

                    if (hasBias)
                    {
                        mv::Data::TensorIterator bias =  dm.getTensor(opIt->get<std::string>("bias"));
                        auto outputShape = opIt->getOutputTensor(0)->getShape();
                        std::vector<int64_t> biasData;
                        double biasOldScale, real_bias;
                        int64_t real_bias_fp16;
                        std::vector<double> weightsScale = opIt->getInputTensor(1)->get<mv::QuantizationParams>("quantParams").getScale();
                        weightsScale = extendToK(outputShape[mv::IO_CHANNEL_DIMENSION], weightsScale, bias->getName());
                        for (size_t k = 0; k < outputShape[mv::IO_CHANNEL_DIMENSION]; k++)
                        {
                            biasOldScale = weightsScale[k] * inputScale[0];
                            real_bias = ((int64_t) bias->at(k)) * biasOldScale;
                            real_bias_fp16 = mv::fp32_to_fp16(real_bias);
                            biasData.push_back(real_bias_fp16);
                        }
                        mv::Data::TensorIterator floatBias;
                        std::string floatBiasName = mv::createBiasName(opIt->getName() + "FP16_bias");
                        floatBias = dm.defineTensor(mv::Tensor(floatBiasName, bias->getShape(),
                                                     mv::DType("Float16"), bias->getOrder(), biasData, {{0},{1},{},{}}));
                        om.eraseAttr(opIt, "bias");
                        om.addAttr(opIt, "bias", floatBiasName);
                        bias->setDType(mv::DType("Float16"));
                    }
                    if (!isHSwish) {
                        for (auto sourceFlow = opIt.leftmostInput(); sourceFlow != om.flowEnd(); ++sourceFlow)
                        {
                            if (sourceFlow.source()->getName() == previousOpIt->getName())
                                om.undefineFlow(sourceFlow);
                        }
                    }
                    om.removeOp(om.getSourceOp(opIt->getInputTensor(1)));
                    opIt->setInputTensor(dequantFP16Weights, 1, false);
                    om.defineFlow(dequantFP16Weights, opIt, 1);
                    om.getSourceOp(opIt->getInputTensor(1))->set<unsigned>("opId", opIt->get<unsigned>("opId"));
                }
            }
        }
    }
}
