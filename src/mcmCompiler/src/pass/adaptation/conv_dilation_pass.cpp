#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/utils/custom_math.hpp"
#include "include/mcm/pass/pass_utils.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/tensor/tiling.hpp"
#include <algorithm>

#define SCALE_RANGE_0_1 0.00392156862745098
#define ZP_RANGE_0_1 0

//NOTE: There are 2 passes implemented for dilation convolution. The one solution is based to the idea that we push
//zero points inside the weight tensor of the convolution in order to simulate dilated conv and this pass is used by emulator.
//This idea might lead to really big kernel sizes so in order to implement dilation in kmb we use the storage element to slice/concat.
static void convDilationUsingWeightsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);
static void convDilationUsingStorageElementFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);
//NOTE: Dilation convolution using storage element, the idea here is that the dilated convolution
//will be splitted to dilation_factor^2 parallel convolutions. The input tensor has distance between its
//elements dilation_factor. Initially the plan was to concatenate them on CMX using storage element
//of the next operation, but this idea was rejected cause sometimes the next operation needs to be streamed.
//On the other hand the concatenation happens always with a generic mechanism of concatenating with 3D-dmas back to DDR.

namespace mv
{
    namespace pass
    {
        MV_REGISTER_PASS(ConvDilationUsingStorageElement)
            .setFunc(convDilationUsingStorageElementFcn)
            .setDescription(
                "This pass dilates a kernel using new method with SEPS");
    }
    namespace pass
    {
        MV_REGISTER_PASS(ConvDilationUsingWeights)
            .setFunc(convDilationUsingWeightsFcn)
            .setDescription(
                "This pass dilates a kernel using new method with SEPS");
    }
}

mv::Data::TensorIterator createDeconvSubConv(
        mv::OpModel& om, mv::Data::OpListIterator opIt, mv::Data::TensorIterator sourceWeights,
        std::array<unsigned short, 4> sub_padding, std::array<unsigned short, 2> anchor, std::string name,
        mv::Shape newShape, size_t subConvIdx, size_t i, size_t j,
        const std::vector<std::vector<std::pair<size_t, size_t>>>& flip_kernel_table) {
    mv::Data::TensorIterator subConv;
    bool hasBias = opIt->hasAttr("bias");
    auto stride = opIt->get<std::array<unsigned short, 2>>("stride")[0];
    auto weightsValue = sourceWeights->getIntData();
    auto quantParams = sourceWeights->getQuantParams();
    std::vector<int64_t> sliceWeightVector(newShape.totalSize());
    mv::Shape srcShape = sourceWeights->getShape();
    auto anchor_x= anchor[0];
    auto anchor_y= anchor[1];
    
    for (unsigned k = 0; k < newShape[mv::KERNEL_OUTPUT_CHANNELS]; ++k)
    {
        for (unsigned c = 0; c < newShape[mv::KERNEL_INPUT_CHANNELS]; ++c)
        {
            for (unsigned h = 0; h < newShape[mv::KERNEL_HEIGHT]; ++h)
            {
                for (unsigned w = 0; w < newShape[mv::KERNEL_WIDTH]; ++w)
                {
                    const size_t dstIdx = (k * newShape[mv::KERNEL_INPUT_CHANNELS] * newShape[mv::KERNEL_WIDTH] * newShape[mv::KERNEL_HEIGHT]) +
                                        (c * newShape[mv::KERNEL_WIDTH] * newShape[mv::KERNEL_HEIGHT]) +
                                        (h * newShape[mv::KERNEL_WIDTH]) +
                                        w;
                    
                    auto i_= (anchor_x - i)%stride + h * stride;
                    auto j_= (anchor_y - j)%stride + w * stride;
                    const size_t srcIdx = (k * srcShape[mv::KERNEL_INPUT_CHANNELS] * srcShape[mv::KERNEL_WIDTH] * srcShape[mv::KERNEL_HEIGHT]) +
                                        (c * srcShape[mv::KERNEL_WIDTH] * srcShape[mv::KERNEL_HEIGHT]) +
                                        flip_kernel_table[i_][j_].first * srcShape[mv::KERNEL_WIDTH] +
                                        flip_kernel_table[i_][j_].second;
                    sliceWeightVector[dstIdx] = weightsValue[srcIdx];
                }
            }
        }
    }

    auto sliceWeight = om.constantInt(
                        opIt->getName() + "_deconvSlice_" + std::to_string(subConvIdx),
                        sliceWeightVector,
                        newShape,
                        sourceWeights->get<mv::DType>("dType"),
                        mv::Order("NCHW"));
    sliceWeight->setQuantParams(quantParams);
    auto sliceWeightOp = om.getSourceOp(sliceWeight);
    mv::QuantizationParams params = opIt->getOutputTensor(mv::IO_TENSOR_OUTPUT)->getQuantParams();
    // Unet fix: make output dtype fp16 to assure Quantize layer (L497) is inserted after DMA-Concat
    auto out_dtype = sourceWeights->isFloatingPointType() ? mv::DType("Float16")
                                                          : opIt->getOutputTensor(mv::IO_TENSOR_OUTPUT)->getDType();

    if (opIt->get<bool>("is_depthwise") == false)
    {
        subConv = om.conv(name,
                opIt->getInputTensor(mv::IO_TENSOR_INPUT),
                sliceWeight,
                {1, 1},
                sub_padding,
                1,
                opIt->get<unsigned>("group"));
        subConv->setDType(out_dtype);
        subConv->setQuantParams(params);
    }
    else
    {
        subConv = om.depthwiseConv(name,
                opIt->getInputTensor(mv::IO_TENSOR_INPUT),
                sliceWeight,
                {1, 1},
                sub_padding,
                1);
        subConv->setDType(out_dtype);
        subConv->setQuantParams(params);
    }

    auto subConvOp = om.getSourceOp(subConv);
    subConvOp->set<bool>("DeconvSubConv", true);
    subConvOp->set<unsigned>("originalDilationFactor", stride);
    subConvOp->set<std::string>("parentOp", opIt->getName());
    subConvOp->set<unsigned>("subConvIndex", subConvIdx);
    subConvOp->set<std::vector<std::size_t>>("subConvsCoordinates", {i, j});
    if(opIt->hasAttr("opId"))
    {
        unsigned currentOpId = opIt->get<unsigned>("opId");
        subConvOp->set<unsigned>("opId", currentOpId);
        sliceWeightOp->set<unsigned>("opId", currentOpId);
    }
    if (hasBias)
    {
        om.addAttr(subConvOp, "bias", opIt->get<std::string>("bias"));
    }
    if (opIt->hasAttr("postOpTypes"))
    {
        subConvOp->set<std::vector<std::string>>("postOpTypes",
                                    opIt->get<std::vector<std::string>>("postOpTypes"));
    }

    return subConv;
}

mv::Data::TensorIterator createDilatedConvSubConv(mv::OpModel & om, mv::Data::OpListIterator opIt, mv::Data::TensorIterator sourceTensor,
                                                    std::array<unsigned short, 4> padding, std::string name, mv::Shape newShape, size_t subConvIdx, size_t i, size_t j)
{
    mv::Data::TensorIterator subConv;
    bool hasBias = opIt->hasAttr("bias");
    //TODO handle stride != 1
    auto stride = opIt->get<std::array<unsigned short, 2>>("stride");

    auto inputQuantParams = sourceTensor->getQuantParams();
    mv::Data::TensorIterator sliceInput = om.slice(opIt->getName() + "_dilatedSlice_" + std::to_string(subConvIdx),
                                 sourceTensor,
                                 {0, 0, 0, 0},
                                 newShape);
    sliceInput->setQuantParams(inputQuantParams);

    sliceInput->set<bool>("dilatedSlice", true);
    auto sliceInputOp = om.getSourceOp(sliceInput);
    sliceInputOp->set<bool>("dilatedSlice", true);

    auto dType = opIt->getOutputTensor(0)->getDType();
    auto quantParams = opIt->getOutputTensor(0)->getQuantParams();

    subConv = om.conv(name,
            sliceInput,
            opIt->getInputTensor(1),
            stride,
            padding,
            1,
            opIt->get<unsigned>("group"));

    subConv->setDType(dType);
    subConv->setQuantParams(quantParams);

    auto subConvOp = om.getSourceOp(subConv);
    subConvOp->set<bool>("DilatedSubConv", true);
    subConvOp->set<unsigned>("originalDilationFactor", opIt->get<unsigned>("dilationFactor"));
    sliceInput->set<mv::Shape>("originalShape", sourceTensor->getShape());
    subConv->set<mv::Shape>("originalShape", sourceTensor->getShape());
    subConvOp->set<mv::Shape>("originalShape", sourceTensor->getShape());
    subConvOp->set<std::string>("parentOp", opIt->getName());
    subConvOp->set<unsigned>("subConvIndex", subConvIdx);
    subConvOp->set<std::vector<std::size_t>>("subConvsCoordinates", {i, j});
    if(opIt->hasAttr("opId"))
    {
        unsigned currentOpId = opIt->get<unsigned>("opId");
        subConvOp->set<unsigned>("opId", currentOpId);
        sliceInputOp->set<unsigned>("opId", currentOpId);
    }
    if (hasBias)
    {
        om.addAttr(subConvOp, "bias", opIt->get<std::string>("bias"));
    }
    if (opIt->hasAttr("postOpTypes"))
    {
        subConvOp->set<std::vector<std::string>>("postOpTypes",
                                    opIt->get<std::vector<std::string>>("postOpTypes"));
    }

    return subConv;
}

std::array<unsigned short, 4> calcNewPadding(mv::Data::OpListIterator opIt, size_t newWidth, size_t newHeight)
{
    auto oldPadding = opIt->get<std::array<unsigned short, 4>>("padding");
    auto oldStride = opIt->get<std::array<unsigned short, 2>>("stride");
    auto kernelShape = opIt->getInputTensor(1)->getShape();

    if (oldPadding[0] == 0 && oldPadding[1] == 0 && oldPadding[2] == 0 && oldPadding[3] == 0)
    {
        //Valid padding
        return oldPadding;
    }
    else
    {

        //same padding
        //case of stride 1
        //p = ((f-1)*s -n +f )/2
        //unsigned int p1 = ((newWidth - 1) * oldStride[0] - kernelShape[mv::KERNEL_WIDTH] + newWidth)/2;
        //unsigned int p2 = ((newHeight - 1) * oldStride[1] - kernelShape[mv::KERNEL_HEIGHT] + newHeight)/2;

        /*
        Formuala for padding taken from here: https://www.pico.net/kb/what-is-the-difference-between-same-and-valid-padding-in-tf-nn-max-pool-of-tensorflow

        For 'SAME' option output dimensions and padding options are computed as:
        out_height 	= ceil(float(in_height) / float(strides[1]))
        out_width 	= ceil(float(in_width) / float(strides[2]))
        pad_along_height 	= max((out_height - 1) * strides[1] + filter_height - in_height, 0)
        pad_along_width 	= max((out_width - 1) * strides[2] + filter_width - in_width, 0)
        pad_top 	= pad_along_height // 2
        pad_bottom 	= pad_along_height - pad_top
        pad_left 	= pad_along_width // 2
        pad_right 	= pad_along_width - pad_left
        */

        unsigned int out_height = ceil(newHeight / oldStride[0]);
        unsigned int out_width = ceil(newWidth / oldStride[1]);
        auto pad_along_height 	= std::max(int((out_height - 1) * oldStride[0] + kernelShape[mv::KERNEL_HEIGHT] - newHeight), 0);
        auto pad_along_width 	= std::max(int((out_width - 1) * oldStride[1] + kernelShape[mv::KERNEL_WIDTH] - newWidth), 0);
        short unsigned int pad_top 	= pad_along_height / 2;
        short unsigned int pad_bottom 	= pad_along_height - pad_top;
        short unsigned int pad_left 	= pad_along_width / 2;
        short unsigned int pad_right 	= pad_along_width - pad_left;

        std::array<unsigned short, 4> padding = {pad_left, pad_right, pad_top, pad_bottom};
        return padding;
    }
}

void convDilationUsingStorageElementFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    using namespace mv;

    mv::OpModel om(model);
    mv::DataModel dm(model);
    auto convOps = om.getOps("Conv");

    for (auto& opIt : convOps)
    {
        auto dilationFactor = opIt->get<unsigned>("dilationFactor");
        if (dilationFactor > 1)
        {
            auto nonDilatedKernel = opIt->getInputTensor(1);
            auto nonDilatedKernelShape = nonDilatedKernel->getShape();
            auto inputTensor = opIt->getInputTensor(0);

            auto name = opIt->getName();

            auto originalShape = inputTensor->getShape();
            std::vector<mv::Data::TensorIterator> subConvs;

            size_t width = originalShape[mv::IO_WIDTH_DIMENSION];
            size_t height = originalShape[mv::IO_HEIGHT_DIMENSION];
            size_t sliceWidth = std::ceil(((double)width)/dilationFactor);
            size_t sliceHeight = std::ceil(((double)height)/dilationFactor);
            std::array<unsigned short, 4> padding = calcNewPadding(opIt, sliceWidth, sliceHeight);

            //Create sub dilated convs
            size_t subConvIdx = 0;
            uint64_t leadingOffset = 0;
            mv::Shape newShape({sliceWidth, sliceHeight, nonDilatedKernelShape[mv::KERNEL_INPUT_CHANNELS], 1});
            for (size_t i = 0; i < dilationFactor; i++)
            {
                mv::Shape subConvShape = newShape;
                if (height%dilationFactor != 0 && i >= (height%dilationFactor))
                {
                    subConvShape[mv::IO_HEIGHT_DIMENSION] = newShape[mv::IO_HEIGHT_DIMENSION] - 1;
                }
                for (size_t j = 0; j < dilationFactor; j++)
                {
                    if (width%dilationFactor != 0 && j >= (width%dilationFactor))
                    {
                        subConvShape[mv::IO_WIDTH_DIMENSION] = newShape[mv::IO_WIDTH_DIMENSION] - 1;
                    }
                    subConvs.push_back(createDilatedConvSubConv(om, opIt, inputTensor, padding,
                        name + "_DilatedSubConv" + std::to_string(i)+"_"+std::to_string(j),
                        subConvShape, subConvIdx++, i, j));
                    subConvs[subConvs.size()-1]->set<uint64_t>("leadingOffset", leadingOffset);
                    leadingOffset += subConvs[subConvs.size()-1]->getShape().totalSize();
                }

            }

            // reconnect children to subgraph
            std::vector<mv::Data::OpListIterator> opsToLink;
            std::vector<std::size_t> inputSlots;
            for (mv::Data::FlowSiblingIterator sinkFlow(opIt.leftmostOutput()); sinkFlow != om.flowEnd(); ++sinkFlow)
            {
                opsToLink.push_back(sinkFlow.sink());
                inputSlots.push_back(sinkFlow->get<std::size_t>("sinkInput"));
            }

            auto quantParams = opIt->getOutputTensor(0)->getQuantParams();
            auto opId = opIt->get<unsigned>("opId");

            om.removeOp(opIt);

            mv::Data::TensorIterator concatIt;
            std::vector<mv::Data::TensorIterator> subConvsPerColumn;
            std::vector<mv::Data::TensorIterator> firstLevelConcats;
            pass.log(mv::Logger::MessageType::Debug, "Dilated Conv concat of concat case " +  name);

            for (size_t i = 0; i < dilationFactor; i++)
            {
                for (size_t j = 0; j < dilationFactor; j++)
                {
                    subConvsPerColumn.push_back(subConvs[i*dilationFactor + j]);
                }
                concatIt = om.implicitConcat(name + std::to_string(i) + "DDR_WIDTH_join",
                                subConvsPerColumn, "W");
                concatIt->setQuantParams(quantParams);
                om.getSourceOp(concatIt)->set<bool>("avoid_cmx_concat", true);

                om.getSourceOp(concatIt)->set<unsigned>("opId", opId);
                om.getSourceOp(concatIt)->set<bool>("dilatedWidthConcat", true);
                om.getSourceOp(concatIt)->set<size_t>("lineofConcatHeight", i);
                om.getSourceOp(concatIt)->set<unsigned>("dilationFactor", dilationFactor);
                firstLevelConcats.push_back(concatIt);
                subConvsPerColumn.clear();
            }
            concatIt = om.implicitConcat(name + "DDR_HEIGHT_join", firstLevelConcats, "H");
            concatIt->setQuantParams(quantParams);
            om.getSourceOp(concatIt)->set<bool>("avoid_cmx_concat", true);

            om.getSourceOp(concatIt)->set<unsigned>("opId", opId);
            om.getSourceOp(concatIt)->set<bool>("joinSimulation", true);
            om.getSourceOp(concatIt)->set<size_t>("dilationSubConvs", dilationFactor * dilationFactor);
            for (unsigned j = 0; j < opsToLink.size(); ++j)
            {
                opsToLink[j]->setInputTensor(concatIt, inputSlots[j], false);
                om.defineFlow(concatIt, opsToLink[j], inputSlots[j]);
                //NOTE: leading to a concat and will have a complex data index
                if ((opsToLink[j]->getOpType() == "Concat" || opsToLink[j]->getOpType() == "ImplicitConcat") &&
                    inputSlots[j] != 0)
                    om.getSourceOp(concatIt)->set<bool>("complexDataIndex", true);

            }
        }
    }

    // Current impelmentation only works for the case when kernel size is N time of stride, both on X and Y
    auto deconvOps = om.getOps("Deconv");
    for (auto & opIt : deconvOps)
    {
        bool is_transInterp = false;
        if (opIt->hasAttr("is_transInterp")) {
            is_transInterp = opIt->get<bool>("is_transInterp");
        }
        if (opIt->get<bool>("is_depthwise") && !is_transInterp)
            continue;
        bool isDepthwise = opIt->get<bool>("is_depthwise");
        auto nextOp = findSinkLayers(dm, opIt->getOutputTensor(0))[0];
        auto deconvKernel = opIt->getInputTensor(1);
        auto deconvKernelOp = om.getSourceOp(deconvKernel);
        auto deconvKernelShape = deconvKernel->getShape();
        auto inputTensor = opIt->getInputTensor(0);
        auto outputTensor = opIt->getOutputTensor(0);
        auto name = opIt->getName();

        std::vector<mv::Data::TensorIterator> subConvs;
        
        auto strides = opIt->get<std::array<unsigned short, 2>>("stride");
        assert(strides[0] == strides[1]);
        auto strideFactor = strides[0];
        auto quantParams = opIt->getOutputTensor(0)->getQuantParams();

        size_t width = deconvKernelShape[mv::IO_WIDTH_DIMENSION];
        size_t height = deconvKernelShape[mv::IO_HEIGHT_DIMENSION];
        size_t sliceWidth = std::ceil(((double)width)/strideFactor);
        size_t sliceHeight = std::ceil(((double)height)/strideFactor);

        // Create sub convs
        size_t subConvIdx = 0;
        uint64_t leadingOffset = 0;
        mv::Shape newShape({sliceWidth, sliceHeight, deconvKernelShape[mv::KERNEL_INPUT_CHANNELS], deconvKernelShape[mv::KERNEL_OUTPUT_CHANNELS]});
        // flip kernel for x and y axis!
        std::vector<std::vector<std::pair<size_t, size_t>>> FLIP_KERNEL_TABLE(height, std::vector<std::pair<size_t, size_t>>(width, {0,0}));
        for (size_t i = 0; i< height; ++i){
            for (size_t j = 0; j< width; ++j){
                FLIP_KERNEL_TABLE[i][j]= {height-1- i, width-1- j};
            }
        }
        // calculate the first sub-kerenl position
        unsigned short width_in= inputTensor->getShape()[mv::IO_WIDTH_DIMENSION];
        unsigned short height_in= inputTensor->getShape()[mv::IO_HEIGHT_DIMENSION];
        unsigned short width_out= outputTensor->getShape()[mv::IO_WIDTH_DIMENSION];
        unsigned short height_out= outputTensor->getShape()[mv::IO_HEIGHT_DIMENSION];
        // INPUT PADDING FORMULA: [ W + (Stride-1) * (W-1) + 2 * W_PAD - K] / 1 + 1 = W_OUT
        unsigned short pad_width= (width_out-1 + width- width_in - (strideFactor-1)*(width_in-1))/2;
        unsigned short pad_height= (height_out-1 + height- height_in - (strideFactor-1)*(height_in-1))/2;
        assert(pad_width == pad_height);
        pad_width= pad_height; // just avoid make warning error
        bool need_sub_padding= (pad_width- strideFactor >=0) ? true: false;
        std::array<unsigned short, 2> anchor;
        anchor[0]= anchor[1] = need_sub_padding ? pad_width - strideFactor :  pad_width;
        std::array<unsigned short, 4> sub_padding = {0, 0, 0 ,0};
        
        for (size_t i = 0; i < strideFactor; i++)
        {
            mv::Shape subConvShape = newShape;
            if (height%strideFactor != 0 && i >= (height%strideFactor))
            {
                subConvShape[mv::IO_HEIGHT_DIMENSION] = newShape[mv::IO_HEIGHT_DIMENSION] - 1;
            }
            for (size_t j = 0; j < strideFactor; j++)
            {
                if (width%strideFactor != 0 && j >= (width%strideFactor))
                {
                    subConvShape[mv::IO_WIDTH_DIMENSION] = newShape[mv::IO_WIDTH_DIMENSION] - 1;
                }
                
                if (need_sub_padding) {
                    // each sub-conv is a size-invariant convolution
                    // so (N + P - K)/1 + 1 == N => P = K-1
                    unsigned short sub_pad_width = subConvShape[mv::IO_WIDTH_DIMENSION] - 1;
                    unsigned short sub_pad_height = subConvShape[mv::IO_HEIGHT_DIMENSION] - 1;
                    if (i <= anchor[0] && j <= anchor[1]) {
                        sub_padding= {sub_pad_width, 0, sub_pad_height, 0};
                    } else if (i <= anchor[0] && j > anchor[1]) {
                        sub_padding= {0, sub_pad_width, sub_pad_height, 0};
                    } else if (i > anchor[0] && j <= anchor[1]) {
                        sub_padding= {sub_pad_width, 0, 0, sub_pad_height};
                    } else {
                        sub_padding= {0, sub_pad_width, 0, sub_pad_height};
                    }
                }
                
                subConvs.push_back(
                            createDeconvSubConv(om, opIt, deconvKernel, sub_padding, anchor,
                                                name + "_DeconvSubConv" + std::to_string(i) + "_" + std::to_string(j),
                                                subConvShape, subConvIdx++, i, j, FLIP_KERNEL_TABLE));
                subConvs[subConvs.size()-1]->set<uint64_t>("leadingOffset", leadingOffset);
                leadingOffset += subConvs[subConvs.size()-1]->getShape().totalSize();
            }
        }

        // reconnect children to subgraph
        std::vector<mv::Data::OpListIterator> opsToLink;
        std::vector<std::size_t> inputSlots;
        for (mv::Data::FlowSiblingIterator sinkFlow(opIt.leftmostOutput()); sinkFlow != om.flowEnd(); ++sinkFlow)
        {
            opsToLink.push_back(sinkFlow.sink());
            inputSlots.push_back(sinkFlow->get<std::size_t>("sinkInput"));
        }

        auto opId = opIt->get<unsigned>("opId");
        om.removeOp(opIt);
        om.removeOp(deconvKernelOp);

        mv::Data::TensorIterator concatIt;
        std::vector<mv::Data::TensorIterator> subConvsPerColumn;
        std::vector<mv::Data::TensorIterator> firstLevelConcats;

        for (size_t i = 0; i < strideFactor; i++)
        {
            for (size_t j = 0; j < strideFactor; j++)
            {
                subConvsPerColumn.push_back(subConvs[i*strideFactor + j]);
            }
            concatIt = om.implicitConcat(name + std::to_string(i) + "DDR_WIDTH_join", subConvsPerColumn, "W");
            concatIt->setQuantParams(quantParams);
            om.getSourceOp(concatIt)->set<unsigned>("opId", opId);
            om.getSourceOp(concatIt)->set<bool>("dilatedWidthConcat", true);
            om.getSourceOp(concatIt)->set<size_t>("lineofConcatHeight", i);
            om.getSourceOp(concatIt)->set<unsigned>("dilationFactor", strideFactor);
            firstLevelConcats.push_back(concatIt);
            subConvsPerColumn.clear();
        }
        concatIt = om.implicitConcat(name + "DDR_HEIGHT_join", firstLevelConcats, "H");
        concatIt->setQuantParams(quantParams);
        om.getSourceOp(concatIt)->set<unsigned>("opId", opId);
        om.getSourceOp(concatIt)->set<bool>("joinSimulation", true);
        om.getSourceOp(concatIt)->set<size_t>("dilationSubConvs", strideFactor * strideFactor);

        if ((isDepthwise == false && concatIt->getDType() == mv::DType("Float16")) && (nextOp->getOpType() != "Output" && nextOp->getOutputTensor(0)->getDType() == mv::DType("UInt8")))
        {
            auto dataUint8 = om.uPATaskQuantize("", {concatIt});
            dataUint8->setDType(mv::DType("UInt8"));
            dataUint8->setQuantParams(quantParams);
            if (concatIt->hasAttr("splitStrategy"))
                dataUint8->set<std::string>("splitStrategy", concatIt->get<std::string>("splitStrategy"));

            auto quantizeOp = om.getSourceOp(dataUint8);
            quantizeOp->set<unsigned>("opId", opId);

            for (unsigned j = 0; j < opsToLink.size(); ++j)
            {
                opsToLink[j]->setInputTensor(dataUint8, inputSlots[j], false);
                om.defineFlow(dataUint8, opsToLink[j], inputSlots[j]);
            }
        }
        else
        {
            for (unsigned j = 0; j < opsToLink.size(); ++j)
            {
                opsToLink[j]->setInputTensor(concatIt, inputSlots[j], false);
                om.defineFlow(concatIt, opsToLink[j], inputSlots[j]);
                if ((opsToLink[j]->getOpType() == "Concat" || opsToLink[j]->getOpType() == "ImplicitConcat") &&
                    inputSlots[j] != 0)
                    om.getSourceOp(concatIt)->set<bool>("complexDataIndex", true);
            }
        }
    }
}

void convDilationUsingWeightsFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{

    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    using namespace mv;

    mv::OpModel om(model);

    for (auto opIt = om.opBegin(); opIt != om.opEnd(); ++opIt)
    {
        if (opIt->getOpType() != "DepthwiseConv") continue;
        auto dilationFactor = opIt->get<unsigned>("dilationFactor");
        const bool dilate = dilationFactor > 1;
        opIt->set<bool>("dilatedUsingWeights", dilate);
        if (!dilate) continue;

        /*Get the kernel attributes*/
        auto nonDilatedKernel = opIt->getInputTensor(1);
        auto nonDilatedKernelWidth = nonDilatedKernel->getShape()[KERNEL_WIDTH];
        auto nonDilatedKernelHeight = nonDilatedKernel->getShape()[KERNEL_HEIGHT];
        auto nonDilatedKernelInputChannels = nonDilatedKernel->getShape()[KERNEL_INPUT_CHANNELS];
        auto nonDilatedKernelOutpuChannels = nonDilatedKernel->getShape()[KERNEL_OUTPUT_CHANNELS];

        /** Calculate dilated kernel shape
             *
             * dilatedWidth = kw + (kw - 1)(df - 1)
             * dilatedHeight = kh + (kh - 1)(df - 1)
             */
        mv::Shape dilatedKernelShape = mv::Shape({nonDilatedKernelWidth + (nonDilatedKernelWidth - 1) * (dilationFactor - 1),
                                                    nonDilatedKernelHeight + (nonDilatedKernelHeight - 1) * (dilationFactor - 1),
                                                    nonDilatedKernelInputChannels, nonDilatedKernelOutpuChannels});
        auto nonDilatedKernelOp = opIt.rightmostParent();
        unsigned currentOpId = nonDilatedKernelOp->get<unsigned>("opId");
        auto quantParams = nonDilatedKernelOp->getOutputTensor(0)->getQuantParams();
        /*Populate dilated tensor with zeros*/

        /*Create Dilated Kernel Tensor*/

        //build the dilated kernel with zero points corresponding to each channel - KMB does not support different zp per channel
        std::vector<int64_t> defaultData;
        if (quantParams.isEmpty() || quantParams.isInitial() || quantParams.isNeutral())
            defaultData = std::vector<int64_t>(dilatedKernelShape.totalSize(), 0);
        else
            defaultData = std::vector<int64_t>(dilatedKernelShape.totalSize(), quantParams.getZeroPoint(0));

        mv::Tensor dilatedKernel("dilatedKernel", dilatedKernelShape, nonDilatedKernel->getDType(), mv::Order(mv::Order::getColMajorID(dilatedKernelShape.ndims())), defaultData);

        for (unsigned oc = 0; oc < nonDilatedKernelOutpuChannels; ++oc)
            for (unsigned ic = 0; ic < nonDilatedKernelInputChannels; ++ic)
                for (unsigned kcolumn = 0; kcolumn < nonDilatedKernelHeight; ++kcolumn)
                    for (unsigned krow = 0; krow < nonDilatedKernelWidth; ++krow)
                        /*Copy non-dilated weights into the dilated kernel*/
                        if (krow != 0 || kcolumn != 0)
                            dilatedKernel.at({krow + (dilationFactor - 1) * krow, kcolumn + (dilationFactor - 1) * kcolumn, ic, oc}) = nonDilatedKernel->at({krow, kcolumn, ic, oc});
                        else
                            dilatedKernel.at({krow, kcolumn, ic, oc}) = nonDilatedKernel->at({krow, kcolumn, ic, oc});

        auto dilatedKernelOp = om.constantDataElement(
            nonDilatedKernelOp->getName() + "_Dilated",
            dilatedKernel.getData(),
            dilatedKernelShape,
            dilatedKernel.getDType(),
            dilatedKernel.getOrder());
        dilatedKernelOp->setQuantParams(quantParams);

        om.removeOp(nonDilatedKernelOp);
        om.defineFlow(dilatedKernelOp, opIt, 1);
        opIt->set<std::array<unsigned short, 2>>("kSize", {
                static_cast<unsigned short>(dilatedKernelShape[KERNEL_WIDTH]),
                static_cast<unsigned short>(dilatedKernelShape[KERNEL_HEIGHT]) } );
        opIt->setInputTensor(dilatedKernelOp, 1, false);
        opIt->set<unsigned>("dilationFactor", 1);
        auto DilatedKernelOpFetched = opIt.rightmostParent();
        DilatedKernelOpFetched->set<unsigned>("opId", currentOpId);
    }

}
