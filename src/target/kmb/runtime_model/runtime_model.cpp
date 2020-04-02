#include "include/mcm/target/kmb/runtime_model/runtime_model.hpp"
#include "include/mcm/op_model.hpp"
#include "contrib/flatbuffers/include/flatbuffers/util.h"
#include "include/mcm/base/exception/argument_error.hpp"
#include "include/mcm/utils/warning_manager.hpp"
#include "include/mcm/utils/custom_math.hpp"
#include "include/mcm/utils/custom_strings.hpp"
#include <fstream>
#include <iostream>

const std::unordered_map<std::string, MVCNN::DType> mv::RuntimeModel::dTypeMapping_ =
{
    {"Float64", MVCNN::DType::DType_FP64},
    {"Float32", MVCNN::DType::DType_FP32},
    {"Float16", MVCNN::DType::DType_FP16},
    {"Float8", MVCNN::DType::DType_FP8},
    {"UInt64", MVCNN::DType::DType_U64},
    {"UInt32", MVCNN::DType::DType_U32},
    {"UInt16", MVCNN::DType::DType_U16},
    {"UInt8", MVCNN::DType::DType_U8},
    {"Int64", MVCNN::DType::DType_I64},
    {"Int32", MVCNN::DType::DType_I32},
    {"Int16", MVCNN::DType::DType_I16},
    {"Int8", MVCNN::DType::DType_I8},
    {"Int4", MVCNN::DType::DType_I4},
    {"Int2", MVCNN::DType::DType_I2},
    {"Int2X", MVCNN::DType::DType_I2X},
    {"Int4X", MVCNN::DType::DType_I4X},
    {"Bin", MVCNN::DType::DType_BIN},
    {"Log", MVCNN::DType::DType_LOG}
};

const std::unordered_map<std::string, MVCNN::MemoryLocation> mv::RuntimeModel::memoryLocationMapping_ =
{
    {"ProgrammableInput", MVCNN::MemoryLocation::MemoryLocation_ProgrammableInput},
    {"ProgrammableOutput", MVCNN::MemoryLocation::MemoryLocation_ProgrammableOutput},
    {"VPU_DDR_Heap", MVCNN::MemoryLocation::MemoryLocation_VPU_DDR_Heap},
    {"GraphFile", MVCNN::MemoryLocation::MemoryLocation_GraphFile},
    {"VPU_CMX_NN", MVCNN::MemoryLocation::MemoryLocation_VPU_CMX_NN},
    {"VPU_CMX_UPA", MVCNN::MemoryLocation::MemoryLocation_VPU_CMX_UPA},
    {"VPU_DDR_BSS", MVCNN::MemoryLocation::MemoryLocation_VPU_DDR_BSS}
};

const std::unordered_map<std::string, MVCNN::DPULayerType> mv::RuntimeModel::dpuLayerMapping_ =
{
    {"Conv",MVCNN::DPULayerType::DPULayerType_CONV},
    {"DepthwiseConv",MVCNN::DPULayerType::DPULayerType_DWCONV},
    {"MaxPool",MVCNN::DPULayerType::DPULayerType_MAXPOOL},
    {"AveragePool",MVCNN::DPULayerType::DPULayerType_AVEPOOL},
    {"FullyConnected",MVCNN::DPULayerType::DPULayerType_FCL},
    {"Eltwise",MVCNN::DPULayerType::DPULayerType_ELTWISE},
    {"Identity",MVCNN::DPULayerType::DPULayerType_IDENTITY},
    {"ChannelMajorConvolution",MVCNN::DPULayerType::DPULayerType_CMCONV}
};

const std::unordered_map<mv::PPELayerTypeEnum, MVCNN::PPELayerType, mv::EnumClassHash> mv::RuntimeModel::ppeLayerTypeMapping_ =
{
   {PPELayerType_STORE, MVCNN::PPELayerType::PPELayerType_STORE},
   {PPELayerType_LOAD, MVCNN::PPELayerType::PPELayerType_LOAD},
   {PPELayerType_CLEAR, MVCNN::PPELayerType::PPELayerType_CLEAR},
   {PPELayerType_NOOP, MVCNN::PPELayerType::PPELayerType_NOOP},
   {PPELayerType_HALT, MVCNN::PPELayerType::PPELayerType_HALT},
   {PPELayerType_ADD, MVCNN::PPELayerType::PPELayerType_ADD},
   {PPELayerType_SUB, MVCNN::PPELayerType::PPELayerType_SUB},
   {PPELayerType_MULT, MVCNN::PPELayerType::PPELayerType_MULT},
   {PPELayerType_RELU, MVCNN::PPELayerType::PPELayerType_RELU},
   {PPELayerType_RELUX, MVCNN::PPELayerType::PPELayerType_RELUX},
   {PPELayerType_LPRELU, MVCNN::PPELayerType::PPELayerType_LPRELU},
   {PPELayerType_MAXIMUM, MVCNN::PPELayerType::PPELayerType_MAXIMUM},
   {PPELayerType_MINIMUM, MVCNN::PPELayerType::PPELayerType_MINIMUM},
   {PPELayerType_CEIL, MVCNN::PPELayerType::PPELayerType_CEIL},
   {PPELayerType_FLOOR, MVCNN::PPELayerType::PPELayerType_FLOOR},
   {PPELayerType_AND, MVCNN::PPELayerType::PPELayerType_AND},
   {PPELayerType_OR, MVCNN::PPELayerType::PPELayerType_OR},
   {PPELayerType_XOR, MVCNN::PPELayerType::PPELayerType_XOR},
   {PPELayerType_NOT, MVCNN::PPELayerType::PPELayerType_NOT},
   {PPELayerType_ABS, MVCNN::PPELayerType::PPELayerType_ABS},
   {PPELayerType_NEG, MVCNN::PPELayerType::PPELayerType_NEG},
   {PPELayerType_POW, MVCNN::PPELayerType::PPELayerType_POW},
   {PPELayerType_EXP, MVCNN::PPELayerType::PPELayerType_EXP},
   {PPELayerType_SIGMOID, MVCNN::PPELayerType::PPELayerType_SIGMOID},
   {PPELayerType_TANH, MVCNN::PPELayerType::PPELayerType_TANH},
   {PPELayerType_SQRT, MVCNN::PPELayerType::PPELayerType_SQRT},
   {PPELayerType_RSQRT, MVCNN::PPELayerType::PPELayerType_RSQRT},
   {PPELayerType_FLEXARB, MVCNN::PPELayerType::PPELayerType_FLEXARB}
};

template <typename T1, typename T2>
void setIfPresent(T1& fieldToFill, mv::Element& compilationDescriptor, const std::string& key)
{
    if(compilationDescriptor.hasAttr(key))
        fieldToFill = compilationDescriptor.get<T2>(key);
}

//Note: 16/8 used below depend on the dtype
void mv::RuntimeModel::alignTensor(mv::ComputationModel& cm, std::unique_ptr<MVCNN::TensorReferenceT>& tensorT, mv::Tensor& tensor, const size_t dimension, bool padFinalOutput)
{

    if(dimension == IO_CHANNEL_DIMENSION) 
    {
        auto globalConfigParams = cm.getGlobalConfigParams();
        int pad = tensor.computeAppropriatePadding();
        std::vector<std::size_t> dimensions = tensor.getShape();
        auto outputChannelsPadded = mv::round_up(dimensions[mv::IO_CHANNEL_DIMENSION], pad);
        dimensions = {dimensions[mv::IO_WIDTH_DIMENSION], dimensions[mv::IO_HEIGHT_DIMENSION], outputChannelsPadded, dimensions[mv::IO_BATCH_DIMENSION]};
        auto numericStrides = tensor.getOrder().computeByteStrides(mv::Shape(dimensions), tensor.getDType().getSizeInBits() / 8);
        numericStrides.push_back(tensor.getDType().getSizeInBits() / 8);
        std::reverse(dimensions.begin(), dimensions.end());
        std::reverse(numericStrides.begin(), numericStrides.end());
        tensorT->strides = numericStrides; // NOTE: Maybe directly bufferIt->computeStrides() in the future
        if(padFinalOutput)
            tensorT->dimensions = std::vector<uint32_t>(dimensions.begin(), dimensions.end());
    }
    else if (dimension == IO_WIDTH_DIMENSION) 
    {
        std::vector<std::size_t> dimensions = tensor.getShape();
        auto widthPadded = mv::round_up(dimensions[mv::IO_WIDTH_DIMENSION], 16);
        dimensions = {widthPadded, dimensions[mv::IO_HEIGHT_DIMENSION],dimensions[mv::IO_CHANNEL_DIMENSION] , dimensions[mv::IO_BATCH_DIMENSION]};
        auto numericStrides = tensor.getOrder().computeByteStrides(mv::Shape(dimensions), tensor.getDType().getSizeInBits() / 8);
        numericStrides.push_back(tensor.getDType().getSizeInBits() / 8);
        std::reverse(dimensions.begin(), dimensions.end());
        std::reverse(numericStrides.begin(), numericStrides.end());
        tensorT->strides = numericStrides; 

    }    
}

MVCNN::DType mv::RuntimeModel::convertDtype(const mv::DType& dtype)
{
    return dTypeMapping_.at(dtype.toString());
}

MVCNN::MemoryLocation mv::RuntimeModel::convertAllocatorToMemoryLocale(const std::string& allocatorName)
{
    return memoryLocationMapping_.at(allocatorName);
}

MVCNN::PPELayerType mv::RuntimeModel::convertPPELayerType(PPELayerTypeEnum ppe)
{
    return ppeLayerTypeMapping_.at(ppe);
}


std::unique_ptr<MVCNN::GraphNodeT> mv::RuntimeModel::buildGraphNodeT(mv::ComputationModel &cm, mv::Element&, mv::Data::OpListIterator op)
{
    std::unique_ptr<MVCNN::GraphNodeT> toBuild = std::unique_ptr<MVCNN::GraphNodeT>(new MVCNN::GraphNodeT());

    mv::OpModel opModel(cm);
    toBuild->name = op->getName();
    toBuild->thisID = op->get<unsigned>("opId");

    for (auto nextChildOp = op.leftmostChild(); nextChildOp != opModel.opEnd(); ++nextChildOp)
        toBuild->sinkID.push_back(nextChildOp->get<unsigned>("opId"));

    for (auto nextParentOp = op.leftmostParent(); nextParentOp != opModel.opEnd(); ++nextParentOp)
        toBuild->sourceID.push_back(nextParentOp->get<unsigned>("opId"));

    return toBuild;
}

std::unique_ptr<MVCNN::SourceStructureT> mv::RuntimeModel::buildSourceStructureT(mv::ComputationModel &cm, mv::Element &compilationDescriptor)
{
    std::unique_ptr<MVCNN::SourceStructureT> toBuild = std::unique_ptr<MVCNN::SourceStructureT>(new MVCNN::SourceStructureT());

    mv::OpModel opModel(cm);
    auto inputOp = opModel.getInput();
    toBuild->first_ID.push_back(inputOp->get<unsigned>("opId"));
    toBuild->nodes = std::vector<std::unique_ptr<MVCNN::GraphNodeT>>(opModel.opsCount());
    unsigned i = 0;

    //auto ops = opModel.topologicalSort();
    for(auto opIt = opModel.opBegin(); opIt != opModel.opEnd(); ++opIt)
        toBuild->nodes[i++] = buildGraphNodeT(cm, compilationDescriptor, opIt);

    return toBuild;
}

std::vector<unsigned> mv::RuntimeModel::reduceQuantVector_(std::vector<unsigned> inVec)
{
    if (inVec.size() > 1)
    {
        auto firstVal = inVec[0];
        auto onlyOneValue = true;
        for (size_t i = 1; i < inVec.size(); i++)
            if (firstVal != inVec[i])
                onlyOneValue = false;
        if (onlyOneValue)
        {
            inVec.clear();
            inVec.push_back(firstVal);
        }
    }
    return inVec;
}

//build tensorReference for Tensors - 1 cluster case
std::unique_ptr<MVCNN::TensorReferenceT> mv::RuntimeModel::buildTensorReferenceT(mv::ComputationModel& model, mv::Element&, mv::Data::TensorIterator t, const std::string &allocatorName)
{
    mv::DataModel dm(model);
    mv::ControlModel cm(model);

    mv::Control::StageIterator stg = cm.getStage(0);
    std::unique_ptr<MVCNN::TensorReferenceT> toBuild = std::unique_ptr<MVCNN::TensorReferenceT>(new MVCNN::TensorReferenceT());

    toBuild->name = t->getName();

    auto tensorAllocators = t->get<std::set<std::string>>("allocators");

    auto tensorAllocatorName = tensorAllocators.begin();
    if(!allocatorName.empty())
        tensorAllocatorName = tensorAllocators.find(allocatorName);

    auto tensorAllocator = dm.getAllocator(*tensorAllocatorName);
    mv::Data::BufferIterator tensorBufferIt = tensorAllocator.getBuffer(0, t); // 0 is the only stage for now, but this will probably change in the future

    auto underlyingTensor = tensorBufferIt->getData();
    std::vector<uint32_t> dimensions = underlyingTensor->getShape();

    auto masterBuffer = tensorAllocator.getTopMasterBuffer(tensorBufferIt);
    std::vector<uint32_t> numericStrides = (*masterBuffer)->getData()->computeNumericStrides();

    numericStrides.push_back(underlyingTensor->getDType().getSizeInBits() / 8);

    //Because according to graphfile order is given as NCHW, which is exactly the reverse of our shape assumption WHCN
    std::reverse(dimensions.begin(), dimensions.end());
    std::reverse(numericStrides.begin(), numericStrides.end());

    toBuild->dimensions = std::vector<uint32_t>(dimensions.begin(), dimensions.end());
    toBuild->strides = numericStrides; // NOTE: Maybe directly bufferIt->computeStrides() in the future?

    toBuild->data = std::unique_ptr<MVCNN::IndirectDataReferenceT>(new MVCNN::IndirectDataReferenceT());
    if (*tensorAllocatorName == "GraphFile")
    {
        toBuild->data->data_index = 0;
        unsigned graphfileIndex = t->get<unsigned>("graphFileIndex");
        toBuild->locale_index = std::vector<unsigned int>(1);
        toBuild->locale_index[0] = graphfileIndex;
    }
    else if(*tensorAllocatorName == "ProgrammableInput" || *tensorAllocatorName == "ProgrammableOutput")
    {
        toBuild->data->data_index = 0;
        auto strides = tensorBufferIt->getStrides();
        //NOTΕ: Leading offset Computation ???
//        auto leading_offset = strides[0] / tensorBufferIt->getDataTypeSize();
        auto leading_offset = strides[0];
        toBuild->locale_index = std::vector<unsigned int>(1,0);
        if (leading_offset)
            toBuild->data->data_index += leading_offset;
        // No need to set sparsity_index for input/output tensor of the network
    }
    else
    {
        auto strides = tensorBufferIt->getStrides();
//        auto leading_offset = strides[0] / tensorBufferIt->getDataTypeSize(); //for some reason we get double the value, for now take the proper one.
        auto leading_offset = strides[0]; //for some reason we get double the value, for now take the proper one.
        toBuild->locale_index = std::vector<unsigned int>(1,0);

        // This part is for concat
        if(t->hasAttr("address"))
            toBuild->data->data_index = t->getAddress();
        else
//            toBuild->data->data_index = tensorBufferIt->getOffset();
            toBuild->data->data_index = (*masterBuffer)->getOffset();

        toBuild->data->data_index += leading_offset;

        if(t->isSparse())
        {
            toBuild->data->sparsity_index = t->getSparsityMap()->getAddress();
            if(!t->isPopulated())
                toBuild->data->storage_element_index = t->getStorageElement()->getAddress();
            else
                toBuild->data->storage_element_index = 0;
        }
    }
    toBuild->locale = convertAllocatorToMemoryLocale(*tensorAllocatorName);
    toBuild->data_dtype = convertDtype(t->getDType());

    // could also be t->hasAttr("quantizationParameters")
    // but in my opinion quantization for a tensor of floats makes very little sense
    // leaving this comment here for future generations
    // future generations say that needs to be serialized even in case of 0s and 1s(z_p, sc)
    if(t->isQuantized())
    {
        auto quantizationParams = t->get<mv::QuantizationParams>("quantParams");

        auto quantZero = quantizationParams.getZeroPoint();
        toBuild->quant_zero = std::vector<unsigned char>(quantZero.begin(), quantZero.end());

        auto quantScale = quantizationParams.getScale();

        toBuild->quant_scale = std::vector<float>(quantScale.begin(), quantScale.end());

        std::vector<unsigned> quantMult = {};
        if (quantizationParams.hasAttr("mult"))
            quantMult = quantizationParams.getMult();
        quantMult = reduceQuantVector_(quantMult);
        toBuild->quant_mult = std::vector<unsigned short int>(quantMult.begin(), quantMult.end());

        std::vector<unsigned> quantShift;
        if (quantizationParams.hasAttr("shift"))
            quantShift = quantizationParams.getShift();
        quantShift = reduceQuantVector_(quantShift);
        toBuild->quant_shift = std::vector<unsigned char>(quantShift.begin(), quantShift.end());
    }

    return toBuild;
}

//build tensorReference for subTensors - multiple clusters
std::unique_ptr<MVCNN::TensorReferenceT> mv::RuntimeModel::buildTensorReferenceT(mv::ComputationModel& cm, mv::Element&, mv::Data::TensorIterator t, unsigned clusterId, const std::string& allocatorName)
{
    mv::DataModel dm(cm);

    auto subtensor = t->getSubTensor(clusterId);

    std::unique_ptr<MVCNN::TensorReferenceT> toBuild = std::unique_ptr<MVCNN::TensorReferenceT>(new MVCNN::TensorReferenceT());

    toBuild->name = subtensor.getName();

    auto tensorAllocators = t->get<std::set<std::string>>("allocators");

    auto tensorAllocatorName = tensorAllocators.begin();
    if(!allocatorName.empty())
        tensorAllocatorName = tensorAllocators.find(allocatorName);
    auto tensorAllocator = dm.getAllocator(*tensorAllocatorName);

    mv::Data::BufferIterator tensorBufferIt = tensorAllocator.getBuffer(0, t); // 0 is the only stage for now, but this will probably change in the future

    // Shape is always of the subtensor
    // If the tensor is broadcasted, then the shape of the subtensor is equal to the shape of the master tensor
    // if not, the subtensor shape is adjusted accordingly
    std::vector<uint32_t> dimensions = subtensor.getShape();

    // Strides are computed depending on the memory location
    // Since subtensors are split only in CMX
    // In CMX we use the strides of the subtensor
    // In DDRs style memory we use the strides of the master tensor
    auto numericStrides = t->computeNumericStrides();
    numericStrides.push_back(t->getDType().getSizeInBits() / 8);
    if(*tensorAllocatorName == "VPU_CMX_NN")
    {
        auto masterBuffer = tensorAllocator.getTopMasterBuffer(tensorBufferIt);
        numericStrides = (*masterBuffer)->getData()->getSubTensor(clusterId).computeNumericStrides();
        numericStrides.push_back(subtensor.getDType().getSizeInBits() / 8);
    }

    //Because according to graphfile order is given as NCHW, which is exactly the reverse of our shape assumption WHCN
    std::reverse(dimensions.begin(), dimensions.end());
    std::reverse(numericStrides.begin(), numericStrides.end());

    toBuild->dimensions = dimensions;
    toBuild->strides = numericStrides;

    toBuild->data = std::unique_ptr<MVCNN::IndirectDataReferenceT>(new MVCNN::IndirectDataReferenceT());
    if (*tensorAllocatorName == "GraphFile")
    {
        if(!t->isSparse())
        {
            //SOK dense weight tensors for HDE support
            if(t->isSplitOverK() && (t->get<mv::DType>("dType") == mv::DType("UInt8") || t->get<mv::DType>("dType") == mv::DType("Int8")))
            {
                unsigned graphfileIndex = subtensor.get<unsigned>("graphFileIndex");
                toBuild->locale_index = std::vector<unsigned int>(1);
                toBuild->locale_index[0] = graphfileIndex;
                toBuild->data->data_index = 0;
            }
            else 
            {
                unsigned graphfileIndex = t->get<unsigned>("graphFileIndex");
                toBuild->locale_index = std::vector<unsigned int>(1);
                toBuild->locale_index[0] = graphfileIndex;

                auto offset = subtensor.get<std::vector<std::size_t>>("offset");
                auto index = t->getOrder().subToInd(t->getShape(), offset);
                auto byte_index = index * t->getDType().getSizeInBits() / 8;
                toBuild->data->data_index = byte_index;
            }
        }
        else //Sparse
        {
            // In case data is sparse, packed subtensors are serialiazed. This simplifies our life a lot.
            // No data index to be provided, just have to take the graphfile index from the subtensor

            unsigned graphfileIndex = subtensor.get<unsigned>("graphFileIndex");
            toBuild->locale_index = std::vector<unsigned int>(1);
            toBuild->locale_index[0] = graphfileIndex;

            toBuild->data->data_index = 0;
        }
    }
    else if(*tensorAllocatorName == "ProgrammableInput" || *tensorAllocatorName == "ProgrammableOutput" ||
            *tensorAllocatorName == "VPU_DDR_BSS" || *tensorAllocatorName == "VPU_DDR_Heap")
    {
        auto offset = subtensor.get<std::vector<std::size_t>>("offset");
        auto index = t->getOrder().subToInd(t->getShape(), offset);
        auto byte_index = index * t->getDType().getSizeInBits() / 8;

        // NOTE: This probably has to be done also when DDR kicks in
        // as CMX is the only memory with the cluster/slice approach
        auto starting_address = 0;
        if(t->hasAttr("address"))
            starting_address = t->get<std::size_t>("address");
        else
        {
            auto masterBuffer = tensorAllocator.getTopMasterBuffer(tensorBufferIt);
            starting_address = (*masterBuffer)->getOffset();
        }

        toBuild->data->data_index = starting_address + byte_index;

        auto strides = tensorBufferIt->getStrides();
        auto leading_offset = strides[0];
        toBuild->locale_index = std::vector<unsigned int>(1,0);
        if (leading_offset)
            toBuild->data->data_index += leading_offset;

    }
    else
    {
        // This part is for concat
        if(t->hasAttr("address"))
            toBuild->data->data_index = subtensor.getAddress();
        else
            toBuild->data->data_index = tensorBufferIt->getOffset();

        toBuild->locale_index = std::vector<unsigned int>(1, clusterId);

        if(t->isSparse())
        {
            toBuild->data->sparsity_index = subtensor.getSparsityMap()->getAddress();
            if(!t->isPopulated())
                toBuild->data->storage_element_index = subtensor.getStorageElement()->getAddress();
            else
                toBuild->data->storage_element_index = 0;
        }
    }

    toBuild->locale = convertAllocatorToMemoryLocale(*tensorAllocatorName);
    toBuild->data_dtype = convertDtype(t->getDType());

    // could also be t->hasAttr("quantizationParameters")
    // but in my opinion quantization for a tensor of floats makes very little sense
    // leaving this comment here for future generations
    if(t->isQuantized())
    {
        auto quantizationParams = t->get<mv::QuantizationParams>("quantParams");

        auto quantZero = quantizationParams.getZeroPoint();
        toBuild->quant_zero = std::vector<unsigned char>(quantZero.begin(), quantZero.end());

        auto quantScale = quantizationParams.getScale();
        toBuild->quant_scale = std::vector<float>(quantScale.begin(), quantScale.end());

        std::vector<unsigned> quantMult = {};
        if (quantizationParams.hasAttr("mult"))
            quantMult = quantizationParams.getMult();
        quantMult = reduceQuantVector_(quantMult);
        toBuild->quant_mult = std::vector<unsigned short int>(quantMult.begin(), quantMult.end());

        std::vector<unsigned> quantShift;
        if (quantizationParams.hasAttr("shift"))
            quantShift = quantizationParams.getShift();
        quantShift = reduceQuantVector_(quantShift);
        toBuild->quant_shift = std::vector<unsigned char>(quantShift.begin(), quantShift.end());

    }

    return toBuild;
}

std::unique_ptr<MVCNN::SummaryHeaderT> mv::RuntimeModel::buildSummaryHeaderMetaInformations(ComputationModel& cm, mv::Element& compilationDescriptor)
{
    mv::OpModel om(cm);

    std::unique_ptr<MVCNN::SummaryHeaderT> toBuild = std::unique_ptr<MVCNN::SummaryHeaderT>(new MVCNN::SummaryHeaderT());

    toBuild->version = buildVersionT(cm, compilationDescriptor);
    toBuild->original_structure = buildSourceStructureT(cm, compilationDescriptor);
    toBuild->layer_count = om.opsCount();

    return toBuild;
}


std::unique_ptr<MVCNN::SummaryHeaderT> mv::RuntimeModel::buildSummaryHeaderT(ComputationModel& cm, mv::Element& compilationDescriptor, std::unique_ptr<MVCNN::SummaryHeaderT> originalHeader)
{
    mv::OpModel om(cm);

    std::unique_ptr<MVCNN::SummaryHeaderT> toBuild = std::unique_ptr<MVCNN::SummaryHeaderT>(new MVCNN::SummaryHeaderT());

    auto globalConfigurationParameters = cm.getGlobalConfigParams();
    auto paddOutput = globalConfigurationParameters->hasAttr("PadOutput") ? globalConfigurationParameters->get<bool>("PadOutput") : false;

    toBuild->version = std::move(originalHeader->version);
    toBuild->original_structure = std::move(originalHeader->original_structure);
    toBuild->resources = buildResourcesT(cm, compilationDescriptor);

    // Just one input for now
    toBuild->net_input = std::vector<std::unique_ptr<MVCNN::TensorReferenceT>>(1);
    toBuild->net_input[0] = buildTensorReferenceT(cm, compilationDescriptor, om.getInput()->getOutputTensor(0));

    toBuild->net_output = std::vector<std::unique_ptr<MVCNN::TensorReferenceT>>(1);
    toBuild->net_output[0] = buildTensorReferenceT(cm, compilationDescriptor, om.getOutput()->getInputTensor(0));
    if (paddOutput && om.getOutput()->getInputTensor(0)->hasAttr("alignment"))
        alignTensor(cm, toBuild->net_output[0], *om.getOutput()->getInputTensor(0), IO_CHANNEL_DIMENSION, paddOutput);
    auto taskCount = [](mv::OpModel m)
    {
        unsigned i = 0;
        for(auto opIt = m.opBegin(); opIt != m.opEnd(); ++opIt)
            if(opIt->getOpType().find("Task") != std::string::npos)
                ++i;
        return i;
    };

    toBuild->options = std::vector<MVCNN::ExecutionFlag>();
    //toBuild->options.push_back(MVCNN::ExecutionFlag_Compiled_For_VPU3);
    if(globalConfigurationParameters->get<std::string>("barrier_index_assignment") == "Dynamic")
        toBuild->options.push_back(MVCNN::ExecutionFlag_DynamicBarriers);

    toBuild->layer_count = originalHeader->layer_count;
    toBuild->task_count = taskCount(om);

    return toBuild;
}

std::unique_ptr<MVCNN::VersionT> mv::RuntimeModel::buildVersionT(ComputationModel&, mv::Element& compilationDescriptor)
{
    std::unique_ptr<MVCNN::VersionT> toBuild = std::unique_ptr<MVCNN::VersionT>(new MVCNN::VersionT());

    setIfPresent<uint32_t, int>(toBuild->majorV, compilationDescriptor, "VersionMajor");
    setIfPresent<uint32_t, int>(toBuild->minorV, compilationDescriptor, "VersionMinor");
    setIfPresent<uint32_t, int>(toBuild->patchV, compilationDescriptor, "VersionPatch");
    setIfPresent<std::string, std::string>(toBuild->hash, compilationDescriptor, "VersionHash");

    return toBuild;
}

std::unique_ptr<MVCNN::ResourcesT> mv::RuntimeModel::buildResourcesT(ComputationModel& cm, mv::Element& compilationDescriptor)
{
    std::unique_ptr<MVCNN::ResourcesT> toBuild = std::unique_ptr<MVCNN::ResourcesT>(new MVCNN::ResourcesT());
    UNUSED(compilationDescriptor);
    auto globalConfigurationParams = cm.getGlobalConfigParams();

    setIfPresent<uint32_t, int>(toBuild->upa_shaves, *globalConfigurationParams , "UpaShaves");
    setIfPresent<int8_t, int>(toBuild->nce1_blocks, *globalConfigurationParams, "NCE1Mask");
    setIfPresent<uint32_t, int>(toBuild->nce2_blocks, *globalConfigurationParams, "Number_of_DPUs");
    setIfPresent<uint32_t, int>(toBuild->upa_shared_cmx, *globalConfigurationParams, "UPASharedCMX");
    uint32_t nn_cmx_per_slice=0;
    setIfPresent<uint32_t, unsigned>(nn_cmx_per_slice, *globalConfigurationParams, "totalCmx");
    toBuild->nn_cmx_per_slice = nn_cmx_per_slice;
    setIfPresent<uint32_t, unsigned>(toBuild->nn_cmx_slice_amount, *globalConfigurationParams, "clusters");
    setIfPresent<uint32_t, int>(toBuild->ddr_scratch, *globalConfigurationParams, "DDRScratch");

    return toBuild;
}

template <typename T>
std::vector<long unsigned int> packToInt64(const std::vector<T>& origData, mv::DType dtype)
{
    unsigned dataSize = origData.size();
    unsigned origDataSize = dtype.getSizeInBits();

    unsigned nElementToPack = 64 / origDataSize;
    unsigned finalLength = mv::ceil_division(dataSize , nElementToPack);

    std::vector<long unsigned int> toReturn(finalLength, 0);

    for(unsigned i = 0; i < finalLength; ++i)
        for(unsigned j = 0; j < nElementToPack; ++j)
            if ((i*nElementToPack + j) < dataSize)
                toReturn[i] ^= origData[i*nElementToPack + j] << (j * origDataSize);
            
    return toReturn;
}

std::unique_ptr<MVCNN::BinaryDataT> mv::RuntimeModel::buildBinaryDataT(ComputationModel&, mv::Element&, mv::Tensor& t, bool huffmanCompression)
{
    std::unique_ptr<MVCNN::BinaryDataT> toBuild = std::unique_ptr<MVCNN::BinaryDataT>(new MVCNN::BinaryDataT());

    //Compress weights
    if((t.getDType().toString() == "UInt8" || t.getDType().toString() == "Int8") && huffmanCompression) 
    {
        auto dataPacked = t.getDataPacked();
        auto weightSizeKb = t.computeTotalSize() / 1024;
        //Minimum size that can be compressed is 4kB
        if(weightSizeKb > 4) {
            auto compressedData = huffmanCompress(dataPacked, t); 
            toBuild->data = packToInt64(compressedData, t.getDType());
            int length = t.get<int>("CompressedSize");
            toBuild->length = length;
            toBuild->underlying_type = MVCNN::DType::DType_U8;         
        }
        else {
            auto dataPacked = t.getDataPacked();
            toBuild->data = packToInt64(dataPacked, t.getDType());
            toBuild->length = dataPacked.size() * t.getDType().getSizeInBits() / 8;
            toBuild->underlying_type = MVCNN::DType::DType_U8;
            t.set<bool>("Compression", false);
        }
    }
    else 
    {
        auto dataPacked = t.getDataPacked();
        toBuild->data = packToInt64(dataPacked, t.getDType());
        toBuild->length = dataPacked.size() * t.getDType().getSizeInBits() / 8;
        toBuild->underlying_type = MVCNN::DType::DType_U8;
        t.set<bool>("Compression", false);
    }

    return toBuild;
}

// We have three taskslist for POC:
// Tasklist 0: Contains all the tasks
// We need to topologically sort the control model graph to get the tasks in the correct order.

std::vector<std::unique_ptr<MVCNN::TaskListT>> mv::RuntimeModel::buildTaskListT(ComputationModel& cm, mv::Element& compilationDescriptor)
{
    mv::OpModel om(cm);
    mv::ControlModel controlModel(cm);
    std::vector<std::unique_ptr<MVCNN::TaskListT>> toBuild = std::vector<std::unique_ptr<MVCNN::TaskListT>>(3);
    toBuild[0] = std::unique_ptr<MVCNN::TaskListT>(new MVCNN::TaskListT());
    toBuild[1] = std::unique_ptr<MVCNN::TaskListT>(new MVCNN::TaskListT());
    toBuild[2] = std::unique_ptr<MVCNN::TaskListT>(new MVCNN::TaskListT());

    auto topologicallySortedOps = controlModel.schedulingSort();

    int initialId = 0;

    for(auto vecIt = topologicallySortedOps.begin(); vecIt != topologicallySortedOps.end(); ++vecIt)
    {
        auto opIt = *vecIt;
        std::unique_ptr<MVCNN::TaskListT> * listToUse = &toBuild[0]; // default to DPU task
        std::string opType = opIt->getOpType();
        if(opType.find("DPU") != std::string::npos)
            listToUse = &toBuild[0];
        if(opType.find("UPA") != std::string::npos)
            listToUse = &toBuild[0];
        else if(opType.find("DMA") != std::string::npos)
            listToUse = &toBuild[1];
        auto tasks = buildTaskT(cm, compilationDescriptor, opIt);
        for(auto& task: tasks)
            (*listToUse)->content.push_back(std::move(task));
    }

    //Barrier task list has to be built in the correct order
    auto barrierTasks = om.getOps("BarrierTask");
    std::sort(
        barrierTasks.begin(),
        barrierTasks.end(),
        [](const mv::Data::OpListIterator& a, const mv::Data::OpListIterator& b) -> bool { return a->get<mv::Barrier>("Barrier").getIndex() < b->get<mv::Barrier>("Barrier").getIndex(); }
        );

    unsigned n = barrierTasks.size();
    for(unsigned i = 0; i < n; ++i)
    {
        auto tasks = buildTaskT(cm, compilationDescriptor, controlModel.switchContext(barrierTasks[i]));
        for(auto& task: tasks)
            toBuild[2]->content.push_back(std::move(task));
    }

    // Filling node IDs
    for(auto& serialTask: toBuild[0]->content)
        serialTask->nodeID = initialId++;
    for(auto& serialTask: toBuild[1]->content)
        serialTask->nodeID = initialId++;
    for(auto& serialTask: toBuild[2]->content)
        serialTask->nodeID = initialId++;

    return toBuild;
}

std::vector<std::unique_ptr<MVCNN::TaskT>> mv::RuntimeModel::buildBarrierTaskT(ComputationModel& cm, Element& compilationDescriptor, Control::OpListIterator opIt)
{
    std::vector<std::unique_ptr<MVCNN::TaskT>> toReturn = std::vector<std::unique_ptr<MVCNN::TaskT>>(1);

    toReturn[0] = std::unique_ptr<MVCNN::TaskT>(new MVCNN::TaskT());
    toReturn[0]->task.type = MVCNN::SpecificTask_ControllerTask;

    auto controllerTask = new MVCNN::ControllerTaskT();
    controllerTask->task.type = MVCNN::ControllerSubTask_BarrierConfigurationTask;

    auto barrierConfigurationTask = new MVCNN::BarrierConfigurationTaskT();
    barrierConfigurationTask->target = buildBarrierT(cm, compilationDescriptor, opIt);

    controllerTask->task.value = barrierConfigurationTask;
    toReturn[0]->task.value = controllerTask;

    return toReturn;
}

std::vector<std::unique_ptr<MVCNN::TaskT>> mv::RuntimeModel::buildSpecificTaskUnion(ComputationModel& cm, mv::Element &compilationDescriptor, Control::OpListIterator opIt)
{
    std::vector<std::unique_ptr<MVCNN::TaskT>> toBuild = std::vector<std::unique_ptr<MVCNN::TaskT>>();
    std::string taskType(opIt->getOpType());

    if(taskType == "MvTensorTask")
        toBuild = buildMvTensorTaskT(cm, compilationDescriptor, opIt);
    else if(taskType == "DMATask")
    {
        auto direction = opIt->get<mv::DmaDirection>("direction");
        if(direction == mv::DmaDirectionEnum::NNCMX2UPACMX ||
           direction == mv::DmaDirectionEnum::UPACMX2NNCMX ||
           direction == mv::DmaDirectionEnum::DDR2UPACMX   ||
           direction == mv::DmaDirectionEnum::UPACMX2DDR)
        {
            toBuild = buildUPADMATaskT(cm, compilationDescriptor, opIt);
        }
        else
        {
            std::string splitting = opIt->getOutputTensor(0)->get<std::string>("splitStrategy");
            toBuild = buildNNDMATaskT(cm, compilationDescriptor, opIt, splitting);
        }
    }
    else if(taskType == "NCE1Task")
        toBuild = buildNCE1TaskT(cm, compilationDescriptor, opIt);
    else if(taskType == "DPUTask")
    {
        std::string splitting = opIt->get<std::string>("splitStrategy");
        toBuild = buildNCE2TaskT(cm, compilationDescriptor, opIt, splitting);
    }
    else if(taskType == "UPATask")
        toBuild = buildUPATask(cm, compilationDescriptor, opIt);
    else if(taskType == "ControllerTask")
        toBuild = buildControllerTaskT(cm, compilationDescriptor, opIt);
    else if(taskType == "BarrierTask")
        toBuild = buildBarrierTaskT(cm, compilationDescriptor, opIt);

    return toBuild;
}

std::vector<std::unique_ptr<MVCNN::TaskT>> mv::RuntimeModel::buildMvTensorTaskT(ComputationModel& cm, mv::Element &compilationDescriptor, Control::OpListIterator opIt)
{
    UNUSED(cm);
    UNUSED(compilationDescriptor);
    UNUSED(opIt);
    std::vector<std::unique_ptr<MVCNN::TaskT>> toReturn = std::vector<std::unique_ptr<MVCNN::TaskT>>(1);
    return toReturn;
}

std::vector<std::unique_ptr<MVCNN::TaskT>> mv::RuntimeModel::buildUPADMATaskT(ComputationModel& cm, mv::Element &compilationDescriptor, Control::OpListIterator opIt)
{
    std::vector<std::unique_ptr<MVCNN::TaskT>> toReturn = std::vector<std::unique_ptr<MVCNN::TaskT>>(1);
    toReturn[0] = std::unique_ptr<MVCNN::TaskT>(new MVCNN::TaskT());
    toReturn[0]->task.type = MVCNN::SpecificTask_UPADMATask;
    auto tmp = new MVCNN::UPADMATaskT();
    tmp->src = buildTensorReferenceT(cm, compilationDescriptor, opIt->getInputTensor(0));
    tmp->dst = buildTensorReferenceT(cm, compilationDescriptor, opIt->getOutputTensor(0));
    toReturn[0]->task.value = tmp;
    return toReturn;
}

void checkUnstridedDMA(mv::Data::TensorIterator src, int i, MVCNN::NNDMATaskT * tmp)
{
    if(tmp->src->locale == MVCNN::MemoryLocation_GraphFile)
    {
        unsigned totalSize = src->getSubTensor(i).getShape().totalSize();
        unsigned totalSizeDst = src->getSubTensor(i).getShape().totalSize();
        if(src->isSparse())
            totalSize = src->getSubTensor(i).dataPackedSize();
        
        if(src->getSubTensor(i).hasAttr("CompressedSize"))
            totalSize = src->getSubTensor(i).get<int>("CompressedSize");
        else 
            totalSize *= src->getDType().getSizeInBits() / 8;
        
        std::vector<uint32_t> dimensions = {totalSize, 1, 1, 1};
        totalSizeDst *= src->getDType().getSizeInBits() / 8;
        std::vector<uint32_t> dimensionsdst = {totalSizeDst, 1, 1, 1};
        std::vector<uint32_t> strides = {1, 1, 1, 1, 1};
        auto dtype = MVCNN::DType::DType_U8;

        tmp->src->dimensions = dimensions;
        tmp->src->strides = strides;
        tmp->src->data_dtype = dtype;


        tmp->dst->dimensions = dimensionsdst;
        tmp->dst->strides = strides;
        tmp->dst->data_dtype = dtype;
    }
}

void mv::RuntimeModel::case1MC(unsigned numTasks, mv::ComputationModel& cm, mv::DmaDirection direction, mv::Element &compilationDescriptor,
                               bool compression, bool padFinalOutput, std::vector<std::unique_ptr<MVCNN::TaskT>>& toReturn, mv::Data::TensorIterator src, mv::Data::TensorIterator dst, const std::string& srcAllocator, const std::string& dstAllocator)
{
    std::unique_ptr<MVCNN::TaskT> toPush = std::unique_ptr<MVCNN::TaskT>(new MVCNN::TaskT());
    auto tmp = new MVCNN::NNDMATaskT();
    toPush->task.type = MVCNN::SpecificTask_NNDMATask;

    tmp->src = buildTensorReferenceT(cm, compilationDescriptor, src, srcAllocator);
    tmp->dst = buildTensorReferenceT(cm, compilationDescriptor, dst, dstAllocator);

    if (direction != mv::DDR2NNCMX)
    {
        if (padFinalOutput && dst->hasAttr("alignment"))
            alignTensor(cm, tmp->dst, *dst, IO_CHANNEL_DIMENSION, padFinalOutput);
    }

    std::vector<unsigned int> locale_index;
    for (unsigned idx = numTasks; idx > 0; idx--)
        locale_index.push_back(idx - 1);

    if(direction == mv::DDR2NNCMX)
        tmp->dst->locale_index = locale_index;

    // Passing -1 as subtensor index, will have us get the full tensor
    checkUnstridedDMA(src, -1, tmp);

    // Check if the HDE engine compressed the weights
    if(tmp->src->dimensions[0] != tmp->dst->dimensions[0])
        tmp->compression =  true;

    toPush->task.value = tmp;

    toReturn.push_back(std::move(toPush));
}

void mv::RuntimeModel::case2MC(unsigned numTasks, ComputationModel& cm,  mv::DmaDirection direction, mv::Element &compilationDescriptor,
                               bool compression, bool padFinalOutput, std::vector<std::unique_ptr<MVCNN::TaskT>>& toReturn,
                               mv::Data::TensorIterator src, mv::Data::TensorIterator dst, const std::string& srcAllocator,
                               const std::string& dstAllocator)
{
    for(unsigned i = 0; i < numTasks; ++i)
    {
        std::unique_ptr<MVCNN::TaskT> toPush = std::unique_ptr<MVCNN::TaskT>(new MVCNN::TaskT());
        auto tmp = new MVCNN::NNDMATaskT();
        toPush->task.type = MVCNN::SpecificTask_NNDMATask;
        tmp->src = buildTensorReferenceT(cm, compilationDescriptor, src, i, srcAllocator);
        tmp->dst = buildTensorReferenceT(cm, compilationDescriptor, dst, i, dstAllocator);

        if (direction != mv::DDR2NNCMX)
        {
            if (padFinalOutput && dst->hasAttr("alignment"))
                alignTensor(cm, tmp->dst, dst->getSubTensor(i), IO_CHANNEL_DIMENSION, padFinalOutput);
        }

        //Check if DMA is DDR2CMX
        if (direction == mv::DDR2NNCMX)
        {
            if (dst->hasAttr("alignWidth")){
                alignTensor(cm, tmp->dst, dst->getSubTensor(i), IO_WIDTH_DIMENSION, false);
            }
        }

        checkUnstridedDMA(src, i, tmp);

        // Check if the HDE engine compressed the weights
        if(tmp->src->dimensions[0] != tmp->dst->dimensions[0])
            tmp->compression =  true;	  

        toPush->task.value = tmp;
        toReturn.push_back(std::move(toPush));
    }
}

std::vector<std::unique_ptr<MVCNN::TaskT>> mv::RuntimeModel::buildNNDMATaskT(ComputationModel& cm, mv::Element &compilationDescriptor, Control::OpListIterator opIt, std::string splitting)
{
    mv::DataModel dm(cm);

    auto direction = opIt->get<mv::DmaDirection>("direction");
    auto globalConfigParams = cm.getGlobalConfigParams();
    unsigned numTasks = globalConfigParams->get<int>("Number_of_Clusters");
    auto padFinalOutput = false;

    auto inputTensor = opIt->getInputTensor(0);
    auto outputTensor = opIt->getOutputTensor(0);
    bool sourceIsBroadCasted = inputTensor->isBroadcasted();

    //NOTE: When strategy is overwritten
    if (opIt->get<mv::DmaDirection>("direction") == mv::DmaDirectionEnum::DDR2NNCMX)
    {
        if (inputTensor->hasAttr("overwriteStrategy"))
        {
            if (inputTensor->get<std::string>("overwriteStrategy") == "ClusteringToSoH")
                sourceIsBroadCasted = false;
            else if (inputTensor->get<std::string>("overwriteStrategy") == "SoHToClustering")
                sourceIsBroadCasted = true;
        }
    }

    bool compression = false;
    if(inputTensor->hasAttr("Compression"))
        compression = inputTensor->get<bool>("Compression"); //This is main tensor for SOK
        
    auto tensorAllocatorName = outputTensor->get<std::set<std::string>>("allocators").begin();
    if (*tensorAllocatorName == "ProgrammableOutput")
        //Only if we are DMA-ing to programmable output check if we need to padd it
        padFinalOutput = cm.getGlobalConfigParams()->hasAttr("PadOutput") ? cm.getGlobalConfigParams()->get<bool>("PadOutput") : false;

    // Case 1 of MC DMAs - Source tensor is broadcasted, i.e. present in it's entirety
    // in all clusters, OR populated tensors going into clustering op
    // (which for some reason are not marked as broadcasted).

    // Weights sparsity maps with new approach should be handled here

    // Strategy: 1 DMA to multiple slices -  multiple slices to 1 place
    // The second condition is necessary because when we spill we replace
    // the strategies, so a SplitOverK unpopulated tensor could potentially
    // not be marked as broadcasted
    if(sourceIsBroadCasted ||
      (splitting == "SplitOverK" && !outputTensor->isPopulated()) ||
      (splitting == "Clustering" && numTasks == 1))
    {
        std::vector<std::unique_ptr<MVCNN::TaskT>> toReturn;

        case1MC(numTasks, cm, direction, compilationDescriptor, compression, padFinalOutput, toReturn, inputTensor, outputTensor);

        if(inputTensor->isSparse())
        {
            // NOTE: First usage ever of the concept one tensor -> Multiple allocators
            auto tensorSparsityMap = dm.getTensor(inputTensor->getSparsityMap()->getName());
            case1MC(numTasks, cm, direction, compilationDescriptor, compression, padFinalOutput, toReturn, tensorSparsityMap, tensorSparsityMap, "GraphFile", "VPU_CMX_NN");
        }
        return toReturn;
    }
    // Case 2 of MC DMAs - All cases that are not case 1 or 2. Mostly applied to SOH tensors for activation
    // and SOK for weights.

    // Weights sparsity maps with new approach has be handled in the future here, for the
    // case of SOK and weights sparsity.

    // Strategy: Multiple DMAs, yuppi!
    else
    {
        std::vector<std::unique_ptr<MVCNN::TaskT>> toReturn;

        case2MC(numTasks, cm, direction, compilationDescriptor, compression, padFinalOutput, toReturn, inputTensor, outputTensor);
        if(inputTensor->isSparse())
        {
            // NOTE: Second usage ever of the concept one tensor -> Multiple allocators
            auto tensorSparsityMap = dm.getTensor(inputTensor->getSparsityMap()->getName());
            case2MC(numTasks, cm, direction, compilationDescriptor, compression, padFinalOutput, toReturn, tensorSparsityMap, tensorSparsityMap, "GraphFile", "VPU_CMX_NN");
        }
        return toReturn;
    }
}

std::vector<std::unique_ptr<MVCNN::TaskT>> mv::RuntimeModel::buildNCE1TaskT(ComputationModel& cm, mv::Element &compilationDescriptor, Control::OpListIterator opIt)
{
    UNUSED(cm);
    UNUSED(compilationDescriptor);
    UNUSED(opIt);
    std::vector<std::unique_ptr<MVCNN::TaskT>> toReturn = std::vector<std::unique_ptr<MVCNN::TaskT>>(1);
    return toReturn;
}

MVCNN::DPULayerType mv::RuntimeModel::convertTaskOp(const std::string& opName)
{
    return dpuLayerMapping_.at(opName);
}


std::unique_ptr<MVCNN::PPEFixedFunctionT> mv::RuntimeModel::buildPPEFixedFunctionT(ComputationModel&, mv::Element&, const mv::PPEFixedFunction& ppeFixedFunction)
{
    std::unique_ptr<MVCNN::PPEFixedFunctionT> toBuild = std::unique_ptr<MVCNN::PPEFixedFunctionT>(new MVCNN::PPEFixedFunctionT());

    auto layers = ppeFixedFunction.getLayers();
    unsigned n = layers.size();
    toBuild->Ops = std::vector<MVCNN::PPELayerType>(n);
    for(unsigned i = 0; i < n; ++i)
        toBuild->Ops[i] = convertPPELayerType(layers[i]);
    toBuild->Clamp_Low = ppeFixedFunction.getLowClamp();
    toBuild->Clamp_High = ppeFixedFunction.getHighClamp();
    toBuild->Lrelu_Mult = ppeFixedFunction.getLReluMult();
    toBuild->Lrelu_Shift = ppeFixedFunction.getLReluShift();
    if (toBuild->Lrelu_Mult > 1)
    {
        auto alpha = toBuild->Lrelu_Mult / pow(2, toBuild->Lrelu_Shift);
        toBuild->Clamp_Low /= alpha;
    }

    return toBuild;
}

std::unique_ptr<MVCNN::PPETaskT> mv::RuntimeModel::buildPPETaskT(ComputationModel& cm, mv::Element& compilationDescriptor, const mv::PPETask& ppeTask)
{
    std::unique_ptr<MVCNN::PPETaskT> toBuild = std::unique_ptr<MVCNN::PPETaskT>(new MVCNN::PPETaskT());

    if(ppeTask.hasAttr("scaleData"))
        toBuild->scale_data = buildTensorReferenceT(cm, compilationDescriptor, ppeTask.getScaleData());
    toBuild->fixed_function = buildPPEFixedFunctionT(cm, compilationDescriptor, ppeTask.getFixedFunction());

    return toBuild;
}

std::unique_ptr<MVCNN::PPETaskT> mv::RuntimeModel::buildPPETaskT()
{
    std::unique_ptr<MVCNN::PPETaskT> toBuild = std::unique_ptr<MVCNN::PPETaskT>(new MVCNN::PPETaskT());
    toBuild->fixed_function = std::unique_ptr<MVCNN::PPEFixedFunctionT>(new MVCNN::PPEFixedFunctionT());
    toBuild->fixed_function->Clamp_High = 2147483647;
    toBuild->fixed_function->Clamp_Low = -2147483648;
    toBuild->fixed_function->Ops = std::vector<MVCNN::PPELayerType>();
    toBuild->fixed_function->Lrelu_Mult = 1;
    toBuild->fixed_function->Lrelu_Shift = 0;
    toBuild->fixed_function->Ops.reserve(5);

    return toBuild;
}

std::unique_ptr<MVCNN::NCEInvariantFieldsT> mv::RuntimeModel::buildNCEInvariantFieldsT(ComputationModel& cm, mv::Element &compilationDescriptor, Control::OpListIterator opIt)
{
    std::unique_ptr<MVCNN::NCEInvariantFieldsT> toBuild = std::unique_ptr<MVCNN::NCEInvariantFieldsT>(new MVCNN::NCEInvariantFieldsT());

    toBuild->dpu_task_type = convertTaskOp(opIt->get<std::string>("taskOp"));

    if(opIt->hasAttr("PPETask"))
        toBuild->ppe_task = buildPPETaskT(cm, compilationDescriptor, opIt->get<PPETask>("PPETask"));
    else
        toBuild->ppe_task = buildPPETaskT();

    if (opIt->hasAttr("kSize"))
    {
        auto kernelShape = opIt->get<std::array<unsigned short, 2>>("kSize");
        toBuild->kernelW = kernelShape[0];
        toBuild->kernelH = kernelShape[1];
    }

    if (opIt->hasAttr("stride"))
    {
        auto kernelStride = opIt->get<std::array<unsigned short, 2>>("stride");
        toBuild->kernel_strideW = kernelStride[0];
        toBuild->kernel_strideH = kernelStride[1];
    }

    if (opIt->hasAttr("padding"))
    {
        auto kernelPadding = opIt->get<std::array<unsigned short, 4>>("padding");
        toBuild->kernel_padLeft = kernelPadding[0];
        toBuild->kernel_padRight = kernelPadding[1];
        toBuild->kernel_padTop = kernelPadding[2];
        toBuild->kernel_padBottom = kernelPadding[3];
    }
    //input
    auto inputTensor = opIt->getInputTensor(0);

    toBuild->input_data = buildTensorReferenceT(cm, compilationDescriptor, inputTensor);
    toBuild->parent_input_tensor = buildTensorReferenceT(cm, compilationDescriptor, inputTensor);
    toBuild->parent_input_tensor->data->sparsity_index = 999999999999999999;
    toBuild->parent_input_tensor->data->storage_element_index = 999999999999999999;

    //output
    auto outputTensor = opIt->getOutputTensor(0);

    toBuild->output_data = buildTensorReferenceT(cm, compilationDescriptor, outputTensor);
    toBuild->parent_output_tensor = buildTensorReferenceT(cm, compilationDescriptor, outputTensor);
    toBuild->parent_output_tensor->data->sparsity_index = 999999999999999999;
    toBuild->parent_output_tensor->data->storage_element_index = 999999999999999999;

    if(opIt->hasAttr("fakeSparsity"))
    {
        auto activationWindowTensorIterator = opIt->getInputTensor(opIt->get<std::size_t>("fakeSparsityIndex"));
        toBuild->activation_window = buildTensorReferenceT(cm, compilationDescriptor, activationWindowTensorIterator);
        toBuild->activation_window_channel_length = activationWindowTensorIterator->get<int>("channelLength");
    }

    if(toBuild->dpu_task_type != MVCNN::DPULayerType_ELTWISE)
    {
        auto weightsTableTensorIterator = opIt->getInputTensor(opIt->get<std::size_t>("weightsTableIndex"));
        toBuild->weights_table = buildTensorReferenceT(cm, compilationDescriptor, weightsTableTensorIterator);
    }

    switch (toBuild->dpu_task_type)
    {
        case MVCNN::DPULayerType_CONV:
        case MVCNN::DPULayerType_DWCONV:
        case MVCNN::DPULayerType_CMCONV:
        case MVCNN::DPULayerType_FCL:
        case MVCNN::DPULayerType_ELTWISE:
            //std::unique_ptr<TensorReferenceT> parent_weights_tensor;
            toBuild->weights_data = buildTensorReferenceT(cm, compilationDescriptor, opIt->getInputTensor(1));
            break;
        default:
            break;
    }

    return toBuild;
}


// Multicluster version
std::unique_ptr<MVCNN::NCEInvariantFieldsT> mv::RuntimeModel::buildNCEInvariantFieldsT(ComputationModel& cm, mv::Element &compilationDescriptor, Control::OpListIterator opIt, int clusterId)
{
    std::unique_ptr<MVCNN::NCEInvariantFieldsT> toBuild = std::unique_ptr<MVCNN::NCEInvariantFieldsT>(new MVCNN::NCEInvariantFieldsT());

    toBuild->dpu_task_type = convertTaskOp(opIt->get<std::string>("taskOp"));

    if(opIt->hasAttr("PPETask"))
        toBuild->ppe_task = buildPPETaskT(cm, compilationDescriptor, opIt->get<PPETask>("PPETask"));
    else
        toBuild->ppe_task = buildPPETaskT();

    if (opIt->hasAttr("kSize"))
    {
        auto kernelShape = opIt->get<std::array<unsigned short, 2>>("kSize");
        toBuild->kernelW = kernelShape[0];
        toBuild->kernelH = kernelShape[1];
    }

    if (opIt->hasAttr("stride"))
    {
        auto kernelStride = opIt->get<std::array<unsigned short, 2>>("stride");
        toBuild->kernel_strideW = kernelStride[0];
        toBuild->kernel_strideH = kernelStride[1];
    }

    if (opIt->hasAttr("padding"))
    {
        auto kernelPadding = opIt->get<std::array<unsigned short, 4>>("padding");
        if (opIt->get<std::string>("taskOp") == "ChannelMajorConvolution")
        {
            unsigned numClusters = cm.getGlobalConfigParams()->get<int>("Number_of_Clusters");
            kernelPadding = getNewPadding(kernelPadding, clusterId, numClusters);
        }

        toBuild->kernel_padLeft = kernelPadding[0];
        toBuild->kernel_padRight = kernelPadding[1];
        toBuild->kernel_padTop = kernelPadding[2];
        toBuild->kernel_padBottom = kernelPadding[3];
    }
    //input
    auto parentInputTensor = opIt->getInputTensor(0);
    if (opIt->get<std::string>("splitStrategy") == "SplitOverK")
    {
        toBuild->input_data = buildTensorReferenceT(cm, compilationDescriptor, parentInputTensor);
        std::vector<unsigned int> locale_index;
        locale_index.push_back(clusterId);
        toBuild->input_data->locale_index = locale_index;
        toBuild->out_channel_offset = 0;
        for (int i = 0; i < clusterId; i++)
            toBuild->out_channel_offset += opIt->getOutputTensor(0)->getSubTensor(i).getShape()[IO_CHANNEL_DIMENSION];
    }
    else
        toBuild->input_data = buildTensorReferenceT(cm, compilationDescriptor, parentInputTensor, clusterId);

    toBuild->parent_input_tensor = buildTensorReferenceT(cm, compilationDescriptor, parentInputTensor);
    toBuild->parent_input_tensor->data->sparsity_index = 999999999999999999;
    toBuild->parent_input_tensor->data->storage_element_index = 999999999999999999;

    //output
    auto parentOutputTensor = opIt->getOutputTensor(0);
    toBuild->output_data = buildTensorReferenceT(cm, compilationDescriptor, parentOutputTensor, clusterId);
    if (opIt->hasAttr("multiCast"))
    {
        if (opIt->get<bool>("multiCast"))
        {
            unsigned numTasks = cm.getGlobalConfigParams()->get<int>("Number_of_Clusters");
            toBuild->output_data = buildTensorReferenceT(cm, compilationDescriptor, parentOutputTensor, clusterId);
            std::vector<unsigned int> locale_index;
            for (unsigned idx = numTasks; idx > 0; idx--)
                locale_index.push_back(idx-1);
            toBuild->output_data->locale_index = locale_index;
            auto numericStrides = parentOutputTensor->computeNumericStrides();
            numericStrides.push_back(parentOutputTensor->getDType().getSizeInBits() / 8);
            std::reverse(numericStrides.begin(), numericStrides.end());
            toBuild->output_data->strides = numericStrides;
        }
    }

    toBuild->parent_output_tensor = buildTensorReferenceT(cm, compilationDescriptor, parentOutputTensor);
    toBuild->parent_output_tensor->data->sparsity_index = 999999999999999999;
    toBuild->parent_output_tensor->data->storage_element_index = 999999999999999999;

    if (opIt->get<bool>("multiCast"))
    {
        if (opIt->get<std::string>("splitStrategy") == "HKSwitch")
        {
            auto outputTensor = opIt->getOutputTensor(0);
            auto subtensor = opIt->getOutputTensor(0)->getSubTensor(clusterId);
            auto offset = subtensor.get<std::vector<std::size_t>>("offset");
            auto index = outputTensor->getOrder().subToInd(outputTensor->getShape(), offset);
            auto byte_index = index * outputTensor->getDType().getSizeInBits() / 8;

            toBuild->output_data->data->data_index += byte_index;
        }
    }

    //OP inputs == n ->
    // n - 2 activation window (when present)
    // n - 1 weights table
    if(opIt->hasAttr("fakeSparsity"))
    {
        auto activationWindowTensorIterator = opIt->getInputTensor(opIt->get<std::size_t>("fakeSparsityIndex"));
        toBuild->activation_window = buildTensorReferenceT(cm, compilationDescriptor, activationWindowTensorIterator, clusterId);
        toBuild->activation_window_channel_length = activationWindowTensorIterator->get<int>("channelLength");
    }

    if(toBuild->dpu_task_type != MVCNN::DPULayerType_ELTWISE)
    {
        auto weightsTableTensorIterator = opIt->getInputTensor(opIt->get<std::size_t>("weightsTableIndex"));
        toBuild->weights_table = buildTensorReferenceT(cm, compilationDescriptor, weightsTableTensorIterator, clusterId);
    }

    switch (toBuild->dpu_task_type)
    {
        case MVCNN::DPULayerType_CONV:
        case MVCNN::DPULayerType_DWCONV:
        case MVCNN::DPULayerType_CMCONV:
        case MVCNN::DPULayerType_FCL:
        case MVCNN::DPULayerType_ELTWISE:
            toBuild->weights_data = buildTensorReferenceT(cm, compilationDescriptor, opIt->getInputTensor(1), clusterId);
            break;
        default:
            break;
    }

    return toBuild;
}

MVCNN::MPE_Mode mv::RuntimeModel::convertMPEMode(mv::MPE_Mode mpe)
{
    switch (mpe)
    {
        case mv::MPE_Mode::Matrix:
            return MVCNN::MPE_Mode::MPE_Mode_MATRIX;
        case mv::MPE_Mode::Vector:
            return MVCNN::MPE_Mode::MPE_Mode_VECTOR;
        case mv::MPE_Mode::Vector_FP16:
            return MVCNN::MPE_Mode::MPE_Mode_VECTOR_FP16;

        default:
            return MVCNN::MPE_Mode::MPE_Mode_VECTOR;
    }
}

bool mv::RuntimeModel::hardwareBugDepthwise(Control::OpListIterator opIt)
{
    auto splitStrategy = opIt->get<std::string>("splitStrategy");
    auto padding = opIt->get<std::array<unsigned short, 4>>("padding");
    auto kernelSize = opIt->get<std::array<unsigned short, 2>>("kSize");
    return (((splitStrategy == "SplitOverH") || splitStrategy == "SplitOverHOverlapped") &&
        (padding[0] % 2 == 1) &&
        (kernelSize[mv::KERNEL_HEIGHT] > 1));
}

void mv::RuntimeModel::getWorkloadPadding(Control::OpListIterator opIt, Workload &workload)
{
    if (opIt->get<std::string>("taskOp") == "Eltwise")
    {
        workload.padLeft = 0;
        workload.padTop = 0;
        workload.padRight = 0;
        workload.padBottom = 0;
    }
    else
    {
        auto padding = opIt->get<std::array<unsigned short, 4>>("padding");
        auto outputWidth = opIt->getOutputTensor(0)->getShape()[mv::IO_WIDTH_DIMENSION];
        auto outputHeight = opIt->getOutputTensor(0)->getShape()[mv::IO_HEIGHT_DIMENSION];
        if (hardwareBugDepthwise(opIt))
        {
            workload.padLeft = (workload.MinX == 0) ? padding[0] : 0;
            workload.padTop = (workload.MinY == 0) ? padding[2] : 0;
            workload.padRight = ((workload.MaxX + unsigned(1)) == outputWidth) ? padding[1] : 0;
            workload.padBottom = ((workload.MaxY + unsigned(1)) == outputHeight) ? padding[3] : 0;
        }
        else
        {
            workload.padLeft = (workload.MinX == 0) ? padding[0] : 0;
            workload.padTop = (workload.MinY == 0) ? padding[2] : 0;
            workload.padRight = ((workload.MaxX + unsigned(1)) == outputWidth) ? padding[1] : 0;
            workload.padBottom = ((workload.MaxY + unsigned(1)) == outputHeight) ? padding[3] : 0;
        }
    }
    return;
}

std::array<unsigned short, 4> mv::RuntimeModel::getNewPadding(std::array<unsigned short, 4> padding, int clusterId, int numClusters)
{
        if (clusterId == 0)
        {
            padding[0] = padding[0];
            padding[1] = padding[1];
            padding[2] = padding[2];
            padding[3] = 0;
        }
        else if (clusterId == numClusters - 1)
        {
            padding[0] = padding[0];
            padding[1] = padding[1];
            padding[3] = padding[3];
            padding[2] = 0;
        }
        else
        {
            padding[0] = padding[0];
            padding[1] = padding[1];
            padding[2] = 0;
            padding[3] = 0;
        }
        return padding;
}

void mv::RuntimeModel::getWorkloadPadding(Control::OpListIterator opIt, Workload &workload, unsigned clusterId)
{
    if (opIt->get<std::string>("taskOp") == "Eltwise")
    {
        workload.padLeft = 0;
        workload.padTop = 0;
        workload.padRight = 0;
        workload.padBottom = 0;
    }
    else
    {
        auto padding = getPadding(opIt, clusterId);
        auto outputWidth = opIt->getOutputTensor(0)->getShape()[mv::IO_WIDTH_DIMENSION];
        auto outputHeight = opIt->getOutputTensor(0)->getShape()[mv::IO_HEIGHT_DIMENSION];

        if (hardwareBugDepthwise(opIt))
        {
            workload.padLeft = (workload.MinX == 0) ? padding[0] : 0;
            workload.padTop = (workload.MinY == 0) ? padding[2] : 0;
            workload.padRight = ((workload.MaxX + unsigned(1)) == outputWidth) ? padding[1] : 0;
            workload.padBottom = ((workload.MaxY + unsigned(1)) == outputHeight) ? padding[3] : 0;
        }

        else
        {
            workload.padLeft = (workload.MinX == 0) ? padding[0] : 0;
            workload.padTop = (workload.MinY == 0) ? padding[2] : 0;
            workload.padRight = ((workload.MaxX + unsigned(1)) == outputWidth) ? padding[1] : 0;
            workload.padBottom = ((workload.MaxY + unsigned(1)) == outputHeight) ? padding[3] : 0;
        }
    }
    return;
}

std::array <unsigned short, 4>  mv::RuntimeModel::getPadding(Control::OpListIterator opIt, unsigned clusterId)
{
    std::array <unsigned short, 4> padding = opIt->get<std::array<unsigned short, 4>>("padding");

    if(clusterId !=0)
    {
        auto subTensor = opIt->getOutputTensor(0)->getSubTensor(clusterId);
        std::vector<std::size_t> offset = subTensor.get<std::vector<std::size_t>>("offset");

        //NOTE:Padding up
        if (offset[1] != 0)
            padding[2] = 0;

        //NOTE:Padding left
        if (offset[0] != 0)
            padding[0] = 0;

        //NOTE:Padding down
        if (subTensor.getShape()[IO_HEIGHT_DIMENSION] + offset[1] != opIt->getOutputTensor(0)->getShape()[IO_HEIGHT_DIMENSION])
            padding[3] = 0;

        if (subTensor.getShape()[IO_WIDTH_DIMENSION] + offset[0] != opIt->getOutputTensor(0)->getShape()[IO_WIDTH_DIMENSION])
            padding[1] = 0;
    }

    return padding;
}

std::unique_ptr<MVCNN::NCEVariantFieldsT> mv::RuntimeModel::buildNCEVariantFieldsT(ComputationModel& , mv::Element &compilationDescriptor, Control::OpListIterator opIt, Workload workload, unsigned clusterId, std::string strategy)
{
    UNUSED (compilationDescriptor);
    std::unique_ptr<MVCNN::NCEVariantFieldsT> toBuild = std::unique_ptr<MVCNN::NCEVariantFieldsT>(new MVCNN::NCEVariantFieldsT());

    toBuild->mpe_mode = convertMPEMode(workload.MPEMode);
    toBuild->workload_start_X = workload.MinX;
    toBuild->workload_start_Y = workload.MinY;
    toBuild->workload_start_Z = workload.MinZ;
    toBuild->workload_end_X = workload.MaxX;
    toBuild->workload_end_Y = workload.MaxY;
    toBuild->workload_end_Z = workload.MaxZ;
    if (strategy != "Clustering")
        //Padding should be computed for every cluster
        getWorkloadPadding(opIt, workload, clusterId);
    else
        getWorkloadPadding(opIt, workload);
    toBuild->padLeft = workload.padLeft;
    toBuild->padRight = workload.padRight;
    toBuild->padTop = workload.padTop;
    toBuild->padBottom = workload.padBottom;

    return toBuild;
}

std::vector<std::unique_ptr<MVCNN::NCEVariantFieldsT>> mv::RuntimeModel::buildNCEVariantFieldsTVector(ComputationModel& cm, mv::Element &compilationDescriptor,
                                                                                                      Control::OpListIterator opIt, unsigned numTask, std::string strategy)
{
    std::vector <mv::Workload> workloads;
    //NOTE: For Clustering SubTensors equal Workloads per Subtensor
    if (strategy == "Clustering" && numTask > 0)
        workloads = opIt->get<mv::Workloads>("Workloads" + std::to_string(0)).getWorkloads();
    else
        workloads = opIt->get<mv::Workloads>("Workloads" + std::to_string(numTask)).getWorkloads();
    unsigned n = workloads.size();
    std::vector<std::unique_ptr<MVCNN::NCEVariantFieldsT>> toBuild = std::vector<std::unique_ptr<MVCNN::NCEVariantFieldsT>>(n);
    for(unsigned i = 0; i < n; ++i)
    {
        toBuild[i] = buildNCEVariantFieldsT(cm, compilationDescriptor, opIt, workloads[i], numTask, strategy);
    }
    return toBuild;
}

std::vector<std::unique_ptr<MVCNN::TaskT>> mv::RuntimeModel::buildNCE2TaskT(ComputationModel& cm, mv::Element &compilationDescriptor, Control::OpListIterator opIt, std::string splitting)
{
    unsigned numTask = 0;
    numTask = cm.getGlobalConfigParams()->get<int>("Number_of_Clusters");

    std::vector<std::unique_ptr<MVCNN::TaskT>> toReturn = std::vector<std::unique_ptr<MVCNN::TaskT>>(numTask);
    if (splitting != "Clustering")
        for(unsigned i = 0; i < numTask; ++i)
        {
            toReturn[i] = std::unique_ptr<MVCNN::TaskT>(new MVCNN::TaskT());
            toReturn[i]->task.type = MVCNN::SpecificTask_NCE2Task;
            auto toBuild = new MVCNN::NCE2TaskT();
            toBuild->variant = buildNCEVariantFieldsTVector(cm, compilationDescriptor, opIt, i, splitting);
            toBuild->invariant = buildNCEInvariantFieldsT(cm, compilationDescriptor, opIt, i);

            auto hash = [](const MVCNN::MPE_Mode &g){ return static_cast<std::size_t>(g); };
            auto comp = [](const MVCNN::MPE_Mode &l, const MVCNN::MPE_Mode &r){ return l == r; };

            std::unordered_map<MVCNN::MPE_Mode, unsigned, decltype(hash), decltype(comp)> frequencyCounter(4, hash, comp);
            for(auto& variantField : toBuild->variant)
                ++frequencyCounter[variantField->mpe_mode];

            unsigned maxFrequency = 0;
            for(auto& frequencyCouple : frequencyCounter)
                if(frequencyCouple.second > maxFrequency)
                    toBuild->invariant->mpe_frequent_mode = frequencyCouple.first;

            toReturn[i]->task.value = toBuild;
        }
    else
        for(unsigned i = 0; i < numTask; ++i)
        {
            //Clustering in multiple clusters has to be executed in every cluster
            toReturn[i] = std::unique_ptr<MVCNN::TaskT>(new MVCNN::TaskT());
            toReturn[i]->task.type = MVCNN::SpecificTask_NCE2Task;
            auto toBuild = new MVCNN::NCE2TaskT();
            toBuild->variant = buildNCEVariantFieldsTVector(cm, compilationDescriptor, opIt, i, splitting);
            toBuild->invariant = buildNCEInvariantFieldsT(cm, compilationDescriptor, opIt);

            auto locale_index = std::vector<unsigned int>(1,i);
            toBuild->invariant->input_data->locale_index = locale_index;
            toBuild->invariant->output_data->locale_index = locale_index;
            if (opIt->get<std::string>("taskOp") != "MaxPool")
                toBuild->invariant->weights_data->locale_index = locale_index;
            else if (opIt->get<std::string>("taskOp") == "MaxPool" ||
                     opIt->get<std::string>("taskOp") == "ChannelMajorConvolution" ||
                     opIt->get<std::string>("taskOp") == "DepthwiseConv")
                toBuild->invariant->activation_window->locale_index = locale_index;
            else if (opIt->get<std::string>("taskOp") == "ElementWise")
                toBuild->invariant->weights_table->locale_index = locale_index;

            auto hash = [](const MVCNN::MPE_Mode &g){ return static_cast<std::size_t>(g); };
            auto comp = [](const MVCNN::MPE_Mode &l, const MVCNN::MPE_Mode &r){ return l == r; };

            std::unordered_map<MVCNN::MPE_Mode, unsigned, decltype(hash), decltype(comp)> frequencyCounter(4, hash, comp);
            for(auto& variantField : toBuild->variant)
                ++frequencyCounter[variantField->mpe_mode];

            unsigned maxFrequency = 0;
            for(auto& frequencyCouple : frequencyCounter)
                if(frequencyCouple.second > maxFrequency)
                    toBuild->invariant->mpe_frequent_mode = frequencyCouple.first;

            toReturn[i]->task.value = toBuild;
        }
    return toReturn;
}

MVCNN::UPALayerTaskT * mv::RuntimeModel::buildUPASoftmaxTask(ComputationModel& cm, Element &compilationDescriptor, Control::OpListIterator opIt)
{
    auto input = opIt->getInputTensor(0);
    auto output = opIt->getOutputTensor(0);
    auto toBuild = new MVCNN::UPALayerTaskT();
    //toBuild->maxShaves = ;
    toBuild->softLayerParams.type = MVCNN::SoftwareLayerParams_SoftmaxParams;
    auto softLayerParamsValue = new MVCNN::SoftmaxParamsT();

    std::string axis = opIt->get<std::string>("axis");
    // TODO: magic numbers
    if (axis.compare(std::string("C")) == 0)
        softLayerParamsValue->axis = 1;
    else if (axis.compare(std::string("H")) == 0)
        softLayerParamsValue->axis = 2;
    toBuild->softLayerParams.value = softLayerParamsValue;

    toBuild->inputs.push_back(std::move(buildTensorReferenceT(cm, compilationDescriptor, input)));

    toBuild->outputs.push_back(std::move(buildTensorReferenceT(cm, compilationDescriptor, output)));

    return toBuild;
}

MVCNN::UPALayerTaskT *mv::RuntimeModel::buildUPANormalizeTask(ComputationModel &cm, Element &compilationDescriptor, Control::OpListIterator opIt)
{
    auto input = opIt->getInputTensor(0);
    auto output = opIt->getOutputTensor(0);
    auto toBuild = new MVCNN::UPALayerTaskT();
    //toBuild->maxShaves = ;
    toBuild->softLayerParams.type = MVCNN::SoftwareLayerParams_NormalizeParams;
    auto softLayerParamsValue = new MVCNN::NormalizeParamsT();

    softLayerParamsValue->eps = static_cast<float>(opIt->get<double>("eps"));
    softLayerParamsValue->across_spatial = static_cast<int32_t>(opIt->get<unsigned>("across_spatial"));
    softLayerParamsValue->channel_shared = static_cast<int32_t>(opIt->get<unsigned>("channel_shared"));

    toBuild->softLayerParams.value = softLayerParamsValue;
    toBuild->inputs.push_back(std::move(buildTensorReferenceT(cm, compilationDescriptor, input)));

    toBuild->outputs.push_back(std::move(buildTensorReferenceT(cm, compilationDescriptor, output)));
    toBuild->inputs.push_back(std::move(buildTensorReferenceT(cm, compilationDescriptor, opIt->getInputTensor(1))));

    return toBuild;
}

MVCNN::UPALayerTaskT *mv::RuntimeModel::buildUPAProposalTask(ComputationModel &cm, Element &compilationDescriptor, Control::OpListIterator opIt)
{

    auto toBuild = new MVCNN::UPALayerTaskT();
    //toBuild->maxShaves = ;
    toBuild->softLayerParams.type = MVCNN::SoftwareLayerParams_ProposalParams;
    auto softLayerParamsValue = new MVCNN::ProposalParamsT();

    // input_tensor mapping:
    // 0 = cls_pred --> UPALayerTask.input_data
    // 1 = bbox_pred --> UPALayerTask.weights_data
    // 2 = im_info --> UPALayerTask.weights_table
    // 3 = scale --> ProposalParams.ratio
    // 4 = ratio --> ProposalParams.scale
    auto cls_pred = opIt->getInputTensor(0);
    auto bbox_pred = opIt->getInputTensor(1);
    auto im_info = opIt->getInputTensor(2);
    auto scale = opIt->getInputTensor(3);
    auto ratio = opIt->getInputTensor(4);

    // output tensor mapping:
    // 0 = output --> UPALayerTask.output_data
    auto output = opIt->getOutputTensor(0);

    // Build scale vector
    auto scale_vector = std::vector<float>();
    for (unsigned i = 0; i < scale->size(); ++i)
        scale_vector.push_back(mv::fp16_to_fp32(static_cast<uint16_t>(scale->getIntData().at(i))));

    // Build ratio vector
    auto ratio_vector = std::vector<float>();
    for (unsigned i = 0; i < ratio->size(); ++i)
        ratio_vector.push_back(mv::fp16_to_fp32(static_cast<uint16_t>(ratio->getIntData().at(i))));

    // Fill in tensors
    toBuild->inputs.push_back(std::move(buildTensorReferenceT(cm, compilationDescriptor, cls_pred)));
    toBuild->inputs.push_back(std::move(buildTensorReferenceT(cm, compilationDescriptor, bbox_pred)));
    toBuild->inputs.push_back(std::move(buildTensorReferenceT(cm, compilationDescriptor, im_info)));
    toBuild->inputs.push_back(std::move(buildTensorReferenceT(cm, compilationDescriptor, scale)));
    toBuild->inputs.push_back(std::move(buildTensorReferenceT(cm, compilationDescriptor, ratio)));

    toBuild->outputs.push_back(std::move(buildTensorReferenceT(cm, compilationDescriptor, output)));

    // Fill in required params
    softLayerParamsValue->ratio = ratio_vector;
    softLayerParamsValue->scale = scale_vector;
    softLayerParamsValue->base_size = opIt->get<unsigned>("base_size");
    softLayerParamsValue->pre_nms_topn = opIt->get<unsigned>("pre_nms_topn");
    softLayerParamsValue->post_nms_topn = opIt->get<unsigned>("post_nms_topn");
    softLayerParamsValue->nms_thresh = static_cast<float>(opIt->get<double>("nms_thresh"));
    softLayerParamsValue->feat_stride = opIt->get<unsigned>("feat_stride");
    softLayerParamsValue->min_size = opIt->get<unsigned>("min_size");

    // Fill in optional params
    softLayerParamsValue->pre_nms_thresh = static_cast<float>(opIt->get<double>("pre_nms_thresh"));
    softLayerParamsValue->clip_before_nms = opIt->get<bool>("clip_before_nms");
    softLayerParamsValue->clip_after_nms = opIt->get<bool>("clip_after_nms");
    softLayerParamsValue->normalize = opIt->get<bool>("normalize");
    softLayerParamsValue->box_size_scale = static_cast<float>(opIt->get<double>("box_size_scale"));
    softLayerParamsValue->box_coordinate_scale = static_cast<float>(opIt->get<double>("box_coordinate_scale"));
    softLayerParamsValue->framework = opIt->get<std::string>("framework");
    softLayerParamsValue->for_deformable = opIt->get<bool>("for_deformable");

    toBuild->softLayerParams.value = softLayerParamsValue;

    return toBuild;
}

MVCNN::UPALayerTaskT * mv::RuntimeModel::buildUPAROIPoolingTask(ComputationModel& cm, Element &compilationDescriptor, Control::OpListIterator opIt)
{

    auto toBuild = new MVCNN::UPALayerTaskT();
    //toBuild->maxShaves = ;
    toBuild->softLayerParams.type = MVCNN::SoftwareLayerParams_ROIPoolingParams;
    auto softLayerParamsValue = new MVCNN::ROIPoolingParamsT();

    auto input = opIt->getInputTensor(0);
    auto coords = opIt->getInputTensor(1);

    auto output = opIt->getOutputTensor(0);

    // Fill in tensors
    toBuild->inputs.push_back(std::move(buildTensorReferenceT(cm, compilationDescriptor, input)));
    toBuild->inputs.push_back(std::move(buildTensorReferenceT(cm, compilationDescriptor, coords)));

    toBuild->outputs.push_back(std::move(buildTensorReferenceT(cm, compilationDescriptor, output)));

    // Fill in required params
    softLayerParamsValue->pooled_w = opIt->get<unsigned>("pooled_w");
    softLayerParamsValue->pooled_h = opIt->get<unsigned>("pooled_h");
    softLayerParamsValue->spatial_scale = static_cast<float>(opIt->get<double>("spatial_scale"));
    softLayerParamsValue->roi_pooling_method = opIt->get<unsigned>("roi_pooling_method");
    softLayerParamsValue->num_rois = opIt->get<unsigned>("num_rois");

    toBuild->softLayerParams.value = softLayerParamsValue;

    return toBuild;
}

MVCNN::UPALayerTaskT * mv::RuntimeModel::buildUPAInterpTask(ComputationModel& cm, Element &compilationDescriptor, Control::OpListIterator opIt)
{

    auto toBuild = new MVCNN::UPALayerTaskT();

    toBuild->softLayerParams.type = MVCNN::SoftwareLayerParams_InterpParams;
    auto softLayerParamsValue = new MVCNN::InterpParamsT();

    auto input = opIt->getInputTensor(0);
    auto output = opIt->getOutputTensor(0);

    // Fill in tensors
    toBuild->inputs.push_back(std::move(buildTensorReferenceT(cm, compilationDescriptor, input)));
    toBuild->outputs.push_back(std::move(buildTensorReferenceT(cm, compilationDescriptor, output)));

    // Fill in required params
    softLayerParamsValue->pad_beg = opIt->get<unsigned>("pad_beg");
    softLayerParamsValue->pad_end = opIt->get<unsigned>("pad_end");
    softLayerParamsValue->align_corners = opIt->get<bool>("align_corners");
    toBuild->softLayerParams.value = softLayerParamsValue;

    return toBuild;
}

MVCNN::UPALayerTaskT * mv::RuntimeModel::buildUPANormTask(ComputationModel& cm, Element &compilationDescriptor, Control::OpListIterator opIt)
{

    auto toBuild = new MVCNN::UPALayerTaskT();

    toBuild->softLayerParams.type = MVCNN::SoftwareLayerParams_NormParams;
    auto softLayerParamsValue = new MVCNN::NormParamsT();

    auto input = opIt->getInputTensor(0);
    auto output = opIt->getOutputTensor(0);

    // Fill in tensors
    toBuild->inputs.push_back(std::move(buildTensorReferenceT(cm, compilationDescriptor, input)));
    toBuild->outputs.push_back(std::move(buildTensorReferenceT(cm, compilationDescriptor, output)));

    // Fill in required params
    softLayerParamsValue->alpha = opIt->get<double>("alpha");
    softLayerParamsValue->beta = opIt->get<double>("beta");
    softLayerParamsValue->region = opIt->get<std::string>("region");
    softLayerParamsValue->local_size = opIt->get<unsigned>("local_size");
    toBuild->softLayerParams.value = softLayerParamsValue;

    return toBuild;
}

MVCNN::UPALayerTaskT * mv::RuntimeModel::buildUPAQuantizeTask(ComputationModel& cm, Element &compilationDescriptor, Control::OpListIterator opIt)
{
    auto input = opIt->getInputTensor(0);
    auto output = opIt->getOutputTensor(0);
    auto toBuild = new MVCNN::UPALayerTaskT();
    //toBuild->maxShaves = ;
    toBuild->softLayerParams.type = MVCNN::SoftwareLayerParams_QuantizeParams;
    auto softLayerParamsValue = new MVCNN::QuantizeParamsT();

    auto quantizationParams = opIt->get<mv::QuantizationParams>("quantParams");
    auto quantScale = quantizationParams.getScale();
    auto quantZero = quantizationParams.getZeroPoint();

    // Convert vectors to fp16
    auto scale_vector = std::vector<unsigned short>();
    for (unsigned i = 0; i < quantScale.size(); ++i)
        scale_vector.push_back(mv::fp32_to_fp16(static_cast<float>(quantScale.at(i))));

    auto zero_vector = std::vector<unsigned short>();
    for (unsigned i = 0; i < quantZero.size(); ++i)
        zero_vector.push_back(mv::fp32_to_fp16(static_cast<float>(quantZero.at(i))));

    toBuild->softLayerParams.value = softLayerParamsValue;
    softLayerParamsValue->scale = std::vector<unsigned short>(scale_vector.begin(), scale_vector.end());
    softLayerParamsValue->zero = std::vector<unsigned short>(zero_vector.begin(), zero_vector.end());

    toBuild->input_data = buildTensorReferenceT(cm, compilationDescriptor, input);
    toBuild->output_data = buildTensorReferenceT(cm, compilationDescriptor, output);

    toBuild->inputs.push_back(std::move(buildTensorReferenceT(cm, compilationDescriptor, input)));

    toBuild->outputs.push_back(std::move(buildTensorReferenceT(cm, compilationDescriptor, output)));

    return toBuild;
}

MVCNN::UPALayerTaskT * mv::RuntimeModel::buildUPAResampleTask(ComputationModel& cm, Element &compilationDescriptor, Control::OpListIterator opIt)
{
    auto input = opIt->getInputTensor(0);
    auto output = opIt->getOutputTensor(0);
    auto toBuild = new MVCNN::UPALayerTaskT();
    //toBuild->maxShaves = ;
    toBuild->softLayerParams.type = MVCNN::SoftwareLayerParams_ResampleParams;
    auto softLayerParamsValue = new MVCNN::ResampleParamsT();
    std::string interpolation = opIt->get<std::string>("interpolation");
    if (interpolation.compare(std::string("BILINEAR")) == 0)
        softLayerParamsValue->interpolation = MVCNN::InterpolationMethod_BILINEAR;
    else if (interpolation.compare(std::string("BICUBIC")) == 0)
        softLayerParamsValue->interpolation = MVCNN::InterpolationMethod_BICUBIC;
    else
        softLayerParamsValue->interpolation = MVCNN::InterpolationMethod_NEAREST;

    softLayerParamsValue->antialias = opIt->get<bool>("antialias");

    toBuild->softLayerParams.value = softLayerParamsValue;

    toBuild->input_data = buildTensorReferenceT(cm, compilationDescriptor, input);
    toBuild->output_data = buildTensorReferenceT(cm, compilationDescriptor, output);

    toBuild->inputs.push_back(std::move(buildTensorReferenceT(cm, compilationDescriptor, input)));

    toBuild->outputs.push_back(std::move(buildTensorReferenceT(cm, compilationDescriptor, output)));

    return toBuild;
}


MVCNN::UPALayerTaskT * mv::RuntimeModel::buildUPAReshapeTask(ComputationModel& cm, Element &compilationDescriptor, Control::OpListIterator opIt)
{
    auto input = opIt->getInputTensor(0);
    auto output = opIt->getOutputTensor(0);
    auto toBuild = new MVCNN::UPALayerTaskT();
    //toBuild->maxShaves = ;
    toBuild->softLayerParams.type = MVCNN::SoftwareLayerParams_ReshapeParams;
    auto softLayerParamsValue = new MVCNN::ReshapeParamsT();

    toBuild->softLayerParams.value = softLayerParamsValue;

    toBuild->input_data = buildTensorReferenceT(cm, compilationDescriptor, input);
    toBuild->output_data = buildTensorReferenceT(cm, compilationDescriptor, output);

    toBuild->inputs.push_back(std::move(buildTensorReferenceT(cm, compilationDescriptor, input)));

    toBuild->outputs.push_back(std::move(buildTensorReferenceT(cm, compilationDescriptor, output)));

    return toBuild;
}

MVCNN::UPALayerTaskT * mv::RuntimeModel::buildUPARegionYoloTask(ComputationModel& cm, Element &compilationDescriptor, Control::OpListIterator opIt)
{
    auto input = opIt->getInputTensor(0);
    auto output = opIt->getOutputTensor(0);
    auto toBuild = new MVCNN::UPALayerTaskT();
    //toBuild->maxShaves = ;
    toBuild->softLayerParams.type = MVCNN::SoftwareLayerParams_RegionYOLOParams;
    auto softLayerParamsValue = new MVCNN::RegionYOLOParamsT();

    toBuild->inputs.push_back(std::move(buildTensorReferenceT(cm, compilationDescriptor, input)));

    toBuild->outputs.push_back(std::move(buildTensorReferenceT(cm, compilationDescriptor, output)));

    softLayerParamsValue->coords = opIt->get<unsigned>("coords");
    softLayerParamsValue->classes = opIt->get<unsigned>("classes");
    softLayerParamsValue->num = opIt->get<unsigned>("num");
    softLayerParamsValue->do_softmax = opIt->get<bool>("do_softmax");
    auto mask_uint = opIt->get<std::vector<unsigned>>("mask");
    std::vector<int> mask(std::begin(mask_uint), std::end(mask_uint));
    softLayerParamsValue->mask = mask;

    toBuild->softLayerParams.value = softLayerParamsValue;

    return toBuild;
}

MVCNN::UPALayerTaskT * mv::RuntimeModel::buildUPAReorgYoloTask(ComputationModel& cm, Element &compilationDescriptor, Control::OpListIterator opIt)
{
    auto input = opIt->getInputTensor(0);
    auto output = opIt->getOutputTensor(0);
    auto toBuild = new MVCNN::UPALayerTaskT();
    //toBuild->maxShaves = ;
    toBuild->softLayerParams.type = MVCNN::SoftwareLayerParams_ReorgYOLOParams;
    auto softLayerParamsValue = new MVCNN::ReorgYOLOParamsT();

    toBuild->inputs.push_back(std::move(buildTensorReferenceT(cm, compilationDescriptor, input)));

    toBuild->outputs.push_back(std::move(buildTensorReferenceT(cm, compilationDescriptor, output)));

    softLayerParamsValue->stride = static_cast<int>(opIt->get<unsigned>("stride"));

    toBuild->softLayerParams.value = softLayerParamsValue;

    return toBuild;
}

MVCNN::UPALayerTaskT * mv::RuntimeModel::buildUPAPermuteTask(ComputationModel& cm, Element &compilationDescriptor, Control::OpListIterator opIt)
{
    auto input = opIt->getInputTensor(0);
    auto output = opIt->getOutputTensor(0);
    auto toBuild = new MVCNN::UPALayerTaskT();
    //toBuild->maxShaves = ;
    toBuild->softLayerParams.type = MVCNN::SoftwareLayerParams_PermuteParams;
    auto softLayerParamsValue = new MVCNN::PermuteParamsT();

    auto x = opIt->get<unsigned>("permute_order_x");
    auto y = opIt->get<unsigned>("permute_order_y");
    auto z = opIt->get<unsigned>("permute_order_z");
    std::unique_ptr<MVCNN::order3> order3 = std::unique_ptr<MVCNN::order3>(new MVCNN::order3(x,y,z));
    softLayerParamsValue->permute_order = std::move(order3);

    toBuild->softLayerParams.value = softLayerParamsValue;

    toBuild->input_data = buildTensorReferenceT(cm, compilationDescriptor, input);
    toBuild->output_data = buildTensorReferenceT(cm, compilationDescriptor, output);

    toBuild->inputs.push_back(std::move(buildTensorReferenceT(cm, compilationDescriptor, input)));

    toBuild->outputs.push_back(std::move(buildTensorReferenceT(cm, compilationDescriptor, output)));

    return toBuild;
}

MVCNN::UPALayerTaskT * mv::RuntimeModel::buildUPADetectionOutputTask(ComputationModel& cm, Element &compilationDescriptor, Control::OpListIterator opIt)
{

    auto toBuild = new MVCNN::UPALayerTaskT();
    //toBuild->maxShaves = ;
    toBuild->softLayerParams.type = MVCNN::SoftwareLayerParams_DetectionOutputParams;
    auto softLayerParamsValue = new MVCNN::DetectionOutputParamsT();

    auto box_logits = opIt->getInputTensor(0);
    auto class_preds = opIt->getInputTensor(1);
    auto proposals = opIt->getInputTensor(2);

    auto output = opIt->getOutputTensor(0);

    // Fill in tensors
    toBuild->inputs.push_back(std::move(buildTensorReferenceT(cm, compilationDescriptor, box_logits)));
    toBuild->inputs.push_back(std::move(buildTensorReferenceT(cm, compilationDescriptor, class_preds)));
    toBuild->inputs.push_back(std::move(buildTensorReferenceT(cm, compilationDescriptor, proposals)));

    toBuild->outputs.push_back(std::move(buildTensorReferenceT(cm, compilationDescriptor, output)));

    // Fill in required params
    softLayerParamsValue->num_classes = opIt->get<int64_t>("num_classes");
    softLayerParamsValue->keep_top_k = opIt->get<int64_t>("keep_top_k");
    softLayerParamsValue->nms_threshold = static_cast<float>(opIt->get<double>("nms_threshold"));
    softLayerParamsValue->background_label_id = opIt->get<int64_t>("background_label_id");
    softLayerParamsValue->top_k = opIt->get<int64_t>("top_k");
    softLayerParamsValue->variance_encoded_in_target = opIt->get<bool>("variance_encoded_in_target");
    softLayerParamsValue->code_type = opIt->get<std::string>("code_type");
    softLayerParamsValue->share_location = opIt->get<bool>("share_location");
    softLayerParamsValue->confidence_threshold = static_cast<float>(opIt->get<double>("confidence_threshold"));
    softLayerParamsValue->clip_before_nms = opIt->get<bool>("clip_before_nms");
    softLayerParamsValue->clip_after_nms = opIt->get<bool>("clip_after_nms");
    softLayerParamsValue->decrease_label_id = opIt->get<int64_t>("decrease_label_id");
    softLayerParamsValue->normalized = opIt->get<bool>("normalized");
    softLayerParamsValue->input_height = opIt->get<int64_t>("input_height");
    softLayerParamsValue->input_width = opIt->get<int64_t>("input_width");
    softLayerParamsValue->objectness_score = static_cast<float>(opIt->get<double>("objectness_score"));

    toBuild->softLayerParams.value = softLayerParamsValue;

    return toBuild;
}

MVCNN::UPALayerTaskT * mv::RuntimeModel::buildUPAPriorboxTask(ComputationModel& cm, Element &compilationDescriptor, Control::OpListIterator opIt)
{

    auto toBuild = new MVCNN::UPALayerTaskT();
    //toBuild->maxShaves = ;
    toBuild->softLayerParams.type = MVCNN::SoftwareLayerParams_PriorboxParams;
    auto softLayerParamsValue = new MVCNN::PriorboxParamsT();

    auto priorbox = opIt->getInputTensor(0);
    auto image = opIt->getInputTensor(1);
    auto min_sizes = opIt->getInputTensor(2);
    auto max_sizes = opIt->getInputTensor(3);
    auto aspect_ratios = opIt->getInputTensor(4);
    auto variances = opIt->getInputTensor(5);

    auto output = opIt->getOutputTensor(0);

    auto min_sizes_vector = std::vector<float>();
    auto min_sizes_data = min_sizes->getData();
    for (unsigned i = 0; i < min_sizes->size(); ++i)
        min_sizes_vector.push_back(mv::fp16_to_fp32(static_cast<uint16_t>(min_sizes->getIntData().at(i))));

    auto max_sizes_vector = std::vector<float>();
    for (unsigned i = 0; i < max_sizes->size(); ++i)
        max_sizes_vector.push_back(mv::fp16_to_fp32(static_cast<uint16_t>(max_sizes->getIntData().at(i))));

    auto aspect_ratios_vector = std::vector<float>();
    for (unsigned i = 0; i < aspect_ratios->size(); ++i)
        aspect_ratios_vector.push_back(mv::fp16_to_fp32(static_cast<uint16_t>(aspect_ratios->getIntData().at(i))));

    auto variances_vector = std::vector<float>();
    for (unsigned i = 0; i < variances->size(); ++i)
        variances_vector.push_back(mv::fp16_to_fp32(static_cast<uint16_t>(variances->getIntData().at(i))));

    // Fill in tensors
    toBuild->inputs.push_back(std::move(buildTensorReferenceT(cm, compilationDescriptor, image)));
    toBuild->inputs.push_back(std::move(buildTensorReferenceT(cm, compilationDescriptor, priorbox)));

    toBuild->outputs.push_back(std::move(buildTensorReferenceT(cm, compilationDescriptor, output)));

    // Fill in required params
    softLayerParamsValue->min_sizes = std::vector<float>(min_sizes_vector.begin(), min_sizes_vector.end());
    softLayerParamsValue->max_sizes = std::vector<float>(max_sizes_vector.begin(), max_sizes_vector.end());
    softLayerParamsValue->aspect_ratios = std::vector<float>(aspect_ratios_vector.begin(), aspect_ratios_vector.end());
    softLayerParamsValue->variances = std::vector<float>(variances_vector.begin(), variances_vector.end());
    softLayerParamsValue->flip = opIt->get<unsigned>("flip");
    softLayerParamsValue->clip = opIt->get<unsigned>("clip");
    softLayerParamsValue->step_w = static_cast<float>(opIt->get<double>("step_w"));
    softLayerParamsValue->step_h = static_cast<float>(opIt->get<double>("step_h"));
    softLayerParamsValue->offset = static_cast<float>(opIt->get<double>("offset"));

    toBuild->softLayerParams.value = softLayerParamsValue;

    return toBuild;
}

MVCNN::UPALayerTaskT * mv::RuntimeModel::buildUPAArgmaxTask(ComputationModel& cm, Element &compilationDescriptor, Control::OpListIterator opIt)
{
    auto input = opIt->getInputTensor(0);
    auto output = opIt->getOutputTensor(0);
    auto toBuild = new MVCNN::UPALayerTaskT();
    //toBuild->maxShaves = ;
    toBuild->softLayerParams.type = MVCNN::SoftwareLayerParams_ArgMaxParams;
    auto softLayerParamsValue = new MVCNN::ArgMaxParamsT();

    auto out_max_val = static_cast<bool>(opIt->get<int64_t>("out_max_val"));
    auto top_k = static_cast<unsigned>(opIt->get<int64_t>("top_k"));
    auto axis = opIt->get<int64_t>("axis");

    softLayerParamsValue->out_max_val = out_max_val;
    softLayerParamsValue->top_k = top_k;
    softLayerParamsValue->axis = axis;

    toBuild->softLayerParams.value = softLayerParamsValue;

    toBuild->input_data = buildTensorReferenceT(cm, compilationDescriptor, input);
    toBuild->output_data = buildTensorReferenceT(cm, compilationDescriptor, output);

    toBuild->inputs.push_back(std::move(buildTensorReferenceT(cm, compilationDescriptor, input)));

    toBuild->outputs.push_back(std::move(buildTensorReferenceT(cm, compilationDescriptor, output)));

    return toBuild;
}

MVCNN::UPALayerTaskT * mv::RuntimeModel::buildUPAPassthroughTask(ComputationModel& cm, Element &compilationDescriptor, Control::OpListIterator opIt)
{
    auto input = opIt->getInputTensor(0);
    auto output = opIt->getOutputTensor(0);
    auto toBuild = new MVCNN::UPALayerTaskT();
    //toBuild->maxShaves = ;
    toBuild->softLayerParams.type = MVCNN::SoftwareLayerParams_PassthroughParams;
    auto softLayerParamsValue = new MVCNN::PassthroughParamsT();
    //softLayerParamsValue->min_delay_us = 1000;
    toBuild->softLayerParams.value = softLayerParamsValue;

    toBuild->inputs.push_back(std::move(buildTensorReferenceT(cm, compilationDescriptor, input)));

    toBuild->outputs.push_back(std::move(buildTensorReferenceT(cm, compilationDescriptor, output)));

    return toBuild;
}

MVCNN::UPALayerTaskT * mv::RuntimeModel::buildUPAEltwiseFP16Task(ComputationModel& cm, Element &compilationDescriptor, Control::OpListIterator opIt)
{
    auto input0 = opIt->getInputTensor(0);
    auto input1 = opIt->getInputTensor(1);
    auto output = opIt->getOutputTensor(0);
    auto toBuild = new MVCNN::UPALayerTaskT();
    //toBuild->maxShaves = ;
    toBuild->softLayerParams.type = MVCNN::SoftwareLayerParams_EltwiseParams;
    auto softLayerParamsValue = new MVCNN::EltwiseParamsT();

    toBuild->softLayerParams.value = softLayerParamsValue;
    std::string operation = opIt->get<std::string>("eltwiseType");
    if (operation.compare(std::string("Add")) == 0)
        softLayerParamsValue->operation = "sum";

    toBuild->inputs.push_back(std::move(buildTensorReferenceT(cm, compilationDescriptor, input0)));
    toBuild->inputs.push_back(std::move(buildTensorReferenceT(cm, compilationDescriptor, input1)));

    toBuild->outputs.push_back(std::move(buildTensorReferenceT(cm, compilationDescriptor, output)));

    return toBuild;
}

MVCNN::UPALayerTaskT * mv::RuntimeModel::buildUPADummyTask(ComputationModel& cm, Element &compilationDescriptor, Control::OpListIterator opIt)
{
    UNUSED(cm);
    UNUSED(compilationDescriptor);
    UNUSED(opIt);
    auto toBuild = new MVCNN::UPALayerTaskT();
    //toBuild->maxShaves = ;
    toBuild->softLayerParams.type = MVCNN::SoftwareLayerParams_DummyParams;
    auto softLayerParamsValue = new MVCNN::DummyParamsT();
    //softLayerParamsValue->message = "Hello!";
    //softLayerParamsValue->executeShaveKernel = false;
    toBuild->softLayerParams.value = softLayerParamsValue;
    return toBuild;
}

MVCNN::UPALayerTaskT * mv::RuntimeModel::buildUPACustomTask(ComputationModel& cm, Element &compilationDescriptor, Control::OpListIterator opIt)
{
    auto toBuild = new MVCNN::UPALayerTaskT();
    toBuild->softLayerParams.type = MVCNN::SoftwareLayerParams_CustomLayerParams;

    for (size_t i = 0; i < opIt->inputSlots(); i++) {
        const auto input = opIt->getInputTensor(i);
        toBuild->inputs.push_back(buildTensorReferenceT(cm, compilationDescriptor, input));
    }

    for (size_t i = 0; i < opIt->outputSlots(); i++) {
        const auto output = opIt->getOutputTensor(i);
        toBuild->outputs.push_back(buildTensorReferenceT(cm, compilationDescriptor, output));
    }

    auto softParams = new MVCNN::CustomLayerParamsT();
    toBuild->softLayerParams.value = softParams;

    softParams->leonPreambleID = -1u;  // unused

    const auto pack = [](const std::vector<uint8_t>& src) {
        auto packed = std::vector<uint64_t>(ceil_division(src.size(), 8));
        for (size_t i = 0; i < src.size(); i++) {
            ((uint8_t *) packed.data())[i] = src[i];
        }
        return packed;
    };

    const auto paramData = opIt->get<std::vector<uint8_t>>("paramData");
    softParams->paramData = std::unique_ptr<MVCNN::BinaryDataT>(new MVCNN::BinaryDataT());
    softParams->paramData->underlying_type = MVCNN::DType::DType_U8;
    softParams->paramData->data = pack(paramData);
    softParams->paramData->length = paramData.size();

    const auto kernelData = opIt->get<std::vector<uint8_t>>("kernelData");
    softParams->kernelData = std::unique_ptr<MVCNN::BinaryDataT>(new MVCNN::BinaryDataT());
    softParams->kernelData->underlying_type = MVCNN::DType::DType_U8;
    softParams->kernelData->data = pack(kernelData);
    softParams->kernelData->length = kernelData.size();

    return toBuild;
}

// For now 1:1 mapping
std::vector<std::unique_ptr<MVCNN::TaskT>> mv::RuntimeModel::buildUPATask(ComputationModel& cm, mv::Element &compilationDescriptor, Control::OpListIterator opIt)
{
    std::vector<std::unique_ptr<MVCNN::TaskT>> toReturn = std::vector<std::unique_ptr<MVCNN::TaskT>>(1);

    toReturn[0] = std::unique_ptr<MVCNN::TaskT>(new MVCNN::TaskT());
    toReturn[0]->task.type = MVCNN::SpecificTask_UPALayerTask;

    std::string underlyingTask = opIt->get<std::string>("taskOp");
    if(underlyingTask == "Identity")
        toReturn[0]->task.value = buildUPAPassthroughTask(cm, compilationDescriptor, opIt);
    else if(underlyingTask == "Dummy")
        toReturn[0]->task.value = buildUPADummyTask(cm, compilationDescriptor, opIt);
    else if(underlyingTask == "Softmax")
        toReturn[0]->task.value = buildUPASoftmaxTask(cm, compilationDescriptor, opIt);
    else if(underlyingTask == "Proposal")
        toReturn[0]->task.value = buildUPAProposalTask(cm, compilationDescriptor, opIt);
    else if(underlyingTask == "ROIPooling")
        toReturn[0]->task.value = buildUPAROIPoolingTask(cm, compilationDescriptor, opIt);
    else if(underlyingTask == "Quantize")
        toReturn[0]->task.value = buildUPAQuantizeTask(cm, compilationDescriptor, opIt);
    else if(underlyingTask == "Resample")
        toReturn[0]->task.value = buildUPAResampleTask(cm, compilationDescriptor, opIt);
    else if(underlyingTask == "Reshape")
        toReturn[0]->task.value = buildUPAReshapeTask(cm, compilationDescriptor, opIt);
    else if(underlyingTask == "RegionYolo")
        toReturn[0]->task.value = buildUPARegionYoloTask(cm, compilationDescriptor, opIt);
    else if(underlyingTask == "ReorgYolo")
        toReturn[0]->task.value = buildUPAReorgYoloTask(cm, compilationDescriptor, opIt);
    else if (underlyingTask == "Normalize")
        toReturn[0]->task.value = buildUPANormalizeTask(cm, compilationDescriptor, opIt);
    else if(underlyingTask == "Permute")
        toReturn[0]->task.value = buildUPAPermuteTask(cm, compilationDescriptor, opIt);
    else if(underlyingTask == "Eltwise")
        toReturn[0]->task.value = buildUPAEltwiseFP16Task(cm, compilationDescriptor, opIt);
    else if(underlyingTask == "Interp")
        toReturn[0]->task.value = buildUPAInterpTask(cm, compilationDescriptor, opIt);
    else if(underlyingTask == "Norm")
        toReturn[0]->task.value = buildUPANormTask(cm, compilationDescriptor, opIt);
    else if(underlyingTask == "DetectionOutput")
        toReturn[0]->task.value = buildUPADetectionOutputTask(cm, compilationDescriptor, opIt);
    else if(underlyingTask == "Priorbox")
        toReturn[0]->task.value = buildUPAPriorboxTask(cm, compilationDescriptor, opIt);
    else if(underlyingTask == "Argmax")
        toReturn[0]->task.value = buildUPAArgmaxTask(cm, compilationDescriptor, opIt);
    else if(underlyingTask == "Custom")
        toReturn[0]->task.value = buildUPACustomTask(cm, compilationDescriptor, opIt);
    // TODO: Add other UPA layers

    return toReturn;
}

std::vector<std::unique_ptr<MVCNN::TaskT>> mv::RuntimeModel::buildControllerTaskT(ComputationModel& cm, mv::Element &compilationDescriptor, Control::OpListIterator opIt)
{
    UNUSED(cm);
    UNUSED(compilationDescriptor);
    UNUSED(opIt);
    std::vector<std::unique_ptr<MVCNN::TaskT>> toReturn = std::vector<std::unique_ptr<MVCNN::TaskT>>(1);
    return toReturn;
}

std::unique_ptr<MVCNN::BarrierReferenceT> mv::RuntimeModel::buildBarrierReferenceT(ComputationModel& , Element& , BarrierDependencies dep)
{
    std::unique_ptr<MVCNN::BarrierReferenceT> toBuild = std::unique_ptr<MVCNN::BarrierReferenceT>(new MVCNN::BarrierReferenceT());
    int waitBarrier = dep.getWait();
    if(waitBarrier != -1)
        toBuild->wait_barriers = {unsigned(waitBarrier)};
    toBuild->update_barriers = dep.getUpdate();
    return toBuild;
}

std::unique_ptr<MVCNN::BarrierReferenceT> mv::RuntimeModel::buildBarrierReferenceT()
{
    std::unique_ptr<MVCNN::BarrierReferenceT> toBuild = std::unique_ptr<MVCNN::BarrierReferenceT>(new MVCNN::BarrierReferenceT());
    return toBuild;
}

std::vector<std::unique_ptr<MVCNN::TaskT>> mv::RuntimeModel::buildTaskT(ComputationModel& cm, mv::Element &compilationDescriptor, Control::OpListIterator opIt)
{

    std::vector<std::unique_ptr<MVCNN::TaskT>> vecToBuild = buildSpecificTaskUnion(cm, compilationDescriptor, opIt);

    for(auto& toBuild: vecToBuild)
    {
        toBuild->name = opIt->getName();

        // NOTE: This might change in the future
        if(opIt->hasAttr("opId"))
            toBuild->sourceTaskIDs = {opIt->get<unsigned>("opId")};

        if(opIt->hasAttr("BarrierDeps"))
            toBuild->associated_barriers = buildBarrierReferenceT(cm, compilationDescriptor, opIt->get<mv::BarrierDependencies>("BarrierDeps"));
        else
            toBuild->associated_barriers = buildBarrierReferenceT();
    }

    return vecToBuild;
}

unsigned mv::RuntimeModel::countProducerConsumerTasks(mv::ComputationModel& cm, mv::Control::OpListIterator opIt)
{
    std::string taskType = opIt->getOpType();
    unsigned toReturn = 0;
    unsigned numClusters = cm.getGlobalConfigParams()->get<int>("Number_of_Clusters");
    if(taskType == "DPUTask")
    {
        if(opIt->hasAttr("splitStrategy"))
        {
            std::string strategy = opIt->get<std::string>("splitStrategy");
            if(strategy != "Clustering")
            {
                for(unsigned i = 0; i < numClusters; ++i)
                {
                    if (opIt->hasAttr("Workloads" + std::to_string(i)))
                    {
                        auto& workloads = opIt->get<mv::Workloads>("Workloads" + std::to_string(i));
                        toReturn += workloads.nWorkloads();
                    }
                    else
                        toReturn = numClusters;
                }
            }
            else
            {
                auto& workloads = opIt->get<mv::Workloads>("Workloads0");
                toReturn = workloads.nWorkloads() * numClusters;
            }
        }
    }
    else if(taskType == "DMATask")
    {
        auto inputTensor = opIt->getInputTensor(0);
        // Weights sparsity new approach: it doesn't matter what strategy is chosen,
        // the number of dma is doubled since the strategy is shared between weights
        // and sparsity map. Techinically the isPopulated() check is not needed
        // because we never transfer sparse activation tensors.

        unsigned multiplicator = 1;
        if(inputTensor->isPopulated() && inputTensor->isSparse())
            multiplicator = 2;
        if(numClusters > 1)
        {
            bool sourceIsBroadCasted = opIt->getInputTensor(0)->isBroadcasted();
            //NOTE: In case I spill from soh and I bring a sok tensor is not broadcasted
            if ((opIt->getInputTensor(0)->get<std::string>("splitStrategy") == "SplitOverH" && opIt->getOutputTensor(0)->get<std::string>("splitStrategy") == "SplitOverK") || (opIt->getInputTensor(0)->get<std::string>("splitStrategy") == "SplitOverHOverlapped" && opIt->getOutputTensor(0)->get<std::string>("splitStrategy") == "SplitOverK"))
                toReturn = 1;
            // NOTE: a sok tensor might come from a different strategy op
            else if(!sourceIsBroadCasted)
                toReturn = numClusters;
            else
                toReturn = 1;
            if ((opIt->getInputTensor(0)->get<std::string>("splitStrategy") == "Clustering"))
                toReturn = 1;
        }
        else
            toReturn = 1;

        toReturn *= multiplicator;
    }
    else if(taskType == "UPATask")
        toReturn = 1;

    return toReturn;
}

std::unique_ptr<MVCNN::BarrierT> mv::RuntimeModel::buildBarrierT(mv::ComputationModel& model, mv::Element& , mv::Control::OpListIterator opIt)
{
    mv::ControlModel cm(model);

    std::unique_ptr<MVCNN::BarrierT> toBuild = std::unique_ptr<MVCNN::BarrierT>(new MVCNN::BarrierT());
    auto barrier = opIt->get<mv::Barrier>("Barrier");

    toBuild->barrier_id = barrier.getIndex();
    toBuild->consumer_count = 0;
    toBuild->producer_count = 0;

    for(auto producer = opIt.leftmostParent(); producer != cm.opEnd(); ++producer)
        toBuild->producer_count += countProducerConsumerTasks(model, producer);

    for(auto consumer = opIt.leftmostChild(); consumer != cm.opEnd(); ++consumer)
        toBuild->consumer_count += countProducerConsumerTasks(model, consumer);

    return toBuild;
}

std::vector<std::unique_ptr<MVCNN::BarrierT>> mv::RuntimeModel::buildBarrierTable(mv::ComputationModel& cm, mv::Element& compilationDescriptor)
{
    mv::OpModel om(cm);
    mv::ControlModel controlModel(cm);

    auto barrierTasks = om.getOps("BarrierTask");
    std::sort(
        barrierTasks.begin(),
        barrierTasks.end(),
        [](const mv::Data::OpListIterator& a, const mv::Data::OpListIterator& b) -> bool { return a->get<mv::Barrier>("Barrier").getIndex() < b->get<mv::Barrier>("Barrier").getIndex(); }
        );

    unsigned n = barrierTasks.size();
    std::vector<std::unique_ptr<MVCNN::BarrierT>> toBuild = std::vector<std::unique_ptr<MVCNN::BarrierT>>(n);
    for(unsigned i = 0; i < n; ++i)
        toBuild[i] = buildBarrierT(cm, compilationDescriptor, controlModel.switchContext(barrierTasks[i]));
    return toBuild;
}

void mv::RuntimeModel::buildHeader(ComputationModel &cm, Element &compilationDescriptor)
{
    //HEADER
    graphFile_.header = buildSummaryHeaderMetaInformations(cm, compilationDescriptor);
}

void mv::RuntimeModel::buildGraphFile(ComputationModel& cm, mv::Element& compilationDescriptor)
{
    mv::OpModel om(cm);
    mv::DataModel dm(cm);

    unsigned numClusters = dm.getGlobalConfigParams()->get<int>("Number_of_Clusters");

    auto globalConfigurationParameters = cm.getGlobalConfigParams();
    auto huffmanCompression = globalConfigurationParameters->get<bool>("HuffmanCompression");

    graphFile_.header = buildSummaryHeaderT(cm, compilationDescriptor, std::move(graphFile_.header));

    // Binary Data
    graphFile_.binary_data = std::vector<std::unique_ptr<MVCNN::BinaryDataT>>();
    std::vector<Tensor *> toSort;
    for(auto opIterator = om.opBegin(); opIterator != om.opEnd(); ++opIterator)
    {
        std::string opType = opIterator->getOpType();
        if (opType == "Constant" || opType == "ConstantInt" || opType == "ConstantDataElement")
        {
            auto tIt = opIterator->getOutputTensor(0);

            // Weights sparsity new approach: there is a separate constant for
            // each cluster
            if(tIt->isSparse())
            {
                auto sparsityMapIterator = dm.getTensor(tIt->getSparsityMap()->getName());
                toSort.push_back(&(*sparsityMapIterator));
                if(tIt->get<std::string>("splitStrategy") == "SplitOverK")
                {
                    for(std::size_t i = 0; i < numClusters; ++i)
                        toSort.push_back(&(tIt->getSubTensor(i)));
                }
                else
                    toSort.push_back(&(*tIt));
            }
            //Serialize SOK weights individually
            if(tIt->isSplitOverK() && (tIt->get<mv::DType>("dType") == mv::DType("UInt8") || tIt->get<mv::DType>("dType") == mv::DType("Int8"))) 
            {
                if(tIt->get<std::string>("splitStrategy") == "SplitOverK")
                {
                    for(std::size_t i = 0; i < numClusters; ++i)
                        toSort.push_back(&(tIt->getSubTensor(i)));
                }
                else
                    toSort.push_back(&(*tIt));
            }
            else
                toSort.push_back(&(*tIt));

        }
    }
    std::sort(toSort.begin(), toSort.end(), [](mv::Tensor * t1, mv::Tensor * t2){return (t1->get<unsigned>("graphFileIndex") < t2->get<unsigned>("graphFileIndex"));});
    for(auto& tIt : toSort)
    {
        //std::cout << "Serializing to binary data section " << tensorIt->getName() << std::endl;
        graphFile_.binary_data.push_back(buildBinaryDataT(cm, compilationDescriptor, *tIt, huffmanCompression));
    }
    // TASKS
    graphFile_.task_lists = buildTaskListT(cm, compilationDescriptor);

    // Barrier Table must be build only on dynamic scheduling
    if(globalConfigurationParameters->get<std::string>("barrier_index_assignment") == "Dynamic")
        graphFile_.barrier_table = buildBarrierTable(cm, compilationDescriptor);

}

void mv::RuntimeModel::serialize()
{
    flatbuffers::FlatBufferBuilder fbb;
    auto offset = MVCNN::CreateGraphFile(fbb, &graphFile_);
    MVCNN::FinishGraphFileBuffer(fbb, offset);
    binaryData_ = std::shared_ptr<std::vector<char>>(new std::vector<char>(fbb.GetSize()));
    std::memcpy(binaryData_->data(), (char*)fbb.GetBufferPointer(), binaryData_->size());
}

void mv::RuntimeModel::serialize(const std::string& filename)
{
    serialize();
    if (flatbuffers::SaveFile((filename).c_str(), binaryData_->data(), binaryData_->size(), true))
        Logger::log(mv::Logger::MessageType::Debug, "RuntimeModel", "File successfully written to: " + filename);
    else
        Logger::log(mv::Logger::MessageType::Error, "RuntimeModel", "File was not created. Check configuration");
}

void mv::RuntimeModel::deserialize(const std::string& path)
{
    std::ifstream ifs(path.c_str(), std::ifstream::binary|std::ifstream::in);
    ifs.seekg(0, std::ios::end);
    unsigned length = ifs.tellg();
    ifs.seekg(0, std::ios::beg);

    binaryData_ = std::shared_ptr<std::vector<char>>(new std::vector<char>(length));

    ifs.read(binaryData_->data(), length);
    ifs.close();
    deserialize(binaryData_->data(), binaryData_->size());
}

void mv::RuntimeModel::deserialize(char * dataBuffer, int length)
{
    flatbuffers::Verifier verifier(reinterpret_cast<const unsigned char*>(dataBuffer), length);
    if (!MVCNN::VerifyGraphFileBuffer(verifier))
        throw ArgumentError("tools:GraphComparator", "file:content", "invalid", "GraphFile verification failed");
    Logger::log(mv::Logger::MessageType::Debug, "RuntimeModel", "GraphFile verification successful");
    const MVCNN::GraphFile *graphPtr = MVCNN::GetGraphFile(dataBuffer);
    graphPtr->UnPackTo(&graphFile_);
}

std::shared_ptr<std::vector<char>> mv::RuntimeModel::getBlob()
{
    return binaryData_;
}

std::vector<int64_t> mv::RuntimeModel::huffmanCompress(std::vector<int64_t>& data, mv::Tensor& t) 
{
    std::vector<uint8_t> uncompressedData(data.begin(),data.end());
    uint32_t uncompressedDataSize = uncompressedData.size();
    auto compressedBufferSize = uncompressedDataSize + 2 * (std::ceil(uncompressedDataSize / 4096.0) + 1);

    std::vector<uint8_t> compressedDataBuffer (compressedBufferSize, 0); 
    uint32_t compressedSize = codec_->huffmanCodecCompressArray(uncompressedDataSize, &uncompressedData[0], &compressedDataBuffer[0]);
    vector<uint8_t>::iterator endDataIterator = compressedDataBuffer.begin() + compressedSize;
    compressedDataBuffer.erase(endDataIterator,compressedDataBuffer.end());

    t.set<int>("CompressedSize", compressedSize);
    std::vector<int64_t> toReturn(compressedDataBuffer.begin(),compressedDataBuffer.end());
       
    return toReturn;
}

std::vector<uint8_t> mv::RuntimeModel::huffmanDecompress(std::vector<uint8_t>& compressedData) 
{
    uint32_t comprssedSize = compressedData.size();
    auto deCompressedBufferSize = compressedData.size() * 5;
    std::vector<uint8_t> deCompressedDataBuffer (deCompressedBufferSize, 0); 
    codec_->huffmanCodecDecompressArray(comprssedSize, &compressedData[0], &deCompressedDataBuffer[0]);

    return deCompressedDataBuffer;
}
