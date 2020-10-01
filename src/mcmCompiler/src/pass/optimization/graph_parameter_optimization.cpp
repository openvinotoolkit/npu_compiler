#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/pass/graphOptimizations/StrategyManager.hpp"
#include "include/mcm/utils/custom_math.hpp"
// #include "include/mcm/utils/compression/hde.hpp"
#include "include/mcm/pass/pass_utils.hpp"
#include "mcm/utils/custom_strings.hpp"


static void GraphParameterOptimizationFcn(const mv::pass::PassEntry&,
    mv::ComputationModel& model,
    mv::TargetDescriptor&, mv::Element& passDesc,
    mv::Element&
);

namespace mv
{

    namespace pass
    {

        MV_REGISTER_PASS(GraphParameterOptimization)
                .setFunc(GraphParameterOptimizationFcn)
                .setDescription("Analyzes graph, and tries to come up with optimal schedule");

    }

}

namespace mv
{

    namespace graphOptimizer
    {

        class StrategyManagerKmb : public StrategyManager
        {

        public:
            StrategyManagerKmb(OpModel& model,mv::Element& passDesc) :
                StrategyManager(model,passDesc)
            {
                auto globalParams = model.getGlobalConfigParams();
                enableChannelMajorConv = globalParams->get<bool>("enable_channel_major_conv");
            }

            size_t totalClusters=4;
            size_t clusterMemoryKb=896;
            size_t dpuPerCluster=5;
            std::string referenceDevice = "A0";
            int ddrBandwidth=128;
            int sysClock=500;
            double sysClockHz=sysClock*1000000;
            bool globalEnableStreaming=true;
            bool globalEnablePipelining = true;
            bool globalEnableActivationSparsity=true;
            bool globalForceActivationSparsity=false;
            bool globalEnableWeightsSparsity=false;
            bool globalForceSpilling=false;
            bool enableChannelMajorConv=false;
            double safetyFactor=1.0;
            double clusterMemory=(double)clusterMemoryKb * 1024.0 * safetyFactor;
            enum class FailCause
            {
                Pass,
                cmxConcatDecision,
                MemorySize,
                StreamAndClusterComp,
                SpillHKSwitch,
                SOKNotAlign16,
                InputNotSpilled,
                OutputNotSpilled,
                StreamingNotSpilled,
                WorkloadLessKernelSOH,
                ChannelMjr1,
                ChannelMjr2,
                DWChannels,
                SOHheight,
                RequiresSparsity,
                RealSparseForFakeSparseOp,
                DilatedSOH,
                DWLargeStrideReplacementSOK,
                SpiltOverHWithStreamOverK,
                SparsityKSegmented,
                SparsitySpilling,
                PipelineNotPossible,
                DeConvSubConvSOKHeight,
                UpaHKSwitch,
                Unknown
            };

            std::unordered_map<FailCause, std::string> failure_causes = {
                {FailCause::Pass, "Pass"},
                {FailCause::cmxConcatDecision, "cmxConcatDecision"},
                {FailCause::MemorySize, "MemorySize"},
                {FailCause::StreamAndClusterComp, "Stream+ClusterComp"},
                {FailCause::SpillHKSwitch, "SpillHKSwitch"},
                {FailCause::SOKNotAlign16, "SOKNotAlign16"},
                {FailCause::InputNotSpilled, "InputNotSpilled"},
                {FailCause::OutputNotSpilled, "OutputNotSpilled"},
                {FailCause::StreamingNotSpilled, "StreamingNotSpilled"},
                {FailCause::WorkloadLessKernelSOH, "Workload<KernelSOH"},
                {FailCause::ChannelMjr1, "ChannelMjr1"},
                {FailCause::ChannelMjr2, "ChannelMjr2"},
                {FailCause::DWChannels, "DWChannels"},
                {FailCause::SOHheight, "SOHheight"},
                {FailCause::RequiresSparsity, "RequiresSparsity"},
                {FailCause::RealSparseForFakeSparseOp, "RealSparseForFakeSparseOp"},
                {FailCause::DilatedSOH, "DilatedSOH"},
                {FailCause::DWLargeStrideReplacementSOK, "DWLargeStrideReplacementSOK"},
                {FailCause::SpiltOverHWithStreamOverK, "SpiltOverHWithStreamOverK"},
                {FailCause::SparsityKSegmented, "SparsityKSegmented"},
                {FailCause::SparsitySpilling, "SparsitySpilling"},
                {FailCause::PipelineNotPossible, "PipelinedNotPossible"},
                {FailCause::DeConvSubConvSOKHeight, "DeConvSubConvSOKHeight"},
                {FailCause::UpaHKSwitch, "UpaHKSwitch"},
                {FailCause::Unknown, "Unknown"}
            };

            void readGlobalConfigs()
            {
                referenceDevice = globalConfig_["referenceDevice"].get<std::string>();
                totalClusters = globalConfig_["totalClusters"].get<int>();
                clusterMemoryKb = globalConfig_["clusterMemory"].get<int>();
                dpuPerCluster = globalConfig_["dpuPerCluster"].get<int>();
                ddrBandwidth = globalConfig_["ddrBandwidth"].get<int>();
                sysClock = globalConfig_["systemClockMhz"].get<int>();
                createStrategyDots = globalConfig_["createStrategyDots"].get<bool>();
                dotFileLocation = globalConfig_["dotFileLocation"].get<std::string>();
                jsonOutFileName = globalConfig_["jsonOutFileName"].get<std::string>();
                safetyFactor = globalConfig_["FathomSafetyFactor"].get<double>();
                //Input is in Kb
                clusterMemory = (double)clusterMemoryKb * 1024.0 * safetyFactor;

                globalEnableStreaming = globalStrategies_["enableStreaming"].get<bool>();
                globalEnablePipelining = globalStrategies_["enablePipelining"].get<bool>();
                globalForceActivationSparsity = globalStrategies_["forceActivationSparsity"].get<bool>();
                globalEnableWeightsSparsity = globalStrategies_["enableWeightsSparsity"].get<bool>();
                globalForceSpilling =  globalStrategies_["forceSpilling"].get<bool>();
            }


            std::size_t realTensorSize(const mv::Data::TensorIterator tensorToSize, const mv::Shape& streamingPool, bool isCMConv)
            {
               mv::Shape worstStreamPool = streamingPool;

                //TODO harmonize this, for now only consider worst shape for nested streams
                if(streamingPool["H"] > 1 && streamingPool["K"] > 1)
                {
                    mv::Shape tensorShape = tensorToSize->getShape();
                    //update the streamingPool to the worst combination, based on slice sizes
                    auto outputSize = tensorShape[mv::IO_HEIGHT_DIMENSION];
                    auto numberOfSplits = streamingPool[mv::IO_HEIGHT_DIMENSION];

                    auto newOutputSizes = tileSpatialOutputSize(outputSize, numberOfSplits);
                    int newOutputSize = newOutputSizes.front();

                    auto worstNumberOfSplits = outputSize/newOutputSize;
                    worstStreamPool[mv::IO_HEIGHT_DIMENSION] = worstNumberOfSplits;
                }

                //TODO add handling for weights case if we dont align it to 16 always
                std::size_t streamDivisor = 1;
                for(std::size_t dim = 0; dim <  worstStreamPool.ndims(); ++dim)
                {
                    streamDivisor = streamDivisor * worstStreamPool[dim];
                }

                if(isCMConv)
                    return tensorToSize->computeTotalSize(16, false, false, false)/streamDivisor;

                return tensorToSize->computeTotalSize(16, false, false, true)/streamDivisor;
            }


            std::size_t activationTensorSize(const mv::Data::TensorIterator tensorToSize, std::string clustering, const mv::Shape& streamingPool, bool isCMConv, mv::Op& op, bool dilation = false)
            {
                auto div = [](unsigned x,unsigned y) -> unsigned { return (x+y-1)/y; };
                auto dtypeMultiplier = std::ceil(tensorToSize->getDType().getSizeInBits()/8.0);

                auto tensorShape = tensorToSize->getShape();
                if (dilation)
                    tensorShape = tensorToSize->get<mv::Shape>("originalShape");
                //Note: For now, all batched operations stream over batch so that N = 1
                size_t streamedBatch = 1;

                size_t fullTensorHeight = tensorShape[mv::IO_HEIGHT_DIMENSION];
                size_t streamedHeight = fullTensorHeight;

                size_t fullTensorChannels = tensorShape[mv::IO_CHANNEL_DIMENSION];
                size_t streamedChannels = fullTensorChannels;

                if(streamingPool["H"] > 1)
                {
                    auto newOutputSizes = tileSpatialOutputSize(fullTensorHeight, streamingPool["H"]);
                    streamedHeight = newOutputSizes.front();
                    if(streamedHeight < newOutputSizes.back())
                        streamedHeight = newOutputSizes.back();

                    // Kernel and padding will add extra lines to final size of streamed portion
                    size_t kHeight = 1;
                    std::array<unsigned short, 4> padding;
                    if(  (op.getOpType() == "Conv") || (op.getOpType() == "DepthwiseConv") )
                        kHeight = op.getInputTensor(1)->getShape()[mv::KERNEL_HEIGHT];
                    else if (op.getOpType() == "MaxPool")
                        kHeight = op.get<std::array<unsigned short, 2>>("kSize")[mv::KERNEL_HEIGHT];
                    if (op.hasAttr("padding"))
                        padding = op.get<std::array<unsigned short, 4>>("padding");
                    else
                        padding = {0, 0, 0, 0};

                    int extraLines = 0;

                    if(extraLines < kHeight-1)
                    {
                        extraLines = kHeight -1;
                    }

                    if(padding[2] > padding[3])
                    {
                        if(padding[2] > extraLines)
                            extraLines = padding[2];
                    }
                    else
                    {
                        if(padding[3] > extraLines)
                            extraLines = padding[3];
                    }

                    streamedHeight += extraLines;
                }
                if(streamingPool["C"] > 1)
                {
                    streamedChannels = div(fullTensorChannels,streamingPool["C"]);
                }
                if (streamingPool["K"] > 1)
                {
                    streamedChannels =  div(fullTensorChannels, streamingPool["K"]);

                    int remainderChannels = fullTensorChannels - (streamedChannels*(streamingPool["K"] -1));
                    if (remainderChannels > streamedChannels)
                        streamedChannels = remainderChannels;

                    streamedChannels = mv::round_up(streamedChannels, 16);
                }

                if(clustering == "SplitOverH")
                {
                    streamedHeight = div(streamedHeight,totalClusters);
                }

                return tensorShape[mv::IO_WIDTH_DIMENSION] * streamedHeight * streamedChannels * streamedBatch * dtypeMultiplier;
            }

            size_t alignedWeightsSize(const mv::Data::TensorIterator tensorToSize, const Shape& streamConfig, std::string clustering){
                auto div = [](unsigned x,unsigned y) -> unsigned { return (x+y-1)/y; };
                auto dtypeMultiplier = std::ceil(tensorToSize->getDType().getSizeInBits()/8.0);
                size_t alignedFullInputChannels = mv::round_up(tensorToSize->getShape()[KERNEL_INPUT_CHANNELS], 16);

                size_t alignedFullOutputChannels = mv::round_up(tensorToSize->getShape()[KERNEL_OUTPUT_CHANNELS], 16);
                size_t alignedStreamedOutputChannels = mv::round_up(alignedFullOutputChannels/streamConfig["K"], 16);

                if(clustering == "SplitOverK")
                {
                    //size_t alignedSplittedOutputChannels = ceil(alignedStreamedOutputChannels/totalClusters)
                    size_t alignedSplittedOutputChannels = div(alignedStreamedOutputChannels,totalClusters);
                    alignedSplittedOutputChannels = mv::round_up(alignedSplittedOutputChannels, 16);

                    return (alignedFullInputChannels * alignedSplittedOutputChannels *
                            tensorToSize->getShape()[KERNEL_WIDTH] * tensorToSize->getShape()[KERNEL_HEIGHT])
                            * dtypeMultiplier;
                }
                else{
                    return (alignedFullInputChannels * alignedStreamedOutputChannels *
                            tensorToSize->getShape()[KERNEL_WIDTH] * tensorToSize->getShape()[KERNEL_HEIGHT])
                            * dtypeMultiplier;
                }
            }

            std::tuple<size_t,size_t,size_t> memorySize(mv::Op& op, const Attribute& clustering, bool inputActivationSparsity,
                                            bool outputActivationSparsity, bool weightsSparsity, const Shape& streamConfig, 
                                            bool fakeSparsity, bool spilling = false, bool parentSpilling = true)
            {
                auto div = [](unsigned x,unsigned y) -> unsigned { return (x+y-1)/y; };

                size_t inputSize = 0;
                size_t outputSize = 0;
                size_t weightSize = 0;
                size_t weightTableSize = 0;
                //NOTE: here is done a trick for the sub-dilated convolutions, if you are
                //dilated on your cmx as input is the original shape tensor which is before
                //the input of the slice...
                bool dilatedLayerInputMemory = false;

                auto opType = op.getOpType();
                auto isCMConv = false;
                auto clusterStrategy = clustering.get<std::string>();

                if(enableChannelMajorConv && op.supportsCMConv())
                    isCMConv = true;

                if (op.hasAttr("DilatedSubConv") && (op.get<bool>("DilatedSubConv")))
                    dilatedLayerInputMemory = true;

                if(opType != "Input" && opType != "Concat")
                {
                    // Note: when an operation is streaming activations, but it's parent didn't spill, the input won't be streamed
                    Shape temporaryStreamConfig = {streamConfig["W"],streamConfig["H"],streamConfig["C"],1,streamConfig["B"]};
                    if(!parentSpilling)
                        temporaryStreamConfig = {1,1,1,1,1};
                    inputSize = activationTensorSize(op.getInputTensor(0),clusterStrategy,temporaryStreamConfig, isCMConv, op, dilatedLayerInputMemory);
                }
                if(opType != "Output")
                {
                    //NOTE: when streaming operations are not spilled, full output (not streamed size) must be counted
                    // Similarly, with explicit concats. We don't call this function for ddr concats, only CMX
                    Shape temporaryStreamConfig = {streamConfig["W"],streamConfig["H"],1,streamConfig["K"],streamConfig["B"]};
                    if (!spilling)
                        temporaryStreamConfig = {1,1,1,1,1};

                    outputSize = activationTensorSize(op.getOutputTensor(0),clusterStrategy,temporaryStreamConfig, isCMConv, op);
                }

                auto software = op.hasAttr("softwareExecuted") && op.get<bool>("softwareExecuted");

                size_t outChannels = op.outputSlots() ? op.getOutputTensor(0)->getShape()[IO_CHANNEL_DIMENSION] : 0;
                size_t alignedFullChannels = mv::round_up(outChannels, 16);
                size_t alignedSplittedChannels = mv::round_up(alignedFullChannels/streamConfig["K"], 16);
                if(clusterStrategy == "SplitOverK") {
                    alignedSplittedChannels =  mv::round_up(alignedSplittedChannels/totalClusters, 16);
                }

                if(opType == "Conv" || opType == "DepthwiseConv")
                {
                    weightTableSize = 16 * alignedSplittedChannels;
                    if (opType == "Conv")
                    {
                        weightSize += alignedWeightsSize(op.getInputTensor(1),{1,1,1,streamConfig["K"],1}, clusterStrategy);
                    }
                    else
                    {
                        weightSize += realTensorSize(op.getInputTensor(1),{1,1,streamConfig["C"],1,1}, isCMConv);
                        if(clusterStrategy == "SplitOverK")
                            weightSize = div(weightSize,totalClusters);
                    }

                }
                else if(opType == "MaxPool")
                {
                    weightTableSize = 16 * alignedSplittedChannels;
                    weightSize = 0;
                }
                else if(opType == "Eltwise" && !software)
                {
                    weightTableSize = 0;
                    weightSize = 0; 
                    inputSize += activationTensorSize(op.getInputTensor(1),clusterStrategy,{streamConfig["W"],streamConfig["H"],streamConfig["C"],1,streamConfig["B"]}, isCMConv, op);
                }

                //Additional memory footprint for sparsity
                if(fakeSparsity)
                {
                    if (opType != "MaxPool" && opType != "DepthwiseConv" && !isCMConv)
                    {
                        throw LogicError(*this, op.getName() + ": Invalid fake Sparsity! Has to be only for MaxPool, DW or CMConv!! opType is " + opType);
                    }
                    uint16_t kernelW, kernelH;


                    auto strides = op.get<std::array<unsigned short, 2>>("stride");

                    if (op.hasAttr("kSize"))
                    {
                        auto kernelShape = op.get<std::array<unsigned short, 2>>("kSize");
                        kernelW = kernelShape[0];
                        kernelH = kernelShape[1];
                    }
                    else
                    {
                        auto weightsShape = op.getInputTensor(1)->getShape();
                        kernelW = weightsShape[mv::KERNEL_WIDTH];
                        kernelH = weightsShape[mv::KERNEL_HEIGHT];
                    }

                    mv::DType dataType =op.getInputTensor(0)->get<mv::DType>("dType");
                    if (opType != "MaxPool")
                        dataType = op.getInputTensor(1)->get<mv::DType>("dType");

                    auto windowsSize = getWindowSize(kernelW, strides[0], dataType);
                    size_t fakeSparsitySize = 0;
                    if ((opType == "MaxPool") || (opType == "DepthwiseConv"))
                    {
                        //inputChannels = 1
                        auto bitpatternSize = windowsSize*kernelH;
                        //ndims = {16 * static_cast<std::size_t>(std::ceil(bitpatternSize / 128.0)), 1, 1, 1};
                        fakeSparsitySize = 16 * static_cast<std::size_t>(std::ceil(bitpatternSize / 128.0));
                    }
                    // Channel Major Convolution doesn't need rounding of channels
                    else if (isCMConv)//isChannelMajorConvolution
                    {

                        std::size_t outputChannels =  op.getOutputTensor(0)->getShape()[IO_CHANNEL_DIMENSION];
                        outputChannels = outputChannels/streamConfig["K"];
                        std::size_t inputChannels = op.getInputTensor(0)->getShape()[IO_CHANNEL_DIMENSION];

                        auto windowSparsitySize = static_cast<std::size_t>(std::ceil(windowsSize/8.0)); //how many bytes we need per window
                        auto NumberOfRowsSparistyBytes = static_cast<std::size_t>(std::ceil((kernelH * inputChannels * windowSparsitySize) / 16.0 ));

                        //ndims = {16, NumberOfRowsSparistyBytes, 1, outputChannels};
                        fakeSparsitySize = 16*NumberOfRowsSparistyBytes*outputChannels;

                    }
                    inputSize += fakeSparsitySize;
                }
                if(inputActivationSparsity){
                    //Alignment due to input channels mult of 16 requirement
                    //Only ZM Conv and Elwise are sparse consumers, both need
                    //input channels mult of 16
                    auto tensorSize = op.getInputTensor(0)->computeTotalSize(16, false, false, true);
                    size_t streamDivisor = streamConfig["W"] * streamConfig["H"] * streamConfig["C"];
                    //Sparsity map calculation, mostly dtype invariant (except for sub 8 bit)
                    auto sparseInputSize = std::ceil((double)tensorSize /
                        (8 * op.getInputTensor(0)->getDType().getSizeInBytes()));
                    //Storage element table calculation, 4 bytes pointers
                    //Bigger with C streaming
                    sparseInputSize += op.getInputTensor(0)->getShape()[IO_WIDTH_DIMENSION] *
                        op.getInputTensor(0)->getShape()[IO_HEIGHT_DIMENSION] *
                        streamConfig["C"] * 4;
                    //Alignment due to bus access requirements
                    sparseInputSize = mv::round_up(sparseInputSize, 16);
                    inputSize += (sparseInputSize / streamDivisor);
                }
                if(outputActivationSparsity){
                    //Alignment due to output channels mult of 16 requirement
                    //Only ZM Conv and Elwise are sparse consumers
                    auto tensorSize = op.getOutputTensor(0)->computeTotalSize(16, false, false, true);
                    size_t streamDivisor = streamConfig["W"] * streamConfig["H"] * streamConfig["K"];
                    //Sparsity map calculation, mostly dtype invariant (except for sub 8 bit)
                    auto sparseOutputSize = std::ceil((double)tensorSize /
                        (8 * op.getOutputTensor(0)->getDType().getSizeInBytes()));
                    //Storage element table calculation, 4 bytes pointers
                    //Bigger with K streaming
                    sparseOutputSize += op.getOutputTensor(0)->getShape()[IO_WIDTH_DIMENSION] *
                        op.getOutputTensor(0)->getShape()[IO_HEIGHT_DIMENSION] *
                        streamConfig["K"] * 4;
                    //Alignment due to bus access requirements
                    sparseOutputSize = mv::round_up(sparseOutputSize, 16);
                    outputSize += (sparseOutputSize / streamDivisor);
                }
                if(weightsSparsity){
                    //Alignment due to output/input channels mult of 16 requirement
                    auto tensorSize = op.getInputTensor(1)->getShape()[KERNEL_WIDTH] *
                        op.getInputTensor(1)->getShape()[KERNEL_HEIGHT] *
                        mv::round_up(op.getInputTensor(1)->getShape()[KERNEL_INPUT_CHANNELS], 16) *
                        alignedSplittedChannels;
                    //Sparsity map calculation, mostly dtype invariant (except for sub 8 bit)
                    auto sparseWeightSize = std::ceil((double)tensorSize / 8);
                    //Sparse pointers taken into account in weight table ...
                    sparseWeightSize = mv::round_up(sparseWeightSize, 16);
                    weightSize += sparseWeightSize;
                }

                weightSize += weightTableSize;

                // Note: for SOH and SOK, division by number of clusters is done in activationTensorSize
                // and alignedWeightsSize, respectively. This allows greater precision than dividing
                // totalClusters. Multiclustering doesn't perfectly split tensor, depends on subtensor size!
                if(clusterStrategy == "SplitOverHOverlapped")
                {
                    inputSize = div(inputSize,totalClusters);
                    outputSize = div(outputSize,totalClusters);
                }

                return std::tuple<std::size_t,std::size_t,std::size_t>(inputSize, outputSize,weightSize);
            }

                        bool isPipeliningPossible(mv::Op& op, StrategySet& strategy, bool parentSpilling)
            {
                if(!globalEnablePipelining)
                    return false;

                // Is this op type enabled for pipelining? For now, default turned on for just conv and dw
                if(!createStrategyFromBool(op, "pipelining"))
                    return false;

                auto stream = strategy["streaming"].get<Shape>();
                auto clustering = strategy["clustering"].get<std::string>();
                auto inputSparsity = strategy["inputSparsity"].get<bool>();
                auto outputSparsity = strategy["outputSparsity"].get<bool>();
                auto weightsSparsity = strategy["weightsSparsity"].get<bool>();
                auto spilling = strategy["spilling"].get<bool>();
                
                //Note: for now, only support pipelining over H and K
                if((stream["N"] * stream["C"]) > 1)
                    return false;

                size_t input, output, weights;
                std::tie(input, output, weights) = memorySize(op,
                                                                clustering,
                                                                inputSparsity,
                                                                outputSparsity,
                                                                weightsSparsity,
                                                                stream,
                                                                requiresFakeActivationSparsity(op),
                                                                spilling,
                                                                parentSpilling);

                if(stream["K"] > 1) // Full activation in CMX, stream weights
                {
                    std::cout << "Considering pipelining for layer " << std::endl;
                    std::cout << "Name: " + op.getName() << std::endl;
                    std::cout << "MCStrategy: " + clustering<< std::endl;
                    std::cout << "Streaming(W,H,C,K,N): " + stream.toString() << std::endl<<std::endl;
                    auto memReq = input + output + 2*weights;
                    if(memReq < clusterMemory)
                    {   
                        std::cout << "Pipelining Possible !" <<std::endl<<std::endl;

                        return true;
                    }
                }
                else if(stream["H"] > 1)
                {
                    if(!spilling && !parentSpilling) //Activations all in CMX, nothing to pipeline
                        return false;

                    double memReq = 0;

                    //Note: memory size function is smart enough to take care of input/output size relative to spilling
                    if(parentSpilling) //streamed input, either full or streamed output
                        memReq = 2*input + weights + output; 
                    else //full input, streamed output
                        memReq = input + weights + 2*output;

                    if(memReq < clusterMemory)
                    {
                        return true;
                    }
                }
                return false; // without streaming, there is no pipelining
            }

            double dmaTime(Op& op,StrategySet& strategySet, bool parentSpilling = false)
            {
                auto opType = op.getOpType();

                double DMA_LATENCY = 36 / sysClockHz; // in c -> s
                double DMA_BANDWIDTH = 25769803776; // gb/s - > b/s

                //Each layer calculates the cost to move it's weights (if they exist), and it's output tensor
                double weightsTime = 0;
                double outputTime = 0;
                //Child will calculate cost for input tensor
                double inputTime = 0;

                auto streamShape = strategySet["streaming"].get<mv::Shape>();
                auto spilling = strategySet["spilling"].get<bool>();

                // Each DMA cost is modelled as latency + size*transfer_rate
                if(opType == "Conv" || opType == "DepthwiseConv")
                {
                    auto weightsSize = op.getInputTensor(1)->computeTotalSize(); // approx for now
                    unsigned stream = 1;

                    if(opType == "Conv")
                    {
                        stream = streamShape["K"];
                    }
                    else // Depthwise
                    {
                        stream = streamShape["C"];
                    }

                    weightsTime = stream*(DMA_LATENCY + (((double)weightsSize/stream) / DMA_BANDWIDTH));
                }

                // If parent tensor stays in CMX, no further cost. Otherwise, calculate bring the input tensor back into CMX
                if(parentSpilling)
                {
                    auto inputSize = op.getInputTensor(0)->computeTotalSize();
                    unsigned stream = 1;
                    if(streamShape["H"] > 1)
                        stream = streamShape["H"];
                    else if(streamShape["C"] > 1)
                        stream = streamShape["C"];

                    inputTime = stream* (DMA_LATENCY + (((double)inputSize/stream) / DMA_BANDWIDTH) );
                }

                // If output tensor stays in CMX, no further dma cost. Otherwise, calculate cost to spill to DDR
                if(spilling)
                {
                    //Each stream is written to DDR. Output activation tensor might be streamed over H or K
                    //TODO consider batch, nested streaming
                    size_t outputSize;
                    if(opType != "Output")
                        outputSize = op.getOutputTensor(0)->computeTotalSize();
                    else
                        outputSize = op.getInputTensor(0)->computeTotalSize();

                    unsigned stream = 1;

                    if(streamShape["H"] > 1)
                        stream = streamShape["H"];
                    else if(streamShape["K"] > 1)
                        stream = streamShape["K"];
                    
                    outputTime = stream*(DMA_LATENCY + (((double)outputSize/stream) / DMA_BANDWIDTH));
                }
                return (inputTime + weightsTime + outputTime) * 1000000; //return in us
            }

            double computeTime(Op& op,StrategySet& strategySet)
            {
                auto opType = op.getOpType();
                auto software = op.hasAttr("softwareExecuted") && op.get<bool>("softwareExecuted");
                if( !op.isHardwarizable() || software)
                    return 0;

                double LATENCY_CMX = 5 / sysClockHz; // in c -> s
                double BANDWIDTH_CMX = (double)(1.5 * 1099511627776); // tb/s -> b/s
                double OPS = 7.168 * 1099511627776; //tops -> o/s
                if(op.getInputTensor(0)->get<mv::DType>("dType") == mv::DType("Float16"))
                    OPS = 1.792 * 1099511627776;

                auto outputShape = op.getOutputTensor(0)->getShape();
                auto clustering = strategySet["clustering"].get<std::string>();
                auto streaming = strategySet["streaming"].get<Shape>();

                unsigned baseKernelCost;

                if ((opType == "Eltwise" && !(software)) || (opType == "Concat"))
                {
                    baseKernelCost = 1;
                }
                else if (opType == "MaxPool")
                {
                    auto kernel = op.get<std::array<unsigned short,2>>("kSize");
                    baseKernelCost = kernel[0] * kernel[1];
                }
                else if ((opType == "DepthwiseConv") || (opType == "Conv"))
                {
                    auto weightsShape = op.getInputTensor(1)->getShape();
                    baseKernelCost = weightsShape[KERNEL_WIDTH] * weightsShape[KERNEL_HEIGHT];
                }
                else if (!(op.hasTypeTrait("optimizable")) || software)
                {
                    baseKernelCost = 1;
                }
                else
                {
                    throw LogicError(*this,"Invalid operation type " + opType);
                }

                bool channelAccum =  (opType == "Conv") ? true : false;
                if(channelAccum)
                {
                    auto weightsShape = op.getInputTensor(1)->getShape();
                    baseKernelCost *= weightsShape[KERNEL_INPUT_CHANNELS];
                }

                auto totalStreams = 1;
                for(unsigned i = 0; i < streaming.ndims(); i++)
                    totalStreams *= streaming[i];

                auto isiDecay = 1.0;
                if (clustering == "SplitOverK" || clustering == "HKSwitch")
                    isiDecay = 0.1  *
                        std::max(1lu, dpuPerCluster - 1) *
                        std::max(1lu, totalClusters - 1);

                //TODO to capture fully, should calculate the cost to bring data to compute from cmx and output back to cmx
                //double readIn = (totalStreams * LATENCY_CMX) +  
                auto totalToCompute = (outputShape.totalSize() / totalStreams);

                // Multiclustering allows parallelism
                if(clustering != "Clustering")
                    totalToCompute = totalToCompute / totalClusters;

                totalToCompute = totalToCompute * isiDecay;
                
                double compTime = ((totalToCompute * baseKernelCost) / OPS);

                return  (totalStreams * compTime) * 1000000; //return in us
            }


            bool requiresActivationSparsity(Op& op, std::string clustering)
            {
                if(requiresRealActivationSparsity(op, clustering))
                    return true;

                if(requiresCompilerActivationSparsity(op))
                    return true;

                return false;
            }

            bool requiresWeightsSparsity(Op& op)
            {
                // If Z-major Conv in Float precision then need to have weights Sparsity
                bool isCMConv = enableChannelMajorConv && op.supportsCMConv();

                if(op.getOpType() == "Conv" &&
                    op.getInputTensor(0)->get<mv::DType>("dType") == mv::DType("Float16") &&
                    !isCMConv && referenceDevice == "A0")
                        return true;

                return false;
            }

            // In these cases parent output sparsity does matter, but child input sparsity must be true
            bool requiresCompilerActivationSparsity(Op& op)
            {
                bool isCMConv = enableChannelMajorConv && op.supportsCMConv();

                if (op.getOpType() == "Conv" && !isCMConv
                        && (op.hasAttr("DilatedSubConv") && op.get<bool>("DilatedSubConv")))
                    return true;

                return false;
            }

            bool requiresRealActivationSparsity(Op& op, std::string clustering){
                //An fp16 Conv Z-major must have activation sparsity
                bool isCMConv = enableChannelMajorConv && op.supportsCMConv();

                if (op.isSparsityConsumer() &&
                    op.getInputTensor(0)->get<mv::DType>("dType") == mv::DType("Float16") &&
                    !isCMConv &&
                    referenceDevice == "A0")
                {
                    return true;
                }


                // Check for need for A0 SOH Sparsity workaround, (SOH conv with kernel > 1)
                // if needed, check memory constraints as for sparse tensor
                if ( op.getOpType() == "Conv" ) {
                    if( clustering == "SplitOverH" &&
                        (op.getInputTensor(1)->getShape()[KERNEL_HEIGHT] > 1 ||
                        op.getInputTensor(1)->getShape()[KERNEL_WIDTH]  > 1) &&
                        !isCMConv &&
                        referenceDevice == "A0")
                        {
                            return true;
                        }
                }

                return false;
            }

             //Channel major conv, pooling and depthwise will get fake sparsity, so need to check memory constraints as if real sparsity
            bool requiresFakeActivationSparsity(Op& op){
                if(enableChannelMajorConv && op.supportsCMConv())
                    return true;

                if(op.getOpType() == "MaxPool")
                    return true;

                if(op.getOpType() == "DepthwiseConv")
                    return true;

                return false;
            }

            int8_t checkInOutSizes(mv::Op& op, size_t input_gate)
            {
                int8_t executableInHW = 0;
                if (op.getInputTensor(input_gate)->getShape()[mv::IO_WIDTH_DIMENSION] > mv::MAX_DIM_SIZE ||
                    op.getInputTensor(input_gate)->getShape()[mv::IO_HEIGHT_DIMENSION] > mv::MAX_DIM_SIZE ||
                    op.getInputTensor(input_gate)->getShape()[mv::IO_CHANNEL_DIMENSION] > mv::MAX_DIM_SIZE ||

                    op.getOutputTensor(input_gate)->getShape()[mv::IO_WIDTH_DIMENSION] > mv::MAX_DIM_SIZE ||
                    op.getOutputTensor(input_gate)->getShape()[mv::IO_HEIGHT_DIMENSION] > mv::MAX_DIM_SIZE ||
                    op.getOutputTensor(input_gate)->getShape()[mv::IO_CHANNEL_DIMENSION] > mv::MAX_DIM_SIZE )
                        executableInHW = 1;
                return executableInHW;
            }

            int8_t checkKernelSizes(mv::Op& op)
            {
                int8_t executableInHW = 0;
                std::array<unsigned short, 4> kernel = {1,1,1,1};//for non conv IN OUT CHANNEL dims = 1
                if (op.hasAttr("kSize"))
                {
                    if (op.getOpType() == "MaxPool" || op.getOpType() == "Eltwise")
                    {
                        kernel[mv::KERNEL_WIDTH] = op.get<std::array<unsigned short, 2>>("kSize")[mv::KERNEL_WIDTH];
                        kernel[mv::KERNEL_HEIGHT] = op.get<std::array<unsigned short, 2>>("kSize")[mv::KERNEL_HEIGHT];
                    }
                    else if (op.getOpType() == "Conv" || op.getOpType() == "DepthwiseConv")
                    {
                        kernel[mv::KERNEL_WIDTH] = op.getInputTensor(1)->getShape()[mv::IO_WIDTH_DIMENSION];
                        kernel[mv::KERNEL_HEIGHT] = op.getInputTensor(1)->getShape()[mv::IO_HEIGHT_DIMENSION];
                        kernel[mv::KERNEL_INPUT_CHANNELS] = op.getInputTensor(1)->getShape()[mv::KERNEL_INPUT_CHANNELS];
                        kernel[mv::KERNEL_OUTPUT_CHANNELS] = op.getInputTensor(1)->getShape()[mv::KERNEL_OUTPUT_CHANNELS];
                    }
                }
                if (kernel[mv::KERNEL_WIDTH] > mv::MAX_KERNEL ||
                     kernel[mv::KERNEL_HEIGHT] > mv::MAX_KERNEL ||
                     kernel[mv::KERNEL_INPUT_CHANNELS] > mv::MAX_DIM_SIZE ||
                     kernel[mv::KERNEL_OUTPUT_CHANNELS] > mv::MAX_DIM_SIZE  )
                    executableInHW = 3;

                if (kernel[mv::KERNEL_WIDTH] != kernel[mv::KERNEL_HEIGHT])
                    log(mv::Logger::MessageType::Debug, op.getName() + "has asymmetric kernel sizes" + \
                                                " k_w" + std::to_string(kernel[mv::KERNEL_WIDTH]) + \
                                                " k_h" + std::to_string(kernel[mv::KERNEL_HEIGHT]));
                return executableInHW;
            }

            int8_t checkStrideSizes(mv::Op& op)
            {
                int8_t executableInHW = 0;
                std::array<unsigned short, 2> stride = {1,1};
                if (op.hasAttr("stride"))
                    stride = op.getAttrs().at("stride").get<std::array<unsigned short, 2>>();
                if (stride[mv::STRIDE_WIDTH] > mv::MAX_STRIDE || stride[mv::STRIDE_HEIGHT] > mv::MAX_STRIDE)
                    executableInHW += 3;

                if (stride[mv::STRIDE_WIDTH] != stride[mv::STRIDE_HEIGHT])
                    log(mv::Logger::MessageType::Debug, op.getName() + "has asymmetric strides" + \
                                                    " s_w" + std::to_string(stride[mv::STRIDE_WIDTH]) + \
                                                    " s_h" + std::to_string(stride[mv::STRIDE_HEIGHT]));
                return executableInHW;
            }

            int8_t checkHWUnsupportedOp(mv::Op& op)
            {
                int8_t executableInHW = 0;
                if (op.isHardwarizable())
                {
                    for (std::size_t input_gates = 0; input_gates < op.getInputTensor().size(); input_gates++)
                    {
                        if (input_gates == 0)
                        {
                            executableInHW += checkInOutSizes(op, input_gates);
                            executableInHW += checkKernelSizes(op); //Note: all the ops have maximum a second input (weights) at G.O stage
                            executableInHW += checkStrideSizes(op);
                        }
                        else if (input_gates == 1 && op.getOpType() == "Eltwise")
                        {
                            if (op.getInputTensor(input_gates)->getShape()[mv::IO_WIDTH_DIMENSION] > mv::MAX_DIM_SIZE ||
                                op.getInputTensor(input_gates)->getShape()[mv::IO_HEIGHT_DIMENSION] > mv::MAX_DIM_SIZE ||
                                op.getInputTensor(input_gates)->getShape()[mv::IO_CHANNEL_DIMENSION] > mv::MAX_DIM_SIZE)
                                executableInHW += 1;
                        }
                    }
                }
                return executableInHW;
            }

            bool opInCMX(mv::Op& op, StrategySet& strategy)
            {
                auto spilling = strategy["spilling"].get<bool>();

                auto opType = op.getOpType();
                if(opType == "Input" || opType == "Output")
                    return false;
    
                if(!op.hasTypeTrait("optimizable"))
                    return false;
   
                if(op.hasAttr("softwareExecuted") && op.get<bool>("softwareExecuted"))
                    return false;
  
                if(opType == "Concat" && spilling)
                    return false;
  
                return true;
            }

            //Check to see if a given stategy is internally consistent for performance
            //Strategies that can only have infinite edges because they are illegal should never be added to the graph
            // Note: IF ADDING A NEW FAILURE CASE, must add new description to failure_causes
            FailCause validateStrategy(mv::Op& op,StrategySet& strategy)
            {
                auto clustering = strategy["clustering"].get<std::string>();
                auto weightsSparsity = strategy["weightsSparsity"].get<bool>();
                auto streamShape = strategy["streaming"].get<Shape>();
                auto spilling = strategy["spilling"].get<bool>();
                auto software = op.hasAttr("softwareExecuted") && op.get<bool>("softwareExecuted");

                //NOTE: funny part you can spill even if you are not streaming, fasten your seatbelts!!
                bool isStreaming = ((streamShape["W"] * streamShape["H"] * streamShape["C"] 
                                                            * streamShape["K"] * streamShape["B"]) > 1) ? true : false;

                // A proper decision on CMX concat for explicit concat or eltwise streaming cannot
                // be made with the information on hand. Will not optimize strategies for these.
                // In store graph optimizer decisions pass, we will mark those that can fit in CMX
                // as CMX-able.
                if((op.getOpType() == "Concat" && !spilling) || 
                    (op.getOpType() == "Eltwise"  && isStreaming && !spilling))
                    return FailCause::cmxConcatDecision;


                if(opInCMX(op, strategy))
                {
                    size_t input, output, weights;
                    std::tie(input, output, weights) = memorySize(op,
                                                                    clustering,
                                                                    strategy["inputSparsity"],
                                                                    strategy["outputSparsity"],
                                                                    weightsSparsity,
                                                                    streamShape,
                                                                    requiresFakeActivationSparsity(op),
                                                                    spilling);
                    if (input + output + weights >= clusterMemory)
                        return FailCause::MemorySize;


                    // To do a CMX concat, the channels must be aligned to 16
                    if (!spilling && isStreaming)
                    {
                        if (op.getOutputTensor()[0]->getShape()[mv::IO_CHANNEL_DIMENSION]%16 != 0)
                        {
                            return FailCause::cmxConcatDecision;
                        }
                        if (op.getInputTensor()[0]->getShape()[mv::IO_CHANNEL_DIMENSION]%16 != 0 &&
                                op.getInputTensor()[0]->getShape()[mv::IO_CHANNEL_DIMENSION] > 16)
                        {
                            return FailCause::cmxConcatDecision;
                        }
                    }
                }

                auto isChanMajor = enableChannelMajorConv && op.supportsCMConv();

                //If spilling, HKSwitch makes no sense
                if( (spilling) && (clustering == "HKSwitch"))
                    return FailCause::SpillHKSwitch;

                if( op.getOpType() == "Conv" || op.getOpType() == "DepthwiseConv")
                {
                    auto weightsShape = op.getInputTensor(1)->getShape();
                    auto numInChannels = weightsShape[KERNEL_INPUT_CHANNELS];
                    auto numOutChannels = weightsShape[KERNEL_OUTPUT_CHANNELS];
                    if (op.getOpType() == "Conv")
                    {
                        if((clustering == "SplitOverK") && (numOutChannels/(streamShape["K"] * totalClusters) < 16))
                            return FailCause::SOKNotAlign16;
                    }
                    else
                    {
                        if((clustering == "SplitOverK") && (numInChannels/(streamShape["C"] * totalClusters) < 16))
                            return FailCause::SOKNotAlign16;
                    }
                    if(clustering == "SplitOverH")
                    {
                        // TODO should we use padding here too?
                        //Try to guess subtensor height, and avoid situations where kernel is bigger than last workload dimension
                        auto outputHeight = op.getOutputTensor(0)->getShape()[IO_HEIGHT_DIMENSION];
                        auto workloadHeight = ceil((double)outputHeight / (double)(totalClusters * streamShape["H"]));
                        if(totalClusters > 1) //last
                            workloadHeight = outputHeight - (workloadHeight * (totalClusters-1)); //get remaining height
                        if(workloadHeight < weightsShape[KERNEL_HEIGHT])
                            return FailCause::WorkloadLessKernelSOH;
                    }
                }

                //Input and Output must have Spilled==True
                if((op.getOpType() == "Input") && (!spilling))
                    return FailCause::InputNotSpilled;

                if((op.getOpType() == "Output") && (!spilling))
                    return FailCause::OutputNotSpilled;

                //Special rules for Channel Major Convolutions
                //No need for SOHOverlapped input unless using channel major
                if(!enableChannelMajorConv && clustering == "SplitOverHOverlapped")
                    return FailCause::ChannelMjr1;

//                if(isChanMajor && clustering == "SplitOverH" && streamShape["H"] > 1)
//                    return FailCause::ChannelMjr2;

                if(isChanMajor && (strategy["inputSparsity"].get<bool>() || strategy["weightsSparsity"].get<bool>()))
                    return FailCause::ChannelMjr2;

                //Guide early on the proposal of a valid strategy
                if (op.getOpType() == "DepthwiseConv")
                {
                    if ((op.getInputTensor(0)->getShape()[mv::IO_CHANNEL_DIMENSION] > 8192)
                            && (streamShape["C"] == 1))
                        return FailCause::DWChannels;
                }

                //For every dpuTask if we splitOverH, workloads are over H dimension, so they need to have at
                //least one line to be assigned with
                if (op.isHardwarizable() && clustering == "SplitOverH")
                {
                    auto outputHeight = op.getOutputTensor(0)->getShape()[IO_HEIGHT_DIMENSION];
                    auto estimatedClusterH = (int)floor((double)outputHeight/totalClusters);
                    if (estimatedClusterH < dpuPerCluster || (outputHeight - (totalClusters - 1) * estimatedClusterH) < dpuPerCluster)
                        return FailCause::SOHheight;
                }

                // TODO: future task tackle SE table allocation to
                // to enable sparse channel segmented ops
                // Additionally each K tile will HAVE to be a power of 2 (IDU restriction)
                // which will create a feedback loop on the K stream rountine
                if(strategy["outputSparsity"].get<bool>() &&
                    (clustering == "SplitOverK" ||
                    clustering == "HKSwitch" ||
                    streamShape["K"] > 1))
                    return FailCause::SparsityKSegmented;

                // TODO: much more harder restriction to overcome
                // joint compiler and runtime coordination needed
                if(strategy["outputSparsity"].get<bool>() &&
                    spilling)
                    return FailCause::SparsitySpilling;

                // TODO: sparsity should be okay with streaming as long as not k segmented!
                if(strategy["outputSparsity"].get<bool>() &&
                    isStreaming)
                    return FailCause::SparsitySpilling;

                if(requiresFakeActivationSparsity(op) && strategy["inputSparsity"].get<bool>())
                    return FailCause::RealSparseForFakeSparseOp;

                if (op.getOpType() == "DepthwiseConv" && op.hasAttr("DWWithReplaceLargeStrides")
                        && op.get<bool>("DWWithReplaceLargeStrides") && (clustering == "SplitOverK"))
                    return FailCause::DWLargeStrideReplacementSOK;

                //NOTE: Subdilation storage element population is not implemented for the SOH case
                if (op.getOpType() == "Conv"  && op.hasAttr("DilatedSubConv")
                        && op.get<bool>("DilatedSubConv")
                        && clustering == "SplitOverH")
                    return FailCause::DilatedSOH;

                if (op.getOpType() == "Conv"  && op.hasAttr("DilatedSubConv")
                        && op.get<bool>("DilatedSubConv")
                        && !spilling)
                    return FailCause::DilatedSOH;

                if (clustering == "SplitOverH" && op.getOpType() == "Conv" && !isChanMajor &&
                    (streamShape["K"]  * streamShape["H"]) > 1 && spilling)
                    return FailCause::SpiltOverHWithStreamOverK;
                //Unet non-DepthwiseDeConv subConv, avoiding splits < # of clusters, to avoid indeterministic outputs on back to back runs
                if(op.getOpType() == "Conv" && op.hasAttr("DeconvSubConv") && clustering == "SplitOverK")
                {
                    auto originalH = op.getOutputTensor(0)->getShape()[IO_HEIGHT_DIMENSION];
                    auto numberOfStreamSplits = streamShape["H"];
                    if ((originalH % streamShape["H"]) != 0)
                    {
                        auto newOutputSizes = tileSpatialOutputSize(originalH, numberOfStreamSplits);
                        int newOutputSize = newOutputSizes.front();
                        int remainderOutputSize = newOutputSizes.back();

                        if (remainderOutputSize < 4)
                        {
                            return FailCause::DeConvSubConvSOKHeight;
                        }
                    }
                }

                if(op.getOpType() == "Eltwise" && op.hasAttr("upaInputs") && clustering == "HKSwitch")
                {
                    return FailCause::UpaHKSwitch;
                }

                return FailCause::Pass; //good strategy
            }
            // CM Conv needs if follows a DW Conv (like OV models) or CM Conv follows another Conv, spilling is needed
            // As OV models have DW->Conv and DW is always 1x1, as long as DW height & width are multiples are 8, no need to spill
            // Reason: CM Conv uses overlaps of rows and so the output subtensors of CM Conv may not be aligned to 8 as needed for next op if conv (and has SOH)
            bool needForceSpillingForCM(Op& parentOp, Op& childOp, std::string& parentClustering, std::string& childClustering)
            {
                bool forceSpill = false;
                if (parentOp.getOpType() == "DepthwiseConv" && childOp.getOpType() == "Conv" && childOp.getInputTensor(1)->getShape()[mv::KERNEL_INPUT_CHANNELS] < 16)
                    if (childClustering == "SplitOverH")
                            forceSpill = true;
                // sorry for a back hack. Spilling needed for TFLite inceptionv3 after CM Conv. But that seems to fail Facenet.
                // so conditon added so that facenet CM Conv doesn't spill its output
                if (parentOp.getOpType() == "Conv" && childOp.getOpType() == "Conv")
                    if (parentClustering == "SplitOverH" && parentOp.getInputTensor(0)->getShape()[mv::IO_HEIGHT_DIMENSION] != 160)
                            forceSpill = true;
                return forceSpill;
            }

            bool violatesClusteringStrategyRules(Op& parentOp,Op& childOp,StrategySet& parent,StrategySet& child)
            {
                std::vector<std::pair<std::string, std::string>>incompatibleStrategiesWithOutSpilling =
                {
                    {"SplitOverHOverlapped", "Clustering"},
                    {"SplitOverHOverlapped", "SplitOverK"},
                    {"SplitOverH", "Clustering"},
                    {"SplitOverH", "SplitOverK"},
                    {"SplitOverK", "SplitOverH"},
                    {"SplitOverK", "HKSwitch"},
                    {"Clustering", "SplitOverH"},
                    {"Clustering", "HKSwitch"},
                    {"HKSwitch", "SplitOverH"},
                    {"HKSwitch", "HKSwitch"}
                };

                // Note: capturing rules from master, TODO revisit
                std::vector<std::pair<std::string, std::string>>incompatibleStrategiesChildConcat =
                {
                    // Parent op, child concat
                    {"SplitOverK", "SplitOverH"},
                    {"HKSwitch", "SplitOverH"},
                    {"Clustering", "SplitOverH"},
                    {"SplitOverH", "SplitOverK"},
                    {"SplitOverH", "Clustering"} //clustering concats follow same rule as SOK concats
                };

                std::vector<std::pair<std::string, std::string>>incompatibleStrategiesParentConcat =
                {
                    // parent concat, child op
                    {"SplitOverH", "SplitOverK"},
                    {"SplitOverK", "SplitOverH"},
                    {"SplitOverK", "HKSwitch"},
                    {"Clustering", "SplitOverH"},//clustering concats follow same rule as SOK concats
                    {"Clustering", "HKSwitch"}
                };

                auto parentClustering = parent["clustering"].get<std::string>();
                auto childClustering = child["clustering"].get<std::string>();
                auto parentOpType = parentOp.getOpType();
                auto childOpType = childOp.getOpType();
                auto parentSpilling = parent["spilling"].get<bool>();
                auto childSpilling = child["spilling"].get<bool>();


                if (parentOpType == "Concat" && childOpType == "Concat")
                {
                    //NOTE: it is not possible to have a parent that is spilling, going to ddr and
                    //the child not spilling there is no concatenation through ddr->cmx concat
                    //when the child is an explicit concat, when it is 2 streaming ops that could happen
                    if (parentSpilling && !childSpilling)
                        return true;


                    //TODO should probably enforce they have same strategy here too..
                }
                else if (parentOpType == "Concat")
                {
                    if (childOpType != "Output")
                    {
                        std::pair<std::string, std::string> possibleCombination(parentClustering, childClustering);
                        for (auto restrictedCombination: incompatibleStrategiesParentConcat)
                        {
                            if (possibleCombination == restrictedCombination)
                            {
                                return true;
                            }
                        }
                    }
                    if(!parentSpilling)
                    {
                        //NOTE: It is impossible to concatenate inside cmx if the input tensors are not aligned // 16
                        for (auto inputTensor : parentOp.getInputTensor())
                        {
                            if (inputTensor->getShape()[mv::IO_CHANNEL_DIMENSION] % 16)
                                return true;
                        }
                    }
                }
                else if (childOpType == "Concat")
                {
                    std::pair<std::string, std::string> possibleCombination(parentClustering, childClustering);
                    for (auto restrictedCombination: incompatibleStrategiesChildConcat)
                    {
                        if (possibleCombination == restrictedCombination)
                        {
                            return true;
                        }
                    }
                    if(!childSpilling)
                    {
                        //NOTE: It is impossible to concatenate inside cmx if the input tensors are not aligned // 16
                        for (auto inputTensor : childOp.getInputTensor())
                        {
                            if (inputTensor->getShape()[mv::IO_CHANNEL_DIMENSION] % 16)
                                return true;
                        }
                    }
                }
                //NOTE: If you Spill a parent a child can be everything...the only thing
                //that has no sense if is your parent is spilling to be HKSwitch as
                //this strategy exists in order to reverse strategies in CMX
                else if (parentSpilling)
                {
                    if (childClustering == "HKSwitch")
                        return true;
                    //NOTE: For now I disable parent spill SOH->child (Clustering, K) for Z major convs
                    if (parentClustering == "SplitOverH" and ((childClustering == "Clustering" and childOpType !=  "Output") ||
                                                              childClustering == "SplitOverK"))
                    {
                        if (!(enableChannelMajorConv and 
                            ((parentOpType == "Conv" and
                            parentOp.getInputTensor(1)->getShape()[mv::KERNEL_INPUT_CHANNELS] < 16) or 
                            (childOpType == "Conv" and
                            childOp.getInputTensor(1)->getShape()[mv::KERNEL_INPUT_CHANNELS] < 16))) )
                            {
                                return true;
                            }
                    }
                }
                else
                {
                    std::pair<std::string, std::string> possibleCombination(parentClustering, childClustering);
                    for (auto restrictedCombination: incompatibleStrategiesWithOutSpilling)
                    {
                        if (possibleCombination == restrictedCombination)
                        {
                            return true;
                        }
                    }
                }

                //Note: last op should not be HKSwitch
                if (childOpType == "Output")
                {
                    if (parentClustering == "HKSwitch")
                        return true;
                }

                return false;
            }

            bool violatesChannelMajorRules(Op& parentOp,Op& childOp,StrategySet& parent,StrategySet& child)
            {
                bool spillForCM = false;
                auto parentClustering = parent["clustering"].get<std::string>();
                auto childClustering = child["clustering"].get<std::string>();
                auto parentOpType = parentOp.getOpType();
                auto childOpType = childOp.getOpType();
                auto parentSpilling = parent["spilling"].get<bool>();


                auto isChildChanMajor = childOpType == "Conv" && 
                                        childOp.getInputTensor(1)->getShape()[KERNEL_INPUT_CHANNELS] < 16;
                auto isParentChanMajor = parentOpType == "Conv" && 
                                        parentOp.getInputTensor(1)->getShape()[KERNEL_INPUT_CHANNELS] < 16;

                if (isParentChanMajor || isChildChanMajor)
                   spillForCM = needForceSpillingForCM(parentOp, childOp, parentClustering, childClustering);

                if (spillForCM and !parentSpilling)
                    return true;

                if( isChildChanMajor )
                {   
                    //Note: If SOHOverlapped input requires SOH CMconv, and vice versa
                    if(childClustering == "SplitOverH" && 
                        (parentOpType == "Input" && parentClustering != "SplitOverHOverlapped"))
                        return true;
                    if(parentClustering == "SplitOverHOverlapped" && childClustering != "SplitOverH")
                        return true;
                }

                // TODO
                // forgive me for this hack. Multiple input topologies are not qualifed for CM Conv, so this hack
                if (parentOpType == "ImplicitInput" or childOpType == "ImplicitInput")
                {
                    if (parentClustering == "SplitOverHOverlapped")
                        return true;
                    if (childClustering == "SplitOverHOverlapped")
                        return true;
                }

                return false;
            }

            double transitionCost(Op& parentOp,Op& childOp,StrategySet& parent,StrategySet& child)
            {
                auto INF = inf_;
                auto parentClustering = parent["clustering"].get<std::string>();
                auto childClustering = child["clustering"].get<std::string>();
                auto parentOpType = parentOp.getOpType();
                auto childOpType = childOp.getOpType();
                auto parentSpilling = parent["spilling"].get<bool>();
                auto childSpilling = child["spilling"].get<bool>();
                auto parentStreamShape = parent["streaming"].get<mv::Shape>();
                auto childStreamShape = child["streaming"].get<mv::Shape>();
                auto parentOutputSparsity = parent["outputSparsity"].get<bool>();
                auto childInputSparsity = child["inputSparsity"].get<bool>();


                //TODO re-enable runtime sparsity in this case, see spatial_split_streaming L143 for issue
                //Dummy slice prevents runtime sparsity from being activated in sparsity pass
                if(parentOutputSparsity && childStreamShape["K"] > 1)
                    return INF;

                // Note: Wasted output sparsity, output sparsity in this context is runtime generated
                if (parentOutputSparsity && !childInputSparsity)
                    return INF;

                // Note: no sense to stream activations if both layers stay in CMX
                if(!parentSpilling && !childSpilling && (childStreamShape["H"] > 1))
                    return INF;

                if( violatesClusteringStrategyRules(parentOp, childOp, parent, child) )
                {
                    log(mv::Logger::MessageType::Debug, parent["name"].toString()+"_"+parent["id"].toString()
                                + " transition to "+ child["name"].toString()+"_"+child["id"].toString() + 
                                " INF caused by incompatible clustering strategies");
                    return INF;
                }

                // Note: when a parent is cmx concating, the input activation tensor will be in CMX, need to recheck memory fits
                bool childActivationStreaming = (childStreamShape["H"] * childStreamShape["C"] * childStreamShape["W"]) > 1 ? true : false;
                if(!parentSpilling && childActivationStreaming)
                {
                    size_t input, output, weights;
                    std::tie(input, output, weights) = memorySize(childOp, childClustering, 
                                                                child["inputSparsity"].get<bool>(), 
                                                                child["outputSparsity"].get<bool>(),
                                                                child["weightsSparsity"].get<bool>(), 
                                                                childStreamShape,
                                                                requiresFakeActivationSparsity(childOp), 
                                                                childSpilling, parentSpilling);
                    if(input + output + weights >= clusterMemory)
                        return INF;
                }

                if(enableChannelMajorConv){
                    if( violatesChannelMajorRules(parentOp, childOp, parent, child) )
                    {
                        log(mv::Logger::MessageType::Debug, parent["name"].toString()+"_"+parent["id"].toString()
                            + " transition to "+ child["name"].toString()+"_"+child["id"].toString() + " INF caused by CM Conv rules");
                        return INF;
                    }
                }

                if(!enableChannelMajorConv && parentOpType == "Input" &&
                        childOpType == "Conv" &&
                        childOp.getInputTensor(1)->getShape()[mv::KERNEL_INPUT_CHANNELS] < 16)
                {
                    if (parentClustering == "SplitOverHOverlapped")
                    {
                        log(mv::Logger::MessageType::Debug, parent["name"].toString()+"_"+parent["id"].toString()
                            + " transition to "+ child["name"].toString()+"_"+child["id"].toString() + " INF caused by no CM Conv but Input has SOHOverlapped");
                        return INF;
                    }

                }
                
                int8_t success = checkHWUnsupportedOp(parentOp);
                if (success != 0)
                {
                    if (success == 1)
                        log(mv::Logger::MessageType::Warning, "The limitation of the tensor dimension 8192 might be hitted with the \
                            operation " + parentOp.getName());
                     else
                        log(mv::Logger::MessageType::Error, "Unsupported kernel/stride combination for DpuTask for \
                            the operation " + parentOp.getName());
                }

                
                auto isChildChanMajor = childOpType == "Conv" && enableChannelMajorConv &&
                                        childOp.getInputTensor(1)->getShape()[KERNEL_INPUT_CHANNELS] < 16;

                if( childOpType == "Conv")
                {

                    if( !isChildChanMajor &&
                        childClustering == "SplitOverH" &&
                        childInputSparsity &&
                        parentOutputSparsity ) // only allow for compiler sparsity, disallow for runtime sparsity
                    {
                        // This should also be solveable with fake compiler provided sparsity
                        // there may very well be cases where sparsity if enforced, but due to this
                        // limitation proper sparsity is not a choice since cluster boundary sparse map
                        // reads will fail due to misalignment
                        // Fake sparsity will provide all 1's sparse map so that probem is solved
                        // from the starts
                        // TODO: enable this case in G.O. decide later if fake or real sparsity
                        // Sparse map has to be contiguously alligned at 16 bytes
                        // for first (N - 1) clusters
                        auto outputTensorShape = parentOp.getOutputTensor(0)->getShape();
                        unsigned int W = outputTensorShape[IO_WIDTH_DIMENSION];
                        unsigned int H = outputTensorShape[IO_HEIGHT_DIMENSION];
                        unsigned int C = outputTensorShape[IO_CHANNEL_DIMENSION];
                        unsigned dy = std::ceil(static_cast<double>(H) / totalClusters);

                        if ((W*dy*C)%128 != 0)
                        {
                            log(mv::Logger::MessageType::Debug, child["name"].toString()+"_"+child["id"].toString() + " INF caused by incorrect SOH");
                            return INF;
                        }
                    }
                }

                //Note: Input clustering strategy should match first layer, if it is Z-major
                if(parentOpType == "Input" && !isChildChanMajor)
                {
                    if(parentClustering != childClustering)
                    {
                        log(mv::Logger::MessageType::Debug, parent["name"].toString()+"_"+parent["id"].toString()
                                + " transition to "+ child["name"].toString()+"_"+child["id"].toString() + " INF caused by input not matching first layer");
                        return INF;
                    }
                }

                auto execTime1 = computeTime(parentOp,parent);
                auto execTime2 = computeTime(childOp,child);

                // Case in which child input sparsity will be provided by compiler
                // Compiler provided sparsity is a dummy sparsity (all 1's sparse map)
                // so no real sparse acceleration will pe provided, only sparse decoding overhead
                auto sparsityOverhead = childOp.getInputTensor(0)->isFloatingPointType() ?
                    0.0625 : 0.125;
                if (!parentOutputSparsity && childInputSparsity)
                    execTime2 += execTime2 * sparsityOverhead;

                //TODO capture sparse speedup potential here if childInputSparsity && parentOutputSparsity both true
                // but probably only enable when activation sparsity is requested from CD. Otherwise, discourage?
                if(childInputSparsity && !requiresActivationSparsity(childOp, childClustering))
                    execTime2 += execTime2 * 0.01; // penalize not needed sparsity

                // cout << parentOp.getName() << " : " << parentStreamShape.toString() << " --> " << childOp.getName() << " : " << childStreamShape.toString() << endl;
                auto commTime1 = dmaTime(parentOp, parent);
                auto commTime2 = dmaTime(childOp, child, parentSpilling);

                // Extremely simplistic overlap computation, assume cost of compute is hidden by pipelining
                if(isPipeliningPossible(childOp, child, parent["spilling"].get<bool>()))
                    execTime2 = 0;

                //TODO if prefetch is possible, eliminate dma time for child op
                
                // std::cout << "Parent ID " << parent["id"].toString() <<"   " << execTime1 << " + " << commTime1 << std::endl;
                // std::cout << "Child ID " << child["id"].toString() << "   " << execTime2 << " + " << commTime2 << std::endl << std::endl;

                return execTime1 + commTime1 + execTime2 + commTime2;
        }


        /************************************************************************
         * Everything below here is to GENERATE STRATEGY SETS
         * 
         * 
         * 
         * 
         * 
         * 
         * 
         * 
         * 
         * 
         * 
         * 
         * 
         * 
         * *********************************************************************/
        std::vector<Attribute> createTFStrategyPoolFromBool(mv::Op op,std::string name)
            {
                auto& streamingStrategy = getStrategy(op,name);

                bool value = streamingStrategy.get<bool>();
                if(value)
                    return std::vector<Attribute>{true,false};
                else
                    return std::vector<Attribute>{false};
            }

            std::vector<mv::Attribute> createTStrategyPoolFromBool(mv::Op op,std::string name)
            {
                auto& streamingStrategy = getStrategy(op,name);

                bool value = streamingStrategy.get<bool>();
                if(value)
                    return std::vector<mv::Attribute>{true};
                else
                    return std::vector<mv::Attribute>{true,false};
            }

            bool createStrategyFromBool(mv::Op op, std::string name){
                auto& streamingStrategy = getStrategy(op,name);

                bool value = streamingStrategy.get<bool>();
                if(value)
                    return true;
                else
                    return false;
            }

            std::vector<mv::Attribute> createStrategyPoolFromStrategySet(mv::Op op, std::string name)
            {
                auto streamingStrategy = getStrategy(op,name);

                std::vector<mv::Attribute> attr;

                for (auto elem : streamingStrategy.get<std::vector<std::string>>())
                {
                    attr.push_back(elem);
                }

                return attr;
            }

            bool decideWeightsSparsity(mv::Op op)
            {
                //These values come from empircal data using three layer tests, in a matrix of size vs. sparsity
                //For performance, implemented as piece-wise if/else rather than polynomial
                double CMX_THRESHOLD_LOW = .05;
                double CMX_THRESHOLD_HIGH = .5;
                double ZEROPOINT_THRESHOLD_LOW = .3;
                double ZEROPOINT_THRESHOLD_HIGH = .2;

                // Only Z-major convolutions support weights sparsity, this is codified in the compilation descriptors
                if( !createStrategyFromBool(op,"weightsSparsity") )
                    return false;

                // If CM convolutions are enabled, don't sparsify these
                if(enableChannelMajorConv && op.supportsCMConv())
                    return false;

                //Size of weights, actual sparsity of tensor determine speedup
                auto weightsSize = realTensorSize(op.getInputTensor(1), {1,1,1,1}, false);

                //If weights are less than 5% of CMX, sparsity benefit does not outweigh overhead
                //no matter how sparse the weights are
                if(weightsSize < (clusterMemory * CMX_THRESHOLD_LOW))
                    return false;

                auto zeroPoints = op.getInputTensor(1)->getNumZeroPoints();
                double actualSparsity = (double) zeroPoints/ (double)weightsSize;

                //Weights between 5% and 50% of CMX, enable sparsity at threshold 30% sparse weights
                if(weightsSize < (clusterMemory * CMX_THRESHOLD_HIGH) &&
                    actualSparsity < ZEROPOINT_THRESHOLD_LOW)
                    return false;

                //Weights larger than 50% of CMX, enable sparsity at threshold 20% sparse weights
                if(weightsSize >= (clusterMemory * CMX_THRESHOLD_HIGH) &&
                    actualSparsity < ZEROPOINT_THRESHOLD_HIGH)
                    return false;

                return true;
            }

            bool validateHStream(mv::Op& op, mv::Attribute clustering, std::size_t splits)
            {
                if( op.getOpType() == "Conv" || op.getOpType() == "DepthwiseConv")
                {
                    if(clustering.get<std::string>() == "SplitOverH")
                    {
                        auto weightsShape = op.getInputTensor(1)->getShape();
                        //Try to guess subtensor height, and avoid situations where kernel is bigger than last workload dimension
                        auto outputHeight = op.getOutputTensor(0)->getShape()[IO_HEIGHT_DIMENSION];
                        auto workloadHeight = ceil((double)outputHeight / (double)(totalClusters * splits));
                        if(totalClusters > 1) //last
                            workloadHeight = outputHeight - (workloadHeight * (totalClusters-1)); //get remaining height
                        if(workloadHeight < weightsShape[KERNEL_HEIGHT])
                            return false;
                    }
                }
                return true;
            }
            // Note: This function will return the potential streams over H for this op. For simplicity, we want to limit
            // the options to reasonable configurations. This always includes H=1, or in other words no streaming over H.
            // If H streaming fits at all (i.e. weights fit), find the H just big enough to fit into CMX. If CMX concat,
            // spilling will be false, and H stream will be higher accordingly.
            std::vector<size_t> getStreamsOverH(mv::Op& op, mv::Attribute clustering, Shape streams, bool iSparsity, bool oSparsity, 
                                                bool wSparsity, bool fSparsity, bool spilling)
            {
                auto minSplitsToFit = getMinStreamOverH(op, clustering, streams, iSparsity, oSparsity, wSparsity, fSparsity, spilling);
                if(minSplitsToFit == 0) // stream over H doesn't fit
                    return {1};

                // Case 0, Not spilling, cmx concat. If pipelined, need room for 2 input slices.
                if(!spilling && globalEnablePipelining)
                {   
                    auto pipelinedMinSplitsToFit =  getMinStreamOverH(op, clustering, streams, iSparsity, oSparsity, wSparsity, fSparsity, spilling, false, true);
                    if(pipelinedMinSplitsToFit > 1 && pipelinedMinSplitsToFit != minSplitsToFit)
                        return {pipelinedMinSplitsToFit, minSplitsToFit, 1};
                }
                else if(spilling) // Case 1 Spilling, ddr concat. Find min stream over H for both input in cmx and input streamed.
                {
                    auto inputCmxMinSplitsToFit =  getMinStreamOverH(op, clustering, streams, iSparsity, oSparsity, wSparsity, fSparsity, spilling, true);                
                    if(inputCmxMinSplitsToFit > 1 && inputCmxMinSplitsToFit != minSplitsToFit)
                        return {inputCmxMinSplitsToFit, minSplitsToFit, 1};
                }

                return {minSplitsToFit, 1};
            }

            // Gives the minimum number of streams over H to fit this layer, or if no number of streams enable streaming
            // (for example, weights don't fit) then return 0
            unsigned getMinStreamOverH(mv::Op& op, mv::Attribute clustering, Shape streams, bool iSparsity, bool oSparsity, 
                                        bool wSparsity, bool fSparsity, bool spilling, bool inputCMX = false, bool pipelined = false)
            {
                size_t input, output, weights;
                std::tie(input, output, weights) = memorySize(op,clustering,iSparsity,oSparsity,wSparsity,streams,fSparsity,spilling);
                auto activationsSize = input + output;
                auto weightsSize = weights;
                double availableMemory = (double) clusterMemory - (double) weightsSize;

                if (availableMemory <= 0) // Weights don't fit, can't stream over H
                    return 0;

                // Keep increasing H until we find one big enough to fit, or we run out of H dimension to stream
                auto outputHeight = op.getOutputTensor(0)->getShape()[IO_HEIGHT_DIMENSION];

                // Every output slice must have at least one line to compute
                unsigned upperBoundH = outputHeight;
                if(clustering.toString() == "SplitOverH") 
                    upperBoundH = upperBoundH/totalClusters;

                // Start searching for min stream at naive requirement for splits to fit, rather than 1
                for(unsigned splits = ceil((double)activationsSize/availableMemory); splits <= upperBoundH; splits++)
                {
                    Shape updatedStreams({1,splits,1,streams["K"],streams["B"]});
                    auto memFitCheck = memorySize(op,clustering,iSparsity,oSparsity,wSparsity,updatedStreams,fSparsity,spilling,!inputCMX);
                    if( pipelined && //TODO inputCMX here too
                        (2*std::get<0>(memFitCheck) + std::get<1>(memFitCheck) + std::get<2>(memFitCheck) < clusterMemory) &&
                        validateHStream(op, clustering, splits) )
                    {
                        return splits;
                    }
                    else if(!pipelined && 
                            (std::get<0>(memFitCheck) + std::get<1>(memFitCheck) + std::get<2>(memFitCheck) < clusterMemory) &&
                            validateHStream(op, clustering, splits))
                    {
                        return splits;
                    }
                }

                return 0;
            }

            unsigned findBestK(unsigned alignedSize, unsigned channels){
                return std::ceil((double)alignedSize / ((alignedSize/2) - channels));
            }
            //Note: this function only used to generate many stream over k options when we nested stream
            std::vector<size_t> getMaxStreamOverK(mv::Op& op)
            {
                auto outputShape = op.getOutputTensor(0)->getShape();
                size_t outputChannelSize = outputShape[IO_CHANNEL_DIMENSION];
                size_t alignedOutputChannelSize = mv::round_up(outputChannelSize, 16);

                std::vector<size_t> splits;

                //Add max split
                splits.push_back(alignedOutputChannelSize/16);

                // For each aligned-to-16 number of output channels possibility, add only the
                // minimum number of streams over k that will be aligned to that number
                for(int channels = (alignedOutputChannelSize/2 -16); channels >= 16; channels=channels-16){
                    auto possibleK = findBestK(alignedOutputChannelSize, channels);
                    if(splits.back() != possibleK and possibleK >= 1)
                        splits.push_back(possibleK);
                }
                if(splits.back() > 2)
                    splits.push_back(2);

                if(splits.back() > 1)
                    splits.push_back(1);

                return splits;
            }

            bool validateKStream(mv::Op& op, mv::Attribute clustering, size_t split, bool spilling)
            {
                if( op.getOpType() == "Conv" && 
                    clustering.get<std::string>() == "SplitOverK")
                {
                    auto weightsShape = op.getInputTensor(1)->getShape();
                    auto numOutChannels = weightsShape[KERNEL_OUTPUT_CHANNELS];
                    if((numOutChannels/split * totalClusters) < 16)
                        return false;
                }
                if(!spilling)
                {
                    auto outputShape = op.getOutputTensor(0)->getShape();
                    size_t outputChannelSize = outputShape[IO_CHANNEL_DIMENSION];
                    //ok it fits, now make sure that if we are !spilling that there's no crop
                    size_t outputChannelSlice = ceil((double)outputChannelSize/(double)split);
                    size_t lastSlice = outputChannelSize - outputChannelSlice*(split - 1);
                    if (!(outputChannelSlice%16 == 0 && lastSlice%16 == 0)) //would need crop
                        return false;
                }
                        
                return true;
            }

            unsigned getMinStreamOverK(mv::Op& op, mv::Attribute clustering, Shape streams, bool iSparsity, bool oSparsity, 
                                                bool wSparsity, bool fSparsity, bool spilling, bool pipelined = false)
            {
                auto outputShape = op.getOutputTensor(0)->getShape();
                size_t outputChannelSize = outputShape[IO_CHANNEL_DIMENSION];
                size_t alignedOutputChannelSize = mv::round_up(outputChannelSize, 16);

                auto maxSplit = alignedOutputChannelSize / 16;

                if(clustering.get<std::string>() == "SplitOverK")
                    maxSplit = maxSplit / totalClusters;

                for(unsigned split = 1; split <= maxSplit; split++)
                {
                    auto memFitCheck = memorySize(op, clustering,iSparsity,oSparsity,wSparsity,{1,1,1,split,streams["B"]},fSparsity, spilling);
                    if( pipelined && //pipelining weights requires 2 weights streams to fit
                        (std::get<0>(memFitCheck) + std::get<1>(memFitCheck) + 2*std::get<2>(memFitCheck) < clusterMemory) &&
                        validateKStream(op, clustering, split, spilling) )
                    {
                        return split;
                    }
                    else if(!pipelined &&
                            (std::get<0>(memFitCheck) + std::get<1>(memFitCheck) + std::get<2>(memFitCheck) < clusterMemory) &&
                            validateKStream(op, clustering, split, spilling) )
                    {
                        return split;
                    }
                }

                return 0;
            }

           // Note: This function produces the potential stream over K strategies for each layer
           // Try to find 2 possible combinations of K, in addition ot K=1 (no streams in this dimension)
           // First, just enough to fit in cmx. Second, enough to enable pipelining.
            std::vector<std::size_t> getStreamsOverK(mv::Op& op, mv::Attribute clustering, Shape streams, bool iSparsity, bool oSparsity, 
                                                bool wSparsity, bool fSparsity, bool spilling)
            {
                auto minSplitsToFit = getMinStreamOverK(op, clustering, streams, iSparsity, oSparsity, wSparsity, fSparsity, spilling);
                if(minSplitsToFit == 0) // No suitable stream over K found
                    return {1};

                std::vector<std::size_t> splits;
                splits.push_back(1);
                if(minSplitsToFit != 1)
                    splits.push_back(minSplitsToFit);

                if(globalEnablePipelining)
                {
                    auto pipelinedMinSplitsToFit = getMinStreamOverK(op, clustering, streams, iSparsity, oSparsity, wSparsity, fSparsity, spilling, true);
                    if(pipelinedMinSplitsToFit > 1 && pipelinedMinSplitsToFit != minSplitsToFit)
                        splits.push_back(pipelinedMinSplitsToFit);
                }

                return splits;
            }

            void generateStrategySetForLayer(mv::Op& op,std::vector<StrategySet>& strategyVec)
            {
                auto findStrategy = [](std::vector<Attribute>& vec,const std::string& str) ->bool { for(const auto elem : vec) if(str==elem.get<std::string>()) return true; return false;};

                std::vector<Attribute> spillingPool;
                if(globalForceSpilling)
                    spillingPool.push_back(true);
                else
                    spillingPool = createTStrategyPoolFromBool(op, "forceSpilling");

                std::vector<Attribute> clusteringStrategyPool;

                if(totalClusters == 1 || op.hasAttr("forceClustering"))
                    clusteringStrategyPool.push_back(std::string("Clustering"));
                else if (totalClusters > 1)
                    clusteringStrategyPool = createStrategyPoolFromStrategySet(op,"clusteringStrategies");
                else
                    throw LogicError(*this, "Graph Optimizer unable to determine number of clusters");

                std::vector<Attribute> streamingStrategyPool = createStrategyPoolFromStrategySet(op,"streamingStrategies");

                bool hasStreamOverK = false;
                bool hasStreamOverH = false;
                bool hasStreamOverC = false;
                bool hasStreamOverN = false;

                if(globalEnableStreaming)
                {
                    hasStreamOverK = findStrategy(streamingStrategyPool,"StreamOverK");
                    hasStreamOverH = findStrategy(streamingStrategyPool,"StreamOverH");
                    hasStreamOverC = findStrategy(streamingStrategyPool,"StreamOverC");
                    hasStreamOverN = findStrategy(streamingStrategyPool,"StreamOverN");
                }


                bool weightsSparsity = false;
                if(requiresWeightsSparsity(op))
                    weightsSparsity = true;
                else if(globalEnableWeightsSparsity)
                    weightsSparsity = decideWeightsSparsity(op);

                //TODO:: replace nested loops with clean cartesian product function
                for(auto spilling : spillingPool)
                {
                for(auto clustering : clusteringStrategyPool)
                {
                    // Make decision about input activation sparsity, depending on clustering strategy
                    std::vector<Attribute> inputActivationSparsity = {false};
                    std::vector<Attribute> outputActivationSparsity = {false};
                    if(globalEnableActivationSparsity)
                    {
                        inputActivationSparsity = createTFStrategyPoolFromBool(op,"inputActivationSparsity");
                        outputActivationSparsity = createTFStrategyPoolFromBool(op,"outputActivationSparsity");
                    }
                    if(globalForceActivationSparsity)
                    {
                        inputActivationSparsity = {createStrategyFromBool(op,"inputActivationSparsity")};
                        outputActivationSparsity = createTFStrategyPoolFromBool(op,"outputActivationSparsity");
                    }

                    if (requiresActivationSparsity(op, clustering.get<std::string>()))
                        inputActivationSparsity = {true};

                    for( auto inputSparsity : inputActivationSparsity)
                    {
                    for( auto outputSparsity : outputActivationSparsity)
                    {
                        bool fakeSparsity = requiresFakeActivationSparsity(op);


                        // Determine streaming options
                        // 0. Determine if streams over H are possible
                        // 1. Determine if streams over N are required
                        // 2. Determine if streams over K are possible
                        // 3. If no streams over H or K will fit, enable nested streaming
                        // 4. Nested loops over generated streaming options to produce all strategy options
                        std::vector<size_t> streamsOverH;
                        if(hasStreamOverH)
                            streamsOverH = getStreamsOverH(op, clustering, {1,1,1,1,1}, inputSparsity.get<bool>(), 
                                                                outputSparsity.get<bool>(), weightsSparsity, fakeSparsity, spilling.get<bool>());
                        else
                            streamsOverH.push_back(1);

                        // Stream over batch, each batch must be it's own stream
                        unsigned n = 1;
                        if(hasStreamOverN && op.getInputTensor(0)->getShape()["N"] > 1)
                            n = op.getInputTensor(0)->getShape()["N"];

                        std::vector<size_t> streamsOverK;
                        if(hasStreamOverK)
                        {
                            streamsOverK = getStreamsOverK(op, clustering, {1,1,1,1,n}, inputSparsity.get<bool>(), 
                                                            outputSparsity.get<bool>(), weightsSparsity, fakeSparsity, spilling.get<bool>());
                        }
                        else
                        {
                            streamsOverK.push_back(1);
                        }

                        std::vector<size_t> streamsOverC;
                        if (hasStreamOverC)
                            streamsOverC = {1,2,3,4}; // TODO calculate properly
                        else
                            streamsOverC.push_back(1);

                        bool enableNestedStreaming = false;
                        auto maxK = streamsOverK.back();
                        auto memK = memorySize(op,clustering,inputSparsity.get<bool>(),outputSparsity.get<bool>(),weightsSparsity,{1,1,1,maxK,n},fakeSparsity, spilling.get<bool>());
                        auto memoryMaxK = std::get<0>(memK) + std::get<1>(memK) + std::get<2>(memK);
                        auto maxH = streamsOverH.front();
                        auto memH = memorySize(op,clustering,inputSparsity.get<bool>(),outputSparsity.get<bool>(),weightsSparsity,{1,maxH,1,1,n},fakeSparsity, spilling.get<bool>());
                        auto memoryMaxH =  std::get<0>(memH) + std::get<1>(memH) + std::get<2>(memH);


                        // If streaming is enabled, but streaming over k or h alone doesn't fit, enable nested streaming
                        //nested streaming is a trick to fit an operation when both weight tensor and input tensors do not fit on cmx
                        //not much sense to try fit the output tensor there, disabling in case of no spilling
                        if( hasStreamOverK && hasStreamOverH && spilling &&
                            (memoryMaxH > clusterMemory) &&
                            (memoryMaxK > clusterMemory) )
                        {
                            //Note: We generate only 1 stream over K possibility now, just enough to fit
                            // If we need nested streaming, generate a range of possibilities as we don't know what
                            // will fit with the range of streams over H
                            streamsOverK = getMaxStreamOverK(op);
                            enableNestedStreaming = true;
                        }
                        for(const auto k : streamsOverK)
                        {
                            if(enableNestedStreaming) // generate h ranges on the fly
                            {
                                streamsOverH = getStreamsOverH(op, clustering, {1,1,1,k,1}, inputSparsity.get<bool>(), 
                                                                outputSparsity.get<bool>(), weightsSparsity, fakeSparsity, spilling.get<bool>());
                            }
                            for(const auto h : streamsOverH)
                            {
                                for(const auto c : streamsOverC)
                                {
                                    if((h > 1) && (c > 1)) //Fast hack to disable nested streaming with C
                                        continue;
                                    if((h > 1) && (n > 1)) //Fast hack to disable nested streaming with n
                                        continue;
                                    if( !enableNestedStreaming && ((h>1) && (k>1))) // Skip nested streams unless necessary
                                        continue;
                                    if( enableNestedStreaming and ((h==1) or (k==1))) // If need nested streams, ignore non-nested
                                       continue;

                                    Shape streamShape({1,h,c,k,n}); //Stream over W is 1. Not implemented.

                                    StrategySet s;
                                    s["name"] = op.getName();
                                    s["id"] = (unique_ctr++);
                                    s["inputSparsity"] = inputSparsity;
                                    s["outputSparsity"] = outputSparsity;
                                    s["weightsSparsity"] = weightsSparsity;
                                    s["spilling"] = spilling;
                                    s["clustering"] = clustering;
                                    s["streaming"] = streamShape;

                                    //Function to prune strategies that will have only infinite edges in or out (or both), improves performance
                                    auto strategyCheck = validateStrategy(op,s);
                                    // if(op.getName() == "yolov2/darknet19_model/max_pool1/MaxPool")
                                    //     std::cout << op.getName() << " : " << clustering.toString() << " : " << streamShape.toString() << " : " << spilling.toString() << " : " << failure_causes[strategyCheck] << std::endl;
                                    if(strategyCheck != FailCause::Pass)
                                        continue;

                                    strategyVec.push_back(s);

                                    //    std::cout << "Name: " + op.getName() << " ID " << s["id"].toString()<< std::endl;
                                    //    std::cout << "Input Sparsity: " + inputSparsity.toString() << std::endl;
                                    //    std::cout << "Output Sparsity: " + outputSparsity.toString() << std::endl;
                                    //    std::cout << "Weights Sparsity: " + weightsSparsity << std::endl;
                                    //    std::cout << "Spilling: " + spilling.toString() << std::endl;
                                    //    std::cout << "MCStrategy: " + clustering.toString() << std::endl;
                                    //    std::cout << "Streaming(W,H,C,K,N): " + streamShape.toString() << std::endl<<std::endl;
                                   
                                }
                            }
                        }
                        }
                        }
                    }
                }
                if(strategyVec.empty())
                    throw LogicError(*this,"No strategies created for layer " + op.getName() + ". Layer possibly unsupported.");
            }

        };

    }

}

static void GraphParameterOptimizationFcn(
    const mv::pass::PassEntry& ,
    mv::ComputationModel& model,
    mv::TargetDescriptor& td, mv::Element& passDesc,
    mv::Element&
)
{
    mv::OpModel om(model);
    mv::graphOptimizer::StrategyManagerKmb strategyManager(om,passDesc);

    strategyManager.updateValuesFromJSON();
    strategyManager.updateDefaultValues();
    strategyManager.readGlobalConfigs();

    strategyManager.graphParameterOptimizations();

    return;
}
