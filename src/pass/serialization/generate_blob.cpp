#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/deployer/serializer.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/target/target_descriptor.hpp"

static void generateBlobFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&td, mv::json::Object& compDesc, mv::json::Object& compOutput);
static void PopulateSerialFieldsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object&, mv::json::Object& compOutput);
//static void writeSerialFieldsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object& compDesc, mv::json::Object& compOutput);

namespace mv
{

    namespace pass
    {


        MV_REGISTER_PASS(PopulateSerialFields)
        .setFunc(PopulateSerialFieldsFcn)
        .setGenre(PassGenre::Serialization)
        .setDescription(
            "Gathers fields for serialization"
        );

        MV_REGISTER_PASS(GenerateBlob)
        .setFunc(generateBlobFcn)
        .setGenre(PassGenre::Serialization)
        .defineArg(json::JSONType::String, "output")
        .setDescription(
            "Generates an executable blob file"
        );

    }

}

void generateBlobFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor& td, mv::json::Object& compDesc, mv::json::Object& compOutput)
{   

    using namespace mv;

    if (compDesc["GenerateBlob"]["output"].get<std::string>().empty())
        throw ArgumentError(model, "output", "", "Unspecified output name for generate dot pass");

    mv::Serializer serializer(mv::mvblob_mode);
    long long result = static_cast<long long>(serializer.serialize(model, td, compDesc["GenerateBlob"]["output"].get<std::string>().c_str()));
    compOutput["blobSize"] = result;

}
void PopulateSerialFieldsFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object&, mv::json::Object& )
{
    mv::OpModel om(model);

    for(auto opIt = om.opBegin(); opIt != om.opEnd(); ++opIt)
    {
        std::cout << "Populating Serial fields for Op{" << opIt->getOpType() << "}" << std::endl;
        //Short term fix: Big if-else acting like a switch
        //Long term solution: Move everything to Target Descriptor

        if(opIt->getOpType() == "Add")
        {
            opIt->set<unsigned>("SerialID", 12);
        }
        else if(opIt->getOpType() == "AveragePool")
        {
            auto fp16_size = 2;

            if (opIt->hasAttr("NCE1_Compatible") && opIt->get<int>("NCE1_Compatible") )
            {
                // Get all attrs:
                auto splits_over_H = opIt->get<size_t>("NCE1_SplitsOverHeight");
                auto DPUmodeVector = opIt->get<std::vector<size_t>>("NCE1_Modes");
                auto splits_over_iC = opIt->get<size_t>("NCE1_SplitsOverInputChannels");
                auto inputChannelsPadded = opIt->get<std::size_t>("NCE1_InputChannelsPadded");
                auto outputChannelsPadded = opIt->get<std::size_t>("NCE1_OutputChannelsPadded");
                auto inputWidthPadded = opIt->get<std::size_t>("NCE1_InputWidthPadded");
                //auto outputWidthPadded = opIt->get<std::size_t>("NCE1_OutputWidthPadded");
                auto desc_count = opIt->get<std::size_t>("NCE1_DescriptorSplits");
                auto streamingMask = opIt->get<std::size_t>("NCE1_StreamingMask");

                auto input_lines_processed = opIt->get<std::vector<size_t>>("NCE1_InputLinesProcessed");
                auto output_lines_processed = opIt->get<std::vector<size_t>>("NCE1_OutputLinesProcessed");
                auto output_line_start = opIt->get<std::vector<size_t>>("NCE1_StartOutputLine");
                auto input_line_start = opIt->get<std::vector<size_t>>("NCE1_StartInputLine");

                auto radixX = opIt->get<std::array<short unsigned, 2>>("kSize")[0];
                auto radixY = opIt->get<std::array<short unsigned, 2>>("kSize")[1];

                opIt->set<unsigned>("SerialID", 34);    // To be moved?

                opIt->set<unsigned>("streamingMask", streamingMask );

                std::size_t total_size = opIt->getInputTensor(0)->getShape().totalSize();
                total_size *= inputChannelsPadded;
                total_size /= opIt->getInputTensor(0)->getShape()[2];
                opIt->set<unsigned>("inputSize", total_size*fp16_size);

                opIt->set<unsigned>("outputSize",
                    opIt->getOutputTensor(0)->getShape().totalSize()*fp16_size);

                opIt->set<unsigned>("concatOffset", 0); // Not Supported...
                opIt->set<unsigned>("unloadCMX", 0); // Not Supported...
                opIt->set<unsigned>("overwriteInput", 0); // Not Supported...
                opIt->set<unsigned>("CMXSize", 256*1024);  // Magic Number...
                opIt->set<unsigned>("reluSHVAcc", 0); // Not Supported...
                opIt->set<unsigned>("shvNegSlope", 0); // Not Supported...
                opIt->set<unsigned>("shvPosSlope", 1065353216); // Magic Number...
                opIt->set<unsigned>("desc_count", desc_count);


                std::vector<unsigned> desc;
                std::vector<cnnConvolutionPoolStructure> descriptors = std::vector<cnnConvolutionPoolStructure>(desc_count);

                int i = -1;
                for (unsigned h = 0; h < splits_over_H; ++h)
                {
                    for (unsigned oc = 0; oc < DPUmodeVector.size(); ++oc)
                    {
                        for (unsigned ic = 0; ic < splits_over_iC; ++ic)
                        {
                            ++i;

                            auto input_width = inputWidthPadded;
                            auto output_channels = outputChannelsPadded;

                            descriptors[i].dataBaseAddr = 2 * input_width * input_line_start[h];    // TODO: Calculate 3f0 (1008)

                            if( opIt->getInputTensor(0)->getOrder().isRowInterleaved() )
                            {
                                descriptors[i].dataBaseAddr *= inputChannelsPadded;    // TODO: Calculate 3f0 (1008)
                                // descriptors[i].dataLnStr = inputBlobTensor.strideY;
                                // descriptors[i].dataChStr = inputBlobTensor.strideZ;
                                descriptors[i].dataLnStr = 42;
                                descriptors[i].dataChStr = 42;
                            }
                            else
                            {
                                // descriptors[i].dataLnStr = inputBlobTensor.strideY;
                                // descriptors[i].dataChStr = inputBlobTensor.strideZ;
                                descriptors[i].dataLnStr = 42;
                                descriptors[i].dataChStr = 42;
                            }
                            descriptors[i].coeffBaseAddr = 0;
                            descriptors[i].biasBaseAddr = 0;
                            descriptors[i].scaleBaseAddr = 0;
                            //HACK FOR CONCAT
                            // descriptors[i].outBaseAddr = outputBlobTensor.strideZ * output_line_start[h];  // TODO: Calculate 3f0 (1008)
                            descriptors[i].outBaseAddr = 42;  // TODO: Calculate 3f0 (1008)

                            if( opIt->getOutputTensor(0)->getOrder().isRowInterleaved() )
                            {
                                descriptors[i].outBaseAddr *= output_channels;    // TODO: Calculate 3f0 (1008)
                                // descriptors[i].outLnStr = outputBlobTensor.strideY;
                                // descriptors[i].outChStr = outputBlobTensor.strideZ;
                                descriptors[i].outLnStr = 42;
                                descriptors[i].outChStr = 42;
                            }
                            else
                            {
                                // descriptors[i].outLnStr = outputBlobTensor.strideY;
                                // descriptors[i].outChStr = outputBlobTensor.strideZ;
                                descriptors[i].outLnStr = 42;
                                descriptors[i].outChStr = 42;
                            }

                            auto weight_4dshape = opIt->getInputTensor(1)->getShape();

                            descriptors[i].coeffChStrIn = weight_4dshape[2]*weight_4dshape[3]*weight_4dshape[4]*2;
                            int inChans = inputChannelsPadded;

                            descriptors[i].coeffChStrOut = radixX * radixY * inChans * 2 * 8; // (fp16)

                            for(unsigned j = 0; j != 32; j++)
                                desc.push_back(((unsigned *) &descriptors[i])[j]);
                        }

                    }

                }

                opIt->set<std::vector<unsigned>>("descriptors", desc);
            }
            else
            {
                opIt->set<unsigned>("SerialID", 2);

                opIt->set<unsigned>("radixX",  opIt->get<std::array<short unsigned, 2>>("kSize")[0]);
                opIt->set<unsigned>("radixY",  opIt->get<std::array<short unsigned, 2>>("kSize")[1]);
                opIt->set<unsigned>("strideX",  opIt->get<std::array<unsigned short, 2>>("stride")[0]);
                opIt->set<unsigned>("strideY",  opIt->get<std::array<unsigned short, 2>>("stride")[1]);
                opIt->set<unsigned>("padX",  opIt->get<std::array<unsigned short, 4>>("padding")[0]);
                opIt->set<unsigned>("padY",  opIt->get<std::array<unsigned short, 4>>("padding")[2]);
                opIt->set<unsigned>("padStyle",  2);

            }
        }
        else if(opIt->getOpType() == "BatchNormalization")
        {

        }
        else if(opIt->getOpType() == "Bias")
        {

        }
        else if(opIt->getOpType() == "Concat")
        {

        }
        else if(opIt->getOpType() == "Constant")
        {

        }
        else if(opIt->getOpType() == "Conv")
        {
            auto fp16_size = 2;

            if (opIt->hasAttr("NCE1_Compatible") && opIt->get<int>("NCE1_Compatible"))
            {
                // Get all attrs:
                auto splits_over_H = opIt->get<size_t>("NCE1_SplitsOverHeight");
                auto DPUmodeVector = opIt->get<std::vector<size_t>>("NCE1_Modes");
                auto splits_over_iC = opIt->get<size_t>("NCE1_SplitsOverInputChannels");
                auto inputChannelsPadded = opIt->get<std::size_t>("NCE1_InputChannelsPadded");
                auto outputChannelsPadded = opIt->get<std::size_t>("NCE1_OutputChannelsPadded");
                auto inputWidthPadded = opIt->get<std::size_t>("NCE1_InputWidthPadded");
                //auto outputWidthPadded = opIt->get<std::size_t>("NCE1_OutputWidthPadded");
                auto desc_count = opIt->get<std::size_t>("NCE1_DescriptorSplits");
                auto streamingMask = opIt->get<std::size_t>("NCE1_StreamingMask");

                auto input_lines_processed = opIt->get<std::vector<size_t>>("NCE1_InputLinesProcessed");
                auto output_lines_processed = opIt->get<std::vector<size_t>>("NCE1_OutputLinesProcessed");
                auto output_line_start = opIt->get<std::vector<size_t>>("NCE1_StartOutputLine");
                auto input_line_start = opIt->get<std::vector<size_t>>("NCE1_StartInputLine");

                auto radixX = opIt->getInputTensor(1)->getShape()[2];
                auto radixY = opIt->getInputTensor(1)->getShape()[3];

                opIt->set<unsigned>("SerialID", 33);    // To be moved?

                opIt->set<unsigned>("streamingMask", streamingMask );

                std::size_t total_size = opIt->getInputTensor(0)->getShape().totalSize();
                total_size *= inputChannelsPadded;
                total_size /= opIt->getInputTensor(0)->getShape()[2];
                opIt->set<unsigned>("inputSize", total_size*fp16_size);

                opIt->set<unsigned>("outputSize",
                    opIt->getOutputTensor(0)->getShape().totalSize()*fp16_size);

                opIt->set<unsigned>("concatOffset", 0); // Not Supported...
                opIt->set<unsigned>("unloadCMX", 0); // Not Supported...
                opIt->set<unsigned>("overwriteInput", 0); // Not Supported...
                opIt->set<unsigned>("CMXSize", 256*1024);  // Magic Number...
                opIt->set<unsigned>("reluSHVAcc", 0); // Not Supported...
                opIt->set<unsigned>("shvNegSlope", 0); // Not Supported...
                opIt->set<unsigned>("shvPosSlope", 1065353216); // Magic Number...
                opIt->set<unsigned>("desc_count", desc_count);

                std::vector<unsigned> desc;
                std::vector<cnnConvolutionPoolStructure> descriptors = std::vector<cnnConvolutionPoolStructure>(desc_count);

                int i = -1;
                for (unsigned h = 0; h < splits_over_H; ++h)
                {
                    for (unsigned oc = 0; oc < DPUmodeVector.size(); ++oc)
                    {
                        for (unsigned ic = 0; ic < splits_over_iC; ++ic)
                        {
                            ++i;

                            auto input_width = inputWidthPadded;
                            auto output_channels = outputChannelsPadded;

                            descriptors[i].dataBaseAddr = 2 * input_width * input_line_start[h];    // TODO: Calculate 3f0 (1008)

                            if( opIt->getInputTensor(0)->getOrder().isRowInterleaved() )
                            {
                                descriptors[i].dataBaseAddr *= inputChannelsPadded;    // TODO: Calculate 3f0 (1008)
                                // descriptors[i].dataLnStr = inputBlobTensor.strideY;
                                // descriptors[i].dataChStr = inputBlobTensor.strideZ;
                                descriptors[i].dataLnStr = 42;
                                descriptors[i].dataChStr = 42;
                            }
                            else
                            {
                                // descriptors[i].dataLnStr = inputBlobTensor.strideY;
                                // descriptors[i].dataChStr = inputBlobTensor.strideZ;
                                descriptors[i].dataLnStr = 42;
                                descriptors[i].dataChStr = 42;
                            }
                            descriptors[i].coeffBaseAddr = 0;
                            descriptors[i].biasBaseAddr = 0;
                            descriptors[i].scaleBaseAddr = 0;
                            //HACK FOR CONCAT
                            // descriptors[i].outBaseAddr = outputBlobTensor.strideZ * output_line_start[h];  // TODO: Calculate 3f0 (1008)
                            descriptors[i].outBaseAddr = 42;  // TODO: Calculate 3f0 (1008)

                            if( opIt->getOutputTensor(0)->getOrder().isRowInterleaved() )
                            {
                                descriptors[i].outBaseAddr *= output_channels;    // TODO: Calculate 3f0 (1008)
                                // descriptors[i].outLnStr = outputBlobTensor.strideY;
                                // descriptors[i].outChStr = outputBlobTensor.strideZ;
                                descriptors[i].outLnStr = 42;
                                descriptors[i].outChStr = 42;
                            }
                            else
                            {
                                // descriptors[i].outLnStr = outputBlobTensor.strideY;
                                // descriptors[i].outChStr = outputBlobTensor.strideZ;
                                descriptors[i].outLnStr = 42;
                                descriptors[i].outChStr = 42;
                            }

                            auto weight_4dshape = opIt->getInputTensor(1)->getShape();

                            descriptors[i].coeffChStrIn = weight_4dshape[2]*weight_4dshape[3]*weight_4dshape[4]*2;
                            int inChans = inputChannelsPadded;

                            descriptors[i].coeffChStrOut = radixX * radixY * inChans * 2 * 8; // (fp16)

                            for(unsigned j = 0; j != 32; j++)
                                desc.push_back(((unsigned *) &descriptors[i])[j]);
                        }
                    }
                }

                opIt->set<std::vector<unsigned>>("descriptors", desc);
            }
            else
            {
                opIt->set<unsigned>("SerialID", 0);
                opIt->set<unsigned>("radixX",  opIt->getInputTensor(1)->getShape()[0]);
                opIt->set<unsigned>("radixY",  opIt->getInputTensor(1)->getShape()[1]);
                opIt->set<unsigned>("strideX",  opIt->get<std::array<unsigned short, 2>>("stride")[0]);
                opIt->set<unsigned>("strideY",  opIt->get<std::array<unsigned short, 2>>("stride")[1]);
                opIt->set<unsigned>("padX",  opIt->get<std::array<unsigned short, 4>>("padding")[0]);
                opIt->set<unsigned>("padY",  opIt->get<std::array<unsigned short, 4>>("padding")[2]);
                opIt->set<unsigned>("padStyle",  2);
                opIt->set<unsigned>("dilation",  1);
            }
        }
        else if(opIt->getOpType() == "Conversion")
        {
            opIt->set<unsigned>("SerialID", 37);
        }
        else if(opIt->getOpType() == "DepthwiseConv")
        {
            //auto fp16_size = 2;

            opIt->set<unsigned>("SerialID", 8);

            opIt->set<unsigned>("radixX",  opIt->getInputTensor(1)->getShape()[0]);
            opIt->set<unsigned>("radixY",  opIt->getInputTensor(1)->getShape()[1]);
            opIt->set<unsigned>("strideX",  opIt->get<std::array<unsigned short, 2>>("stride")[0]);
            opIt->set<unsigned>("strideY",  opIt->get<std::array<unsigned short, 2>>("stride")[1]);
            opIt->set<unsigned>("padX",  opIt->get<std::array<unsigned short, 4>>("padding")[0]);
            opIt->set<unsigned>("padY",  opIt->get<std::array<unsigned short, 4>>("padding")[2]);
            opIt->set<unsigned>("padStyle",  2);
            opIt->set<unsigned>("dilation",  1);
        }
        else if(opIt->getOpType() == "Divide")
        {
            opIt->set<unsigned>("SerialID", 13);
        }
        else if(opIt->getOpType() == "Dropout")
        {

        }
        else if(opIt->getOpType() == "Fullyconnected")
        {
            auto fp16_size = 2;

            if (opIt->hasAttr("NCE1_Compatible") && opIt->get<int>("NCE1_Compatible") )
            {
                opIt->set<unsigned>("SerialID", 35);
                // Get all attrs:
                auto splits_over_H = opIt->get<size_t>("NCE1_SplitsOverHeight");
                auto DPUmodeVector = opIt->get<std::vector<size_t>>("NCE1_Modes");
                auto splits_over_iC = opIt->get<size_t>("NCE1_SplitsOverInputChannels");
                auto inputChannelsPadded = opIt->get<std::size_t>("NCE1_InputChannelsPadded");
                auto outputChannelsPadded = opIt->get<std::size_t>("NCE1_OutputChannelsPadded");
                auto inputWidthPadded = opIt->get<std::size_t>("NCE1_InputWidthPadded");
                //auto outputWidthPadded = opIt->get<std::size_t>("NCE1_OutputWidthPadded");
                auto desc_count = opIt->get<std::size_t>("NCE1_DescriptorSplits");
                auto streamingMask = opIt->get<std::size_t>("NCE1_StreamingMask");

                auto input_lines_processed = opIt->get<std::vector<size_t>>("NCE1_InputLinesProcessed");
                auto output_lines_processed = opIt->get<std::vector<size_t>>("NCE1_OutputLinesProcessed");
                auto output_line_start = opIt->get<std::vector<size_t>>("NCE1_StartOutputLine");
                auto input_line_start = opIt->get<std::vector<size_t>>("NCE1_StartInputLine");

                auto radixX = opIt->getInputTensor(1)->getShape()[2];
                auto radixY = opIt->getInputTensor(1)->getShape()[3];

                opIt->set<unsigned>("SerialID", 34);    // To be moved?

                opIt->set<unsigned>("streamingMask", streamingMask );

                std::size_t total_size = opIt->getInputTensor(0)->getShape().totalSize();
                total_size *= inputChannelsPadded;
                total_size /= opIt->getInputTensor(0)->getShape()[2];
                opIt->set<unsigned>("inputSize", total_size*fp16_size);

                opIt->set<unsigned>("outputSize",
                    opIt->getOutputTensor(0)->getShape().totalSize()*fp16_size);

                opIt->set<unsigned>("concatOffset", 0); // Not Supported...
                opIt->set<unsigned>("unloadCMX", 0); // Not Supported...
                opIt->set<unsigned>("overwriteInput", 0); // Not Supported...
                opIt->set<unsigned>("CMXSize", 256*1024);  // Magic Number...
                opIt->set<unsigned>("reluSHVAcc", 0); // Not Supported...
                opIt->set<unsigned>("shvNegSlope", 0); // Not Supported...
                opIt->set<unsigned>("shvPosSlope", 1065353216); // Magic Number...
                opIt->set<unsigned>("desc_count", desc_count);


                std::vector<unsigned> desc;
                std::vector<cnnConvolutionPoolStructure> descriptors = std::vector<cnnConvolutionPoolStructure>(desc_count);

                int i = -1;
                for (unsigned h = 0; h < splits_over_H; ++h)
                {
                    for (unsigned oc = 0; oc < DPUmodeVector.size(); ++oc)
                    {
                        for (unsigned ic = 0; ic < splits_over_iC; ++ic)
                        {
                            ++i;

                            auto input_width = inputWidthPadded;
                            auto output_channels = outputChannelsPadded;

                            descriptors[i].dataBaseAddr = 2 * input_width * input_line_start[h];    // TODO: Calculate 3f0 (1008)

                            if( opIt->getInputTensor(0)->getOrder().isRowInterleaved() )
                            {
                                descriptors[i].dataBaseAddr *= inputChannelsPadded;    // TODO: Calculate 3f0 (1008)
                                // descriptors[i].dataLnStr = inputBlobTensor.strideY;
                                // descriptors[i].dataChStr = inputBlobTensor.strideZ;
                                descriptors[i].dataLnStr = 42;
                                descriptors[i].dataChStr = 42;
                            }
                            else
                            {
                                // descriptors[i].dataLnStr = inputBlobTensor.strideY;
                                // descriptors[i].dataChStr = inputBlobTensor.strideZ;
                                descriptors[i].dataLnStr = 42;
                                descriptors[i].dataChStr = 42;
                            }
                            descriptors[i].coeffBaseAddr = 0;
                            descriptors[i].biasBaseAddr = 0;
                            descriptors[i].scaleBaseAddr = 0;
                            //HACK FOR CONCAT
                            // descriptors[i].outBaseAddr = outputBlobTensor.strideZ * output_line_start[h];  // TODO: Calculate 3f0 (1008)
                            descriptors[i].outBaseAddr = 42;  // TODO: Calculate 3f0 (1008)

                            if( opIt->getOutputTensor(0)->getOrder().isRowInterleaved() )
                            {
                                descriptors[i].outBaseAddr *= output_channels;    // TODO: Calculate 3f0 (1008)
                                // descriptors[i].outLnStr = outputBlobTensor.strideY;
                                // descriptors[i].outChStr = outputBlobTensor.strideZ;
                                descriptors[i].outLnStr = 42;
                                descriptors[i].outChStr = 42;
                            }
                            else
                            {
                                // descriptors[i].outLnStr = outputBlobTensor.strideY;
                                // descriptors[i].outChStr = outputBlobTensor.strideZ;
                                descriptors[i].outLnStr = 42;
                                descriptors[i].outChStr = 42;
                            }

                            auto weight_4dshape = opIt->getInputTensor(1)->getShape();

                            descriptors[i].coeffChStrIn = weight_4dshape[2]*weight_4dshape[3]*weight_4dshape[4]*2;
                            int inChans = inputChannelsPadded;

                            descriptors[i].coeffChStrOut = radixX * radixY * inChans * 2 * 8; // (fp16)

                            for(unsigned j = 0; j != 32; j++)
                                desc.push_back(((unsigned *) &descriptors[i])[j]);
                        }

                    }

                }

                opIt->set<std::vector<unsigned>>("descriptors", desc);
            }
            else
            {
                opIt->set<unsigned>("SerialID", 4);
            }
        }
        else if(opIt->getOpType() == "Input")
        {

        }
        else if(opIt->getOpType() == "MatMul")
        {
            opIt->set<unsigned>("SerialID", 8);
        }
        else if(opIt->getOpType() == "MaxPool")
        {
            auto fp16_size = 2;

            if (opIt->hasAttr("NCE1_Compatible") && opIt->get<int>("NCE1_Compatible") )
            {
                // Get all attrs:
                auto splits_over_H = opIt->get<size_t>("NCE1_SplitsOverHeight");
                auto DPUmodeVector = opIt->get<std::vector<size_t>>("NCE1_Modes");
                auto splits_over_iC = opIt->get<size_t>("NCE1_SplitsOverInputChannels");
                auto inputChannelsPadded = opIt->get<std::size_t>("NCE1_InputChannelsPadded");
                auto outputChannelsPadded = opIt->get<std::size_t>("NCE1_OutputChannelsPadded");
                auto inputWidthPadded = opIt->get<std::size_t>("NCE1_InputWidthPadded");
                //auto outputWidthPadded = opIt->get<std::size_t>("NCE1_OutputWidthPadded");
                auto desc_count = opIt->get<std::size_t>("NCE1_DescriptorSplits");
                auto streamingMask = opIt->get<std::size_t>("NCE1_StreamingMask");

                auto input_lines_processed = opIt->get<std::vector<size_t>>("NCE1_InputLinesProcessed");
                auto output_lines_processed = opIt->get<std::vector<size_t>>("NCE1_OutputLinesProcessed");
                auto output_line_start = opIt->get<std::vector<size_t>>("NCE1_StartOutputLine");
                auto input_line_start = opIt->get<std::vector<size_t>>("NCE1_StartInputLine");

                //auto radixX = opIt->get<std::array<short unsigned, 2>>("kSize")[0];
                //auto radixY = opIt->get<std::array<short unsigned, 2>>("kSize")[1];

                opIt->set<unsigned>("SerialID", 34);    // To be moved?

                opIt->set<unsigned>("streamingMask", streamingMask );

                std::size_t total_size = opIt->getInputTensor(0)->getShape().totalSize();
                total_size *= inputChannelsPadded;
                total_size /= opIt->getInputTensor(0)->getShape()[2];
                opIt->set<unsigned>("inputSize", total_size*fp16_size);

                opIt->set<unsigned>("outputSize",
                    opIt->getOutputTensor(0)->getShape().totalSize()*fp16_size);

                opIt->set<unsigned>("concatOffset", 0); // Not Supported...
                opIt->set<unsigned>("unloadCMX", 0); // Not Supported...
                opIt->set<unsigned>("overwriteInput", 0); // Not Supported...
                opIt->set<unsigned>("CMXSize", 256*1024);  // Magic Number...
                opIt->set<unsigned>("reluSHVAcc", 0); // Not Supported...
                opIt->set<unsigned>("shvNegSlope", 0); // Not Supported...
                opIt->set<unsigned>("shvPosSlope", 1065353216); // Magic Number...
                opIt->set<unsigned>("desc_count", desc_count);


                std::vector<unsigned> desc;
                std::vector<cnnConvolutionPoolStructure> descriptors = std::vector<cnnConvolutionPoolStructure>(desc_count);

                int i = -1;
                for (unsigned h = 0; h < splits_over_H; ++h)
                {
                    for (unsigned oc = 0; oc < DPUmodeVector.size(); ++oc)
                    {
                        for (unsigned ic = 0; ic < splits_over_iC; ++ic)
                        {
                            ++i;

                            auto input_width = inputWidthPadded;
                            auto output_channels = outputChannelsPadded;

                            descriptors[i].dataBaseAddr = 2 * input_width * input_line_start[h];    // TODO: Calculate 3f0 (1008)

                            if( opIt->getInputTensor(0)->getOrder().isRowInterleaved() )
                            {
                                descriptors[i].dataBaseAddr *= inputChannelsPadded;    // TODO: Calculate 3f0 (1008)
                                // descriptors[i].dataLnStr = inputBlobTensor.strideY;
                                // descriptors[i].dataChStr = inputBlobTensor.strideZ;
                                descriptors[i].dataLnStr = 42;
                                descriptors[i].dataChStr = 42;
                            }
                            else
                            {
                                // descriptors[i].dataLnStr = inputBlobTensor.strideY;
                                // descriptors[i].dataChStr = inputBlobTensor.strideZ;
                                descriptors[i].dataLnStr = 42;
                                descriptors[i].dataChStr = 42;
                            }
                            descriptors[i].coeffBaseAddr = 0;
                            descriptors[i].biasBaseAddr = 0;
                            descriptors[i].scaleBaseAddr = 0;
                            //HACK FOR CONCAT
                            // descriptors[i].outBaseAddr = outputBlobTensor.strideZ * output_line_start[h];  // TODO: Calculate 3f0 (1008)
                            descriptors[i].outBaseAddr = 42;  // TODO: Calculate 3f0 (1008)

                            if( opIt->getOutputTensor(0)->getOrder().isRowInterleaved() )
                            {
                                descriptors[i].outBaseAddr *= output_channels;    // TODO: Calculate 3f0 (1008)
                                // descriptors[i].outLnStr = outputBlobTensor.strideY;
                                // descriptors[i].outChStr = outputBlobTensor.strideZ;
                                descriptors[i].outLnStr = 42;
                                descriptors[i].outChStr = 42;
                            }
                            else
                            {
                                // descriptors[i].outLnStr = outputBlobTensor.strideY;
                                // descriptors[i].outChStr = outputBlobTensor.strideZ;
                                descriptors[i].outLnStr = 42;
                                descriptors[i].outChStr = 42;
                            }

                            //int inChans = inputChannelsPadded;
                            for(unsigned j = 0; j != 32; j++)
                                desc.push_back(((unsigned *) &descriptors[i])[j]);
                                
                        }

                    }

                }

                opIt->set<std::vector<unsigned>>("descriptors", desc);
            }
            else
            {
                opIt->set<unsigned>("SerialID", 1);

                opIt->set<unsigned>("radixX",  opIt->get<std::array<short unsigned, 2>>("kSize")[0]);
                opIt->set<unsigned>("radixY",  opIt->get<std::array<short unsigned, 2>>("kSize")[1]);
                opIt->set<unsigned>("strideX",  opIt->get<std::array<unsigned short, 2>>("stride")[0]);
                opIt->set<unsigned>("strideY",  opIt->get<std::array<unsigned short, 2>>("stride")[1]);
                opIt->set<unsigned>("padX",  opIt->get<std::array<unsigned short, 4>>("padding")[0]);
                opIt->set<unsigned>("padY",  opIt->get<std::array<unsigned short, 4>>("padding")[2]);
                opIt->set<unsigned>("padStyle",  2);

            }
        }
        else if(opIt->getOpType() == "Multiply")
        {
            opIt->set<unsigned>("SerialID", 13);
        }
        else if(opIt->getOpType() == "Output")
        {

        }
        else if(opIt->getOpType() == "Prelu")
        {
            opIt->set<unsigned>("serialID", 10);
        }
        else if(opIt->getOpType() == "Relu")
        {
            opIt->set<unsigned>("opX", 0);
            opIt->set<unsigned>("strideX", 0);
            opIt->set<unsigned>("strideY", 0);
            opIt->set<unsigned>("SerialID", 6);
        }
        else if(opIt->getOpType() == "Reshape")
        {

        }
        else if(opIt->getOpType() == "Scale")
        {
            opIt->set<unsigned>("serialID", 15);
        }
        else if(opIt->getOpType() == "Softmax")
        {
            opIt->set<unsigned>("axis", 1);
            opIt->set<unsigned>("SerialID", 3);
        }
        else if(opIt->getOpType() == "Subtract")
        {
            opIt->set<unsigned>("SerialID", 12);
        }
        else
            std::cout << "Unsupported serialization operation " << opIt->getOpType() << std::endl;
    }
}
