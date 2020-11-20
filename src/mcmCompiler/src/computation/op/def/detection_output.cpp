#include "include/mcm/computation/op/op_registry.hpp"

namespace mv
{

    namespace op_detection_output
    {

        static std::function<std::pair<bool, std::size_t>(const std::vector<Data::TensorIterator>&,
            const std::map<std::string, Attribute>&, std::string&)> inputCheckFcn =
            [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>& args,
            std::string& errMsg) -> std::pair<bool, std::size_t>
        {

            auto input = inputs[0];
            if (inputs.size() != 3)
            {
                std::stringstream err;
                err << "Incorrect number of inputs (must be 3): " << inputs.size();
                errMsg = err.str();
                return {false, 0};
            }

            auto keep_top_k = args.at("keep_top_k").get<int64_t>();
            if (keep_top_k < 0)
            {
                std::stringstream err;
                err << "keep_top_k must be a valid positive number.";
                errMsg = err.str();
                return {false, 0};
            }

            return {true, 0};

        };

        static std::function<void(const std::vector<Data::TensorIterator>&, const std::map<std::string, Attribute>&,
            std::vector<Tensor>&)> outputDefFcn =
            [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>& args, std::vector<Tensor>& outputs)
        {

            auto input = inputs[0];
            auto outputOrder = input->getOrder();
            auto inputShape = input->getShape();
            auto ndims = inputShape.ndims();
            mv::Shape outputShape(ndims);

            // Calculate output shape
            auto keep_top_k = args.at("keep_top_k").get<int64_t>();
            unsigned long int max_detections = keep_top_k;
            outputShape = {7,max_detections,1,1};

            outputs.emplace_back(":0",  outputShape, inputs[0]->getDType(), outputOrder);
        };
    }

    namespace op
    {
        MV_REGISTER_OP(DetectionOutput)
        .setInputs({"inputs"})
        .setOutputs({"output"})
        .setArg<int64_t>("num_classes")
        .setArg<int64_t>("keep_top_k")
        .setArg<double>("nms_threshold")
        .setArg<int64_t>("background_label_id")
        .setArg<int64_t>("top_k")
        .setArg<bool>("variance_encoded_in_target")
        .setArg<std::string>("code_type")
        .setArg<bool>("share_location")
        .setArg<double>("confidence_threshold")
        .setArg<bool>("clip_before_nms")
        .setArg<bool>("clip_after_nms")
        .setArg<int64_t>("decrease_label_id")
        .setArg<bool>("normalized")
        .setArg<int64_t>("input_height")
        .setArg<int64_t>("input_width")
        .setArg<double>("objectness_score")
        .setInputCheck(op_detection_output::inputCheckFcn)
        .setOutputDef(op_detection_output::outputDefFcn)
        .setTypeTrait({"executable", "exposed"})
        .setVariableInputNum(true);
    }

}
