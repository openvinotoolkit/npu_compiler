#ifndef MV_OP_HPP_
#define MV_OP_HPP_

#include <string>
#include <array>
#include <map>
#include "include/mcm/computation/model/model_element.hpp"
#include "include/mcm/tensor/shape.hpp"
#include "include/mcm/tensor/tensor.hpp"
#include "include/mcm/base/exception/op_error.hpp"
#include "include/mcm/base/exception/index_error.hpp"
#include "include/mcm/computation/model/iterator/data_context.hpp"
#include "include/mcm/computation/op/op_registry.hpp"

namespace mv
{

    class Op : public ModelElement
    {

        std::vector<Data::TensorIterator> inputs_;
        std::vector<Data::TensorIterator> outputs_;

    public:

        Op(ComputationModel& model, const std::string& opType, const std::string& name, 
            const std::vector<Data::TensorIterator>& inputs, std::initializer_list<std::pair<std::string, Attribute>> args = {});

        virtual ~Op();

        std::string getOpType() const;

        void setInputTensor(Data::TensorIterator tensor, std::size_t idx);

        Data::TensorIterator getInputTensor(std::size_t idx);
        Data::TensorIterator getInputTensor(const std::string& label);
        std::vector<Data::TensorIterator> getInputTensor();
        Data::TensorIterator getOutputTensor(std::size_t idx);
        Data::TensorIterator getOutputTensor(const std::string& label);
        std::vector<Data::TensorIterator> getOutputTensor();

        std::size_t inputSlots();
        std::size_t outputSlots();

        std::string getLogID() const override;

    };

}

#endif // COMPUTATION_OP_HPP_