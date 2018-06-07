#ifndef MODEL_TENSOR_HPP_
#define MODEL_TENSOR_HPP_

#include "include/fathom/computation/tensor/shape.hpp"
#include "include/fathom/computation/model/element.hpp"
#include "include/fathom/computation/logger/logger.hpp"
#include "include/fathom/computation/tensor/tensor.hpp"

namespace mv
{

    class ModelTensor : public Tensor, public ComputationElement
    {

    public:

        ModelTensor(const Logger &logger, const string &name, const Shape &shape, DType dType, Order order);
        ModelTensor(const Logger &logger, const string &name, const ConstantTensor &other);
        ModelTensor(const ModelTensor &other);
        ModelTensor(const Logger &logger);
        //ModelTensor& operator=(const ModelTensor &other);
        virtual ~ModelTensor() = 0;

        virtual string toString() const;

    };

}

#endif // MODEL_TENSOR_HPP_