#ifndef MV_RUNTIME_MODEL_TENSOR_REFERENCE_
#define MV_RUNTIME_MODEL_TENSOR_REFERENCE_

#include <vector>
#include <cstdint>
#include "include/mcm/compiler/runtime/runtime_model_memory_location.hpp"
#include "include/mcm/compiler/runtime/runtime_model_dtypes.hpp"
#include "KeemBayFBSchema/compiledSchemas/graphfile_generated.h"

namespace mv
{

    struct RuntimeModelTensorReference
    {
        std::vector<unsigned> * dimensions;
        std::vector<unsigned> * strides;
        unsigned leadingOffset;
        unsigned trailingOffset;
        unsigned dataIndex;
        unsigned sparsityIndex;
        RuntimeModelMemoryLocation locale;
        RuntimeModelDType dtype;
        std::vector<unsigned> * quantScale;
        std::vector<unsigned> * quantZero;
        std::vector<unsigned> * quantShift;
    };

    std::vector<flatbuffers::Offset<MVCNN::TensorReference>> convertToFlatbuffer(std::vector<RuntimeModelTensorReference> * ref, flatbuffers::FlatBufferBuilder * fbb)
    {
        std::vector<flatbuffers::Offset<MVCNN::TensorReference>> toReturn;
        for(unsigned i = 0; i < ref->size(); ++i)
            toReturn.push_back(convertToFlatbuffer(ref->at(i), fbb));
        return toReturn;
    }


    flatbuffers::Offset<MVCNN::TensorReference> convertToFlatbuffer(RuntimeModelTensorReference * ref, flatbuffers::FlatBufferBuilder * fbb)
    {

        return CreateTensorReferenceDirect(fbb,
                                    ref->dimensions,
                                    ref->strides,
                                    ref->leadingOffset,
                                    ref->trailingOffset,
                                    CreateIndirectDataReference(fbb, ref->dataIndex, ref->sparsityIndex),
                                    ref->locale,
                                    ref->dtype,
                                    ref->quantScale,
                                    ref->quantZero,
                                    ref->quantShift);
    }
}


#endif //MV_RUNTIME_MODEL_TENSOR_REFERENCE_
