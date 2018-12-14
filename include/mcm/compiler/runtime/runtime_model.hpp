#ifndef MV_RUNTIME_MODEL_
#define MV_RUNTIME_MODEL_

#include "include/mcm/compiler/runtime/runtime_model_header.hpp"
#include "include/mcm/compiler/runtime/tasks/runtime_model_task.hpp"
#include "include/mcm/compiler/runtime/runtime_model_barrier.hpp"
#include "include/mcm/compiler/runtime/runtime_model_binary_data.hpp"
#include <vector>

namespace mv
{
    struct RuntimeModel
    {
        RuntimeModelHeader * header;
        std::vector<std::vector<RuntimeModelTask>> * taskLists;
        std::vector<RuntimeModelBarrier> * barrierTable;
        std::vector<std::vector<RuntimeModelBinaryData>> * binaryData;
    };
}

#endif
