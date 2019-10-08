// Define base tasks/OPs //
#include    "src/computation/op/op_registry.cpp"

// Kmb specific tasks //

#include    "src/target/kmb/tasks/def/barrier_task.cpp"
#include    "src/target/kmb/tasks/def/deallocate_task.cpp"
#include    "src/target/kmb/tasks/def/dma_task.cpp"
#include    "src/target/kmb/tasks/def/dpu_task.cpp"
#include    "src/target/kmb/tasks/def/upa_task.cpp"
#include    "src/target/kmb/tasks/def/placeholder_task.cpp"
#include    "src/target/kmb/tasks/def/sparsity_map.cpp"
#include    "src/target/kmb/tasks/def/weights_table.cpp"
#include    "src/target/kmb/tasks/def/implicit_concat.cpp"
