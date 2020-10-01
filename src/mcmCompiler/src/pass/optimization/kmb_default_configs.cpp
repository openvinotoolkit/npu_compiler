#include "include/mcm/pass/graphOptimizations/StrategyRegistry.hpp"

namespace mv {
namespace graphOptimizer {

//################## DEFAULT GLOBAL CONFIG'S FOR Kmb ####################
MV_OPTIMIZER_GLOBAL_CONFIG_REGISTRY(KmbOptGlCongigRefDev)
    .enter("referenceDevice").set("A0");

MV_OPTIMIZER_GLOBAL_CONFIG_REGISTRY(KmbOptGlCongigTotClusters)
    .enter("totalClusters").set(1);

MV_OPTIMIZER_GLOBAL_CONFIG_REGISTRY(KmbOptGlCongigClusterMem)
    .enter("clusterMemory").set(3584);

MV_OPTIMIZER_GLOBAL_CONFIG_REGISTRY(KmbOptGlCongigDpuPerCluster)
    .enter("dpuPerCluster").set(1);

MV_OPTIMIZER_GLOBAL_CONFIG_REGISTRY(KmbOptGlCongigDdrBw)
    .enter("ddrBandwidth").set(1);

MV_OPTIMIZER_GLOBAL_CONFIG_REGISTRY(KmbOptGlCongigSysClk)
    .enter("systemClockMhz").set(500);

//##################DEFAULT GLOBAL STRATEGIES FOR Kmb ###################

MV_OPTIMIZER_GLOBAL_STRATEGY_REGISTRY(KmbOptGlStrSpilling)
    .enter("forceSpilling").set(true);

MV_OPTIMIZER_GLOBAL_STRATEGY_REGISTRY(KmbOptGlStrStreaming)
    .enter("enableStreaming").set(true);

MV_OPTIMIZER_GLOBAL_STRATEGY_REGISTRY(KmbOptGlStrPipelining)
    .enter("enablePipelining").set(true);

MV_OPTIMIZER_GLOBAL_STRATEGY_REGISTRY(KmbOptGlStrBuffering)
    .enter("doubleBuffering").set(false);

MV_OPTIMIZER_GLOBAL_STRATEGY_REGISTRY(KmbOptGlStrSparsity)
    .enter("enableSparsity").set(false);

MV_OPTIMIZER_GLOBAL_STRATEGY_REGISTRY(KmbOptGlStrAtomicClustering)
    .enter("clusteringStrategy").set("Automatic");

//################# DEFAULT LAYER STRATEGIES FOR Kmb #####################

MV_OPTIMIZER_LAYER_STRATEGY_REGISTRY(KmbOptLayerInitConv)
    .enter("Conv")
    .registerSet("streamingStrategies").insert(vector<string>{"StreamOverH","StreamOverK","StreamOverN"})
    .registerSet("clusteringStrategies").insert(vector<string>{"Clustering","SplitOverH", "SplitOverK"})
    .registerSet("inputActivationSparsity").insert(true)
    .registerSet("outputActivationSparsity").insert(true)
    .registerSet("weightsSparsity").insert(true)
    .registerSet("pipelining").insert(true);

MV_OPTIMIZER_LAYER_STRATEGY_REGISTRY(KmbOptLayerInitDwConv)
    .enter("DepthwiseConv")
    .registerSet("streamingStrategies").insert(vector<string>{"StreamOverH", "StreamOverC"})
    .registerSet("ClusteringStrategies").insert(vector<string>{"Clustering","SplitOverH", "SplitOverK"})
    .registerSet("inputActivationSparsity").insert(false)
    .registerSet("outputActivationSparsity").insert(true)
    .registerSet("weightsSparsity").insert(false)
    .registerSet("pipelining").insert(true);

MV_OPTIMIZER_LAYER_STRATEGY_REGISTRY(KmbOptLayerInitMaxPool)
    .enter("MaxPool")
    .registerSet("streamingStrategies").insert(vector<string>{"StreamOverH"})
    .registerSet("clusteringStrategies").insert(vector<string>{"Clustering","SplitOverH","HKSwitch"})
    .registerSet("inputActivationSparsity").insert(false)
    .registerSet("outputActivationSparsity").insert(true)
    .registerSet("weightsSparsity").insert(false)
    .registerSet("pipelining").insert(false);

MV_OPTIMIZER_LAYER_STRATEGY_REGISTRY(KmbOptLayerInitEltwise)
    .enter("Eltwise")
    .registerSet("streamingStrategies").insert(vector<string>{"StreamOverH"})
    .registerSet("clusteringStrategies").insert(vector<string>{"Clustering","SplitOverH","HKSwitch"})
    .registerSet("inputActivationSparsity").insert(true)
    .registerSet("outputActivationSparsity").insert(true)
    .registerSet("weightsSparsity").insert(false)
    .registerSet("pipelining").insert(false);

MV_OPTIMIZER_LAYER_STRATEGY_REGISTRY(KmbOptLayerInitConcat)
    .enter("Concat")
    .registerSet("streamingStrategies").insert(vector<string>(0))
    .registerSet("clusteringStrategies").insert(vector<string>{"Clustering"})
    .registerSet("forceSpilling").insert(true)
    .registerSet("inputActivationSparsity").insert(false)
    .registerSet("outputActivationSparsity").insert(false)
    .registerSet("weightsSparsity").insert(false)
    .registerSet("pipelining").insert(false);

MV_OPTIMIZER_LAYER_STRATEGY_REGISTRY(KmbOptLayerInitInput)
    .enter("Input")
    .registerSet("streamingStrategies").insert(vector<string>(0))
    .registerSet("clusteringStrategies").insert(vector<string>{"Clustering", "SplitOverK", "SplitOverH", "SplitOverHOverlapped"})
    .registerSet("forceSpilling").insert(false)
    .registerSet("inputActivationSparsity").insert(false)
    .registerSet("outputActivationSparsity").insert(false)
    .registerSet("weightsSparsity").insert(false)
    .registerSet("pipelining").insert(false);

MV_OPTIMIZER_LAYER_STRATEGY_REGISTRY(KmbOptLayerInitImplicitInput)
    .enter("ImplicitInput")
    .registerSet("streamingStrategies").insert(vector<string>(0))
    .registerSet("clusteringStrategies").insert(vector<string>{"Clustering", "SplitOverK", "SplitOverH", "SplitOverHOverlapped"})
    .registerSet("forceSpilling").insert(false)
    .registerSet("inputActivationSparsity").insert(false)
    .registerSet("outputActivationSparsity").insert(false)
    .registerSet("weightsSparsity").insert(false)
    .registerSet("pipelining").insert(false);

MV_OPTIMIZER_LAYER_STRATEGY_REGISTRY(KmbOptLayerInitOutput)
    .enter("Output")
    .registerSet("streamingStrategies").insert(vector<string>(0))
    .registerSet("clusteringStrategies").insert(vector<string>{"Clustering", "SplitOverH", "SplitOverK"})
    .registerSet("forceSpilling").insert(true)
    .registerSet("inputActivationSparsity").insert(false)
    .registerSet("outputActivationSparsity").insert(false)
    .registerSet("weightsSparsity").insert(false)
    .registerSet("pipelining").insert(false);



//################# DEFAULT SW layers #####################
MV_OPTIMIZER_LAYER_STRATEGY_REGISTRY(KmbOptLayerInitDefault)
   .enter("Default")
   .registerSet("streamingStrategies").insert(vector<string>(0))
   .registerSet("clusteringStrategies").insert(vector<string>{"Clustering"})
   .registerSet("forceSpilling").insert(true)
   .registerSet("inputActivationSparsity").insert(false)
   .registerSet("outputActivationSparsity").insert(false)
   .registerSet("weightsSparsity").insert(false)
   .registerSet("pipelining").insert(false);

}
}
