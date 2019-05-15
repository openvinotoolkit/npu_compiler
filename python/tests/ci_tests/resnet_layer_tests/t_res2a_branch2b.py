#!/usr/bin/python3

import sys
sys.path.append("..")
from test_helper import *

net_folder = "network_csvs/"
sch_folder = "POC_config_csvs/"

root_file = "res2a_branch2b"

model = generate_model(net_folder+root_file+"_MIG.csv")

graphfile, _, _ = compile_graphFile(model, 1, 1, sch_folder+"SOH_POC.csv", cmx=RESNET_CMX, emulator=False)
_, s_result, _ = compile_graphFile(model, 1, 1, sch_folder+"simple_POC.csv", cmx=MAX_UINT32)

t_result = execute_network(graphfile)
validate_files(s_result, t_result)