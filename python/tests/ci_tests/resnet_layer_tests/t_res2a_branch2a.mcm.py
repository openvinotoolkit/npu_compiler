#!/usr/bin/python3

import sys, os
sys.path.append("..")
from test_helper import *

net_folder = "network_csvs/"
sch_folder = "POC_config_csvs/"
root_file = "res2a_branch2a"

model = generate_model(net_folder+root_file+"_MIG.csv")
graphfile, s_result, _ = compile_graphFile(model, 1, 1, sch_folder+"simple_POC.csv")

os.remove(graphfile)  # delete blob from previous run
graphfile, s_result, _ = compile_graphFile(model, 1, 1, sch_folder+"simple_POC.csv", cpp=True)
t_result = execute_network(graphfile)
validate_files(s_result, t_result)