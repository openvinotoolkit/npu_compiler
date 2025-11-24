# Side load strategy generator

Tool to generate C++ file with definitions of layer hashes from dumped MC stragies
To run
```
Required arguments:
  -i         Required. Path json files with dumped stragies
```
Example:
```
  python3 ./generate_mc_sideloader.py -i model_1_strategy.json model_2_strategy.json model_3_strategy.json
```
