#!/bin/bash

nohup python nlds/lorenz_rslds.py --out-dir "/mnt/fs5/nclkong/rslds/" --seed 1 --cuda --plot-dir "./plots/" >> outs/out_2 &
#nohup python nlds/lorenz_rslds.py --out-dir "/home/nclkong/nlsys/temp_params" --seed 1 --cuda --plot-dir "./plots/" >> outs/out &

