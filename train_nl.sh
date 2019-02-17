#!/bin/bash

nohup python nlds/lorenz_rsnlds.py --out-dir "/mnt/fs5/nclkong/rsnlds/" --seed 1 --cuda --plot-dir "./plots_nl/" >> outs/out_nl &

