 #!/bin/bash
 for x in 0 1 2 3 4 5 6 7 8 9
 do 
    for y in 0 1 2 3 4 5 6 7 8 9
    do
        CUDA_VISIBLE_DEVICES=9 python src/navierstokes/train_ode_single.py --system navier_stokes --x-coord $x --y-coord $y;
    done
 done
 
 
