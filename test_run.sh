for seed in 123 231 321
do
    for env in GoToObjMazeS4 GoToObjMazeS7
    do
        python run_leap.py --seed $seed --horizon 10 --context_length 10 --sample_iteration 30 --epochs 200 --env $env --model_type 'naive' --batch_size 64
    done

    for env in GoToLocalS7N5 GoToLocalS8N7 
    do
        python run_leap.py --seed $seed --horizon 5 --context_length 5 --sample_iteration 10 --epochs 200 --env $env --model_type 'naive' --batch_size 64
    done

    for env in PickupLoc
    do
        python run_leap.py --seed $seed --horizon 5 --context_length 5 --sample_iteration 10 --epochs 200 --env $env --model_type 'naive' --batch_size 64
    done

    for env in GoToObjMazeS4R2Close
    do
        python run_leap.py --seed $seed --horizon 5 --context_length 5 --sample_iteration 10 --epochs 200 --env $env --model_type 'naive' --batch_size 64
    done
done
