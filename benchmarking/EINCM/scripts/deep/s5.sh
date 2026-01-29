#! /usr/bin/env bash
cd ../src

python -m experiments.e16\
    --config-path=./configs --config-name=main\
    dataset=deep\
    root_dir="/datasets/deep"\
    sequence_name=s5\
    des_n_events=700000\
    alpha=1000\
    beta=1000\
    gamma=0.0025\
    run_full_sequence=True\
    solver_params.n_repeat_solve=1\
    solver_params.theta_opt.maxiter=25\
    solver_params.theta_opt.n_extra_attempts.pyr_lvl_0=2\
    solver_params.theta_opt.n_extra_attempts.pyr_lvl_1=1\
    n_pyr_lvls=3\
    pyramid_bases=[4,4,4]\
    edge_extraction.canny.threshold_1=30\
    edge_extraction.canny.threshold_2=80\
    experiment_settings.solver.enable=True\
    experiment_settings.theta_evaluation.enable=True\
    experiment_settings.plot.enable=False\
    experiment_settings.theta_evaluation.print_eval_results_at_sample=False\
    experiment_settings.solver.checkpoints.enable=True\
    experiment_settings.solver.checkpoints.delete_after_final_save=True
