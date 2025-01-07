# BOFormer
for ft in RBF_0.05 matern52_0.05; do
    echo python q_value_transformer_testing_offpolicy.py \
    --f_num 2 --T 100 --function_type $ft --device 2 --model_episode 3000 --initial_sample 1  --record_idx 4 --n_positions 31 --n_layer 8 --n_head 4
done | xargs -d '\n' -P 2 -I {} sh -c '{}'

for ft in BCD; do
    echo python q_value_transformer_testing_offpolicy.py \
    --f_num 3 --T 100 --function_type $ft --device 0 --model_episode 1000 --initial_sample 1 --testing_episode 50 --n_positions 31
done | xargs -d '\n' -P 1 -I {} sh -c '{}'

for ft in lego materials; do
    echo python q_value_transformer_testing_offpolicy.py \
    --device 2 --f_num 2 --T 30 --function_type NERF_synthetic --model_episode 3000 --gamma 0.95 --n_positions 31 --NERF_scene $ft --perturb_noise_level 0.01 --observation_noise_level 0.0 --record_idx 4 --n_layer 8 --domain_size 1440
done | xargs -d '\n' -P 2 -I {} sh -c '{}'

for ft in chairs; do
    echo python q_value_transformer_testing_offpolicy.py \
    --device 2 --f_num 2 --T 30 --function_type NERF_synthetic --model_episode 3000 --gamma 0.95 --n_positions 31 --NERF_scene $ft --perturb_noise_level 0.01 --observation_noise_level 0.0 --record_idx 4 --n_layer 8 --testing_episode 65
done | xargs -d '\n' -P 1 -I {} sh -c '{}'

for ft in mic ship; do
    echo python q_value_transformer_testing_offpolicy.py \
    --device 2 --f_num 3 --T 30 --function_type NERF_synthetic_fnum_3 --model_episode 3400 --gamma 0.95 --n_positions 31 --NERF_scene $ft --perturb_noise_level 0.01 --observation_noise_level 0.0 --record_idx 5 --n_layer 8 --domain_size 1440
done | xargs -d '\n' -P 2 -I {} sh -c '{}'

# Baseline
for learner in qNEHVI qParEGO JES qHVKG; do
    for ft in AR ARa BC DR RBF_0.05 matern52_0.05; do
        echo python test_botorch_baseline.py \
        --f_num 2 --T 100 --function_type $ft --learner $learner --discrete 0
    done
done | xargs -d '\n' -P 1 -I {} sh -c '{}'

for learner in qNEHVI qParEGO JES qHVKG; do
    for ft in BCD; do
        echo python test_botorch_baseline.py \
        --f_num 3 --T 100 --function_type $ft --learner $learner
    done
done | xargs -d '\n' -P 2 -I {} sh -c '{}'

for learner in qNEHVI qParEGO JES qHVKG; do
    for ft in lego materials; do
        echo python test_botorch_baseline.py \
        --f_num 2 --T 30 --function_type NERF_synthetic --NERF_scene $ft --learner $learner --perturb_noise_level 0.01 --observation_noise_level 0.0 --domain_size 1440
    done
done | xargs -d '\n' -P 1 -I {} sh -c '{}'

for learner in qNEHVI qParEGO JES qHVKG; do
    for ft in chairs; do
        echo python test_botorch_baseline.py \
        --f_num 2 --T 30 --function_type NERF_synthetic --NERF_scene $ft --learner $learner --perturb_noise_level 0.01 --observation_noise_level 0.0 --episode 65
    done
done | xargs -d '\n' -P 1 -I {} sh -c '{}'

for learner in qNEHVI qParEGO JES qHVKG; do
    for ft in mic ship; do
        echo python test_botorch_baseline.py \
        --f_num 3 --T 30 --function_type NERF_synthetic_fnum_3 --NERF_scene $ft --learner $learner --perturb_noise_level 0.01 --observation_noise_level 0.0 --domain_size 1440
    done
done | xargs -d '\n' -P 1 -I {} sh -c '{}'

# NSGA-II
for ft in AR ARa BC DR RBF_0.05 matern52_0.05; do
    echo python test_NSGAII.py --f_num 2 --T 100 --function_type $ft --perturb_noise_level 0.1 --observation_noise_level 0.1 --domain_size 1000
done | xargs -d '\n' -P 2 -I {} sh -c '{}'

for ft in BCD; do
    echo python test_NSGAII.py --f_num 3 --T 100 --function_type $ft --perturb_noise_level 0.1 --observation_noise_level 0.1 --domain_size 1000
done | xargs -d '\n' -P 2 -I {} sh -c '{}'

for ft in lego materials; do
    echo python test_NSGAII.py --f_num 2 --T 30 --function_type NERF_synthetic --NERF_scene $ft --perturb_noise_level 0.01 --observation_noise_level 0.0 --domain_size 1440
done | xargs -d '\n' -P 2 -I {} sh -c '{}'

for ft in chairs; do
    echo python test_NSGAII.py --f_num 2 --T 30 --function_type NERF_synthetic --NERF_scene $ft --perturb_noise_level 0.01 --observation_noise_level 0.0 --episode 65
done | xargs -d '\n' -P 2 -I {} sh -c '{}'

for ft in mic ship; do
    echo python test_NSGAII.py --f_num 3 --T 30 --function_type NERF_synthetic_fnum_3 --NERF_scene $ft --perturb_noise_level 0.01 --observation_noise_level 0.0 --domain_size 1440
done | xargs -d '\n' -P 2 -I {} sh -c '{}'

# FSAF
for ft in AR ARa BC DR RBF_0.05 matern52_0.05; do
    echo python run_DQN_MAML.py --f_num 2 --T 100 --load_model_episode 500 --function_type $ft 
done | xargs -d '\n' -P 4 -I {} sh -c '{}'

for ft in BCD; do
    echo python run_DQN_MAML.py --f_num 3 --T 100 --load_model_episode 300 --function_type $ft 
done | xargs -d '\n' -P 3 -I {} sh -c '{}'

for ft in lego materials; do
    echo python run_DQN_MAML.py --device 0 --f_num 2 --T 30 --load_model_episode 500 --function_type NERF_synthetic --NERF_scene $ft --perturb_noise_level 0.01 --observation_noise_level 0.0 --domain_size 1440
done | xargs -d '\n' -P 1 -I {} sh -c '{}'

for ft in chairs; do
    echo python run_DQN_MAML.py --device 0 --f_num 2 --T 30 --load_model_episode 500 --function_type NERF_synthetic --NERF_scene $ft --perturb_noise_level 0.01 --observation_noise_level 0.0 --domain_size 1440 --episode 65
done | xargs -d '\n' -P 1 -I {} sh -c '{}'

for ft in mic ship; do
    echo python run_DQN_MAML.py --device 0 --f_num 3 --T 30 --load_model_episode 300 --function_type NERF_synthetic_fnum_3 --NERF_scene $ft --perturb_noise_level 0.01 --observation_noise_level 0.0 --domain_size 1440
done | xargs -d '\n' -P 1 -I {} sh -c '{}'

# optformer
for ft in AR ARa BC DR RBF_0.05 matern52_0.05; do
    echo python optformer_testing.py --f_num 2 --T 100 --function_type $ft 
done | xargs -d '\n' -P 4 -I {} sh -c '{}'

for ft in BCD; do
    echo python optformer_testing.py --f_num 3 --T 100 --function_type $ft 
done | xargs -d '\n' -P 3 -I {} sh -c '{}'

for ft in lego materials; do
    echo python optformer_testing.py --f_num 2 --T 30 --function_type NERF_synthetic --NERF_scene $ft --perturb_noise_level 0.01 --observation_noise_level 0.0
done | xargs -d '\n' -P 2 -I {} sh -c '{}'

for ft in chairs; do
    echo python optformer_testing.py --f_num 2 --T 30 --function_type NERF_synthetic --NERF_scene $ft --perturb_noise_level 0.01 --observation_noise_level 0.0 --episode 65
done | xargs -d '\n' -P 2 -I {} sh -c '{}'

for ft in mic ship; do
    echo python optformer_testing.py --f_num 3 --T 30 --function_type NERF_synthetic_fnum_3 --NERF_scene $ft --perturb_noise_level 0.01 --observation_noise_level 0.0
done | xargs -d '\n' -P 2 -I {} sh -c '{}'

# QT
for ft in AR ARa BC DR RBF_0.05 matern52_0.05; do
    echo python q_transformer_testing.py --f_num 2 --T 100 --function_type $ft --episode 100
done | xargs -d '\n' -P 1 -I {} sh -c '{}'

for ft in BCD; do
    echo python q_transformer_testing.py --f_num 3 --T 100 --function_type $ft --episode 100
done | xargs -d '\n' -P 2 -I {} sh -c '{}'

for ft in lego materials; do
    echo python q_transformer_testing.py --f_num 2 --T 30 --function_type NERF_synthetic --NERF_scene $ft --perturb_noise_level 0.01 --observation_noise_level 0.0 --domain_size 1440 --episode 100
done | xargs -d '\n' -P 2 -I {} sh -c '{}'

for ft in chairs; do
    echo python q_transformer_testing.py --f_num 2 --T 30 --function_type NERF_synthetic --NERF_scene $ft --perturb_noise_level 0.01 --observation_noise_level 0.0 --episode 65
done | xargs -d '\n' -P 2 -I {} sh -c '{}'

for ft in mic ship; do
    echo python q_transformer_testing.py --f_num 3 --T 30 --function_type NERF_synthetic_fnum_3 --NERF_scene $ft --perturb_noise_level 0.01 --observation_noise_level 0.0 --domain_size 1440 --episode 100
done | xargs -d '\n' -P 2 -I {} sh -c '{}'

# DT
for ft in AR ARa BC DR RBF_0.05 matern52_0.05; do
    echo python Decision_Transformer/testing_experiment.py --f_num 2 --function_type $ft --num_eval_episodes 50
done | xargs -d '\n' -P 1 -I {} sh -c '{}'

for ft in BCD; do
    echo python Decision_Transformer/testing_experiment.py --f_num 3 --function_type $ft --num_eval_episodes 50
done | xargs -d '\n' -P 2 -I {} sh -c '{}'

for ft in lego materials; do
    echo python Decision_Transformer/testing_experiment.py --f_num 2 --function_type NERF_synthetic --NERF_scene $ft --num_eval_episodes 100 --perturb_noise_level 0.01 --observation_noise_level 0.0 --domain_size 1440
done | xargs -d '\n' -P 1 -I {} sh -c '{}'

for ft in chairs; do
    echo python Decision_Transformer/testing_experiment.py --f_num 2 --function_type NERF_synthetic --NERF_scene $ft --num_eval_episodes 65 --perturb_noise_level 0.01 --observation_noise_level 0.0 --domain_size 1440
done | xargs -d '\n' -P 1 -I {} sh -c '{}'

for ft in mic ship; do
    echo python Decision_Transformer/testing_experiment.py --f_num 3 --function_type NERF_synthetic_fnum_3 --NERF_scene $ft --num_eval_episodes 100 --perturb_noise_level 0.01 --observation_noise_level 0.0 --domain_size 1440
done | xargs -d '\n' -P 1 -I {} sh -c '{}'

# training 
python optformer.py --f_num 2 
python optformer.py --f_num 3 
python q_transformer.py --f_num 2 
python q_transformer.py --f_num 3
python q_value_transformer_offpolicy.py --f_num 2 --update_step 10 
python q_value_transformer_offpolicy.py --f_num 3 --update_step 10 --device 1
