from utils.utils_cffl import (
    FedAvgGradients,
    reconstruct_client_models,
    compute_CFFL_validation,
    compute_CFFL_reward,
    build_effective_reward_for_reset_client,
)
from utils.save_cffl_metrics import write_server_metrics, write_client_metrics
from utils.fl_types import TrainingInstructions, EvaluationInstructions, StopInstructions, ClientTestResult, ClientTrainResult
from collections import defaultdict
from mpi4py import MPI
import argparse
import os
import numpy as np
import numpy.typing as npt


# FL Training Configs
DATASET = ""
PARTIONING = "IID"
SEED = 12345
BATCH_SIZE = 128
FL_ROUNDS = 1
LOCAL_EPOCHS = 10
LEARNING_RATE = 0.001
LR_GAMMA = 0.99
GRAD_SPARSIFICATION_MODE = ""
CENTRALIZED_EVALUATION = False
EPS = 1e-12

# SST2 dataset specific config
TOKEN_TO_TID_MAP = {}

# CFFL Configs
PUNISHMENT_FACTOR = 1.00

# MPI Communication Tags:
TRAIN_RESULTS = 10_000
TEST_RESULTS = 20_000

# Reward ablation controls
REWARD_ALLOCATION_MODE = "paper_topk"   # "paper_topk" or "importance_only_topk"
KEEP_LOCAL_GRADIENT = True              # True = paper-like, False = ablation

def main():
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    number_of_clients = size - 1
    args = parse_args()
    global DATASET, PARTIONING, SEED, BATCH_SIZE, FL_ROUNDS, LOCAL_EPOCHS, LEARNING_RATE, LR_GAMMA, CENTRALIZED_EVALUATION
    global GRAD_SPARSIFICATION_MODE
    global PUNISHMENT_FACTOR
    global REWARD_ALLOCATION_MODE
    global KEEP_LOCAL_GRADIENT

    REWARD_ALLOCATION_MODE = args.reward_allocation_mode
    KEEP_LOCAL_GRADIENT = (args.keep_local_gradient == "true")
    DATASET = args.dataset
    PARTIONING = args.partitioning
    SEED = args.seed
    BATCH_SIZE = args.batch_size
    FL_ROUNDS = args.rounds
    LOCAL_EPOCHS  = int(args.local_epochs)
    LEARNING_RATE = args.learning_rate
    LR_GAMMA = args.lr_gamma
    centralized_evaluation = args.centralized_evaluation

    if centralized_evaluation == "true":
        CENTRALIZED_EVALUATION = True

    PUNISHMENT_FACTOR = args.punishment_factor

    GRAD_SPARSIFICATION_MODE = args.sparsification_mode

    # Uses the same model as defined in the CGSV paper
    if DATASET == "CIFAR10":
        from models.CIFAR10_model import Net, train, test, add_update_to_model, model_to_numpy, load_model
        from dataset_loaders.CIFAR10_centralized_datasetloader import load_centralized_data
        if PARTIONING == "IID":
            from dataset_loaders.CIFAR10_IID_with_validation_datasetloader import load_data
        elif PARTIONING == "POW":
            from dataset_loaders.CIFAR10_POW_with_validation_datasetloader import load_data          
        elif PARTIONING == "CLA":
            from dataset_loaders.CIFAR10_CLA_with_validation_datasetloader import load_data, get_partition_summary
        elif PARTIONING == "DIR":
            from dataset_loaders.CIFAR10_DIR_with_validation_datasetloader import load_data

    # https://www.kaggle.com/code/ektasharma/simple-cifar10-cnn-keras-code-with-88-accuracy
    elif DATASET == "Moderate_CIFAR10":
        from models.Moderate_CIFAR10_model import Net, train, test, add_update_to_model, load_model, model_to_numpy
        from dataset_loaders.CIFAR10_centralized_datasetloader import load_centralized_data
        if PARTIONING == "IID":
            from dataset_loaders.CIFAR10_IID_with_validation_datasetloader import load_data
        elif PARTIONING == "POW":
            from dataset_loaders.CIFAR10_POW_with_validation_datasetloader import load_data   
        elif PARTIONING == "CLA":
            from dataset_loaders.CIFAR10_CLA_with_validation_datasetloader import load_data, get_partition_summary
        elif PARTIONING == "DIR":
            from dataset_loaders.CIFAR10_DIR_with_validation_datasetloader import load_data

    # Uses the same model as defined in the CGSV paper
    elif DATASET == "MNIST":
        from models.MNIST_model import Net, train, test, add_update_to_model, model_to_numpy, load_model
        from dataset_loaders.MNIST_centralized_datasetloader import load_centralized_data
        if PARTIONING == "IID":
            from dataset_loaders.MNIST_IID_with_validation_datasetloader import load_data
        elif PARTIONING == "POW":
            from dataset_loaders.MNIST_POW_with_validation_datasetloader import load_data        
        elif PARTIONING == "CLA":
            from dataset_loaders.MNIST_CLA_with_validation_datasetloader import load_data, get_partition_summary
        elif PARTIONING == "DIR":
            from dataset_loaders.MNIST_DIR_with_validation_datasetloader import load_data

    # Uses the same model as defined in the FedProb paper
    elif DATASET == "FASHION_MNIST":
        from models.FASHION_MNIST_model import Net, train, test, add_update_to_model, model_to_numpy, load_model
        from dataset_loaders.FASHION_MNIST_centralized_datasetloader import load_centralized_data
        if PARTIONING == "IID":
            from dataset_loaders.FASHION_MNIST_IID_with_validation_datasetloader import load_data
        elif PARTIONING == "POW":
            from dataset_loaders.FASHION_MNIST_POW_with_validation_datasetloader import load_data
        elif PARTIONING == "CLA":
            from dataset_loaders.FASHION_MNIST_CLA_with_validation_datasetloader import load_data, get_partition_summary
        elif PARTIONING == "DIR":
            from dataset_loaders.FASHION_MNIST_DIR_with_validation_datasetloader import load_data

    elif DATASET == "SST2":
        from models.SST2_model import Net, train, test, add_update_to_model, model_to_numpy, load_model
        from dataset_loaders.SST2_centralized_datasetloader import load_centralized_data
        if PARTIONING == "IID":
            from dataset_loaders.SST2_IID_with_validation_datasetloader import build_global_token_tid_map, load_data
        elif PARTIONING == "POW":
            from dataset_loaders.SST2_POW_with_validation_datasetloader import build_global_token_tid_map, load_data

        if rank == 0:
            global TOKEN_TO_TID_MAP
            TOKEN_TO_TID_MAP = build_global_token_tid_map(seed=SEED, embed_num=20_000)
        else:
            TOKEN_TO_TID_MAP = None

        TOKEN_TO_TID_MAP = comm.bcast(TOKEN_TO_TID_MAP, root=0)
        
    # Uses same model as in "Federated Learning on Non-IID Data Silos: An Experimental Study" paper
    elif DATASET == "SVHN":
        from models.SVHN_model import Net, train, test, add_update_to_model, load_model, model_to_numpy
        from dataset_loaders.SVHN_centralized_datasetloader import load_centralized_data
        if PARTIONING == "IID":
            from dataset_loaders.SVHN_IID_with_validation_datasetloader import load_data
        elif PARTIONING == "POW":
            from dataset_loaders.SVHN_POW_with_validation_datasetloader import load_data
        elif PARTIONING == "CLA":
            from dataset_loaders.SVHN_CLA_with_validation_datasetloader import load_data, get_partition_summary
        elif PARTIONING == "DIR":
            from dataset_loaders.SVHN_DIR_with_validation_datasetloader import load_data

    if rank == 0:
        print(f"[INIT] Hub-and-Spoke Architecture; size={size}, (num_clients={number_of_clients})")
        print(f"Server Initialized the model")

        net = Net()

        if (DATASET == "SST2"):
            _, validation_dataset, _ = load_data(partition_id=0,
                                                 num_partitions=number_of_clients,
                                                 global_token_to_tid=TOKEN_TO_TID_MAP,
                                                 max_length=64,
                                                 seed=SEED,
                                                 batch_size=BATCH_SIZE)
            
            centralized_testloader = load_centralized_data(partition_id=0,
                                                           num_partitions=1,
                                                           global_token_to_tid=TOKEN_TO_TID_MAP,
                                                           max_length=64,
                                                           seed=SEED,
                                                           batch_size=BATCH_SIZE)
        else:
            _, validation_dataset, _ = load_data(partition_id=0,
                                                 num_partitions=number_of_clients,
                                                 seed=SEED,
                                                 batch_size=BATCH_SIZE)
            
            centralized_testloader = load_centralized_data(partition_id=0,
                                                           num_partitions=1,
                                                           seed=SEED,
                                                           batch_size=BATCH_SIZE)

        total_time = 0.0
        total_time_spent_in_server_blocking_communication = 0.0
        total_time_spent_in_server_computation = 0.0
        contribution_c_vals = defaultdict(dict)
        contribution_c_vals = {
            client_id: 1.0 / number_of_clients for client_id in range(number_of_clients)
        }

        device = "cpu"
        server_model = model_to_numpy(net)
        client_models: dict[int, list[npt.NDArray[np.generic]]] = {}

        # Sets clients model to initialize as the server model
        for client_id in range(number_of_clients):
             client_models[client_id] = [w.copy() for w in server_model]


        # Saving Metrics
        os.makedirs(args.output_dir, exist_ok=True)
        base = f"{number_of_clients}_clients_run_{args.run_id}_hub_and_spoke"
        prefix = f"CFFL_{DATASET}_{PARTIONING}"

        server_csv = os.path.join(args.output_dir, f"{prefix}_{base}_server_metrics.csv")
        client_csv = os.path.join(args.output_dir, f"{prefix}_{base}_client_metrics.csv")
        
        for rnd in range(FL_ROUNDS):
            round_duration = 0.0

            # Server sends training instructions to FL Clients
            lr_rnd = LEARNING_RATE * (LR_GAMMA ** rnd)
            round_start_time = MPI.Wtime()
            training_instructions: TrainingInstructions = {"phase": "training",
                                     "round": rnd,
                                     "local_epochs": LOCAL_EPOCHS,
                                     "learning_rate": lr_rnd,
                                     "has_model": (rnd == 0),
                                     "model": server_model if rnd == 0 else None,
                                     }
            
            train_instructions_broadcast_start_time = MPI.Wtime()
            comm.bcast(obj=training_instructions, root=0)
            train_instructions_broadcast_end_time = MPI.Wtime()
            time_spent_in_train_instructions_broadcast = train_instructions_broadcast_end_time - train_instructions_broadcast_start_time

            # Server collects the training results from clients
            # Each client result is: client_results_train = {"client_id": ... , "model_update": ... , "local_training_dataset_size"}
            client_results_train: list[ClientTrainResult] = []
            t0 = MPI.Wtime()
            for cid in range(number_of_clients):
                result: ClientTrainResult = comm.recv(source=cid + 1, tag=TRAIN_RESULTS + rnd)
                client_results_train.append(result)
            t1 = MPI.Wtime()
            time_spent_in_collecting_training_results = t1 - t0

            if (PARTIONING == "CLA"):
                classes_per_client = [int(client["class_j"]) for client in client_results_train]
                max_class_count = max(classes_per_client)
            else:
                max_class_count = 1

            # Aggregates the gradients using FedAvg (u_(N,t))
            t0 = MPI.Wtime()
            aggregated_gradient = FedAvgGradients(client_results=client_results_train, partitioning=PARTIONING, max_class_count=max_class_count)
            t1 = MPI.Wtime()
            time_spent_aggregating_gradients = t1 - t0

            # Forward pass on the Server Model
            t0 = MPI.Wtime()
            server_model = add_update_to_model(server_model, aggregated_gradient)
            load_model(net, server_model)
            t1 = MPI.Wtime()
            time_spent_updating_server_model = t1 - t0

            # CFFL Build Client models
            t0 = MPI.Wtime()
            client_models = reconstruct_client_models(client_results=client_results_train,
                                                      client_models=client_models,
                                                      add_update_to_model=add_update_to_model)
            t1 = MPI.Wtime()
            time_spent_reconstructing_client_models = t1 - t0

            # CONDUCT CFFL validation
            (vloss_values,
             vacc_values, 
             sum_vacc_val,
             contribution_c_vals,
             time_spent_testing_client_models,
             time_spent_computing_contribution_values,
             time_spent_normalizing_contribution_values) = compute_CFFL_validation(client_results=client_results_train,
                                                                                   client_models=client_models,
                                                                                   load_model=load_model,
                                                                                   net=net,
                                                                                   test=test,
                                                                                   validation_dataset=validation_dataset,
                                                                                   device=device,
                                                                                   punishment_factor=PUNISHMENT_FACTOR,
                                                                                   contribution_c_vals=contribution_c_vals,
                                                                                   eps=EPS,)

            # Compute sparsified rewards ONCE for all clients
            (v_by_client,
                discounted_layers_by_client,
                n_j_vals,
                max_n_j_val,
                n_ratio_by_client,
                q_by_client,
                K_by_client,
                v_non_zero_frac_by_client,
                v_sparsity_by_client,
                time_spent_in_reward_preparation,
                time_spent_computing_sparsified_gradients) = compute_CFFL_reward(
                    client_results=client_results_train,
                    aggregated_gradient=aggregated_gradient,
                    contribution_c_vals=contribution_c_vals,
                    eps=EPS,
                    mode=GRAD_SPARSIFICATION_MODE,
                    partitioning=PARTIONING,
                    max_class_count=max_class_count,
                    reward_allocation_mode=REWARD_ALLOCATION_MODE,
                )

            client_models_before_round = {
                cid: [w.copy() for w in client_models[cid]]
                for cid in range(number_of_clients)
            }

            train_result_by_id = {r["client_id"]: r for r in client_results_train}

            reward_update_by_client: dict[int, list[npt.NDArray[np.generic]]] = {}
            for cid in range(number_of_clients):
                reward_update_by_client[cid] = build_effective_reward_for_reset_client(
                    allocated_update=v_by_client[cid],
                    local_update=train_result_by_id[cid]["model_update"],
                    self_ratio=n_ratio_by_client[cid],
                    emulate_keep_local_gradient=True,   # regular CFFL
                )

            for cid in range(number_of_clients):
                client_models[cid] = add_update_to_model(
                    client_models_before_round[cid],
                    reward_update_by_client[cid],
                )

            evaluation_instructions: EvaluationInstructions = {"phase": "evaluation", "round": rnd}
            t0 = MPI.Wtime()
            comm.bcast(obj=evaluation_instructions, root=0)
            t1 = MPI.Wtime()
            time_spent_in_eval_instructions_broadcast = t1 - t0

            t0 = MPI.Wtime()
            scatter_payload_eval: list[npt.NDArray[np.generic]] = []
            scatter_payload_eval.append(None)  # for the server
            for cid in range(number_of_clients):
                if cid in v_by_client:
                    scatter_payload_eval.append(reward_update_by_client[cid])
                else:
                    zero_update = [np.zeros_like(w) for w in server_model]
                    scatter_payload_eval.append(zero_update)

            t1 = MPI.Wtime()
            time_spent_constructing_rewards_payload = t1 - t0

            t0 = MPI.Wtime()
            _ = comm.scatter(sendobj=scatter_payload_eval, root=0)
            t1 = MPI.Wtime()
            time_spent_in_scattering_rewards = t1 - t0

            print(f"Server started evaluation on centralized test dataset")
            t0 = MPI.Wtime()
            server_central_dataset_test_loss, server_central_dataset_test_accuracy = test(net, centralized_testloader, device)
            t1 = MPI.Wtime()
            time_spent_testing_global_model = t1 - t0
            print(f"Server finished evaluation on centralized test dataset")

            client_testing_results: list[ClientTestResult] = []
            t0 = MPI.Wtime()
            for cid in range(number_of_clients):
                result = comm.recv(source=cid + 1, tag=TEST_RESULTS + rnd)
                client_testing_results.append(result)
            t1 = MPI.Wtime()
            time_spent_in_collecting_testing_results = t1 - t0

            round_end_time = MPI.Wtime()
            round_duration = round_end_time - round_start_time
            round_clients_local_train_average_accuracy = float(np.mean([r["local_dataset_train_accuracy"] for r in client_testing_results])) if client_testing_results else 0.0
            if CENTRALIZED_EVALUATION == True:
                round_clients_local_test_average_accuracy  = 0.0
                round_clients_central_test_average_accuracy  = float(np.mean([r["central_dataset_test_accuracy"] for r in client_testing_results])) if client_testing_results else 0.0
            else:
                round_clients_local_test_average_accuracy  = float(np.mean([r["local_dataset_test_accuracy"] for r in client_testing_results])) if client_testing_results else 0.0
                round_clients_central_test_average_accuracy  = 0.0
           
            round_time_spent_in_server_blocking_communication = (time_spent_in_train_instructions_broadcast +
                                                                 time_spent_in_collecting_training_results + 
                                                                 time_spent_in_eval_instructions_broadcast + 
                                                                 time_spent_in_scattering_rewards + 
                                                                 time_spent_in_collecting_testing_results)

            round_time_spent_in_server_computation = (time_spent_aggregating_gradients +
                                                      time_spent_updating_server_model +
                                                      time_spent_reconstructing_client_models +
                                                      time_spent_testing_client_models +
                                                      time_spent_computing_contribution_values +
                                                      time_spent_normalizing_contribution_values +
                                                      time_spent_in_reward_preparation +
                                                      time_spent_computing_sparsified_gradients + 
                                                      time_spent_constructing_rewards_payload + 
                                                      time_spent_testing_global_model)
            
            total_time_spent_in_server_computation += round_time_spent_in_server_computation
            total_time_spent_in_server_blocking_communication += round_time_spent_in_server_blocking_communication
            total_time += round_duration
            
            # ---- write server-level metrics ----
            write_server_metrics(
                path=server_csv,
                round_idx=rnd,
                time_spent_aggregating_gradients=time_spent_aggregating_gradients,
                time_spent_updating_server_model=time_spent_updating_server_model,
                time_spent_reconstructing_client_models=time_spent_reconstructing_client_models,
                time_spent_testing_client_models=time_spent_testing_client_models,
                time_spent_computing_contribution_values=time_spent_computing_contribution_values,
                time_spent_normalizing_contribution_values=time_spent_normalizing_contribution_values,
                time_spent_in_reward_preparation=time_spent_in_reward_preparation,
                time_spent_computing_sparsified_gradients=time_spent_computing_sparsified_gradients,
                time_spent_constructing_rewards_payload=time_spent_constructing_rewards_payload,
                time_spent_testing_global_model=time_spent_testing_global_model,
                time_spent_in_train_instructions_broadcast=time_spent_in_train_instructions_broadcast,
                time_spent_in_collecting_training_results=time_spent_in_collecting_training_results,
                time_spent_in_eval_instructions_broadcast=time_spent_in_eval_instructions_broadcast,
                time_spent_in_scattering_rewards=time_spent_in_scattering_rewards,
                time_spent_in_collecting_testing_results=time_spent_in_collecting_testing_results,
                round_time_spent_in_server_computation=round_time_spent_in_server_computation,
                round_time_spent_in_server_blocking_communication=round_time_spent_in_server_blocking_communication,
                round_duration=round_duration,
                total_time_spent_in_server_computation=total_time_spent_in_server_computation,
                total_time_spent_in_server_blocking_communication=total_time_spent_in_server_blocking_communication,
                total_time=total_time,
                round_clients_local_training_dataset_average_accuracy=round_clients_local_train_average_accuracy,
                round_clients_local_testing_dataset_average_accuracy=round_clients_local_test_average_accuracy,
                round_clients_central_testing_dataset_average_accuracy=round_clients_central_test_average_accuracy,
                round_server_central_testing_dataset_loss=server_central_dataset_test_loss,
                round_server_central_testing_dataset_accuracy=server_central_dataset_test_accuracy,
            )

            # ---- write per-client metrics ----
            write_client_metrics(
                path=client_csv,
                round_idx=rnd,
                client_results_test=client_testing_results,
                contribution_c_vals=contribution_c_vals,
                local_epochs=LOCAL_EPOCHS,
                learning_rate=lr_rnd,
                vacc_values=vacc_values,
                n_j_vals=n_j_vals,
                max_n_j_val=max_n_j_val,
                q_by_client=q_by_client,
                K_by_client=K_by_client,
                reward_non_zero_frac_by_client=v_non_zero_frac_by_client,
                reward_sparsity_by_client=v_sparsity_by_client,
            )

        # After iterating through all rounds, stops client loop
        comm.bcast({"phase": "stop"}, root=0)
        print(f"Server finished all {FL_ROUNDS} rounds. Round server-only-computation: {round_time_spent_in_server_computation:.4f} seconds.")
        print(f"Server finished all {FL_ROUNDS} rounds. Total time: {total_time:.4f} seconds. Average time per round: {total_time/FL_ROUNDS:.4f} seconds.")

    else:
        client_id = rank - 1
        device = "cpu"

        if CENTRALIZED_EVALUATION == True:
            if DATASET == "SST2":
                centralized_testloader = load_centralized_data(
                    partition_id=0,
                    num_partitions=1,
                    global_token_to_tid=TOKEN_TO_TID_MAP,
                    max_length=64,
                    seed=SEED,
                    batch_size=BATCH_SIZE,
                )
            else:
                centralized_testloader = load_centralized_data(
                    partition_id=0,
                    num_partitions=1,
                    seed=SEED,
                    batch_size=BATCH_SIZE,
                )

        if (DATASET == "SST2"):
            net = Net()             
            trainloader, _, testloader = load_data(partition_id=client_id,
                                                num_partitions=number_of_clients,
                                                global_token_to_tid=TOKEN_TO_TID_MAP,
                                                max_length=64, 
                                                seed=SEED, 
                                                batch_size=BATCH_SIZE)
        else:
            net = Net()
            trainloader, _, testloader = load_data(partition_id=client_id,
                                                num_partitions=number_of_clients,
                                                seed=SEED,
                                                batch_size=BATCH_SIZE)       

        if (PARTIONING == "CLA"):
            partition_summary = get_partition_summary()
            local_class_count = int(partition_summary[client_id]["num_classes"])


        while(True):
            # Clients collects the server instructions
            control = comm.bcast(obj=None, root=0)

            phase = control["phase"]

            if (phase == "training"):
                print(f"Client {client_id} initialized round {control['round']} training. Local training happening for {control['local_epochs']} epochs")

                if control.get("has_model", False):
                    # rnd == 0 : everyone receives the same server model
                    load_model(net=net, parameters=control["model"])
                lr_rnd = control["learning_rate"]
                local_dataset_train_loss, local_dataset_train_accuracy, model_update = train(net, trainloader, control["local_epochs"], device, lr=lr_rnd)

                # Client sends only the necessary information
                client_results_train: ClientTrainResult = {
                    "client_id": client_id,
                    "model_update": model_update,
                    "local_training_dataset_size": len(trainloader.dataset),
                    "class_j": local_class_count if (PARTIONING == "CLA") else None
                }

                comm.send(obj=client_results_train, dest=0, tag=TRAIN_RESULTS + control["round"])
                print(f"Client {client_id} sent results to server for round {control['round']}")

            elif (phase == "evaluation"):
                # Updating local model
                reward_update = comm.scatter(sendobj=None, root=0)

                current_params = model_to_numpy(net)   # reset model base
                updated_params = add_update_to_model(current_params, reward_update)
                load_model(net, updated_params)
                print(f"Client {client_id} updated its local model")
                
                if CENTRALIZED_EVALUATION == True:
                    print(f"Client {client_id} started evaluation on centralized test dataset")
                    central_dataset_test_loss, central_dataset_test_accuracy = test(net, centralized_testloader, device)
                    print(f"Client {client_id} finished evaluation on centralized test dataset")
                    print(f"Client {client_id} testing results on centralized dataset -- Test Loss: {central_dataset_test_loss}, Test Accuracy: {central_dataset_test_accuracy}")
                    
                    # Preparing payload with training and testing metrics
                    client_testing_results: ClientTestResult = {
                        "client_id": client_id,
                        "local_training_dataset_size": len(trainloader.dataset),
                        "class_j": local_class_count if (PARTIONING == "CLA") else None,
                        "local_dataset_train_loss": local_dataset_train_loss,
                        "local_dataset_train_accuracy": local_dataset_train_accuracy,
                        "local_dataset_test_loss": 0,
                        "local_dataset_test_accuracy": 0,
                        "local_testing_dataset_size": len(testloader.dataset),
                        "central_dataset_test_loss": central_dataset_test_loss,
                        "central_dataset_test_accuracy": central_dataset_test_accuracy,
                        "corrupted_labels": False,
                        "corruption_fraction": 0.0
                    }

                else:
                    # Preparing payload with training and testing metrics
                    print(f"Client {client_id} started evaluation on local test dataset")
                    local_dataset_test_loss, local_dataset_test_accuracy = test(net, testloader, device)
                    print(f"Client {client_id} finished evaluation on local test dataset")
                    print(f"Client {client_id} testing results on local dataset -- Test Loss: {local_dataset_test_loss}, Test Accuracy: {local_dataset_test_accuracy}")
                    client_testing_results: ClientTestResult = {
                        "client_id": client_id,
                        "local_training_dataset_size": len(trainloader.dataset),
                        "local_dataset_train_loss": local_dataset_train_loss,
                        "local_dataset_train_accuracy": local_dataset_train_accuracy,
                        "local_dataset_test_loss": local_dataset_test_loss,
                        "local_dataset_test_accuracy": local_dataset_test_accuracy,
                        "local_testing_dataset_size": len(testloader.dataset),
                        "central_dataset_test_loss": 0,
                        "central_dataset_test_accuracy": 0,
                        "corrupted_labels": False,
                        "corruption_fraction": 0.0
                    }

                comm.send(obj=client_testing_results, dest=0, tag=TEST_RESULTS + control["round"])
                print(f"Client {client_id} sent testing results to server")

            elif phase == "stop":
                break

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="MNIST")
    p.add_argument("--partitioning", type=str, default="IID")
    p.add_argument("--alpha", type=float, default=1.65911332899)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--rounds", type=int, default=200,)
    p.add_argument("--local_epochs", type=int, default=10)
    p.add_argument("--learning_rate", type=float, default=0.001)
    p.add_argument("--lr_gamma", type=float, default=0.99)
    p.add_argument("--centralized_evaluation", type=str, default="false")
    p.add_argument("--sparsification_mode", type=str, default="topk")
    p.add_argument("--punishment_factor", type=float, default=1)
    p.add_argument("--run_id", type=int, default=1, )
    p.add_argument("--output_dir", type=str, default="results")
    p.add_argument("--reward_allocation_mode", type=str, default="paper_topk")
    p.add_argument("--keep_local_gradient", type=str, default="true")
    return p.parse_args()

if __name__ == "__main__":
    main()
