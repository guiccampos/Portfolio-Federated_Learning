from utils.save_metrics import write_server_metrics, write_client_metrics
from utils.fl_types import TrainingInstructions, EvaluationInstructions, StopInstructions, ClientTestResult, ClientTrainResult
import argparse
import os
import numpy as np
import numpy.typing as npt
import torch

import mpi4py
# mpi4py.rc.initialize = False
# mpi4py.rc.finalize = False
from mpi4py import MPI

# FL Training Configs
DATASET = ""
PARTIONING = "IID"
SEED = 12345
BATCH_SIZE = 128
FL_ROUNDS = 1
LOCAL_EPOCHS = 10
LEARNING_RATE = 0.001
LR_GAMMA = 0.99
CENTRALIZED_EVALUATION = False

# SST2 dataset specific config
TOKEN_TO_TID_MAP ={}

# MPI Communication Tags:
TRAIN_RESULTS = 10_000
TEST_RESULTS = 20_000

def main():
    # Let PyTorch/ROCm settle before MPI init
    pre_mpi_cuda_available = torch.cuda.is_available()
    pre_mpi_device_count = torch.cuda.device_count()

    if not MPI.Is_initialized():
        MPI.Init()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    print(
        f"[RANK {rank}] pre-MPI-init torch view: "
        f"cuda_available={pre_mpi_cuda_available}, "
        f"device_count={pre_mpi_device_count}"
    )

    number_of_clients = size - 1
    args = parse_args()
    global DATASET, PARTIONING, SEED, BATCH_SIZE, FL_ROUNDS, LOCAL_EPOCHS, LEARNING_RATE, LR_GAMMA, CENTRALIZED_EVALUATION

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

    # Uses the same model as defined in the CGSV paper
    if DATASET == "MNIST":
        from models.MNIST_model import Net, train, test, add_update_to_model, model_to_numpy, load_model
        from dataset_loaders.MNIST_centralized_datasetloader import load_centralized_data
        if PARTIONING == "IID":
            from dataset_loaders.MNIST_IID_with_validation_datasetloader import load_data
        elif PARTIONING == "POW":
            from dataset_loaders.MNIST_POW_with_validation_datasetloader import load_data        
        elif PARTIONING == "CLA":
            from dataset_loaders.MNIST_CLA_with_validation_datasetloader import load_data
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
            from dataset_loaders.FASHION_MNIST_CLA_with_validation_datasetloader import load_data
        elif PARTIONING == "DIR":
            from dataset_loaders.FASHION_MNIST_DIR_with_validation_datasetloader import load_data

    # Uses the same model as defined in the CGSV paper
    elif DATASET == "CIFAR10":
        from models.CIFAR10_model import Net, train, test, add_update_to_model, model_to_numpy, load_model
        from dataset_loaders.CIFAR10_centralized_datasetloader import load_centralized_data
        if PARTIONING == "IID":
            from dataset_loaders.CIFAR10_IID_with_validation_datasetloader import load_data
        elif PARTIONING == "POW":
            from dataset_loaders.CIFAR10_POW_with_validation_datasetloader import load_data          
        elif PARTIONING == "CLA":
            from dataset_loaders.CIFAR10_CLA_with_validation_datasetloader import load_data
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
            from dataset_loaders.CIFAR10_CLA_with_validation_datasetloader import load_data
        elif PARTIONING == "DIR":
            from dataset_loaders.CIFAR10_DIR_with_validation_datasetloader import load_data

    elif DATASET == "SST2":
        from models.SST2_model import Net, train, test, add_update_to_model, load_model, model_to_numpy
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
            from dataset_loaders.SVHN_CLA_with_validation_datasetloader import load_data
        elif PARTIONING == "DIR":
            from dataset_loaders.SVHN_DIR_with_validation_datasetloader import load_data

    if rank == 0:
        print(f"[INIT] Hub-and-Spoke Architecture; size={size}, (num_clients={number_of_clients})")
        print(f"Server Initialized the model")

        if (DATASET == "SST2"):
            centralized_testloader = load_centralized_data(partition_id=0,
                                                           num_partitions=1,
                                                           global_token_to_tid=TOKEN_TO_TID_MAP,
                                                           max_length=64,
                                                           seed=SEED,
                                                           batch_size=BATCH_SIZE)
        else:
            centralized_testloader = load_centralized_data(partition_id=0,
                                                           num_partitions=1,
                                                           seed=SEED,
                                                           batch_size=BATCH_SIZE)        

        total_time = 0.0
        total_time_spent_in_server_computation = 0.0
        total_time_spent_in_server_blocking_communication = 0.0

        device = "cpu"

        net = Net().to(device)

        global_model = model_to_numpy(net)

        # Saving Metrics
        os.makedirs(args.output_dir, exist_ok=True)
        base = f"{number_of_clients}_clients_run_{args.run_id}_hub_and_spoke"
        prefix = F"Standalone_{DATASET}_{PARTIONING}"

        server_csv = os.path.join(args.output_dir, f"{prefix}_{base}_server_metrics.csv")
        client_csv = os.path.join(args.output_dir, f"{prefix}_{base}_client_metrics.csv")   
    
        for rnd in range(FL_ROUNDS):
            round_server_only_computation_time = 0.0
            round_duration = 0.0

            # Server sends training instructions to FL Clients
            round_start_time = MPI.Wtime()
            training_instructions: TrainingInstructions = {"phase": "training",
                                     "round": rnd,
                                     "local_epochs": LOCAL_EPOCHS,
                                     "has_model": (rnd == 0),
                                     "model": global_model if rnd == 0 else None,
                                     }

            # Interpret these times to bcast and scatter as collective blocking time
            train_instructions_broadcast_start_time = MPI.Wtime()
            comm.bcast(obj=training_instructions, root=0)
            train_instructions_broadcast_end_time = MPI.Wtime()
            time_spent_in_train_instructions_broadcast = train_instructions_broadcast_end_time - train_instructions_broadcast_start_time

            # Server collects the training results from clients
            client_results_train: list[ClientTrainResult] = []
            train_results_collection_start_time = MPI.Wtime()
            for cid in range(number_of_clients):
                result: ClientTrainResult = comm.recv(source=cid + 1, tag=TRAIN_RESULTS + rnd)
                client_results_train.append(result)
            train_results_collection_end_time = MPI.Wtime()
            time_spent_in_collecting_training_results = train_results_collection_end_time - train_results_collection_start_time

            # --- Independent mode: NO FedAvg ---
            # Each client trains its own model on its own partition.
            # Clients reset their weights after train(), so we broadcast each client's trained model
            # back during evaluation, and clients load THEIR OWN model before testing.
            models_by_client = {r["client_id"]: r["model_update"] for r in client_results_train}

            # No aggregation / no server-side model update / no server-side global test
            time_spent_aggregating_gradients = 0.0
            time_spent_updating_server_model = 0.0
            time_spent_testing_global_model = 0.0

            evaluation_instructions: EvaluationInstructions = {"phase": "evaluation", "round": rnd, "models_by_client": models_by_client}
            eval_instructions_broadcast_start_time = MPI.Wtime()
            comm.bcast(obj=evaluation_instructions, root=0)
            eval_instructions_broadcast_end_time = MPI.Wtime()
            time_spent_in_eval_instructions_broadcast = eval_instructions_broadcast_end_time - eval_instructions_broadcast_start_time

            client_testing_results: list[ClientTestResult] = []
            test_results_collection_start_time = MPI.Wtime()
            for cid in range(number_of_clients):
                result = comm.recv(source=cid + 1, tag=TEST_RESULTS + rnd)
                client_testing_results.append(result)
            test_results_collection_end_time = MPI.Wtime()
            time_spent_in_collecting_testing_results = test_results_collection_end_time - test_results_collection_start_time

            round_end_time = MPI.Wtime()
            round_duration = round_end_time - round_start_time

            round_clients_local_training_dataset_average_accuracy = float(np.mean([r["local_dataset_train_accuracy"] for r in client_testing_results])) if client_testing_results else 0.0
            if CENTRALIZED_EVALUATION == True:
                round_clients_local_testing_dataset_average_accuracy  = 0.0
                round_clients_central_testing_dataset_average_accuracy  = float(np.mean([r["central_dataset_test_accuracy"] for r in client_testing_results])) if client_testing_results else 0.0
            else:
                round_clients_local_testing_dataset_average_accuracy  = float(np.mean([r["local_dataset_test_accuracy"] for r in client_testing_results])) if client_testing_results else 0.0
                round_clients_central_testing_dataset_average_accuracy  = 0.0

            # No aggregation / no server-side model update / no server-side global test
            round_time_spent_in_server_blocking_communication = (time_spent_in_train_instructions_broadcast +
                                                                 time_spent_in_collecting_training_results + 
                                                                 time_spent_in_eval_instructions_broadcast + 
                                                                 time_spent_in_collecting_testing_results)
            
            round_time_spent_in_server_computation = (time_spent_aggregating_gradients +
                                                      time_spent_updating_server_model +
                                                      time_spent_testing_global_model)
            
            total_time_spent_in_server_computation += round_time_spent_in_server_computation
            total_time_spent_in_server_blocking_communication += round_time_spent_in_server_blocking_communication
            total_time += round_duration

            # ---- write server-level metrics ----
            write_server_metrics(
                path=server_csv,
                round=rnd,
                time_spent_aggregating_gradients=time_spent_aggregating_gradients,
                time_spent_updating_server_model=time_spent_updating_server_model,
                time_spent_testing_global_model=time_spent_testing_global_model,
                time_spent_in_train_instructions_broadcast=time_spent_in_train_instructions_broadcast,
                time_spent_in_collecting_training_results=time_spent_in_collecting_training_results,
                time_spent_in_eval_instructions_broadcast=time_spent_in_eval_instructions_broadcast,
                time_spent_in_collecting_testing_results=time_spent_in_collecting_testing_results,
                round_time_spent_in_server_computation=round_time_spent_in_server_computation,
                round_time_spent_in_server_blocking_communication=round_time_spent_in_server_blocking_communication,
                round_duration=round_duration,
                total_time_spent_in_server_computation=total_time_spent_in_server_computation,
                total_time_spent_in_server_blocking_communication=total_time_spent_in_server_blocking_communication,
                total_time=total_time,
                round_clients_local_training_dataset_average_accuracy=round_clients_local_training_dataset_average_accuracy,
                round_clients_local_testing_dataset_average_accuracy=round_clients_local_testing_dataset_average_accuracy,
                round_clients_central_testing_dataset_average_accuracy=round_clients_central_testing_dataset_average_accuracy,
                round_server_central_testing_dataset_loss=0.0,
                round_server_central_testing_dataset_accuracy=0.0,
            )

            # ---- write per-client metrics ----
            write_client_metrics(
                path=client_csv,
                round_idx=rnd,
                client_results_test=client_testing_results,
            )

            print(
                f"[ROUND {rnd}] avg_train_acc={round_clients_local_training_dataset_average_accuracy:.4f} | "
                f"avg_test_acc_local={round_clients_local_testing_dataset_average_accuracy:.4f} | "
                f"avg_test_acc_central={round_clients_central_testing_dataset_average_accuracy:.4f} | "
                f"round_time={round_duration:.3f}s"
            )

        # After iterating through all rounds, stops client loop
        if (PARTIONING == "CLA"):
            _ = load_data(
                partition_id=0,
                num_partitions=number_of_clients,
                seed=SEED,
                batch_size=BATCH_SIZE,
            )

            # Save the imbalance information
            # save_partition_summary_csv(f"{args.output_dir}/CLA_summary_{DATASET}.csv")
        comm.bcast({"phase": "stop"}, root=0)
        print(f"Server finished all {FL_ROUNDS} rounds. Total server-only-computation: {round_server_only_computation_time:.4f} seconds.")
        print(f"Server finished all {FL_ROUNDS} rounds. Total time: {total_time:.4f} seconds. Average time per round: {total_time/FL_ROUNDS:.4f} seconds.")

    else:
        client_id = rank - 1

        # Added this just for model validation
        # GPUs have some sort o per-GPU context limit. AMD GPUs 7900 have about 8 simultaneous context limit.
        # If I add more than 8 processes to the GPU, training stalls
        if number_of_clients == 1:
            device = "cuda:0" if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu"
        else:
            device = "cpu"
        print(device)

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
            net = Net().to(device)
            optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=LR_GAMMA)

            trainloader, _, testloader = load_data(
                partition_id=client_id,
                num_partitions=number_of_clients,
                global_token_to_tid=TOKEN_TO_TID_MAP,
                max_length=64,
                seed=SEED,
                batch_size=BATCH_SIZE,
            )
        else:
            net = Net().to(device)
            optimizer = torch.optim.SGD(net.parameters(), lr=LEARNING_RATE)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=LR_GAMMA)
            trainloader, _, testloader = load_data(partition_id=client_id,
                                                    num_partitions=number_of_clients,
                                                    seed=SEED,
                                                    batch_size=BATCH_SIZE,
                                                    )  
            local_dataset_train_loss = float("nan")
            local_dataset_train_accuracy = float("nan")



        print(
            f"[RANK {rank}] device={device} "
            f"cuda_available={torch.cuda.is_available()} "
            f"device_count={torch.cuda.device_count()} "
            f"current_device={(torch.cuda.current_device() if torch.cuda.is_available() else 'cpu')}"
        )
        if torch.cuda.is_available():
            print(f"[RANK {rank}] gpu_name={torch.cuda.get_device_name(0)}")

        while(True):
            # Clients collects the server instructions
            control = comm.bcast(obj=None, root=0)

            phase = control["phase"]
            
            if (phase == "training"):
                print(f"Client {client_id} initialized round {control['round']} training. Local training happening for {control['local_epochs']} epochs")

                if control.get("model") is not None:
                    load_model(net=net, parameters=control["model"])

                local_dataset_train_loss, local_dataset_train_accuracy, final_params = train(
                        net,
                        trainloader,
                        control["local_epochs"],
                        device,
                        optimizer=optimizer,
                        scheduler=scheduler,
                    )

                # Client sends only the necessary information
                client_results_train: ClientTrainResult = {
                    "client_id": client_id,
                    "model_update": final_params,
                    "local_training_dataset_size": len(trainloader.dataset)
                }

                comm.send(obj=client_results_train, dest=0, tag=TRAIN_RESULTS + control["round"])
                print(f"Client {client_id} sent results to server for round {control['round']}")

            elif (phase == "evaluation"):
                # Updating local model
                client_model = control["models_by_client"][client_id]
                load_model(net, client_model)
                
                if CENTRALIZED_EVALUATION == True:
                    print(f"Client {client_id} started evaluation on centralized test dataset")
                    central_dataset_test_loss, central_dataset_test_accuracy = test(net, centralized_testloader, device)
                    print(f"Client {client_id} finished evaluation on centralized test dataset")
                    print(f"Client {client_id} testing results on centralized dataset -- Test Loss: {central_dataset_test_loss}, Test Accuracy: {central_dataset_test_accuracy}")
                    
                    # Preparing payload with training and testing metrics
                    client_testing_results: ClientTestResult = {
                        "client_id": client_id,
                        "local_training_dataset_size": len(trainloader.dataset),
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
    p.add_argument("--run_id", type=int, default=1, )
    p.add_argument("--output_dir", type=str, default="results")
    return p.parse_args()

if __name__ == "__main__":
    main()
    # try:
    #     main()
    # finally:
    #     if MPI.Is_initialized() and not MPI.Is_finalized():
    #         try:
    #             MPI.COMM_WORLD.Barrier()
    #         except Exception:
    #             pass
    #         MPI.Finalize()
