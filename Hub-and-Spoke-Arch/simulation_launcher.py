from utils.save_metrics import write_server_metrics ,write_client_metrics
from utils.fl_types import TrainingInstructions, EvaluationInstructions, StopInstructions, ClientTestResult, ClientTrainResult
from aggregation_protocols.FederatedAveragingGradients import FedAvgGradients
from collections import defaultdict
from mpi4py import MPI
import argparse
import os
import numpy as np
import numpy.typing as npt

# FL Training Configs
DATASET=""
PARTIONING="IID"
SEED = 12345
BATCH_SIZE = 128
FL_ROUNDS = 1
LOCAL_EPOCHS = 10
LEARNING_RATE = 0.001
CENTRALIZED_EVALUATION = False

# SST2 dataset specific config
TOKEN_TO_TID_MAP ={}

# CGSV Configs
ALPHA = 0.95
BETA = 1.0
EPS = 1e-12

# MPI Communication Tags:
TRAIN_RESULTS = 10_000
TEST_RESULTS = 20_000

def main():
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    number_of_clients = size - 1

    args = parse_args()
    global DATASET
    global PARTIONING
    global SEED
    global BATCH_SIZE
    global FL_ROUNDS
    global LOCAL_EPOCHS
    global LEARNING_RATE
    global CENTRALIZED_EVALUATION

    DATASET = args.dataset
    PARTIONING = args.partitioning
    SEED = args.seed
    BATCH_SIZE = args.batch_size
    FL_ROUNDS = args.rounds
    LOCAL_EPOCHS  = int(args.local_epochs)
    LEARNING_RATE = args.learning_rate

    centralized_evaluation = args.centralized_evaluation

    if centralized_evaluation == "true":
        CENTRALIZED_EVALUATION = True
    
    # Uses the same model as defined in the CGSV paper
    if DATASET == "CIFAR10":
        from models.CIFAR10_model import Net, train, test, add_update_to_model, local_model, collect_global_model
        from dataset_loaders.CIFAR10_centralized_datasetloader import load_centralized_data
        if PARTIONING == "IID":
            from dataset_loaders.CIFAR10_IID_datasetloader import load_data
        elif PARTIONING == "POW":
            from dataset_loaders.CIFAR10_POW_datasetloader import load_data          

    # Uses the same model as defined in the CGSV paper
    elif DATASET == "MNIST":
        from models.MNIST_model import Net, train, test, add_update_to_model, local_model, collect_global_model
        from dataset_loaders.MNIST_centralized_datasetloader import load_centralized_data
        if PARTIONING == "IID":
            from dataset_loaders.MNIST_IID_datasetloader import load_data
        elif PARTIONING == "POW":
            from dataset_loaders.MNIST_POW_datasetloader import load_data        

    # Uses the same model as defined in the FedProb paper
    elif DATASET == "FASHION_MNIST":
        from models.FASHION_MNIST_model import Net, train, test, add_update_to_model, local_model, collect_global_model
        from dataset_loaders.FASHION_MNIST_centralized_datasetloader import load_centralized_data
        if PARTIONING == "IID":
            from dataset_loaders.FASHION_MNIST_IID_datasetloader import load_data
        elif PARTIONING == "POW":
            from dataset_loaders.FASHION_MNIST_POW_datasetloader import load_data

    elif DATASET == "SST2":
        from models.SST2_model import Net, train, test, add_update_to_model, local_model, collect_global_model
        from dataset_loaders.SST2_centralized_datasetloader import load_centralized_data
        if PARTIONING == "IID":
            from dataset_loaders.SST2_IID_datasetloader import build_global_token_tid_map, load_data
        elif PARTIONING == "POW":
            from dataset_loaders.SST2_POW_datasetloader import build_global_token_tid_map, load_data

        if rank == 0:
            global TOKEN_TO_TID_MAP
            TOKEN_TO_TID_MAP = build_global_token_tid_map(seed=SEED, embed_num=20_000)
        else:
            TOKEN_TO_TID_MAP = None

        TOKEN_TO_TID_MAP = comm.bcast(TOKEN_TO_TID_MAP, root=0)
        
    if rank == 0:
        print(f"[INIT] Hub-and-Spoke Architecture; size={size}, (num_clients={number_of_clients})")
        print(f"Server Initialized the model")

        # By default I'm seeting the device to CPU because I used a CPU cluster to train my models, you can swap it according to your needs
        device = "cpu"

        net = Net()
        server_model = local_model(net)

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
        total_server_only_computation_time = 0.0

        # Preparing .csv files to save metrics
        os.makedirs(args.output_dir, exist_ok=True)
        base = f"{number_of_clients}_clients_run_{args.run_id}_hub_and_spoke"
        server_csv = os.path.join(args.output_dir, f"FedAvg_{DATASET}_{PARTIONING}_{base}_server_metrics.csv")
        client_csv = os.path.join(args.output_dir, f"FedAvg_{DATASET}_{PARTIONING}_{base}_client_metrics.csv")

        for rnd in range(FL_ROUNDS):
            round_server_only_computation_time = 0.0
            round_duration = 0.0

            # Server sends training instructions to FL Clients
            round_start_time = MPI.Wtime()
            training_instructions: TrainingInstructions = {"phase": "training",
                                                           "round": rnd,
                                                           "local_epochs": LOCAL_EPOCHS,
                                                           "has_model": (rnd == 0),
                                                           "model": server_model if rnd == 0 else None,
                                                           }

            train_instructions_broadcast_start_time = MPI.Wtime()
            comm.bcast(obj=training_instructions, root=0)
            train_instructions_broadcast_end_time = MPI.Wtime()
            train_instructions_broadcast_time = train_instructions_broadcast_end_time - train_instructions_broadcast_start_time
            print("Server broadcasted training instructions to clients")

            # Server collects the training results from clients
            client_results_train: list[ClientTrainResult] = []
            for client_id in range(number_of_clients):
                result: ClientTrainResult = comm.recv(source=client_id + 1, tag=TRAIN_RESULTS + rnd)
                client_results_train.append(result)
            print("Server received training results from clients")

            # Aggregates the gradients using FedAvg (u_(N,t))
            print("Server started gradient aggregation using Federatd Averaging")
            aggregation_start_time = MPI.Wtime()
            aggregated_gradient = FedAvgGradients(client_results=client_results_train)
            aggregation_end_time = MPI.Wtime()
            aggregation_time = aggregation_end_time - aggregation_start_time
            print(f"Server finished gradient aggregation in {aggregation_time:.4f} seconds.")

            # Forward pass on the Server Model
            print("Server started global model forward pass")
            server_forward_pass_start_time = MPI.Wtime()
            server_model = add_update_to_model(server_model, aggregated_gradient)
            collect_global_model(net, server_model)
            server_forward_pass_end_time = MPI.Wtime()
            server_forward_pass_time = server_forward_pass_end_time - server_forward_pass_start_time
            print(f"Server finished global model forward pass in {server_forward_pass_time:.4f} seconds.")
            
            evaluation_instructions: EvaluationInstructions = {"phase": "evaluation", "round": rnd}
            eval_instructions_broadcast_start_time = MPI.Wtime()
            comm.bcast(obj=evaluation_instructions, root=0)
            eval_instructions_broadcast_end_time = MPI.Wtime()
            eval_instructions_broadcast_time = eval_instructions_broadcast_end_time - eval_instructions_broadcast_start_time
            print("Server broadcasted evaluation instructions to clients")

            scatter_payload_eval: list[npt.NDArray[np.generic]] = []
            scatter_payload_eval.append(None)  # for the server
            for cid in range(number_of_clients):
                scatter_payload_eval.append(aggregated_gradient)
            
            model_eval_scattering_start_time = MPI.Wtime()
            _ = comm.scatter(sendobj=scatter_payload_eval, root=0)
            model_eval_scattering_end_time = MPI.Wtime()
            model_eval_scattering_time = model_eval_scattering_end_time - model_eval_scattering_start_time
            print("Server broadcasted aggregated gradient to clients")

            if CENTRALIZED_EVALUATION == True:
                print(f"Server started evaluation of global model on centralized test dataset")
                server_central_evaluation_loss, server_central_evaluation_accuracy = test(net, centralized_testloader, device)
                print(f"Server finished evaluation of global model on centralized test dataset")
            else:
                print(f"Global model evaluation was disabled")
                server_central_evaluation_loss, server_central_evaluation_accuracy = (0.00, 0.00)

            client_testing_results: list[ClientTestResult] = []
            for cid in range(number_of_clients):
                result = comm.recv(source=cid + 1, tag=TEST_RESULTS + rnd)
                client_testing_results.append(result)
            print("Server received evaluation results from clients")

            round_end_time = MPI.Wtime()

            round_clients_local_train_accuracy_values: list[float] = []
            round_clients_local_test_accuracy_values:  list[float] = []
            round_clients_central_test_accuracy_values:  list[float] = []

            for result in client_testing_results:
                round_clients_local_train_accuracy_values.append(result["training_accuracy"])
                round_clients_local_test_accuracy_values.append(result["evaluation_accuracy"])
                round_clients_central_test_accuracy_values.append(result["central_evaluation_accuracy"])
                

            round_clients_local_train_average_accuracy = float(np.mean(round_clients_local_train_accuracy_values))
            round_clients_local_test_average_accuracy  = float(np.mean(round_clients_local_test_accuracy_values))
            round_clients_central_test_average_accuracy  = float(np.mean(round_clients_central_test_accuracy_values))
            
            # Acquiring some server metrics
            round_server_only_computation_time = (aggregation_time + server_forward_pass_time)
            round_server_communication_time = (train_instructions_broadcast_time + eval_instructions_broadcast_time + model_eval_scattering_time)
            
            round_duration = round_end_time - round_start_time

            total_server_only_computation_time += round_server_only_computation_time
            total_time += round_duration

            # ---- write server-level metrics ----
            write_server_metrics(
                path=server_csv,
                round_idx=rnd,
                time_to_aggregate_gradients=aggregation_time,
                time_to_update_server_model=server_forward_pass_time,
                round_server_only_computation_time=round_server_only_computation_time,
                round_duration=round_duration,
                total_server_only_computation_time=total_server_only_computation_time,
                total_time=total_time,
                round_clients_local_training_dataset_average_accuracy=round_clients_local_train_average_accuracy,
                round_clients_local_testing_dataset_average_accuracy=round_clients_local_test_average_accuracy,
                round_clients_central_testing_dataset_average_accuracy=round_clients_central_test_average_accuracy,
                round_server_central_testing_dataset_loss=server_central_evaluation_loss,
                round_server_central_testing_dataset_accuracy=server_central_evaluation_accuracy,
            )

            # ---- write per-client metrics ----
            write_client_metrics(
                path=client_csv,
                round_idx=rnd,
                client_testing_results=client_testing_results,
                local_epochs=LOCAL_EPOCHS,
                learning_rate=LEARNING_RATE,
            )

        # After iterating through all rounds, stops client loop
        comm.bcast({"phase": "stop"}, root=0)
        print(f"Server finished all {FL_ROUNDS} rounds. Total server-only-computation: {round_server_only_computation_time:.4f} seconds.")
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
            trainloader, testloader = load_data(partition_id=client_id,
                                                num_partitions=number_of_clients,
                                                global_token_to_tid=TOKEN_TO_TID_MAP,
                                                max_length=64, 
                                                seed=SEED, 
                                                batch_size=BATCH_SIZE)
        else:
            net = Net()
            trainloader, testloader = load_data(partition_id=client_id,
                                                num_partitions=number_of_clients,
                                                seed=SEED,
                                                batch_size=BATCH_SIZE)  

        while(True):
            control = comm.bcast(obj=None, root=0)

            phase = control["phase"]

            if (phase == "training"):
                print(f"Client {client_id} initialized round {control['round']} training. Local training happening for {control['local_epochs']} epochs")

                if control.get("has_model", False):
                    # rnd == 0 : everyone receives the same server model
                    collect_global_model(net, control["model"])

                training_loss, training_accuracy, model_update = train(net, trainloader, control["local_epochs"], device, lr=LEARNING_RATE)

                # Client sends only the necessary information
                client_results_train: ClientTrainResult = {
                    "client_id": client_id,
                    "model_update": model_update,
                    "training_dataset_size": len(trainloader.dataset)
                }

                comm.send(obj=client_results_train, dest=0, tag=TRAIN_RESULTS + control["round"])
                print(f"Client {client_id} sent results to server for round {control['round']}")

            elif (phase == "evaluation"):
                
                # Updating local model
                my_gradient_reward = comm.scatter(sendobj=None, root=0)
                current_params = local_model(net)
                reward = add_update_to_model(current_params, my_gradient_reward)
                collect_global_model(net, reward)
                print(f"Client {client_id} updated its local model")
                
                if CENTRALIZED_EVALUATION == True:
                    print(f"Client {client_id} started evaluation on centralized test dataset")
                    central_evaluation_loss, central_evaluation_accuracy = test(net, centralized_testloader, device)
                    print(f"Client {client_id} finished evaluation on centralized test dataset")
                    
                    # Preparing payload with training and testing metrics
                    client_testing_results: ClientTestResult = {
                        "client_id": client_id,
                        "training_dataset_size": len(trainloader.dataset),
                        "training_loss": training_loss,
                        "training_accuracy": training_accuracy,
                        "evaluation_loss": 0,
                        "evaluation_accuracy": 0,
                        "evaluation_dataset_size": 0,
                        "central_evaluation_loss": central_evaluation_loss,
                        "central_evaluation_accuracy": central_evaluation_accuracy,
                    }

                else:
                    print(f"Client {client_id} started evaluation on local test dataset")
                    evaluation_loss, evaluation_accuracy = test(net, testloader, device)
                    print(f"Client {client_id} finished evaluation on local test dataset")
                    
                    # Preparing payload with training and testing metrics
                    client_testing_results: ClientTestResult = {
                        "client_id": client_id,
                        "training_dataset_size": len(trainloader.dataset),
                        "training_loss": training_loss,
                        "training_accuracy": training_accuracy,
                        "evaluation_loss": evaluation_loss,
                        "evaluation_accuracy": evaluation_accuracy,
                        "evaluation_dataset_size": len(testloader.dataset),
                        "central_evaluation_loss": 0,
                        "central_evaluation_accuracy": 0,
                    }
                
                if CENTRALIZED_EVALUATION == True:
                    print(f"Client {client_id} testing results on centralized dataset -- Test Loss: {central_evaluation_loss}, Test Accuracy: {central_evaluation_accuracy}")
                else:
                    print(f"Client {client_id} testing results on local dataset -- Test Loss: {evaluation_loss}, Test Accuracy: {evaluation_accuracy}")

                comm.send(obj=client_testing_results, dest=0, tag=TEST_RESULTS + control["round"])
                print(f"Client {client_id} sent testing results to server")

            elif phase == "stop":
                break

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="MNIST")
    p.add_argument("--partitioning", type=str, default="IID")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--rounds", type=int, default=200,)
    p.add_argument("--local_epochs", type=int, default=10)
    p.add_argument("--learning_rate", type=float, default=0.001)
    p.add_argument("--centralized_evaluation", type=str, default="false")
    p.add_argument("--run_id", type=int, default=1, )
    p.add_argument("--output_dir", type=str, default="results")
    return p.parse_args()

if __name__ == "__main__":
    main()
