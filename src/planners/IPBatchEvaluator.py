"""
Batch Evaluation module for motion planning algorithms.

This module provides functionality to run multiple motion planning algorithms
on a set of benchmarks and collect performance metrics.
"""

import pandas as pd
import time
import numpy as np
from tqdm.notebook import tqdm
from IPMultiGoalPlannerRunner import MultiGoalPlannerRunner


class BatchEvaluator:
    """
    Evaluates multiple motion planning algorithms on a set of benchmarks.

    This class provides static methods to execute planners on benchmark scenarios
    multiple times and collect comprehensive performance metrics including success rates,
    execution times, path lengths, and failure information.
    """

    @staticmethod
    def _calc_path_distance(graph, path):
        """
        Calculate the Euclidean distance of a path in the planning graph.

        Computes the cumulative Euclidean distance between consecutive nodes
        in the given path by extracting their positions from the graph.

        Args:
            graph: NetworkX graph containing nodes with 'pos' attributes
            path: List of node identifiers representing the path

        Returns:
            float: Total Euclidean distance of the path. Returns np.nan if path
                   is empty, contains invalid nodes, or an error occurs.
        """
        dist = 0.0
        try:
            for k in range(len(path) - 1):
                u, v = path[k], path[k+1]
                # Verify that nodes exist in the graph before accessing them
                if u in graph.nodes and v in graph.nodes:
                    p1 = np.array(graph.nodes[u]['pos'])
                    p2 = np.array(graph.nodes[v]['pos'])
                    dist += np.linalg.norm(p2 - p1)
        except Exception:
            return np.nan
        return dist
    
    @staticmethod
    def run_experiment(planner_factory, bench_list, num_runs=10):
        """
        Execute multiple motion planning algorithms on benchmark scenarios.

        For each planner and benchmark combination, the planner is instantiated
        from scratch and executed num_runs times to collect statistical data.
        This ensures a clean state for each run without side effects from previous
        evaluations.

        Args:
            planner_factory (dict): Dictionary mapping planner names to tuples of
                (planner_class, config, metadata). The planner_class is instantiated
                with a CollisionChecker object for each run.
            bench_list (list): List of Benchmark objects to evaluate the planners on.
            num_runs (int, optional): Number of independent executions per planner-benchmark
                combination. Defaults to 10.

        Returns:
            pd.DataFrame: DataFrame containing evaluation results with columns:
                - Planner: Name of the planner algorithm
                - Benchmark: Name of the benchmark scenario
                - RunID: Sequential run number (0-indexed)
                - Success: Boolean indicating whether planning succeeded
                - Time: Execution time in seconds
                - PathNodes: Number of nodes in the planned path (NaN if failed)
                - PathEuclid: Euclidean distance of the planned path (NaN if failed)
                - GraphNodes: Total nodes in the constructed planning graph
                - FailStage: Success status or failure information (e.g., "Seg 1: PICK")
        """
        data = []
        
        # Calculate total iterations for progress tracking
        total_iterations = len(planner_factory) * len(bench_list) * num_runs
        pbar = tqdm(total=total_iterations, desc="Running Batch Evaluation")
        
        for planner_name, (planner_class, config, _) in planner_factory.items():
            print("=" * 50)
            print(f"{planner_name}:")
            for bench in bench_list:
                for run_id in range(num_runs):
                    print(f"    Running Benchmark {bench.name} {run_id + 1}/{num_runs}...")
                    
                    # Instantiate planner with fresh state to avoid carry-over effects
                    # from previous runs
                    planner = planner_class(bench.collisionChecker)
                    
                    # Initialize measurement variables
                    start_time = time.time()
                    success = False
                    path_length_nodes = np.nan
                    path_length_euclid = np.nan
                    nodes_count = 0
                    fail_stage_label = "Success"
                    
                    try:
                        # Execute the benchmark scenario
                        full_path, _, status = MultiGoalPlannerRunner.run_benchmark(planner, bench, config)
                        
                        # Collect metrics from the executed benchmark
                        success = status["success"]
                        nodes_count = planner.graph.number_of_nodes()

                        if success:
                            path_length_nodes = len(full_path)
                            path_length_euclid = BatchEvaluator._calc_path_distance(planner.graph, full_path)
                            fail_stage_label = "Success"
                        else:
                            # Determine which action (segment) caused the failure
                            seg_idx = status['fail_segment']
                            target_entry = bench.goalList[seg_idx]
                            
                            # Extract action name from goal list entry (e.g., "PICK" or "PLACE")
                            action_name = "MOVE"
                            if isinstance(target_entry, (tuple, list)) and len(target_entry) >= 2:
                                action_name = target_entry[1]
                            
                            # Create descriptive failure label (e.g., "Seg 1: PICK")
                            fail_stage_label = f"Seg {seg_idx}: {action_name}"
                            
                            # For failed runs, use NaN for path metrics since planning was incomplete
                            path_length_nodes = np.nan
                            path_length_euclid = np.nan
                        
                    except Exception as e:
                        # Capture any exceptions that occur during benchmark execution
                        success = False
                        fail_stage_label = "Exception/Crash"
                    
                    duration = time.time() - start_time
                    
                    # Store collected metrics for this run
                    entry = {
                        "Planner": planner_name,
                        "Benchmark": bench.name,
                        "RunID": run_id,
                        "Success": success,
                        "Time": duration,
                        "PathNodes": path_length_nodes,
                        "PathEuclid": path_length_euclid,
                        "GraphNodes": nodes_count,
                        "FailStage": fail_stage_label
                    }
                    data.append(entry)
                    pbar.update(1)
                    
        pbar.close()
        
        # Return collected data as a pandas DataFrame for further analysis
        return pd.DataFrame(data)