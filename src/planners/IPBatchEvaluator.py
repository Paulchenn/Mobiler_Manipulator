import pandas as pd
import time
import numpy as np
from tqdm.notebook import tqdm # Schöner Progress-Bar für Notebooks
from IPMultiGoalPlannerRunner import MultiGoalPlannerRunner

class BatchEvaluator:
    @staticmethod
    def run_experiment(planner_factory, bench_list, num_runs=10):
        """
        Führt jeden Planner auf jedem Benchmark N-mal aus.
        Instanziiert den Planner für jeden Run komplett neu (WICHTIG!).
        """
        data = []
        
        # Gesamtanzahl der Iterationen für den Progress Bar berechnen
        total_iterations = len(planner_factory) * len(bench_list) * num_runs
        pbar = tqdm(total=total_iterations, desc="Running Batch Evaluation")
        
        for planner_name, (planner_class, config, _) in planner_factory.items():
            print("="*50)
            print(f"{planner_name}:")
            for bench in bench_list:
                for run_id in range(num_runs):
                    print(f"    Running Benchmark {bench.name} {run_id+1}/{num_runs}...")
                    
                    # 1. Planner NEU instanziieren (Reset State)
                    # Wir übergeben den CollisionChecker des Benchmarks
                    planner = planner_class(bench.collisionChecker)
                    
                    # Variablen für Messung
                    start_time = time.time()
                    success = False
                    path_length = np.nan
                    nodes_count = 0
                    fail_stage_label = "Success" # Default
                    
                    try:
                        # 2. Benchmark ausführen
                        # IPPerfMonitor.clearData() # Falls du interne Timer tracken willst
                        
                        full_path, _, status = MultiGoalPlannerRunner.run_benchmark(planner, bench, config)
                        
                        # 3. Metriken erfassen (bei Erfolg)
                        success = status["success"]
                        nodes_count = planner.graph.number_of_nodes()

                        if success:
                            path_length = len(full_path)
                            fail_stage_label = "Success"
                        else:
                            # Wir schauen nach, welche Action bei diesem Segment geplant war
                            seg_idx = status['fail_segment']
                            target_entry = bench.goalList[seg_idx]
                            
                            # Action Namen extrahieren
                            action_name = "MOVE"
                            if isinstance(target_entry, (tuple, list)) and len(target_entry) >= 2:
                                action_name = target_entry[1]
                            
                            # Label bauen: z.B. "Seg 1: PICK"
                            fail_stage_label = f"Seg {seg_idx}: {action_name}"
                            
                            # Auch bei Teil-Pfaden haben wir eine Länge, 
                            # für Statistik ist aber NaN besser oder wir speichern partial_length separat
                            path_length = np.nan
                        
                    except Exception as e:
                        # 3b. Fehler erfassen
                        success = False
                        fail_stage_label = "Exception/Crash"
                    
                    duration = time.time() - start_time
                    
                    # 4. Daten speichern
                    entry = {
                        "Planner": planner_name,
                        "Benchmark": bench.name,
                        "RunID": run_id,
                        "Success": success,
                        "Time": duration,
                        "PathLength": path_length,
                        "GraphNodes": nodes_count,
                        "FailStage": fail_stage_label # NEUE SPALTE
                    }
                    data.append(entry)
                    pbar.update(1)
                    
        pbar.close()
        
        # Als Pandas DataFrame zurückgeben
        return pd.DataFrame(data)