import matplotlib.pyplot as plt
import seaborn as sns

class BatchPlotter:
    """
    Helper class to visualize Batch Evaluation results (Boxplots & Statistics).
    """

    @staticmethod
    def visualize(df_results, save_plots=False, output_dir="plots"):
        """
        Generates 3x2 grid plots for each benchmark in the dataframe.
        
        Layout:
        [Success Rate]   [Failure Stages]
        [Planning Time]  [Path Nodes]
        [Path Euclid]    [Roadmap Size]
        """
        
        # Theme setzen
        sns.set_theme(style="whitegrid")

        # Konsistente Reihenfolge der Planner
        planner_order = df_results["Planner"].unique()
        unique_benchmarks = df_results["Benchmark"].unique()

        for bench_name in unique_benchmarks:
            df_bench = df_results[df_results["Benchmark"] == bench_name]
            has_successes = df_bench["Success"].any()
            
            # Setup Figure: 3 Zeilen, 2 Spalten
            fig, axes = plt.subplots(3, 2, figsize=(14, 15))
            fig.suptitle(f"Batch Results: {bench_name}", fontsize=16, fontweight='bold')
            
            # Flatten axes array for easy indexing [0..5]
            ax_flat = axes.flatten()

            # --- 1. SUCCESS RATE (Barplot) ---
            success_rates = df_bench.groupby("Planner")["Success"].mean().reset_index()
            success_rates["Success"] *= 100 
            
            sns.barplot(data=success_rates, x="Planner", y="Success", ax=ax_flat[0], 
                        order=planner_order, hue="Planner", palette="viridis", legend=False)
            ax_flat[0].set_title("Success Rate (%)")
            ax_flat[0].set_ylim(0, 105)
            ax_flat[0].set_ylabel("Success Rate [%]")

            # Werte als Text anzeigen
            for index, row in success_rates.iterrows():
                # Hinweis: Text-Positionierung ist bei Seaborn Barplots manchmal tricky, 
                # hier eine vereinfachte Variante
                pass 

            # --- 2. FAILURE ANALYSIS (Countplot) ---
            df_failures = df_bench[df_bench["Success"] == False]
            
            if not df_failures.empty:
                sns.countplot(data=df_failures, x="Planner", hue="FailStage", ax=ax_flat[1], 
                              order=planner_order, palette="magma")
                ax_flat[1].set_title("Failure Analysis (Where did it break?)")
                ax_flat[1].set_ylabel("Count of Failures")
                ax_flat[1].legend(title="Failed at:", loc='upper right', fontsize='small')
            else:
                ax_flat[1].text(0.5, 0.5, "100% SUCCESS", ha='center', va='center', color='green', fontweight='bold', fontsize=14)
                ax_flat[1].set_title("Failure Analysis")

            # --- 3. PLANNING TIME (Boxplot) ---
            sns.boxplot(data=df_bench, x="Planner", y="Time", ax=ax_flat[2], 
                        order=planner_order, hue="Planner", palette="Blues", legend=False)
            sns.stripplot(data=df_bench, x="Planner", y="Time", color=".3", size=3, ax=ax_flat[2], order=planner_order)
            ax_flat[2].set_title("Calculation Time (s)")

            # --- 4. PATH LENGTH NODES (Boxplot) ---
            if has_successes:
                sns.boxplot(data=df_bench, x="Planner", y="PathNodes", ax=ax_flat[3], 
                            order=planner_order, hue="Planner", palette="Reds", legend=False)
                sns.stripplot(data=df_bench, x="Planner", y="PathNodes", color=".3", size=3, ax=ax_flat[3], order=planner_order)
                ax_flat[3].set_title("Path Length (Count of Nodes)")
            else:
                ax_flat[3].text(0.5, 0.5, "NO SUCCESS", ha='center', color='red')
                ax_flat[3].set_title("Path Length (Nodes)")

            # --- 5. PATH LENGTH EUCLIDEAN (Boxplot) ---
            if has_successes and "PathEuclid" in df_bench.columns:
                sns.boxplot(data=df_bench, x="Planner", y="PathEuclid", ax=ax_flat[4], 
                            order=planner_order, hue="Planner", palette="Greens", legend=False)
                sns.stripplot(data=df_bench, x="Planner", y="PathEuclid", color=".3", size=3, ax=ax_flat[4], order=planner_order)
                ax_flat[4].set_title("Path Length (Euclidean Distance)")
            else:
                msg = "NO SUCCESS" if not has_successes else "Data Missing"
                ax_flat[4].text(0.5, 0.5, msg, ha='center', color='red')
                ax_flat[4].set_title("Path Length (Euclidean)")

            # --- 6. ROADMAP SIZE (Boxplot) ---
            sns.boxplot(data=df_bench, x="Planner", y="GraphNodes", ax=ax_flat[5], 
                        order=planner_order, hue="Planner", palette="Purples", legend=False)
            sns.stripplot(data=df_bench, x="Planner", y="GraphNodes", color=".3", size=3, ax=ax_flat[5], order=planner_order)
            ax_flat[5].set_title("Roadmap Size (Total Nodes)")

            plt.tight_layout()

            # 4. Speichern (Optional)
            if save_plots:
                # Dateiname bereinigen (Leerzeichen zu Unterstrichen)
                clean_b = bench_name.replace(" ", "_")
                filename = f"{output_dir}/{clean_b}_benchmark_multiRun.png"
                
                plt.savefig(filename, dpi=150, bbox_inches='tight')
                print(f"Saved: {filename}")
            else:
                # Nur anzeigen
                plt.show()