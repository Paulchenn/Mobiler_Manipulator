import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import traceback

class BenchmarkPlotter:
    """
    Helper class to visualize Benchmark results (Bar Charts with multiple axes).
    """

    @staticmethod
    def _add_labels(ax, rects, color, fmt='{:.0f}', offset=3):
        """Internal helper to attach a text label above each bar."""
        for rect in rects:
            height = rect.get_height()
            if height == 0: continue 
            ax.annotate(fmt.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, offset),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        color=color, fontsize=8, fontweight='bold')

    @staticmethod
    def _get_path_dist(res):
        """Internal helper to calculate euclidean path distance."""
        d = 0.0
        if res.solution:
            # Zugriff auf den Graphen im Result-Objekt
            g = res.planner.graph
            # Filtern, falls Nodes nicht im Graph sind (sollte nicht passieren, aber sicher ist sicher)
            pts = [np.array(g.nodes[n]['pos']) for n in res.solution if n in g.nodes]
            for k in range(len(pts)-1): 
                d += np.linalg.norm(pts[k+1]-pts[k])
        return d

    @staticmethod
    def visualize(resultList, benchList, save_plots=False, output_dir="plots"):
        """
        Generates individual figures for each benchmark comparing all planners.
        
        Args:
            resultList: List of ResultCollection objects.
            benchList: List of Benchmark objects.
            save_plots: Boolean, if True saves images as PNG.
            save_prefix: String, prefix for filenames.
        """
        
        # Grid zurücksetzen, falls Seaborn aktiv war
        try:
            import seaborn as sns
            sns.reset_orig()
        except ImportError:
            pass

        # --- GLOBALE MAXIMA BERECHNEN (für einheitliche Skalierung) ---
        # Wir nutzen Generatoren für Speicher-Effizienz
        all_max_nodes = max((len(r.solution) for r in resultList), default=0) * 1.1
        
        # Zeit extrahieren (pandas)
        all_max_time = max((r.perfDataFrame.groupby(["name"]).sum(numeric_only=True)["time"]["planPath"] for r in resultList), default=0) * 1.1
        
        all_max_size = max((r.planner.graph.size() for r in resultList), default=0) * 1.1
        
        all_max_dist = max((BenchmarkPlotter._get_path_dist(r) for r in resultList), default=0) * 1.1

        # --- SCHLEIFE ÜBER BENCHMARKS ---
        for i, bench in enumerate(benchList):
            title = bench.name
            
            # --- Neue Figure pro Benchmark ---
            fig, ax = plt.subplots(figsize=(12, 5))
            plt.subplots_adjust(right=0.65) # Platz für die Achsen rechts
            
            pathNodes    = dict() 
            euclidLength = dict() 
            planningTime = dict()
            roadmapSize  = dict()
            
            try:
                # --- 1. Data Collection ---
                for result in resultList:
                    if result.benchmark.name == bench.name:
                        name = result.plannerFactoryName
                        
                        pathNodes[name]    = len(result.solution)
                        euclidLength[name] = BenchmarkPlotter._get_path_dist(result)
                        planningTime[name] = result.perfDataFrame.groupby(["name"]).sum(numeric_only=True)["time"]["planPath"]
                        roadmapSize[name]  = result.planner.graph.size()

                if not pathNodes:
                    print(f"No results for benchmark '{title}'")
                    plt.close(fig)
                    continue

                # --- 2. Axis Setup ---
                ax.grid(False) 
                width = 0.15 
                
                # X-Axis
                # Keys sind die Planner Namen
                planner_names = list(pathNodes.keys())
                ax.set_xticks(np.arange(len(planner_names)) + 1.5 * width)
                ax.set_xticklabels(planner_names, fontsize=11, fontweight='bold')
                ax.set_title(f"Benchmark Result: {title}", fontsize=14, pad=20)
                
                # Ghost Axis
                ax.set_yticks([])
                ax.spines['left'].set_visible(True)
                ax.spines['left'].set_color("black")
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.spines['bottom'].set_visible(False)

                # --- 3. BLUE AXIS: Node Count ---
                ax_blue = ax.twinx() 
                ax_blue.grid(False)
                ax_blue.set_ylim(0, all_max_nodes)
                blue_bars = ax_blue.bar(np.arange(len(planner_names)), pathNodes.values(), width, color="blue")
                ax_blue.set_ylabel("Path Nodes (Count)", color="blue")
                
                ax_blue.spines['right'].set_color("blue")
                ax_blue.tick_params(axis='y', colors='blue')
                ax_blue.spines['left'].set_visible(False)
                ax_blue.margins(y=0.15)
                BenchmarkPlotter._add_labels(ax_blue, blue_bars, "blue", fmt='{:.0f}')

                # --- 4. GREEN AXIS: Euclidean Length ---
                ax_green = ax.twinx()
                ax_green.grid(False)
                ax_green.set_ylim(0, all_max_dist)
                green_bars = ax_green.bar(np.arange(len(planner_names)) + width, euclidLength.values(), width, color="green")
                ax_green.set_ylabel("Path Length (Euclidean)", color="green")
                
                ax_green.spines['right'].set_position(('axes', 1.15))
                ax_green.spines['right'].set_visible(True)
                ax_green.spines['right'].set_color("green")
                ax_green.tick_params(axis='y', colors='green')
                ax_green.spines['left'].set_visible(False)
                ax_green.margins(y=0.15)
                BenchmarkPlotter._add_labels(ax_green, green_bars, "green", fmt='{:.1f}')

                # --- 5. DARKGOLDENROD AXIS: Time ---
                ax_gold = ax.twinx()
                ax_gold.grid(False)
                ax_gold.set_ylim(0, all_max_time)
                
                hatches = ['xx' if length==0 else '' for length in pathNodes.values()]
                bar_colors = ['darkgoldenrod' if length==0 else 'darkgoldenrod' for length in pathNodes.values()]
                
                gold_bars = ax_gold.bar(np.arange(len(planner_names)) + 2*width, planningTime.values(), width, color="red")
                
                for j, thisbar in enumerate(gold_bars.patches):
                    thisbar.set_facecolor(bar_colors[j])
                    thisbar.set_hatch(hatches[j])
                    
                ax_gold.set_ylabel("Planning Time [s]", color="darkgoldenrod") 
                
                ax_gold.spines['right'].set_position(('axes', 1.30))
                ax_gold.spines['right'].set_visible(True)
                ax_gold.spines['right'].set_color("darkgoldenrod")
                ax_gold.tick_params(axis='y', colors='darkgoldenrod')
                ax_gold.spines['left'].set_visible(False)
                ax_gold.margins(y=0.15)
                BenchmarkPlotter._add_labels(ax_gold, gold_bars, "darkgoldenrod", fmt='{:.3f}')

                # --- 6. PURPLE AXIS: Roadmap Size ---
                ax_purple = ax.twinx()
                ax_purple.grid(False)
                ax_purple.set_ylim(0, all_max_size)
                purple_bars = ax_purple.bar(np.arange(len(planner_names)) + 3*width, roadmapSize.values(), width, color="purple")
                ax_purple.set_ylabel("Roadmap Size", color="purple")
                
                ax_purple.spines['right'].set_position(('axes', 1.45))
                ax_purple.spines['right'].set_visible(True)
                ax_purple.spines['right'].set_color("purple")
                ax_purple.tick_params(axis='y', colors='purple')
                ax_purple.spines['left'].set_visible(False)
                ax_purple.margins(y=0.15)
                BenchmarkPlotter._add_labels(ax_purple, purple_bars, "purple", fmt='{:.0f}')

                # --- Legend ---
                legend_handles = [
                    mpatches.Patch(facecolor='darkgoldenrod', label='Success'),
                    mpatches.Patch(facecolor='darkgoldenrod', hatch='xx', label='Failed')
                ]
                ax_purple.legend(handles=legend_handles, loc='upper center', fontsize='small', framealpha=0.8)

                # 4. Speichern (Optional)
                if save_plots:
                    # Dateiname bereinigen (Leerzeichen zu Unterstrichen)
                    clean_b = bench.name.replace(" ", "_")
                    filename = f"{output_dir}/{clean_b}_benchmark_singleRun.png"
                    
                    plt.savefig(filename, dpi=150, bbox_inches='tight')
                    print(f"Saved: {filename}")
                else:
                    # Nur anzeigen
                    plt.show()

            except Exception as e:
                print(f"Error processing benchmark '{bench.name}': {e}")
                traceback.print_exc()
                pass

            # plt.show()