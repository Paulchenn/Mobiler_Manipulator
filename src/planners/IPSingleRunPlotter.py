import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

class SingleRunPlotter:
    """
    Helper class to plot single planning results individually.
    """

    @staticmethod
    def visualize_and_save(resultList, plotList, plannerNames, benchNames, plannerFactory, save_plots=False, output_dir="plots"):
        """
        Iterates through the results and creates one figure per run.
        """
        
        # Iterator für die Ergebnisse
        resultList_iter = iter(resultList)
        
        # Ordner erstellen, falls gespeichert werden soll
        if save_plots and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Über alle geplanten Kombinationen iterieren
        # plotList enthält 1 (Success/Partial) oder 0 (Error)
        for i, status in enumerate(plotList):
            
            # Indizes berechnen (Mapping wie in deinem Code)
            # row_idx -> Benchmark
            # col_idx -> Planner
            num_benchmarks = len(benchNames)
            row_idx = i % num_benchmarks
            col_idx = i // num_benchmarks
            
            planner_name = plannerNames[col_idx]
            bench_name = benchNames[row_idx]
            
            title = f"{planner_name} - {bench_name}"

            # Falls Fehler beim Planen war (Status 0), überspringen wir es
            # (Oder man könnte ein leeres Bild mit "Error" speichern, hier skippen wir)
            if status == 0:
                print(f"Skipping {title} (Planning Error)")
                continue
            
            # Nächstes Ergebnis holen
            try:
                result = next(resultList_iter)
            except StopIteration:
                break

            # Warnung im Titel, falls kein Pfad gefunden wurde
            full_title = title
            if not result.solution:
                full_title += " (No path found!)"

            # --- PLOTTING ---
            # Neue Figure für JEDEN Plot einzeln
            fig, ax = plt.subplots(figsize=(8, 8))
            
            ax.set_title(full_title, fontsize=12, fontweight='bold')

            try:
                # 1. Die Factory-Methode zum Zeichnen aufrufen
                # plannerFactory[Name] = [Class, Config, VisualizeFunc] -> wir nehmen Index 2
                visualize_func = plannerFactory[result.plannerFactoryName][2]
                visualize_func(result.planner, result.solution, result.actions, ax=ax, nodeSize=20, plot_only_solution=False)
                
                # 2. Limits und Seitenverhältnis aus CollisionChecker
                limits = result.benchmark.collisionChecker.getEnvironmentLimits()
                ax.set_xlim(limits[0])
                ax.set_ylim(limits[1])
                ax.set_aspect('equal')

                # 3. Grid und Ticks (Deine Konfiguration)
                ax.axis('on')
                ax.xaxis.set_major_locator(ticker.MultipleLocator(1.0))
                ax.yaxis.set_major_locator(ticker.MultipleLocator(1.0))
                ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
                ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.5))
                
                ax.grid(which='major', alpha=0.5, color='gray', linestyle='-')
                ax.grid(which='minor', alpha=0.2, color='gray', linestyle='--')
                ax.tick_params(axis='both', which='both', labelbottom=True, labelleft=True)

                # 4. Speichern (Optional)
                if save_plots:
                    # Dateiname bereinigen (Leerzeichen zu Unterstrichen)
                    clean_p = planner_name.replace(" ", "_")
                    clean_b = bench_name.replace(" ", "_")
                    filename = f"{output_dir}/{clean_b}_{clean_p}.png"
                    
                    plt.savefig(filename, dpi=150, bbox_inches='tight')
                    print(f"Saved: {filename}")
                else:
                    # Nur anzeigen
                    plt.show()
                
                # Figure schließen, um Speicher freizugeben (wichtig bei vielen Plots!)
                plt.close(fig)

            except Exception as e:
                print(f"Error plotting {title}: {e}")
                import traceback
                traceback.print_exc()
                plt.close(fig)