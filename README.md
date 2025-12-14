# Projekt: Mobiler Manipulator (CollisionChecker)

Dieses Repository beinhaltet die Implementierung und Evaluierung eines Systems für einen mobilen Roboter, entwickelt im Rahmen der Vorlesung **"Roboterprogammierung"** an der Hochschule Karlsruhe (HKA).

**Semester:** Wintersemester 2025/2026  
**Dozent:** Prof. Dr. Björn Hein  
**Datum der Aufgabenstellung:** 08.12.2025

---

## 📋 Projektübersicht

Ziel ist die Implementierung eines **CollisionCheckers** für einen ebenen mobilen Roboter, der aus einer Basis und einem Arm mit rotatorischen Gelenken besteht. Darauf aufbauend werden Benchmark-Tests durchgeführt und ein Pick-and-Place Szenario realisiert.

Die Planungsverfahren selbst (z.B. PRM) werden nicht verändert; der Fokus liegt auf der Kollisionsprüfung und der Modellierung des Roboters.

## 🚀 Aufgaben & Features

### 1. Implementierung des CollisionCheckers
Der `CollisionChecker` ermöglicht die Planung für einen Roboter $(x,y)$ mit einem Arm (2 rotatorische Gelenke) unter Berücksichtigung von Hindernissen.

* **Roboterbasis:**
    * Frei definierbare Form (Shape).
    * Startposition der Basis im Raum $(x,y)$ ist vorgebbar.
    * Der Arm beginnt an einer definierten Position auf der Basis.
* **Arm-Konfiguration:**
    * Definition über eine Liste von Segmenten.
    * Format: `[Länge, Dicke, [Min_Winkel, Max_Winkel]]`
    * *Beispiel:* `[5.1, 1, [-3.14, 3.14], [-3.14, 3.14]]` (Länge 5.1, Dicke 1, Limits in Radians (Gelenk 1), Limits in Radians (Gelenk 2)).
* **Kollisionsprüfung:**
    * Implementierung der Hinderniserkennung.
    * **Feature:** Ein-/Ausschalten von Eigenkollisionen (insbesondere Arm vs. Roboterbasis).
* **Visualisierung:**
    * Funktion zur Darstellung von Hindernissen und Roboter in einer gegebenen Konfiguration (analog zu `drawObstacles`).

### 2. Evaluierung & Benchmarking
Vergleich der Algorithmen **LazyPRM** und **VisibilityPRM** in mindestens 5 verschiedenen Benchmark-Umgebungen mit unterschiedlichem Schwierigkeitsgrad.

- [ ] Vergleich des Verhaltens **mit** und **ohne** Eigenkollisionen.
- [ ] Diskussion der Ergebnisse (siehe `docs/Endbericht`).
- [ ] **Animationen:**
    -   Bewegung des Roboters im Arbeitsraum.
    -   Darstellung der Pfade im Konfigurationsraum (für 2-DoF / 3-DoF Systeme).

### 3. Pick-and-Place Szenario
Erweiterung des Systems, um Interaktionen mit der Umgebung zu simulieren.

* Der CollisionChecker wurde erweitert, sodass die Spitze des letzten Armsegments ein Hindernis "greifen" (anhängen) kann.
* **Demo:** Ein mobiler Roboter greift einen Block an Position A und legt ihn an Position B ab.
* *Hinweis:* Positionen werden explizit vorgegeben (keine inverse Kinematik notwendig).

---

## 📝 Endbericht & Theorie

Der Endbericht (zu finden unter `docs/` oder als PDF) umfasst mindestens eine Seite und beantwortet zusätzlich folgende theoretische Fragen:

1.  **Erweiterung auf translatorische Gelenke:**
    * Wie müsste das System erweitert werden, um auch Schubgelenke zu berücksichtigen?
    * Welche Stellen im Code müssten konkret verändert werden?
2.  **Bahnoptimierung:**
    * Wie können die Bewegungsbahnen optimiert oder geglättet werden?
    * Kurze Erläuterung einer möglichen Vorgehensweise.

*Referenzierte Notebooks für Profiling:* `IP-X-0-Benchmarking-concept.ipynb` und `IP-X-1-Automated_PlanerTest.ipynb`.

---

## 🛠 Installation & Nutzung

Voraussetzungen: Python 3.x, Jupyter Notebook, Matplotlib, Numpy (und ggf. weitere Robotik-Bibliotheken der Vorlesung).

1.  **Repository klonen:**
    ```bash
    git clone [https://github.com/USERNAME/REPO-NAME.git](https://github.com/USERNAME/REPO-NAME.git)
    ```

2.  **Abhängigkeiten installieren:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Projekt ausführen:**
    Öffnen Sie die entsprechenden `.ipynb` Dateien im Ordner `notebooks/`, um die Simulationen und Benchmarks zu starten.

## 📂 Dateistruktur

```text
├── assets/             # Bilder und Benchmark-Maps
├── docs/               # Endbericht und Dokumentation
├── notebooks/          # Jupyter Notebooks (Simulation & Tests)
├── src/                # Python Source Code (CollisionChecker Klasse)
├── README.md           # Projektübersicht
└── requirements.txt    # Python Dependencies
