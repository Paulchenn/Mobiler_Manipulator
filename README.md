# Mobile Manipulator: Collision Checking & Path Planning

**Course:** Robot Programming (Roboterprogrammierung)  
**Institution:** Hochschule Karlsruhe (HKA) – Master Robotics and AI in Production (RKIM)  
**Semester:** Winter Semester 2025/2026  
**Lecturer:** Prof. Dr. Björn Hein

## 📖 Project Overview

This repository contains the implementation and evaluation of a planning system for a **planar mobile manipulator**. The system consists of a mobile base (3 Degrees of Freedom: $x, y, \theta$) and a robotic arm with two rotational joints (2 Degrees of Freedom), resulting in a 5-DOF configuration space.

The core objective of this project is the development of a robust **Collision Checker** that handles:

* **Base Collision:** Arbitrary geometric shapes for the mobile base against static obstacles.
* **Arm Collision:** Multi-segment arm collision detection.
* **Self-Collision:** Detection of collisions between the robot's arm and its own base.

Furthermore, the project benchmarks probabilistic sampling-based planning algorithms—specifically **LazyPRM** and **VisibilityPRM**—and simulates a **Pick-and-Place** scenario without the use of inverse kinematics.

## ✨ Key Features

### 1. Custom Collision Checker

A geometric collision detection engine (using `shapely` and `numpy`) that supports:

* **Configurable Robot Design:** Define base shape, arm segment lengths, thicknesses, and joint limits.
* **Environment Interaction:** Detects collisions with static obstacles defined in benchmark maps.
* **Self-Collision Logic:** Toggleable checks to prevent the manipulator from clipping through the mobile base.

### 2. Planning Algorithms & Benchmarking

Integration with sampling-based planners to evaluate performance in complex environments:

* **Algorithms:** `LazyPRM` (multi-query, lazy evaluation) vs. `VisibilityPRM` (optimized for narrow passages).
* **Metrics:** Success rate, path length, number of nodes/edges, and computation time.
* **Batch Evaluation:** Automated runner for statistical analysis over multiple runs ($N=10+$).

### 3. Interactive Simulation

* **Jupyter Notebook Interface:** Full control over the simulation via interactive widgets (sliders, dropdowns).
* **Visualization:** Real-time plotting of the robot configuration ($q_1, q_2, x, y, \theta$), obstacles, start/goal states, and resulting paths.
* **Pick-and-Place:** Simulation of attaching an object to the end-effector and transporting it to a target zone.

## 📂 Repository Structure

The project is organized as follows:

```text
├── assets/                 # Images, icons, and benchmark map definitions
├── docs/                   # Documentation and LaTeX source for the final report
│   ├── LaTeX/              # Thesis/Project report source files
│   └── ...
├── notebooks/              # Jupyter Notebooks (Main entry point)
│   └── Mobile_Manipulator_Main.ipynb  # <--- START HERE
├── src/                    # Source code for CollisionChecker and Planners
│   ├── planners/           # Implementations of PRM, RRT, and Benchmarking tools
│   ├── collision_checker.py# Core collision detection logic
│   ├── IPAnimator.py       # Visualization tools
│   ├── IPTestSuite.py      # Benchmark scenarios definition
│   └── ...
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## 🚀 Getting Started

Follow these steps to set up the environment and run the simulation.

**Prerequisites**

* **Python 3.8+**
* **Jupyter Notebook** or **JupyterLab**

**Installation**

1. **Clone the Repository**

    ```bash
    git clone [https://github.com/YOUR_USERNAME/Mobiler_Manipulator.git](https://github.com/YOUR_USERNAME/Mobiler_Manipulator.git)
    cd Mobiler_Manipulator
    ```

1. **Create a Virtual Environment (Recommended)**

    ```bash
    python -m venv .venv
    # Windows
    .venv\Scripts\activate
    # Linux/Mac
    source .venv/bin/activate
    ```

1. **Install Dependencies** Install the required libraries (including ```numpy```, ```matplotlib```, ```networkx```, ```shapely```, ```ipympl```):

    ```bash
    pip install -r requirements.txt
    ```

## 💻 Usage

The entire project is controlled via the main Jupyter Notebook.

1. **Launch the Environment**

    Start Jupyter in the repository root:

    ```bash
    jupyter notebook
    ```

    Or however you youse Jupyter-Notebooks.

1. **Open the Main Controller**

    Navigate to the ```notebooks/``` folder and open ```Mobile_Manipulator_Main.ipynb```.

1. **Run the Simulation**

    Execute the cells in order. The notebook is structured into specific phases:

    * **Initialization:** Loads the planner factory and benchmark scenarios (defined in ```IPTestSuite.py```).

    * **Interactive Visualization:** Use the provided UI Widgets to manually move the robot joints and base.

        * Test the Attach Object checkbox to simulate gripping.

        * Observe the "COLLISION" or "FREE" status indicator in real-time.

    * **Planning & Benchmarking:**

        * Run the „Planning“ cells to execute ```LazyPRM``` and ```VisibilityPRM``` on all loaded benchmarks.

        * View the generated paths and success/failure logs.

    * **Evaluation:**

        * Run the ”Batch Evaluator“ to perform repeated tests (default: 10 runs).

        * Generate boxplots and statistical data comparing the planners.

## ⚙️ Configuration

You can customize the robot's physics, the test environments, and collision rules by editing the file ```src/IPTestSuite.py```.

* **Self-Collision Check:** Toggle ```SELF_CHECK = True/False``` to enable or disable collision detection between the robot arm and its own base.

* **Robot Geometry:**

    * Modify ```ROBOT_BASE_SHAPE``` to change the polygon defining the mobile base.

    * Adjust ```ROBOT_ARM_CONFIG``` to change arm segment lengths, thicknesses, or joint limits.

* **Benchmark Scenarios:** You can add or modify obstacles in the ```benchList```. Each benchmark defines specific ```obstacles``` (Polygons/Points) and Start/Goal configurations.

## 📊 Documentation

For a deep dive into the theoretical background, the extension to prismatic joints, and path optimization strategies, please refer to the project report located in the ```docs/``` directory.

The LaTeX documentation covers:

1. **System Modeling:** Kinematic chains and configuration space.

1. **Algorithm Analysis:** Comparison of Lazy vs. Visibility strategies.

1. **Future Work:** Theoretical expansion to linear axes and trajectory smoothing.

## 👥 Authors & Acknowledgments

Development: Paul Glaser, Tim Schaefer, Felix Wietschel (Students of the Master's program Robotics and AI in Production (HKA)).

Supervision: Prof. Dr. Björn Hein.

Developed for the ”Roboterprogrammierung“ module, Winter Semester 2025/2026.
