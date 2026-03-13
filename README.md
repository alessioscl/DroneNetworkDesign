Optimization framework for the **Drone-based Medical Network Design Problem (DMNDP)** — designing drone logistics networks to transport medical supplies (organs, blood, lab samples, medications, fluids) between hospitals via intermediate hubs and battery-swap facilities.
 
The repository implements three solution approaches with increasing levels of approximation:
 
1. **Mathematical Model** (`MathematicalModel`) — exact flow-based MILP formulation solved with Gurobi
2. **Matheuristic** (`MathEuristic`) — path-based MIP solved with Gurobi
3. **Heuristic** (`Heuristic`) — Heuristic with no solver dependency at runtime
 
## Repository Structure
 
```
DroneNetworkDesign/
├── model/
│   ├── instancegenerator.py   # Instance generation and loading
│   ├── mathmodel.py           # Exact MILP formulation
│   ├── matheuristic.py        # Path-based matheuristic
│   └── heuristic.py           # Heuristic
├── data/
│   ├── roma5/                 # Rome urban instances 
│   ├── milano5/               # Milan urban instances
│   ├── napoli5/               # Naples urban instances
│   └── nord/                   # Northern Italy regional instances 
├── results/                   # Computational results (.xlsx)
├── solution/                  # Saved solution files
│   ├── roma5/                 
│   ├── milano5/               
│   ├── napoli5/               
│   └── nord/                   
├── experiment_regional.py              # Batch runner for regional instances (Nord)
└── experiment_urban.py        # Batch runner for urban instances (Roma, Milano, Napoli)
```

## Data Format

**Edit-adapt the `input_dir` and file paths at the top of each script to switch between dataset.**
 
Each city folder contains:
 
- A **node file** (e.g., `roma5.csv`) with columns `id, name, lat, lon, type` where type is `hospital`, `facility`, or `hub`.
- **Commodity files** (e.g., `ROMA-10-1-c.csv`) with columns `commodity_id, origin, destination, ready_time, due_time, drone_req, penalty, supply_type, quantity, origin_drones`. File names encode `CITY-K-seed-scenario` where `K` is the number of commodities and scenario is `c` (critical) or `nc` (non-critical).
 
A `population_density.csv` file with columns `lat, lon, density` is also needed by the instance generator to enforce no-fly zones over densely populated areas.

## Requirements
 
- Python 3.9+
- [Gurobi](https://www.gurobi.com/) with a valid license (required for the Mathematical Model and Matheuristic)
- Python packages:
 
```
pip install gurobipy pandas numpy networkx matplotlib
```

## Full Example: Running All Three Algorithms on a Single Instance
 
```python
import pandas as pd
from model.instancegenerator import InstanceGenerator
from model.mathmodel import MathematicalModel
from model.matheuristic import MathEuristic
from model.heuristic import Heuristic
 
# --- 1. Load instance ---
city = "roma5"
nodes_df = pd.read_csv(f"data/{city}/{city}.csv")
density_df = pd.read_csv("data/population_density.csv")
 
commodity_file = f"data/{city}/ROMA-10-1-nc.csv"
 
generator = InstanceGenerator(nodes_df, density_df)
generator.d_max = 15     # max drone range in km (urban setting)
generator.v = 30         # average drone speed in km/h
instance = generator.load_commodities(commodity_file)
 
# --- 2. Exact Mathematical Model (Flow-based Gurobi MILP) ---
math_model = MathematicalModel(instance, nodes_df)
mm_solution = math_model.solve()
 
# --- 3. Matheuristic (Path-based Gurobi MIP) ---
matheuristic = MathEuristic(instance, K_paths=50, nodes_df=nodes_df, density_df=density_df)
mh_solution = matheuristic.solve()
 
# --- 4. Heuristic ---
heuristic = Heuristic(instance, s_max=20)
h_solution = heuristic.run_heuristic()
```
