import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
import numpy as np

import networkx as nx

import warnings
warnings.filterwarnings('ignore')

import gurobipy as gp
from gurobipy import GRB

from collections import Counter

import matplotlib.pyplot as plt
from model.instancegenerator import InstanceGenerator

#import model.crawford_network as nt


def prepare_data(inst, K_paths):
    """Prepare data for the matheuristic model."""
    facilities = inst['nodes']['facilities']

    K_j = {}
    for j in inst['nodes']['hubs']:
        K_j[j] = set()
        for k in inst['constants']['K']:
            i_k = inst['origins'][k]  # origine della commodity k
            if (j, i_k) in inst['arcs']['A']:
                K_j[j].add(k)

    U_k = {}
    for k in inst['constants']['K']:
        U_k[k] = set()
        i_k = inst['origins'][k]  # origine della commodity k
        for j in inst['nodes']['hubs']:
            if (j, i_k) in inst['arcs']['A']:
                U_k[k].add(j)

    G = nx.DiGraph()
    A = inst['arcs']['A']
    dist = inst['matrices']['distance']
    hospitals = set(inst['nodes']['hospitals'])


    for (i,j) in A:
        if i in hospitals and j in hospitals:
            continue
        
        weight = (inst['constants']['tau_i'] * inst['facility_indicators'].get(j,0) 
                + dist.loc[i,j] / inst['constants']['v'])
        G.add_edge(i, j, weight=weight)
    P_ik_jk = {}
    P_j_ik_jk = {}

    for k in inst['constants']['K']:
        i_k = inst['commodities'][k][0]
        j_k = inst['commodities'][k][1]
        s_k = inst['commodities'][k][4]  
        n_ik = inst['drones']['available_drones'][i_k]  
        
        
        if s_k <= n_ik:
            path_generator = nx.shortest_simple_paths(G, i_k, j_k, weight='weight')
            valid_paths = []

            if (i_k, j_k) in A:
                valid_paths.append([i_k, j_k])
            
            
            for path in path_generator:
                
                valid = True
                for node in path[1:-1]:
                    if node in hospitals:
                        valid = False
                        break
                
                if valid:
                    valid_paths.append(path)
                    if len(valid_paths) >= K_paths:
                        break
            
            P_ik_jk[k] = valid_paths
            
        else:
        #if s_k > n_ik:   
            P_j_ik_jk[k] = {}
            for j in U_k[k]:  
                
                path_generator = nx.shortest_simple_paths(G, i_k, j_k, weight='weight')
                valid_paths = []
                if (i_k, j_k) in A:
                    valid_paths.append([i_k, j_k])
                facilities_j_ik = []
                if (j,i_k) in inst['arcs']['F_ij'].keys():
                    facilities_j_ik = list(inst['arcs']['F_ij'][(j, i_k)])
                    facilities_j_ik.sort(key=lambda f: inst['matrices']['distance'].loc[j,f])
                for path in path_generator:
                    
                    valid = True
                    for node in path[1:-1]:
                        if node in hospitals:
                            valid = False
                            break
                    
                    if valid:
                        valid_paths.append(path)
                        if len(valid_paths) >= K_paths:
                            break
                
                
                P_j_ik_jk[k][j] = []
                for path in valid_paths:

                    complete_path = [j] # hub j
                    complete_path.extend(facilities_j_ik) # j to i_k
                    complete_path.extend(path)  # i_k to j_k
                    P_j_ik_jk[k][j].append(complete_path)

    A_p= {}
    F_p = {}
    P = {}

    # P_{i^kj^k}(K)
    for k in inst['constants']['K']:
        if k in P_ik_jk:
            for idx, path in enumerate(P_ik_jk[k]):
                
                path_id = f"P_ik_jk_{k}_{idx}"
                P[path_id] = path
                
                arcs = []
                nodes = []
                for i in range(len(path) - 1):
                    arcs.append((path[i], path[i+1]))
                
                
                A_p[path_id] = set(arcs)
                F_p[path_id] = set(node for node in path if node in facilities)

    # P_{ji^kj^k}(K)
    for k in inst['constants']['K']:
        if k in P_j_ik_jk:
            for j in P_j_ik_jk[k]:
                for idx, path in enumerate(P_j_ik_jk[k][j]):
                    
                    path_id = f"P_j_ik_jk_{k}_{j}_{idx}"
                    P[path_id] = path
                    
                    arcs = []
                    nodes = []
                    
                    if isinstance(path[0], tuple):
                        arcs.append(path[0])  
                        
                        
                        for i in range(1, len(path) - 1):
                            if isinstance(path[i], tuple):
                                arcs.append(path[i])
                            else:
                                arcs.append((path[i], path[i+1]))
                    else:
                        
                        for i in range(len(path) - 1):
                            arcs.append((path[i], path[i+1]))

                    F_p[path_id] = set([node for node in path if node in facilities])
                    
                    
                    A_p[path_id] = set(arcs)

    
    
    t_p = {}
    for path_id, path in P.items():
        if 'P_ik_jk_' in path_id:
            k = int(path_id.split('_')[3])
            path_type = 'direct'
        elif 'P_j_ik_jk_' in path_id:
            k = int(path_id.split('_')[4])
            path_type = 'hub'
        else:
            continue
        

        l_k = inst['commodities'][k][3]  
        pi_k = inst['commodities'][k][5]     
        s_k = inst['commodities'][k][4]  
        e_k = inst['commodities'][k][2]  
        i_k = inst['origins'][k]         
        n_ik = inst['drones']['available_drones'][i_k]  
        
        if path_type == 'direct':
            facility_time = inst['constants']['tau_i'] * len(F_p[path_id])
            travel_time = sum(inst['matrices']['distance'].loc[i, j] / inst['constants']['v'] 
                            for i, j in A_p[path_id])
            
            t_jk = e_k+facility_time + travel_time
            delay = max(0, t_jk - l_k)
            penalty = pi_k * delay
            
            t_p[path_id] = s_k * (facility_time + travel_time + penalty)
            
        else:  # path_type == 'hub'
            ik_index = path.index(i_k)
            
            # Split path into segments
            #hub_to_ik = path[:ik_index+1]  # Including i_k
            ik_to_jk = path[ik_index:]     # Including i_k and destination

            # Complete path calculation (hub -> j_k)
            facilities_complete = [node for node in path if node in facilities]
            #print(path,facilities_complete)
            facility_time_complete = inst['constants']['tau_i'] * len(facilities_complete)
            
            travel_time_complete = 0
            for i, j in zip(path[:-1], path[1:]):
                travel_time_complete += inst['matrices']['distance'].loc[i, j] / inst['constants']['v']
            
            t_jk_complete = e_k + facility_time_complete + travel_time_complete
            delay_complete = max(0, t_jk_complete - l_k)
            
            
            s_k_p_complete = s_k - n_ik
            cost_complete = 0
            if s_k_p_complete > 0:
                cost_complete = s_k_p_complete * (facility_time_complete + travel_time_complete)
            
            # Partial path calculation (i_k -> j_k)
            facilities_partial = [node for node in ik_to_jk if node in facilities]
            facility_time_partial = inst['constants']['tau_i'] * len(facilities_partial)
            
            travel_time_partial = 0
            for i, j in zip(ik_to_jk[:-1], ik_to_jk[1:]):
                travel_time_partial += inst['matrices']['distance'].loc[i, j] / inst['constants']['v']
            
            t_jk_partial = e_k + facility_time_partial + travel_time_partial
            delay_partial = max(0, t_jk_partial - l_k)
            
            
            s_k_p_partial = n_ik
            cost_partial = 0
            if s_k_p_partial >= 0:
                cost_partial = s_k_p_partial * (facility_time_partial + travel_time_partial)# + penalty_partial)
        
            penalty = (n_ik * pi_k * delay_partial) + ((s_k - n_ik) * pi_k * delay_complete)
            #print(f"Path ID: {path_id}, time: {cost_complete + cost_partial} penalty: {penalty_complete + penalty_partial}")
            #print(delay_complete+delay_partial, cost_complete + cost_partial)
            t_p[path_id] = cost_complete + cost_partial + penalty


    c = {}
    for p in P:
        c[p] = sum(inst['costs']['facility_costs'][f] for f in F_p[p])
        
    return K_j, U_k, P_ik_jk, P_j_ik_jk, A_p, F_p, P, t_p, c

def build_model(inst, P, t_p, c, U_k, K_j, F_p):
    """Build the path formulation model for the matheuristic."""

    model = gp.Model('matheuristic')

    x = {}

    for p in P:
        x[p] = model.addVar(vtype=GRB.BINARY, name=p)

    y = {}
    for j in inst['nodes']['hubs']:
        y[j] = model.addVar(vtype=GRB.CONTINUOUS,lb = 0,ub = 1, name=f"used_hub_{j}")
    
    z = {}
    for i in inst['nodes']['facilities']:
        z[i] = model.addVar(vtype=GRB.CONTINUOUS,lb = 0,ub = 1, name=f"used_facility_{i}")


    obj = gp.quicksum(t_p[p] * x[p] for p in P)
    model.setObjective(obj, GRB.MINIMIZE)

    for k in inst['constants']['K']:
        i_k = inst['commodities'][k][0]  # origine della commodity k
        s_k = inst['commodities'][k][4]  # numero di droni richiesti
        n_ik = inst['drones']['available_drones'][i_k]  # droni disponibili all'origine
        if s_k <= n_ik:
            paths = [p for p in P if 'P_ik_jk_' in p and k == int(p.split('_')[3])]
            model.addConstr(sum(x[p] for p in paths) == 1, name=f'constr(2)_{k}')

        elif s_k > n_ik:
            paths = [p for p in P if 'P_j_ik_jk_' in p and k == int(p.split('_')[4])]
            
            for j in U_k[k]:
                paths = [p for p in P if 'P_j_ik_jk_' in p and (k == int(p.split('_')[4]) and j == p.split('_')[5]+"_"+p.split('_')[6])]
                model.addConstr(sum(x[p] for p in paths) <= 1, name=f'constr(3)_{k}_{j}')
                
            paths = [p for p in P if 'P_j_ik_jk_' in p and k == int(p.split('_')[4])]
            model.addConstr(gp.quicksum(min(len(inst['drones']['hub_drones_sets'][j]),s_k - n_ik) * x[p] for j in U_k[k] for p in paths if j in p) == s_k - n_ik, name=f'constr(4)_{k}')
    
    for j in inst['nodes']['hubs']:
        paths = [p for p in P if 'P_j_ik_jk_' in p and j == p.split('_')[5]+"_"+p.split('_')[6]]
        model.addConstr(sum([min(len(inst['drones']['hub_drones_sets'][j]),inst['commodities'][k][4] - inst['commodities'][k][8]) * x[p] for k in K_j[j] for p in paths if inst['commodities'][k][4] > inst['commodities'][k][8] and k == int(p.split('_')[4])])<= len(inst['drones']['hub_drones_sets'][j]), name=f'constr(5)_{j}')
    
    for j in inst['nodes']['hubs']:
        paths_using_j = [p for p in P if 'P_j_ik_jk_' in p and (j in p)]
        for p in paths_using_j:
            model.addConstr(x[p] <= y[j], name=f"link_path_{p}_to_hub_{j}")
    model.addConstr(gp.quicksum(y[j] * inst['costs']['hub_costs'][j] for j in inst['nodes']['hubs']) <= inst['constants']['b_U'], name="hub_budget")


    for i in inst['nodes']['facilities']:
        paths_using_i = [p for p in P if i in F_p[p]]
        for p in paths_using_i:
            model.addConstr(x[p] <= z[i], name=f"link_path_{p}_to_facility_{i}")

    model.addConstr(gp.quicksum(z[i] * inst['costs']['facility_costs'][i] for i in inst['nodes']['facilities']) <= inst['constants']['b_F'],name="facility_budget")
    

    model.setParam('OutputFlag', 0)  
    model.setParam('TimeLimit', 60)

    model.update()
    model.write('initial_path_formulation.lp')
    return model, x


def save_results(nodes_df, P, x, F_p, model, inst):
    """Save computational results to CSV file."""

    active_paths = [p for p in P if x[p].X > 0.5]
    
    facility_sets = [F_p[p] for p in active_paths]
    used_facilities = set.union(*facility_sets) if facility_sets else set()
    avg_facilities_per_path = (sum(len(F_p[p]) for p in active_paths) / len(active_paths)) if active_paths else 0
    used_hubs = set()
    for p in active_paths:
        if 'P_j_ik_jk_' in p:
            j = p.split('_')[5] + "_" + p.split('_')[6]
            used_hubs.add(j)

    facilities_cost = sum(150000 for _ in used_facilities)
    hubs_cost = sum(50000 for _ in used_hubs)
    

    # Instance features
    instance_features = {
        "N.H. o-d": len(set((k[0], k[1]) for k in inst['commodities'].values())),
        "N.C.": len(inst['constants']['K']),
        "N.H.": len(inst['nodes']['hospitals']),
        "N.F.": len(inst['nodes']['facilities']),
        "N. HUBS": len(inst['nodes']['hubs']),
        "N.A.": len(active_paths),
        "N.D.H": len(inst['drones']['hospital_drones']),
        "N.D.HUBS": len(inst['drones']['hub_drones']),
        "N.D.": len(inst['drones']['hospital_drones']) + len(inst['drones']['hub_drones']), 
        "N.D.R": sum(inst['commodities'][k][4] for k in inst['constants']['K']),
        "B.U.": inst['constants']['b_U'],
        "B.F.": inst['constants']['b_F']
    }

    # Results metrics
    results = {
        "OBJ": model.ObjVal,
        "GAP": model.MIPGap,
        "LB": model.ObjBound,
        "F.C": facilities_cost,
        "H.C": hubs_cost,
        #"Operational Costs": operational_cost,
        "N.F.A": len(used_facilities),
        "N.HUBS.A": len(used_hubs),
        "N. ARCS": len(active_paths),
        "N. HUBS ARCS": len([p for p in active_paths if 'P_j_ik_jk_' in p]),
        "AVG.F.PATH": round(avg_facilities_per_path, 2)
    }

    # Computational performance
    computational_performance = {
        "TIME(s)": model.Runtime
    }

    data = {**instance_features, **results, **computational_performance}
    df_results = pd.DataFrame([data])

    results_filename = 'DMNDP-MH-rome.csv'
    df_results.to_csv(results_filename, mode='a', 
                     header=not pd.io.common.file_exists(results_filename),
                     index=False)
    print(f"Results saved to {results_filename}")

def get_base_node_id(node_id: str) -> str:
    """
    Extract the base node ID by removing _origin_k or _dest_k suffixes.
    
    Examples:
        'H_001_origin_5' -> 'H_001'
        'F_010' -> 'F_010'
        'U_hub1' -> 'U_hub1'
    """
    if '_origin_' in node_id:
        return node_id.split('_origin_')[0]
    elif '_dest_' in node_id:
        return node_id.split('_dest_')[0]
    return node_id


def get_node_coordinates(node_id: str, nodes_df: pd.DataFrame) -> tuple:
    """
    Get latitude and longitude for a node from the nodes DataFrame.
    Handles various node ID formats.
    """
    base_node = get_base_node_id(node_id)
    
    # Direct match
    node_row = nodes_df[nodes_df['id'] == base_node]
    if not node_row.empty:
        return node_row['lat'].values[0], node_row['lon'].values[0]
    
    # Try removing common prefixes
    for prefix in ['H_', 'F_', 'U_']:
        if base_node.startswith(prefix):
            alt_id = base_node[len(prefix):]
            node_row = nodes_df[nodes_df['id'] == alt_id]
            if not node_row.empty:
                return node_row['lat'].values[0], node_row['lon'].values[0]
    
    return None, None


def save_solution(nodes_df: pd.DataFrame, inst: dict, P: dict, x: dict, 
                  F_p: dict, A_p: dict, filename: str = "solution.txt"):
    """
    Save the optimal solution to a file with node and arc information.
    
    Format:
        NODES: #numero di nodi attivati
        ARCS: #numero di archi di collegamento
        NODE
        Idn  name         x         y        Type
        1    hospital_X  2.35     10.57      HUB
        ...
        ARC
        Ida Idn_o Idn_d Avg_ttime[min.] Label
        1        2         5          123                USED
        ...
    
    Parameters
    ----------
    nodes_df : pd.DataFrame
        DataFrame with columns ['id', 'name', 'lat', 'lon', 'type'] for all nodes
    inst : dict
        Instance dictionary containing problem data (commodities, nodes, matrices, etc.)
    P : dict
        Dictionary of paths {path_id: [list of nodes in path]}
    x : dict
        Dictionary of Gurobi decision variables for path selection
    F_p : dict
        Dictionary of facilities used in each path {path_id: set of facilities}
    A_p : dict
        Dictionary of arcs in each path {path_id: set of (i,j) arcs}
    filename : str
        Output filename (default: "solution.txt")
    
    Returns
    -------
    tuple
        (node_info dict, all_arcs list)
    """
    
    # 1. Identify active paths in the solution
    active_paths = [p for p in P if x[p].X > 0.5]
    
    if not active_paths:
        print("Warning: No active paths found in solution.")
        return {}, []
    
    # 2. Collect activated nodes and used arcs from the solution
    activated_nodes = set()
    used_arcs = set()  # Arcs actually used in routing
    
    for p in active_paths:
        path = P[p]
        for node in path:
            #base_node = get_base_node_id(node)
            activated_nodes.add(node)
        # Collect used arcs from the path
        for arc in A_p[p]:
            used_arcs.add(arc)
    
    # 3. Identify node types
    # Get origins and destinations from all active commodities
    origins = set()
    destinations = set()
    active_commodities = set()
    
    for p in active_paths:
        if 'P_ik_jk_' in p:
            k = int(p.split('_')[3])
        elif 'P_j_ik_jk_' in p:
            k = int(p.split('_')[4])
        else:
            continue
        active_commodities.add(k)
        origins.add(inst['commodities'][k][0])
        destinations.add(inst['commodities'][k][1])
    
    # Get hubs from active paths
    used_hubs = set()
    for p in active_paths:
        if 'P_j_ik_jk_' in p:
            parts = p.split('_')
            hub_id = parts[5] + "_" + parts[6]
            used_hubs.add(hub_id)
    
    # Get facilities from active paths
    used_facilities = set()
    for p in active_paths:
        used_facilities.update(F_p[p])
    
    # 4. Build node_info dictionary with coordinates and types
    node_info = {}
    
    for node in activated_nodes:
        lat, lon = get_node_coordinates(node, nodes_df)
        
        if lat is None:
            print(f"Warning: Could not find coordinates for node {node}")
            continue
        
        base_node = get_base_node_id(node)
        node_row = nodes_df[nodes_df['id'] == base_node]
        node_name = node_row['name'].values[0] if not node_row.empty else node
        
        # Determine node type based on role in solution
        if node in used_hubs or node in inst['nodes']['hubs']:
            node_type = "HUB"
        elif node in origins:
            node_type = "H_ORIGIN"
        elif node in destinations:
            node_type = "H_DESTINATION"
        elif node in used_facilities or node in inst['nodes']['facilities']:
            node_type = "FACILITY"
        elif '_origin_' in node:
            node_type = "H_ORIGIN"
        elif '_dest_' in node:
            node_type = "H_DESTINATION"
        else:
            # Check base type from nodes_df
            base_node = get_base_node_id(node)
            node_row = nodes_df[nodes_df['id'] == base_node]
            #node_name = node_row['name'].values[0] if not node_row.empty else node
            if not node_row.empty:
                df_type = node_row['type'].values[0]
                if df_type == 'hospital':
                    node_type = "H_ORIGIN"  # Default for hospitals
                elif df_type == 'hub':
                    node_type = "HUB"
                elif df_type == 'station':
                    node_type = "FACILITY"
                else:
                    node_type = "FACILITY"
            else:
                node_type = "FACILITY"
        
        node_info[node] = {
            'name': node_name,
            'x': lat,
            'y': lon,
            'type': node_type
        }
    
    # 5. Build all possible arcs between activated nodes
    all_arcs = []
    arc_id = 1
    distance_matrix = inst['matrices']['distance']
    v = inst['constants']['v']  # drone speed in km/h
    
    activated_nodes_list = list(activated_nodes)
    
    for node_o in activated_nodes_list:
        for node_d in activated_nodes_list:
            if node_o == node_d:
                continue
            
            # Check if this arc exists in the instance arcs
            if (node_o, node_d) not in inst['arcs']['A'] and (node_o, node_d) not in inst['arcs']['A_U']:
                continue
            
            # Calculate travel time in minutes
            try:
                dist_km = float(distance_matrix.loc[node_o, node_d])
                travel_time_hours = dist_km / v
                travel_time_min = travel_time_hours * 60
            except (KeyError, ValueError, TypeError) as e:
                continue
            
            # Determine if arc is used in solution routing
            label = "USED" if (node_o, node_d) in used_arcs else "NOT USED"
            
            all_arcs.append({
                'Ida': arc_id,
                'Idn_o': node_o,
                'Idn_d': node_d,
                'Avg_ttime': round(travel_time_min, 2),
                'Label': label
            })
            arc_id += 1
    
    # 6. Write to file
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"NODES: {len(node_info)}\n")
        f.write(f"ARCS: {len(all_arcs)}\n")
        f.write("\n")
        
        # Write nodes section
        f.write("NODE\n")
        f.write(f"{'Idn':<40} {'name':<30} {'x':<15} {'y':<15} {'Type':<15}\n")
        
        # Sort nodes by type then by ID for better readability
        type_order = {'HUB': 0, 'H_ORIGIN': 1, 'H_DESTINATION': 2, 'FACILITY': 3}
        sorted_nodes = sorted(node_info.items(), 
                             key=lambda x: (type_order.get(x[1]['type'], 4), x[0]))
        
        for node_id, info in sorted_nodes:
            f.write(f"{node_id:<40} {info['name']:<30} {info['x']:<15.6f} {info['y']:<15.6f} {info['type']:<15}\n")
        
        f.write("\n")
        
        # Write arcs section
        f.write("ARC\n")
        f.write(f"{'Ida':<8} {'Idn_o':<40} {'Idn_d':<40} {'Avg_ttime[min.]':<20} {'Label':<10}\n")
        
        # Sort arcs: USED first, then by ID
        sorted_arcs = sorted(all_arcs, key=lambda a: (0 if a['Label'] == 'USED' else 1, a['Ida']))
        
        for arc in sorted_arcs:
            f.write(f"{arc['Ida']:<8} {arc['Idn_o']:<40} {arc['Idn_d']:<40} {arc['Avg_ttime']:<20.2f} {arc['Label']:<10}\n")
        
        f.write("\n")
        
        # Write legend
        f.write("LEGENDA\n")
        f.write("Idn: id del nodo\n")
        f.write("Idn_o: id del nodo origine dell'arco\n")
        f.write("Idn_d: id del nodo destinazione dell'arco\n")
        f.write("Ida: id arco\n")
        f.write("name: nome dell'ospedale/struttura\n")
        f.write("x: coordinata geografica latitudine\n")
        f.write("y: coordinata geografica longitudine\n")
        f.write("Type: nodo HUB, H_ORIGIN (Ospedale di origine), H_DESTINATION (Ospedale di destinazione), FACILITY\n")
        f.write("Avg_ttime[min.]: tempo medio di viaggio del drone sull'arco, espresso in minuti\n")
        f.write("Label: USED se l'arco è utilizzato nella soluzione ottima; NOT USED, viceversa.\n")
    
    # Print summary
    n_used = sum(1 for a in all_arcs if a['Label'] == 'USED')
    n_not_used = sum(1 for a in all_arcs if a['Label'] == 'NOT USED')
    
    print(f"\n{'='*60}")
    print(f"Solution saved to: {filename}")
    print(f"{'='*60}")
    print(f"  Activated nodes: {len(node_info)}")
    print(f"    - HUBs: {sum(1 for n in node_info.values() if n['type'] == 'HUB')}")
    print(f"    - H_ORIGIN: {sum(1 for n in node_info.values() if n['type'] == 'H_ORIGIN')}")
    print(f"    - H_DESTINATION: {sum(1 for n in node_info.values() if n['type'] == 'H_DESTINATION')}")
    print(f"    - FACILITY: {sum(1 for n in node_info.values() if n['type'] == 'FACILITY')}")
    print(f"  Total arcs: {len(all_arcs)}")
    print(f"    - USED: {n_used}")
    print(f"    - NOT USED: {n_not_used}")
    print(f"{'='*60}\n")
    
    return node_info, all_arcs



def visualize_solution(nodes_df, P, x, F_p):
    """
    Visualize the network solution with nodes from DataFrame.
    """
    fig, ax = plt.subplots(figsize=(12,6))
    

    G = nx.DiGraph()
    
    node_mapping = {}
    
    for idx, row in nodes_df.iterrows():
        node_id = row['id']
        G.add_node(node_id, 
                  pos=(row['lon'], row['lat']),
                  type=row['type'])
        
        node_mapping[f"H_{node_id}"] = node_id
        node_mapping[f"F_{node_id}"] = node_id
        node_mapping[f"U_{node_id}"] = node_id
        node_mapping[node_id] = node_id

    hospitals = [n for n in G.nodes if G.nodes[n]['type'] == 'hospital']
    stations = [n for n in G.nodes if G.nodes[n]['type'] == 'station']
    hubs = [n for n in G.nodes if G.nodes[n]['type'] == 'hub']

    active_paths = [p for p in P if x[p].X > 0.5]
    print(f"Number of active paths: {len(active_paths)}")
    
    hub_edges = []
    direct_edges = []
    
    for p in active_paths:
            if p in P:
                path = P[p]
                if isinstance(path, list) and len(path) > 1:
                    edges = []
                    for i in range(len(path)-1):

                        node1 = path[i].split('_origin_')[0].split('_dest_')[0]
                        node2 = path[i+1].split('_origin_')[0].split('_dest_')[0]
                        
                        if node1 in node_mapping and node2 in node_mapping:
                            edge = (node_mapping[node1], node_mapping[node2])
                            edges.append(edge)
                    

                    if 'P_j_ik_jk_' in p:  
                        hub_edges.extend(edges)
                    else:
                        direct_edges.extend(edges)
    
    
    G.add_edges_from(hub_edges + direct_edges)

    pos = nx.get_node_attributes(G, 'pos')

    nx.draw_networkx_nodes(G, pos, nodelist=hospitals, 
                          node_color='green', node_size=500,
                          edgecolors='black', linewidths=0.5,
                          node_shape='o', label='Hospital')
    
    nx.draw_networkx_nodes(G, pos, nodelist=stations,
                          node_color='white', node_size=300,
                          edgecolors='black', linewidths=0.5,
                          node_shape='^', label='Facility')
    
    nx.draw_networkx_nodes(G, pos, nodelist=hubs,
                          node_color='blue', node_size=500,
                          edgecolors='black', linewidths=0.5,
                          node_shape='s', label='Hub')
    

    if hub_edges:
        nx.draw_networkx_edges(G, pos, edgelist=hub_edges,
                             edge_color='black', style='dashed',
                             width=0.8, arrowsize=10, arrowstyle='-|>')
    
    if direct_edges:
        nx.draw_networkx_edges(G, pos, edgelist=direct_edges,
                             edge_color='black', style='solid',
                             width=0.8, arrowsize=10, arrowstyle='-|>')
    
  
    nx.draw_networkx_labels(G, pos, font_size=8)
    
    plt.title("Network Solution Visualization")
    plt.legend()
    plt.axis('equal')
    
    # Remove axis spines
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # Calculate statistics
    used_hubs = set(h for e in hub_edges for h in e if G.nodes[h]['type'] == 'hub')
    used_stations = set(s for e in direct_edges + hub_edges 
                       for s in e if G.nodes[s]['type'] == 'station')
    
    # Add solution statistics
    stats_text = (
        f"Solution Statistics:\n"
        f"Active paths: {len(active_paths)}\n"
        f"Used hubs: {len(used_hubs)}\n"
        f"Used facilities: {len(used_stations)}\n"
        f"Total facility cost: {150000*len(used_stations)}\n"
        f"Total hub cost: {sum(50000 for _ in used_hubs)}"
    )
    
    fig.text(0.02, 0.02, stats_text, fontsize=10, family='monospace')
    plt.tight_layout()
    plt.savefig('solution_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()


class MathEuristic():
    def __init__(self, inst, K_paths, nodes_df,density_df):
        self.network = InstanceGenerator(nodes_df,density_df)
        if not inst:
            self.inst = self.network.generate()
        else:
            self.inst = inst
        self.K_paths = K_paths
        self.nodes_df = nodes_df
        self.K_j, self.U_k, self.P_ik_jk, self.P_j_ik_jk, self.A_p, self.F_p, self.P, self.t_p, self.c = prepare_data(self.inst, self.K_paths)
        self.model, self.x = build_model(inst, self.P, self.t_p, self.c, self.U_k, self.K_j, self.F_p)

    def solve(self, max_attempts=3, K_paths_increment=2, file_name = None):

        attempt = 0
        while attempt < max_attempts:
            model = self.model
            model.optimize()

            cost = 0
            if model.status != GRB.INFEASIBLE:
                save_results(self.nodes_df, self.P, self.x, self.F_p, self.model, self.inst)
                if file_name:
                    #save_solution(self.nodes_df, self.P, self.x, self.F_p, self.A_p, file_name)
                    save_solution(
                        nodes_df=self.nodes_df,
                        inst=self.inst,
                        P=self.P,
                        x=self.x,
                        F_p=self.F_p,
                        A_p=self.A_p,
                        filename=file_name
                    )
                print(f"Optimal solution found ---> {model.ObjVal:.2f}")
                for v in model.getVars():
                    if v.X > 0 and ('P_j_ik_jk_' in v.VarName or 'P_ik_jk_' in v.VarName):
                        print(f"{v.VarName}: {v.X}, {self.P[v.VarName]}, {self.t_p[v.VarName]:.2f}")
                        cost += self.c[v.VarName]
                print(f"Total cost: {cost}", "Facility budget:", self.inst['constants']['b_F'], cost<= self.inst['constants']['b_F'])

                selected_paths = [p for p in self.P if self.x[p].X > 0.5]
                facility_sets = [self.F_p[p] for p in selected_paths]
                print(f"Active facility cost : {150000*len(set.union(*facility_sets))}, {150000*len(set.union(*facility_sets))<= self.inst['constants']['b_F']}")
                
                used_hubs = set()
                for v in model.getVars():
                    if v.X > 0 and 'P_j_ik_jk_' in v.VarName:
                        used_hubs.add(v.VarName.split('_')[5]+"_"+v.VarName.split('_')[6])
                hubs_cost = sum(self.inst['costs']['hub_costs'][h] for h in used_hubs)
                print(f"Hub cost: {hubs_cost}", "Hub budget:", self.inst['constants']['b_U'], hubs_cost<= self.inst['constants']['b_U'])
                return model
            else:
                print(f"Model infeasible with K_paths={self.K_paths}. Retrying with more paths...")
                self.K_paths *= K_paths_increment
                self.K_j, self.U_k, self.P_ik_jk, self.P_j_ik_jk, self.A_p, self.F_p, self.P, self.t_p, self.c = prepare_data(self.inst, self.K_paths)
                self.model, self.x = build_model(self.inst, self.P, self.t_p, self.c, self.U_k, self.K_j, self.F_p)
                attempt += 1
        print("Max attempts reached. No feasible solution found.")
        return model

    

