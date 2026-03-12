import pandas as pd
import numpy as np
import math
import time
from typing import Dict, Set, Tuple, List, Optional, Any
import gurobipy as gp
from gurobipy import GRB
import heapq
import math
import networkx as nx
import matplotlib.pyplot as plt


class InstanceGenerator:
    """
    Generates instances for the medical drone distribution problem.
    Handles node data, density data, and commodity generation.
    """
    
    def __init__(self, nodes_df: pd.DataFrame, density_data: pd.DataFrame):
        """Initialize instance generator with node and density data."""
        self.nodes_df = nodes_df
        self.density = density_data
        self.instance = None
        
        # Physical constraints
        self.d_max = 30      # Maximum drone distance (km)
        self.d_max_empty = 40  # Maximum drone distance when empty (km)
        self.v = 35          # Average drone speed (km/h)
        self.H_max = 10000    # Population density limit
        self.max_weight = 2  # Maximum drone weight capacity
        self.tau_i = 1/12    # Average battery change time
        self.reduction_factor = 0.9
        
        # Drone limits
        self.max_drones_hospital = 3
        self.max_drones_hub = 40
        
        # Commodity settings
        self.num_commodity = 10
        self.medical_supplies = {
            'organ': {
                'due_time': 1.5, 'unit_weight': 2.0, 'max_quantity': 1,
                'ready_time': 0.0, 'base_penalty': 100
            },
            'blood': {
                'due_time': 4.0, 'unit_weight': 0.5, 'max_quantity': 4,
                'ready_time': 0.25, 'base_penalty': 80
            },
            'fluids': {
                'due_time': 8.0, 'unit_weight': 0.5, 'max_quantity': 4,
                'ready_time': 0.5, 'base_penalty': 40
            },
            'lab_sample': {
                'due_time': 10.0, 'unit_weight': 0.2, 'max_quantity': 10,
                'ready_time': 0.3, 'base_penalty': 60
            },
            'medication': {
                'due_time': 10.0, 'unit_weight': 0.3, 'max_quantity': 6,
                'ready_time': 0.4, 'base_penalty': 50
            }
        }

    @staticmethod
    def haversine(lat1, lon1, lat2, lon2):
        """Calculate the great circle distance between two points on Earth."""
        R = 6371.0
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c

    @staticmethod
    def euclidean_distance_geo(lat1, lon1, lat2, lon2):
        """Calculate Euclidean distance between two geographic points."""
        km_per_degree = 111
        delta_lat = lat2 - lat1
        delta_lon = lon2 - lon1
        mean_lat = (lat1 + lat2) / 2
        delta_x = delta_lat * km_per_degree
        delta_y = delta_lon * km_per_degree
        return math.sqrt(delta_x**2 + delta_y**2)

    @staticmethod
    def average_density_along_arc(lat1, lon1, lat2, lon2, density_points):
        """Calculate average population density along an arc."""
        points_on_arc = density_points[
            (density_points['lat'] >= min(lat1, lat2)) & 
            (density_points['lat'] <= max(lat1, lat2)) &
            (density_points['lon'] >= min(lon1, lon2)) & 
            (density_points['lon'] <= max(lon1, lon2))
        ]
        return points_on_arc['density'].mean() if not points_on_arc.empty else np.nan

    @staticmethod
    def shortest_path(start: str, end: str, A: Set, V: Set, 
                 distance_matrix: pd.DataFrame, tau_i: float, 
                 a_f_i: Dict, v: float, forbidden_nodes: Set[str] = None) -> Tuple[List[Tuple[str, str]], float, Set[str]]:
        """Calculate shortest path between two nodes considering facilities, avoiding forbidden nodes."""
        if forbidden_nodes is None:
            forbidden_nodes = set()
        
        min_heap = [(0, start, [], set())]
        shortest_times = {node: float('inf') for node in V}
        shortest_times[start] = 0
        
        while min_heap:
            current_time, current_node, path, facility_nodes = heapq.heappop(min_heap)
            
            if current_node == end:
                return path, current_time, facility_nodes
            
            for (i, j) in A:
                if i == current_node:
                    # Skip if j is a forbidden node (but allow the destination)
                    if j in forbidden_nodes and j != end:
                        continue
                        
                    travel_time_arc = tau_i * a_f_i[j] + distance_matrix.loc[i,j] / v
                    new_time = current_time + travel_time_arc
                    new_facility_nodes = facility_nodes.copy()
                    if a_f_i.get(j, 0) == 1:
                        new_facility_nodes.add(j)
                    
                    if new_time < shortest_times[j]:
                        shortest_times[j] = new_time
                        heapq.heappush(min_heap, (new_time, j, path + [(i, j)], new_facility_nodes))
        
        return None, float('inf'), set()


    def distance(self, node1, node2):
        """Calcola la distanza in km tra due nodi usando la formula di Haversine."""
        # Recupera le coordinate del nodo1
        node1_info = self.nodes_df.loc[self.nodes_df['id'] == node1]
        lat1 = node1_info['lat'].values[0]
        lon1 = node1_info['lon'].values[0]
        
        # Recupera le coordinate del nodo2
        node2_info = self.nodes_df.loc[self.nodes_df['id'] == node2]
        lat2 = node2_info['lat'].values[0]
        lon2 = node2_info['lon'].values[0]
        return self.haversine(lat1, lon1, lat2, lon2)

    def generate(self, scenario: str = 'non-critical', commodity_df: pd.DataFrame = None) -> Dict:
        """
        Generate complete medical transport scenario.
        Se commodity_df è fornito, l'istanza viene generata a partire dai dati contenuti nel DataFrame;
        altrimenti, le commodity vengono create casualmente.
        """
        K = set(range(1, self.num_commodity + 1))
        total_commodities = len(K)
        pi_k = {}
        
        # Se commodity_df non è fornito, genera le commodity casualmente
        if commodity_df is None:
            if scenario == 'critical':
                min_urgent = math.ceil(0.10 * total_commodities)
                max_urgent = math.floor(0.30 * total_commodities)
                urgent_count = np.random.randint(min_urgent, max_urgent + 1)
            else:
                urgent_count = np.random.randint(0, math.ceil(0.10 * total_commodities) + 1)
            
            urgent_indices = set(np.random.choice(list(K), size=urgent_count, replace=False))
            commodities = {}
            n_i = {}
            
            for k in K:
                valid_pair = False
                attempt_count = 0
                while not valid_pair:
                    attempt_count += 1
                    if attempt_count > 100:
                        raise ValueError("Impossibile trovare una coppia di ospedali che rispetti i vincoli di distanza dopo 100 tentativi.")
                    i_k = self.nodes_df.loc[self.nodes_df['type'] == 'hospital']['id'].sample(1).values[0]
                    j_k = i_k
                    while j_k == i_k:
                        j_k = self.nodes_df.loc[self.nodes_df['type'] == 'hospital']['id'].sample(1).values[0]
                    
                    n_i[i_k] = np.random.randint(1, self.max_drones_hospital)
                    
                    if k in urgent_indices:
                        supply_type = np.random.choice(['organ', 'blood'])
                    else:
                        supply_type = np.random.choice(['fluids', 'lab_sample', 'medication'])
                    
                    supply_info = self.medical_supplies[supply_type]
                    max_distance = self.v * (supply_info['due_time'] - supply_info['ready_time']) * self.reduction_factor
                    actual_distance = self.distance(i_k, j_k)
        
                    if actual_distance <= max_distance:
                        valid_pair = True
                
                quantity = 1 if supply_type == 'organ' else np.random.randint(1, 3 * supply_info['max_quantity'] + 1)
                total_weight = quantity * supply_info['unit_weight']
                s_k = math.ceil(total_weight / self.max_weight)
                penalty = supply_info['base_penalty']
                pi_k[k] = penalty
                commodities[k] = [i_k, j_k, supply_info['ready_time'], 
                                  supply_info['due_time'], s_k, penalty, supply_type, quantity, n_i[i_k]]
            
            # Creazione di copie per i nodi ospedale (origine e destinazione)
            new_commodities = {}
            origin_nodes = {commodities[k][0] for k in K}
            dest_nodes = {commodities[k][1] for k in K}
            new_n_i = {}  # Nuovo dizionario per i droni

            for k in K:
                i_k, j_k = commodities[k][0], commodities[k][1]
                values = list(commodities[k])
                if i_k in origin_nodes:
                    new_node = f"{i_k}_origin_{k}"
                    values[0] = new_node
                    new_n_i[new_node] = n_i[i_k]  # Copia il numero di droni per il nuovo nodo
                    values[8] = n_i[i_k]
                if j_k in dest_nodes:
                    new_node = f"{j_k}_dest_{k}"
                    values[1] = new_node
                new_commodities[k] = values

            n_i = new_n_i
                
        else:
            # Se commodity_df è fornito, convertiamo il DataFrame nel formato atteso
            #df = commodity_df.set_index("commodity_id")
            commodity_dict = commodity_df.to_dict(orient="index")
            new_commodities = {}
            K = set(range(1, len(commodity_df)+1))
            for k, v in commodity_dict.items():
                new_commodities[int(k)] = [
                    v['origin'],
                    v['destination'],
                    v['ready_time'],
                    v['due_time'],
                    v['drone_req'],
                    v['penalty'],
                    v['supply_type'],
                    v['quantity'],
                    v['origin_drones']
                ]
            for k in new_commodities:
                pi_k[k] = new_commodities[k][5]
            
            n_i = {}
            for k, values in new_commodities.items():
                n_i[values[0]] = values[8]
        
        # Generazione del resto dell'istanza a partire dalle commodity
        O_k_H = {k: new_commodities[k][0] for k in K}
        D_k_H = {k: new_commodities[k][1] for k in K}
        H = set().union(O_k_H.values()).union(D_k_H.values())
        F = set(self.nodes_df.loc[self.nodes_df['type'] == 'facility']['id'])
        U = set(self.nodes_df.loc[self.nodes_df['type'] == 'hub']['id'].values)
        V = H.union(F).union(U)
        
        #n_i = {}
        #for hospital in O_k_H.values():
        #    n_i[hospital] = np.random.randint(1, 3)

        drone_sets = {}
        for hospital, num_drones in n_i.items():
            drone_sets[hospital] = {f'D{i+1}_{hospital}' for i in range(num_drones)}

        DELTA_H = set().union({f'D{i+1}_{hospital}' for hospital, num_drones in n_i.items() for i in range(num_drones)})
        DELTA_u = {hub: self.max_drones_hub for hub in U}
        delta_u = set().union({f'D{i+1}_{hub}' for hub, num_drones in DELTA_u.items() for i in range(num_drones)})
        delta_hub = {hub: [f'D{i+1}_{hub}' for i in range(num_drones)] for hub, num_drones in DELTA_u.items()}

        coordinates = self.nodes_df[['id', 'lat', 'lon']].set_index('id')
        V_list = list(V)
        distance_matrix = pd.DataFrame(index=V_list, columns=V_list)
        housing_matrix = pd.DataFrame(index=V_list, columns=V_list)

        for i in V_list:
            for j in V_list:
                i_base = i.split('_origin_')[0].split('_dest_')[0]
                j_base = j.split('_origin_')[0].split('_dest_')[0]
                lat1, lon1 = coordinates.loc[i_base, ['lat', 'lon']]
                lat2, lon2 = coordinates.loc[j_base, ['lat', 'lon']]
                distance_matrix.loc[i, j] = self.haversine(lat1, lon1, lat2, lon2)
                housing_matrix.loc[i, j] = self.average_density_along_arc(lat1, lon1, lat2, lon2, self.density)
        housing_matrix = housing_matrix.fillna(0)

        A_U = {(i, j) for i in V for j in V if i not in D_k_H.values() and i != j 
               and distance_matrix.loc[i, j] <= self.d_max and housing_matrix.loc[i, j] <= self.H_max}
        A_0 = {(i, j) for i in (V - U) for j in (V - U) if i not in D_k_H.values() and i != j 
               and distance_matrix.loc[i, j] <= self.d_max and housing_matrix.loc[i, j] <= self.H_max}
        A_1 = {(i, j) for i in F for j in O_k_H.values() if distance_matrix.loc[i, j] <= self.d_max 
               and housing_matrix.loc[i, j] <= self.H_max}
        all_origins = set(O_k_H.values())
        A_2 = set()
        a_F_i = {node: 1 if node in F else 0 for node in V}
        for u in U:
            for j in O_k_H.values():
                path, _, _ = self.shortest_path(u, j, A_U, V, distance_matrix, self.tau_i, a_F_i, self.v, all_origins - {j})
                if path and all(distance_matrix.loc[s, t] <= self.d_max_empty and housing_matrix.loc[s, t] <= self.H_max for s, t in path):
                    A_2.add((u, j))
        A_3 = {(u, j) for (u, j) in A_2 if any(f in self.shortest_path(u, j, A_U, V, distance_matrix, self.tau_i, a_F_i, self.v, all_origins - {j})[2] for f in F)}
        A = (A_0 - A_1) | A_2
        F_ij = {(u, j): self.shortest_path(u, j, A_U, V, distance_matrix, self.tau_i, a_F_i, self.v,all_origins - {j})[2] for (u, j) in A_3}

        c_F_i = {facility: 150000.0 for facility in F}
        c_U_i = {hub: 50000.0 for hub in U}
        b_F = 150000 * len(F) * 0.5
        b_U = 50000 * len(U) * 0.7

        self.instance = {
            'constants': {
                'd_max': self.d_max,
                'd_max_empty': self.d_max_empty,
                'v': self.v,
                'H_max': self.H_max,
                'max_weight': self.max_weight,
                'tau_i': self.tau_i,
                'K': K,
                'b_F': b_F,
                'b_U': b_U
            },
            'medical_supplies': self.medical_supplies,
            'commodities': new_commodities,
            'penalties': pi_k,
            'nodes': {
                'hospitals': H,
                'facilities': F,
                'hubs': U,
                'all': V
            },
            'drones': {
                'hospital_drones': DELTA_H,
                'hospital_drones_sets': drone_sets,
                'hub_drones': delta_u,
                'hub_drones_sets': delta_hub,
                'available_drones': n_i
            },
            'costs': {
                'facility_costs': c_F_i,
                'hub_costs': c_U_i
            },
            'matrices': {
                'distance': distance_matrix,
                'housing': housing_matrix
            },
            'arcs': {
                'A_U': A_U,
                'A_0': A_0,
                'A_1': A_1,
                'A_2': A_2,
                'A_3': A_3,
                'A': A,
                'F_ij': F_ij
            },
            'facility_indicators': a_F_i,
            'origins': O_k_H,
            'destinations': D_k_H
        }
        print("Istanza generata correttamente. Usa .summary() per visualizzarne i dettagli")
        return self.instance

    def save(self, filename: str = "commodity_instance.csv") -> None:
        """
        Salva il dizionario delle commodity in un file CSV.
        
        Args:
            filename (str): Il nome del file CSV di destinazione.
        """
        if self.instance is None or 'commodities' not in self.instance:
            raise ValueError("Nessuna istanza generata. Esegui prima generate().")
        
        df = pd.DataFrame.from_dict(self.instance['commodities'], orient='index',
                                    columns=['origin', 'destination', 'ready_time',
                                             'due_time', 'drone_req', 'penalty', 'supply_type', 'quantity','origin_drones'])
        df.index.name = "commodity_id"
        df.to_csv(filename)
        print(f"Commodity salvate in {filename}")

    def load_commodities(self, filename: str = "commodity_instance.csv") -> None:
        """
        Carica il dizionario delle commodity da un file CSV e genera l'istanza.
        
        Args:
            filename (str): Il nome del file CSV da cui caricare le commodity.
        """
        df = pd.read_csv(filename, index_col="commodity_id")
        self.instance = self.generate(commodity_df=df)
        print(f"Commodity caricate da {filename}")
        if hasattr(self, 'model'):
                delattr(self, 'model')
        return self.instance

    def summary(self) -> dict:
        """
        Restituisce un riepilogo dell'istanza corrente.
        
        Returns:
            dict: Statistiche riepilogative dell'istanza.
        """
        if self.instance is None:
            raise ValueError("Nessuna istanza generata. Esegui prima generate().")

        print("="*50)
        print("MEDICAL TRANSPORT INSTANCE OVERVIEW")
        print("="*50)
        print("\nCONSTANTS:")
        for key, value in self.instance['constants'].items():
            print(f"{key}: {value}")
        print("\nCOMMODITIES:")
        print(pd.DataFrame(self.instance['commodities']).T)
        print("\nNODE COUNTS:")
        print(f"Hospitals: {len(self.instance['nodes']['hospitals'])}")
        print(f"Facilities: {len(self.instance['nodes']['facilities'])}")
        print(f"Hubs: {len(self.instance['nodes']['hubs'])}")
        print(f"Total nodes: {len(self.instance['nodes']['all'])}")
        print("\nARC COUNTS:")
        for arc_type, arcs in self.instance['arcs'].items():
            if isinstance(arcs, set):
                print(f"{arc_type}: {len(arcs)} arcs")
        print("\nDRONE INFORMATION:")
        print(f"Total hospital drones: {len(self.instance['drones']['hospital_drones'])}")
        for x, y in self.instance['drones']['hospital_drones_sets'].items():
            print(f" {x}: {len(y)}")
        print(f"Total hub drones: {len(self.instance['drones']['hub_drones'])}")