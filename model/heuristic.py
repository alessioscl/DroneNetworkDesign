import pandas as pd
import numpy as np
import networkx as nx
from typing import List, Tuple, Set, Dict
from collections import defaultdict
import matplotlib.pyplot as plt
import math
from itertools import combinations
import time
import copy
import random

import warnings
warnings.filterwarnings('ignore')


class PathDistanceMetric:    
    def __init__(self, graph: nx.DiGraph):
        self.graph = graph
    
    def is_subpath(self, path_jj: List[int], p: List[int], p_prime: List[int]) -> bool:
        """
        Check if path_jj is a subpath of p or p'.
        
        Args:
            path_jj: Path to check
            p: First path
            p_prime: Second path
            
        Returns:
            True if path_jj is a subpath of p or p'
        """
        if len(path_jj) <= 1:
            return False
            
        # Check if path_jj is a subpath of p
        for i in range(len(p) - len(path_jj) + 1):
            if p[i:i+len(path_jj)] == path_jj:
                return True
                
        # Check if path_jj is a subpath of p'
        for i in range(len(p_prime) - len(path_jj) + 1):
            if p_prime[i:i+len(path_jj)] == path_jj:
                return True
                
        return False
    
    def get_distinct_node_pairs(self, p: List[int], p_prime: List[int]) -> Set[Tuple[int, int]]:
        """
        
        Args:
            p: First path
            p_prime: Second path
            
        Returns:
            Set of ordered pairs (j, j') where j ∈ p, j' ∈ p'\{i^{k'}}, j ≠ j'
        """
        pairs = set()
        # p' \ {i^{k'}} means p' without the first node (origin node)
        p_prime_without_origin = p_prime[1:] if len(p_prime) > 1 else []
        
        for j in p:
            for j_prime in p_prime_without_origin:
                if j != j_prime:
                    if not ((j[0] == 'H' and j_prime[0] == 'H') and (j.split('_')[0:3] == j_prime.split('_')[0:3])):
                        pairs.add((j, j_prime))
        return pairs
    
    def get_valid_esps(self, p: List[int], p_prime: List[int]) -> Tuple[Set[Tuple[int, int]], Set[Tuple[int, int]]]:
        """
        Get P_{==2}(p,p') and P_{>2}(p,p') sets according to the definition.
        
        Args:
            p: First path
            p_prime: Second path
            
        Returns:
            Tuple of (P_{==2}, P_{>2}) - sets of node pairs with direct/indirect connections
        """
        distinct_pairs = self.get_distinct_node_pairs(p, p_prime)
        
        p_eq_2 = set()  # Direct connections (|p_{jj'}| = 2)
        p_gt_2 = set()  # Indirect connections (|p_{jj'}| > 2)
        
        for j, j_prime in distinct_pairs:
            try:
                path_jj =  nx.shortest_path(self.graph, j, j_prime, weight= 'weight')
            except nx.NetworkXNoPath:
                path_jj = []
            
            if not path_jj:
                # No path exists - skip this pair
                continue
                #p_eq_2.add((j, j_prime))
                
            # Check if path is a subpath of p or p'
            if self.is_subpath(path_jj, p, p_prime):
                continue
                
            path_length = len(path_jj)
            
            if path_length == 2:
                # Direct connection
                p_eq_2.add((j, j_prime))
            elif path_length > 2:
                # Indirect connection
                p_gt_2.add((j, j_prime))
        
        return p_eq_2, p_gt_2
    
    def calculate_metric(self, p: List[int], p_prime: List[int]) -> float:
        """
        Calculate the distance metric between two ESPs according to the formula:
        d(p,p') = |P_{>2}(p,p')| / max{|P_{==2}(p,p')|, 1} ∈ [0, |p|·(|p'|-1)]
        
        Args:
            p: First path (ESP)
            p_prime: Second path (ESP)
            
        Returns:
            Distance metric d(p,p')
        """

        if p[0].split('_')[0:3] == p_prime[0].split('_')[0:3] and p[-1].split('_')[0:3] == p_prime[-1].split('_')[0:3]:
            return 0.0

        p_eq_2, p_gt_2 = self.get_valid_esps(p, p_prime)
        
        # Calculate metric according to the formula
        numerator = len(p_gt_2)
        denominator = max(len(p_eq_2), 1)
        
        return numerator /  (numerator+denominator)
    
    def get_detailed_analysis(self, p: List[int], p_prime: List[int]) -> Dict:
        """
        Get detailed analysis of the metric calculation for debugging/understanding.
        
        Returns:
            Dictionary with detailed breakdown of the calculation
        """
        distinct_pairs = self.get_distinct_node_pairs(p, p_prime)
        p_eq_2, p_gt_2 = self.get_valid_esps(p, p_prime)
        
        analysis = {
            'paths': {
                'p': p,
                'p_prime': p_prime,
                'p_prime_without_origin': p_prime[1:] if len(p_prime) > 1 else []
            },
            'distinct_pairs': list(distinct_pairs),
            'direct_connections_P_eq_2': list(p_eq_2),
            'indirect_connections_P_gt_2': list(p_gt_2),
            'esp_details': [],
            'metric_components': {
                'numerator': len(p_gt_2),
                'denominator': max(len(p_eq_2), 1),
                'metric': len(p_gt_2) / max(len(p_eq_2), 1)
            }
        }
        
        # Add detailed ESP analysis for each pair
        for j, j_prime in distinct_pairs:
            try:
                path_jj =  nx.shortest_path(self.graph, j, j_prime)
            except nx.NetworkXNoPath:
                path_jj = []
            is_subpath_flag = self.is_subpath(path_jj, p, p_prime) if path_jj else False
            
            connection_type = "no path in A (excluded)"
            if path_jj and not is_subpath_flag:
                #if len(path_jj) == 0:
                #    connection_type = "direct"
                if len(path_jj) == 2:
                    connection_type = "direct (=2)"
                elif len(path_jj) > 2:
                    connection_type = "indirect (>=2)"
            elif is_subpath_flag:
                connection_type = "subpath (excluded)"
            
            analysis['esp_details'].append({
                'pair': (j, j_prime),
                'shortest_path': path_jj,
                'path_length': len(path_jj) if path_jj else 0,
                'is_subpath': is_subpath_flag,
                'connection_type': connection_type
            })
        
        return analysis

    def build_distance_matrix(self, paths: Dict[int, List[int]]) -> Dict[Tuple[int, int], float]:
        """
        Build the distance matrix D for all pairs of ESPs.
        D_{kk'} = d(p,p') if k ≠ k', D_{kk} = 0 for all k ∈ K
        
        Args:
            paths: Dictionary mapping commodity k to its ESP path
            
        Returns:
            Dictionary representing the distance matrix
        """
        distance_matrix = {}
        commodities = list(paths.keys())
        
        for k in commodities:
            for k_prime in commodities:
                if k == k_prime:
                    distance_matrix[(k, k_prime)] = 0.0
                else:
                    distance_matrix[(k, k_prime)] = self.calculate_metric(paths[k], paths[k_prime])
        
        df = pd.Series(distance_matrix).unstack()
        return distance_matrix, df
    


    def visualize_analysis(self, analysis: dict, nodes_df):
        """
        Visualizza i risultati di get_detailed_analysis.
        - Stampa un riepilogo testuale
        - Disegna il grafo con evidenziati p, p_prime e le connessioni distinte
        """
        print("\n=== ANALISI DETTAGLIATA ===")
        print("Percorso p:", analysis['paths']['p'])
        print("Percorso p':", analysis['paths']['p_prime'])
        print("Coppie distinte:", analysis['distinct_pairs'])
        print("Connessioni dirette (|ESP|=2):", analysis['direct_connections_P_eq_2'])
        print("Connessioni indirette (|ESP|>2):", analysis['indirect_connections_P_gt_2'])
        print("\n--- ESP Details ---")
        for esp in analysis['esp_details']:
            print(f"  {esp['pair']}: path={esp['shortest_path']} "
                f"(len={esp['path_length']}), subpath={esp['is_subpath']}, tipo={esp['connection_type']}")
        print("\n--- Metric ---")
        print(f"numeratore={analysis['metric_components']['numerator']}, "
            f"denominatore={analysis['metric_components']['denominator']}, "
            f"metric={analysis['metric_components']['metric']:.3f}")

        for idx, row in nodes_df.iterrows():
            node_id = row['id']
        self.graph.add_node(node_id, 
                  pos=(row['lon'], row['lat']),
                  type=row['type'])

class NeighborhoodConstructor:
    """
    Classe per la costruzione di vicinati semplici ed estesi secondo la procedura
    Neighborhoods_Construction descritta nel paper.
    """
    
    def __init__(self, paths: Dict[int, List[str]], distance_matrix: pd.DataFrame):
        """
        Inizializza il costruttore di vicinati.
        
        Args:
            paths: Dictionary mapping commodity k to its ESP path
            distance_matrix: DataFrame with distance matrix D
        """
        self.paths = paths
        self.distance_matrix = distance_matrix
        self.commodities = list(paths.keys())
        
    def get_sorted_neighbors(self, k: int) -> List[Tuple[int, float]]:
        """
        Costruisce la lista ℒ_p ordinata per valori non decrescenti della metrica d(p,p_j).
        
        Args:
            k: Commodity ID for which to find neighbors
            
        Returns:
            List of (commodity_id, distance) tuples sorted by increasing distance
        """
        if k not in self.commodities:
            return []
        
        # Get distances from commodity k to all others
        distances = []
        for k_j in self.commodities:
            if k_j != k:  # Exclude self
                distance = self.distance_matrix.loc[k, k_j]
                distances.append((k_j, distance))
        
        # Sort by distance (non-decreasing)
        distances.sort(key=lambda x: x[1])
        return distances
    
    def get_simple_neighborhood(self, k: int, s: int = 1) -> Set[int]:
        """
        Costruisce il vicinato 𝒩_s(p) dei primi s ESPs più prossimi a p.
        Secondo la procedura, per s=1 è un vicinato "semplice".
        
        Args:
            k: Commodity ID
            s: Size of neighborhood (number of closest ESPs)
            
        Returns:
            Set of commodity IDs representing the s closest ESPs
        """
        sorted_neighbors = self.get_sorted_neighbors(k)
        
        # Take first s neighbors
        neighborhood = set()
        for i in range(min(s, len(sorted_neighbors))):
            commodity_id, _ = sorted_neighbors[i]
            neighborhood.add(commodity_id)
        
        neighborhood.add(k)
        
        return neighborhood
    
    def get_extended_neighborhood(self, k_start: int, s: int) -> Set[int]:
        """
        Args:
            k_start: Starting commodity ID
            s: Size of extended neighborhood (chain length)
            
        Returns:
            Set of all commodity IDs in the extended neighborhood
        """
        if s == 1:
            # Per s=1, è un vicinato semplice
            return self.get_simple_neighborhood(k_start, s)
        
        extended_neighborhood = set()
        current_k = k_start
        visited = set()  # To avoid infinite loops
        
        for step in range(s):
            if current_k in visited:
                break  # Avoid cycles
                
            visited.add(current_k)
            
            # Get simple neighborhood (size 1) of current commodity
            simple_neighbors = self.get_simple_neighborhood(current_k, s)
            extended_neighborhood.update(simple_neighbors)
            
            # Move to the closest neighbor for next iteration
            if simple_neighbors:
                current_k = next(iter(simple_neighbors))  # Get the single neighbor
            else:
                break  # No more neighbors
        extended_neighborhood.add(k_start)
        return extended_neighborhood
    
    def build_all_neighborhoods(self, s_max: int = 3, use_extended: bool = False) -> Dict[int, Dict[int, Set[int]]]:
        """
        Costruisce tutti i vicinati per tutte le commodity e per tutti i valori di s.
        
        Args:
            s_max: Maximum neighborhood size to compute
            use_extended: If True, use extended neighborhoods; if False, use simple neighborhoods
            
        Returns:
            Dictionary: {commodity_id: {s: neighborhood_set}}
        """
        all_neighborhoods = {}
        
        for k in self.commodities:
            all_neighborhoods[k] = {}
            for s in range(1, s_max + 1):
                if use_extended:
                    all_neighborhoods[k][s] = self.get_extended_neighborhood(k, s)
                else:
                    all_neighborhoods[k][s] = self.get_simple_neighborhood(k, s)
        
        return all_neighborhoods
    
    def calculate_neighborhood_distance(self, neighborhood: Set[int]) -> float:
        """
        Calcola d(N_s(p)) = min_{(p',p'') ∈ N_s(p) x N_s(p): p' ≠ p''}{d(p',p'')}
        
        Args:
            neighborhood: Set di commodity IDs nel vicinato
            
        Returns:
            Distanza minima tra percorsi del vicinato
        """
        #if len(neighborhood) < 2:
        #    return float('inf')
        
        min_distance = float('inf')
        neighborhood_list = list(neighborhood)
        
        # Per ogni coppia (p', p'') nel vicinato con p' ≠ p''
        for i in range(len(neighborhood_list)):
            for j in range(len(neighborhood_list)):
                k1, k2 = neighborhood_list[i], neighborhood_list[j]
                if k1 != k2:
                
                # d(p', p'')
                    distance_12 = self.distance_matrix.loc[k1, k2]
                    # d(p'', p')
                    distance_21 = self.distance_matrix.loc[k2, k1]
                
                # Prendi il minimo tra le due direzioni
                    pair_min_distance = min(distance_12, distance_21)
                    min_distance = min(min_distance, pair_min_distance)
        
        return min_distance
    
    def get_neighborhoods_sorted_by_distance(self, s: int, use_extended: bool = False) -> List[Tuple[int, Set[int], float]]:
        """
        Restituisce tutti i vicinati di dimensione s ordinati per d(N_s(p)) crescente.
        Questo supporta l'algoritmo nel trovare il vicinato con ordine specificato.
        
        Args:
            s: Neighborhood size
            use_extended: If True, use extended neighborhoods; if False, use simple neighborhoods
            
        Returns:
            List of (commodity_id, neighborhood, distance) tuples sorted by distance
        """
        neighborhoods_with_distances = []
        
        for k in self.commodities:
            if use_extended:
                neighborhood = self.get_extended_neighborhood(k, s)
            else:
                neighborhood = self.get_simple_neighborhood(k, s)
            
            if len(neighborhood) > 0:  # Solo se il vicinato non è vuoto
                distance = self.calculate_neighborhood_distance(neighborhood)
                neighborhoods_with_distances.append((k, neighborhood, distance))
        
        # Ordina per distanza crescente
        neighborhoods_with_distances.sort(key=lambda x: x[2])
        return neighborhoods_with_distances

    def get_best_neighborhood_by_order(self, s: int, order: int = 1, use_extended: bool = False) -> Tuple[int, Set[int], float]:
        """
        Restituisce il vicinato con l'ordine specificato secondo d(N_s(p)).
        
        Per vicinati di dimensione 1, usa la distanza diretta tra commodity originale e vicino.
        """
        if s == 1:
            # Per vicinati semplici di dimensione 1, usa approccio diverso
            commodity_distances = []
            
            for k in self.commodities:
                sorted_neighbors = self.get_sorted_neighbors(k)
                if sorted_neighbors:
                    # Il vicino più prossimo
                    closest_neighbor, direct_distance = sorted_neighbors[0]
                    neighborhood = {closest_neighbor}
                    
                    # Per s=1, la "distanza del vicinato" è la distanza diretta
                    commodity_distances.append((k, neighborhood, direct_distance))
            
            # Ordina per distanza crescente
            commodity_distances.sort(key=lambda x: x[2])
            
            if 1 <= order <= len(commodity_distances):
                return commodity_distances[order - 1]
            else:
                return (None, set(), float('inf'))
        
        else:
            # Per s > 1, usa la logica originale
            neighborhoods_with_distances = []
            
            for k in self.commodities:
                if use_extended:
                    neighborhood = self.get_extended_neighborhood(k, s)
                else:
                    neighborhood = self.get_simple_neighborhood(k, s= 1)
                
                if len(neighborhood) > 0:
                    distance = self.calculate_neighborhood_distance(neighborhood)
                    neighborhoods_with_distances.append((k, neighborhood, distance))
            
            neighborhoods_with_distances.sort(key=lambda x: x[2])
            #return neighborhood[order]
            
            if 1 <= order <= len(neighborhoods_with_distances):
                return neighborhoods_with_distances[order - 1]
            else:
                return (None, set(), float('inf'))
            
    def n_construction_s(self, s: int, use_extended: bool = False) -> Dict[int, Set[int]]:
        """
        Implementa la procedura N_Construction_s del paper.
        
        Args:
            s: Neighborhood size
            use_extended: If True, constructs extended neighborhoods; if False, simple neighborhoods
            
        Returns:
            Dictionary mapping each commodity to its neighborhood of size s
        """
        neighborhoods = {}
        
        for k in self.commodities:
            if use_extended:
                neighborhoods[k] = self.get_extended_neighborhood(k, s)
            else:
                neighborhoods[k] = self.get_simple_neighborhood(k, s)
        
        return neighborhoods
    
    def print_neighborhood_info(self, s: int, use_extended: bool = False):
        """
        Stampa informazioni sui vicinati per debugging.
        """
        print(f"\n=== Vicinati di dimensione s={s} ({'estesi' if use_extended else 'semplici'}) ===")
        
        neighborhoods = self.n_construction_s(s, use_extended)
        
        for k in self.commodities:
            neighborhood = neighborhoods[k]
            if len(neighborhood) > 0:
                distance = self.calculate_neighborhood_distance(neighborhood)
                print(f"Commodity {k}: vicinato = {neighborhood}, d(N_s(p)) = {distance:.3f}")
            else:
                print(f"Commodity {k}: vicinato vuoto")
        
        # Mostra l'ordinamento per distanza
        sorted_neighborhoods = self.get_neighborhoods_sorted_by_distance(s, use_extended)
        print(f"\nOrdinamento per d(N_s(p)) crescente:")
        for i, (k, neighborhood, distance) in enumerate(sorted_neighborhoods, 1):
            print(f"  Ordine {i}: Commodity {k}, d(N_s(p)) = {distance:.3f}")

class NeighborhoodExplorer:

    
    def __init__(self, graph: nx.DiGraph, paths: Dict[int, List[str]]):
        """
        Inizializza l'esploratore di vicinati.
        
        Args:
            graph: Grafo originale G = (V, A)
            paths: Dictionary mapping commodity k to its ESP path
        """
        self.original_graph = graph.copy()
        self.paths = paths.copy()
        
    def calculate_facility_traversal_time(self, facility: str, path: List[str]) -> float:
        """
        Calcola τ^p'_c(j) = t^p'_ρ(j),j + t^p'_j,σ(j)
        
        Args:
            facility: Facility j
            path: ESP p' che contiene la facility
            
        Returns:
            Tempo totale di attraversamento
        """
        if facility not in path:
            return 0.0
        
        facility_index = path.index(facility)
        
        # Se la facility è all'inizio o alla fine, non ha predecessore/successore
        if facility_index == 0 or facility_index == len(path) - 1:
            return 0.0
        
        predecessor = path[facility_index - 1]  # ρ(j)
        successor = path[facility_index + 1]    # σ(j)
        
        # Ottieni i tempi degli archi
        try:
            time_in = self.original_graph[predecessor][facility].get('weight', 1.0)   # t^p'_ρ(j),j
            time_out = self.original_graph[facility][successor].get('weight', 1.0)    # t^p'_j,σ(j)
            return time_in + time_out
        except KeyError:
            return float('inf')
    
    def get_all_facilities_in_neighborhood(self, neighborhood: Set[int]) -> List[Tuple[str, int, float]]:
        """
        Ottiene tutte le facility del vicinato con i loro tempi di attraversamento.
        Ordinate per valori NON CRESCENTI di τ^p'_c(j).
        
        Args:
            neighborhood: Set di commodity IDs nel vicinato
            
        Returns:
            Lista di (facility, commodity_id, traversal_time) ordinata per tempo DECRESCENTE
        """
        facility_times = []
        
        for k in neighborhood:
            if k in self.paths:
                path = self.paths[k]
                facilities = [node for node in path if node.startswith('F_')]
                
                for facility in facilities:
                    traversal_time = self.calculate_facility_traversal_time(facility, path)
                    facility_times.append((facility, k, traversal_time))
        
        # Ordina per tempo DECRESCENTE (valori non crescenti)
        facility_times.sort(key=lambda x: x[2], reverse=True)
        return facility_times
    
    def get_all_direct_connections(self, neighborhood: Set[int]) -> Set[Tuple[str, str]]:
        """
        Calcola ⋃_{(p',p'') ∈ N_s(p) × N_s(p): p' ≠ p''} P_{==2}(p',p'')
        
        Args:
            neighborhood: Set di commodity IDs nel vicinato
            
        Returns:
            Set di tutti gli archi diretti tra ESPs del vicinato
        """
        
        
        direct_arcs = set()
        neighborhood_list = list(neighborhood)
        path_metric = PathDistanceMetric(self.original_graph)
        
        # Per ogni coppia (p', p'') nel vicinato con p' ≠ p''
        for k1 in neighborhood_list:
            for k2 in neighborhood_list:
                if k1 != k2 and k1 in self.paths and k2 in self.paths:
                    # Calcola P_{==2}(p', p'')
                    p_eq_2_12, _ = path_metric.get_valid_esps(self.paths[k1], self.paths[k2])
                    # Calcola P_{==2}(p'', p')
                    p_eq_2_21, _ = path_metric.get_valid_esps(self.paths[k2], self.paths[k1])
                    
                    # Aggiungi tutti gli archi diretti
                    for j, j_prime in p_eq_2_12:
                        if self.original_graph.has_edge(j, j_prime):
                            direct_arcs.add((j, j_prime))
                    
                    for j, j_prime in p_eq_2_21:
                        if self.original_graph.has_edge(j, j_prime):
                            direct_arcs.add((j, j_prime))
        #print(direct_arcs)
        
        return direct_arcs
    

    def explore_neighborhood(self, neighborhood: Set[int], p_commodity: int = None) -> Tuple[Set[str], Dict[int, List[str]], Dict[int, List[str]]]:
        """
        Implementa la procedura N_EXPLORATION secondo il documento.
        Salta l'esplorazione se p_commodity ha un path diretto (len == 2).
        
        Args:
            neighborhood: Set di commodity IDs nel vicinato N_s(p)
            p_commodity: ID della commodity p (se non è già in neighborhood)
        
        Returns:
            Tuple (F̄, P(N_s(p)), P̄(N_s(p)))
        """
        # CONTROLLO PATH DIRETTO
        # Se p_commodity ha un path di lunghezza 2 (origine->destinazione), skip esplorazione
        if p_commodity is not None and p_commodity in self.paths:
            p_path = self.paths[p_commodity]
            if len(p_path) == 2:
                #print(f"  Commodity {p_commodity} ha path diretto {p_path[0]} -> {p_path[1]}, skip esplorazione")
                
                # Ritorna lo stato corrente senza modifiche
                current_global_facilities = set()
                for k, path in self.paths.items():
                    path_facilities = [node for node in path if node.startswith('F_')]
                    current_global_facilities.update(path_facilities)
                
                # Ritorna paths originali senza modifiche
                original_paths = {k: self.paths[k].copy() for k in neighborhood if k in self.paths}
                return current_global_facilities, original_paths, original_paths
        #
        ## Assicurati che p sia incluso nel vicinato (N_s(p) ∪ {p})
        extended_neighborhood = neighborhood.copy()
        if p_commodity is not None and p_commodity not in extended_neighborhood:
            extended_neighborhood.add(p_commodity)
    
        
        # Salva i paths originali del vicinato
        original_paths_neighborhood = {k: self.paths[k].copy() for k in extended_neighborhood if k in self.paths}
        
        # Ottieni tutte le facility ordinate per tempo di attraversamento decrescente
        facility_times = self.get_all_facilities_in_neighborhood(extended_neighborhood)
        
        if not facility_times:
            # Nessuna facility nel vicinato
            current_global_facilities = set()
            for k, path in self.paths.items():
                path_facilities = [node for node in path if node.startswith('F_')]
                current_global_facilities.update(path_facilities)
            return current_global_facilities, original_paths_neighborhood, original_paths_neighborhood
        
        # Seleziona la facility j con τ massimo
        #print(facility_times)
        facility_to_remove, commodity_with_facility, max_time = facility_times[0]
        
        # Trova il path p' che attraversa questa facility
        p_prime = self.paths[commodity_with_facility]
        
        # COSTRUZIONE DI G^j
        # Passo 1: Costruisci V^j e A^j
        gj = nx.DiGraph()
        
        # V^j = tutti i nodi degli ESPs in N_s(p) ∪ {p}
        nodes_vj = set()
        for k in extended_neighborhood:
            if k in self.paths:
                nodes_vj.update(self.paths[k])
        gj.add_nodes_from(nodes_vj)
        
        # A^j parte 1: archi degli ESPs originali
        for k in extended_neighborhood:
            if k in self.paths:
                path = self.paths[k]
                for i in range(len(path) - 1):
                    from_node, to_node = path[i], path[i + 1]
                    if self.original_graph.has_edge(from_node, to_node):
                        edge_data = self.original_graph[from_node][to_node].copy()
                        gj.add_edge(from_node, to_node, **edge_data)
        
        # A^j parte 2: archi diretti P_{==2} tra coppie di ESPs
        direct_connections = self.get_all_direct_connections(extended_neighborhood)
        for from_node, to_node in direct_connections:
            if from_node in nodes_vj and to_node in nodes_vj:
                if self.original_graph.has_edge(from_node, to_node):
                    if not gj.has_edge(from_node, to_node):
                        edge_data = self.original_graph[from_node][to_node].copy()
                        gj.add_edge(from_node, to_node, **edge_data)
        
        # Passo 2: Imposta t^p'_ρ(j),j ← +∞ e t^p'_j,σ(j) ← +∞
        # SOLO per gli archi specifici lungo il path p' che contiene j
        if facility_to_remove in p_prime:
            facility_index = p_prime.index(facility_to_remove)
            
            # Se j ha un predecessore ρ(j) in p'
            #if facility_index > 0:
            #    predecessor = p_prime[facility_index - 1]
            #    if gj.has_edge(predecessor, facility_to_remove):
            #        gj[predecessor][facility_to_remove]['weight'] = float('inf')
            #
            ## Se j ha un successore σ(j) in p'
            #if facility_index < len(p_prime) - 1:
            #    successor = p_prime[facility_index + 1]
            #    if gj.has_edge(facility_to_remove, successor):
            #        gj[facility_to_remove][successor]['weight'] = float('inf')

            for pred in gj.predecessors(facility_to_remove):
                if gj.has_edge(pred, facility_to_remove):
                    gj[pred][facility_to_remove]['weight'] = float('inf')

            for succ in gj.successors(facility_to_remove):
                if gj.has_edge(facility_to_remove, succ):
                    gj[facility_to_remove][succ]['weight'] = float('inf')
        
        # Ricalcola gli ESPs sul grafo modificato G^j
        modified_paths = {}
        for k in extended_neighborhood:
            if k in self.paths:
                original_path = self.paths[k]
                source = original_path[0]
                target = original_path[-1]
                if len(original_path) ==2: #self.original_graph.has_edge(source, target):
                        # C'è un arco diretto disponibile
                        modified_paths[k] = [source, target]
                        #print(f"    Commodity {k}: usa arco diretto {source} -> {target}")
                else:
                    try:
                            # Calcola nuovo shortest path su G^j
                            new_path = nx.shortest_path(gj, source, target, weight='weight')
                            modified_paths[k] = new_path
                    except nx.NetworkXNoPath:
                        # Se non esiste percorso, mantieni quello originale
                        modified_paths[k] = original_path.copy()
        
        # Calcola F̄ = insieme globale delle facility dopo la modifica
        facility_bar = set()
        
        # Aggiungi facility dai paths NON nel vicinato esteso (rimangono invariati)
        #for k, path in self.paths.items():
        #    if k not in extended_neighborhood:
        #        path_facilities = [node for node in path if node.startswith('F_')]
        #        facility_bar.update(path_facilities)
        #
        ## Aggiungi facility dai paths modificati del vicinato esteso
        #for k in extended_neighborhood:
        #    if k in modified_paths:
        #        path_facilities = [node for node in modified_paths[k] if node.startswith('F_')]
        #        facility_bar.update(path_facilities)
        for k, path in self.paths.items():
            if k not in extended_neighborhood:
                path_facilities = [node for node in path if node.startswith('F_') and node != facility_to_remove]
                facility_bar.update(path_facilities)

        # Aggiungi facility dai paths modificati del vicinato esteso
        for k in extended_neighborhood:
            if k in modified_paths:
                path_facilities = [node for node in modified_paths[k] if node.startswith('F_') and node != facility_to_remove]
                facility_bar.update(path_facilities)
        
        # P(N_s(p)) = paths originali del vicinato
        # P̄(N_s(p)) = paths modificati del vicinato
        paths_before = {k: self.paths[k] for k in extended_neighborhood if k in self.paths}
        paths_after = modified_paths.copy()
        
        return facility_bar, paths_before, paths_after
    
    def compute_total_traversal_time(self, paths_subset: Dict[int, List[str]]) -> float:
        """
        Calcola la somma dei tempi di attraversamento τ^p_c(j) di tutte le facility nei path forniti.
        """
        total_time = 0.0
        for k, path in paths_subset.items():
            facilities = [node for node in path if node.startswith('F_')]
            for facility in facilities:
                total_time += self.calculate_facility_traversal_time(facility, path)
        return total_time

class HubActivation:
    """
    Classe per la determinazione degli hub da attivare secondo la procedura HUB_ACTIVATION.
    """
    
    def __init__(self, graph: nx.DiGraph, inst, final_paths: Dict[int, List[str]]):#, 
                 #drone_requirements: Dict[int, int], hospital_drones: Dict[str, int],
                 #hub_drones: Dict[str, list], budget_hubs: int, hub_cost: int = 50000):
        """
        Inizializza l'attivatore di hub.
        
        Args:
            graph: Grafo completo G = (V, A)
            final_paths: ESPs finali dopo exploration
            drone_requirements: s^k per ogni commodity k (droni richiesti)
            hospital_drones: n_{i^k} per ogni ospedale origine (droni disponibili)
            hub_capacities: Δ^U(i) per ogni hub i (capacità hub)
            budget_hubs: b^U (budget per hub)
            hub_cost: c (costo costante per hub)
        """
        self.graph = graph
        self.final_paths = final_paths
        self.inst = inst

        inst_df = pd.DataFrame(inst['commodities']).T
        self.drone_requirements = inst_df[4]
        self.hospital_drones = self.inst['drones']['available_drones']
        self.hub_drones = self.inst['drones']['hub_drones_sets']
        self.budget_hubs = self.inst['constants']['b_U']
        self.hub_cost = self.inst['costs']['hub_costs']['HUB_1']
        self.hub_capacities = {i: len(drones) for i, drones in self.hub_drones.items()}
        

    
    def get_origin_hospitals_with_deficit(self) -> Dict[str, int]:
        """
        Identifica ospedali origine con deficit di droni: s^k > n_{i^k}.
        
        Returns:
            Dictionary {origin_node: deficit_amount}
        """
        hospitals_with_deficit = {}
        
        for k, path in self.final_paths.items():
            if k in self.drone_requirements:
                # Primo nodo è l'ospedale origine
                origin_node = path[0]
                
                required_drones = self.drone_requirements[k]  # s^k
                available_drones = self.hospital_drones.get(origin_node, 0)  # n_{i^k}
                
                if required_drones > available_drones:
                    deficit = required_drones - available_drones
                    if origin_node in hospitals_with_deficit:
                        hospitals_with_deficit[origin_node] += deficit
                    else:
                        hospitals_with_deficit[origin_node] = deficit
        
        return hospitals_with_deficit
    
    def get_hub_nodes(self) -> Set[str]:
        """
        Identifica tutti i nodi hub nel grafo.
        
        Returns:
            Set di nodi hub (che iniziano con 'U_')
        """
        return {node for node in self.graph.nodes() if node.startswith('HUB_')}
    
    def build_hub_subgraph(self) -> Tuple[nx.DiGraph, Dict[str, int]]:
        """
        Costruisce il sottografo G^U = (V^U, A^U).
        
        Returns:
            Tuple (G^U, hospitals_deficit) dove:
            - G^U: sottografo hub → ospedali con deficit
            - hospitals_deficit: deficit per ogni ospedale
        """
        hub_nodes = self.get_hub_nodes()
        hospitals_with_deficit = self.get_origin_hospitals_with_deficit()
        
        # V^U = hub ∪ ospedali con deficit
        nodes_vu = set(hub_nodes)
        nodes_vu.update(hospitals_with_deficit.keys())
        
        # Crea sottografo
        subgraph = nx.DiGraph()
        subgraph.add_nodes_from(nodes_vu)
        
        # A^U = archi da hub a ospedali con deficit
        for hub in hub_nodes:
            for hospital in hospitals_with_deficit.keys():
                if self.graph.has_edge(hub, hospital):
                    # Copia arco con tempo di viaggio dei droni
                    edge_data = self.graph[hub][hospital].copy()
                    subgraph.add_edge(hub, hospital, **edge_data)
        
        return subgraph, hospitals_with_deficit
    
    
    def capacitated_kmeans_clustering(self, subgraph: nx.DiGraph, 
                                    hospitals_deficit: Dict[str, int]) -> Dict[str, Set[str]]:
        """
        Implementa un algoritmo completo di capacitated k-means clustering con minimizzazione tempi di viaggio.
        
        Args:
            subgraph: Sottografo G^U contenente hub e ospedali con deficit
            hospitals_deficit: Dictionary {ospedale: deficit_droni}
            
        Returns:
            Dictionary {hub_seed: set_of_assigned_hospitals} - cluster finali
        """
        # 1. Estrai nodi hub e ospedali dal sottografo
        hub_nodes = list(self.inst['nodes']['hubs'])
        hospital_nodes = list(hospitals_deficit.keys())
        
        if not hub_nodes or not hospital_nodes:
            return {}
        
        # 2. Prepara dati per clustering
        weight_matrix = pd.DataFrame(index=hospital_nodes, columns=hub_nodes)
        
        # Popola weight_matrix con pesi diretti (per backup)
        for c in hub_nodes:
            for h in hospital_nodes:
                if h in subgraph[c]:
                    weight_matrix.loc[h, c] = subgraph[c][h].get('weight', None)
                else:
                    weight_matrix.loc[h, c] = None

        weight_matrix['n_ik'] = self.inst['drones']['available_drones']
        s_k_map = {v[0]: v[4] for v in self.inst['commodities'].values()}
        weight_matrix['s_k'] = weight_matrix.index.map(s_k_map)
        weight_matrix['demand'] = weight_matrix['s_k'] - weight_matrix['n_ik']

        def calculate_path_travel_time(hub, hospital):
            """Calcola il tempo di viaggio totale lungo il path hub -> facilities -> hospital"""
            facilities = []
            key = (hub, hospital)
            if key in self.inst['arcs']['F_ij']:
                facilities = list(self.inst['arcs']['F_ij'][key])
                facilities.sort(key=lambda f: self.inst['matrices']['distance'].loc[hub, f])
            
            path = [hub] + facilities + [hospital]
            total_time = 0
            
            # Calcola tempo totale lungo il path
            #for i in range(len(path) - 1):
            #    from_node = path[i]
            #    to_node = path[i + 1]
            #    if from_node in self.inst['matrices']['distance'].index and to_node in self.inst['matrices']['distance'].columns:
            #        total_time += self.inst['matrices']['distance'].loc[from_node, to_node]
            #    else:
            #        return float('inf')  # Path non valido

            facilities_in_path = [node for node in path if node in facilities]
            facility_time = self.inst['constants']['tau_i'] * len(facilities_in_path)
            travel_time = 0
            for i in range(len(path) - 1):
                from_node, to_node = path[i], path[i + 1]
                travel_time += self.inst['matrices']['distance'].loc[from_node, to_node] / self.inst['constants']['v']

            total_time = travel_time + facility_time

                    
            return total_time

        def build_hub_hospital_connections(assigned_hospitals, hub_list):
            """Costruisce le connessioni hub-hospital per una data configurazione"""
            connections = {}
            total_travel_time = 0
            
            for hospital, assigned_hub in assigned_hospitals.items():
                if assigned_hub is not None:
                    facilities = []
                    key = (assigned_hub, hospital)
                    if key in self.inst['arcs']['F_ij']:
                        facilities = list(self.inst['arcs']['F_ij'][key])
                        facilities.sort(key=lambda f: self.inst['matrices']['distance'].loc[assigned_hub, f])
                    
                    path = [assigned_hub] + facilities + [hospital]
                    facilities_time = self.inst['constants']['tau_i'] * len(facilities)
                    # Calcola tempo di viaggio per questo path
                    path_time = 0
                    for i in range(len(path) - 1):
                        from_node = path[i]
                        to_node = path[i + 1]
                        if (from_node in self.inst['matrices']['distance'].index and 
                            to_node in self.inst['matrices']['distance'].columns):
                            path_time += self.inst['matrices']['distance'].loc[from_node, to_node] / self.inst['constants']['v']
                        else:
                            path_time = float('inf')
                            break
                    
                    total_travel_time += (path_time + facilities_time)
                    
                    # Trova il numero della commodity associata all'ospedale
                    commodity_idx = None
                    for k, v in self.inst['commodities'].items():
                        if v[0] == hospital or v[1] == hospital:
                            commodity_idx = k
                            break
                            
                    if commodity_idx is not None:
                        connections[commodity_idx] = path
            
            return connections, total_travel_time

        # 3. Prova tutte le combinazioni di 2 hub
        clusters = {}
        for (hub1, hub2) in combinations(hub_nodes, 2):
            assigned_demand = {hub1: 0, hub2: 0}
            assigned_hospitals = {}
            
            for idx, row in weight_matrix.iterrows():
                # Calcola tempi di viaggio totali per entrambi gli hub
                time1 = calculate_path_travel_time(hub1, idx)
                time2 = calculate_path_travel_time(hub2, idx)
                
                # Scegli hub con tempo minimo
                if time1 == float('inf') and time2 == float('inf'):
                    nearest = None
                elif time1 == float('inf'):
                    nearest = hub2
                elif time2 == float('inf'):
                    nearest = hub1
                else:
                    nearest = hub1 if time1 <= time2 else hub2

                demand = row['demand'] if nearest is not None else 0
                
                # Controlla la capacità dell'hub selezionato
                if nearest is not None and assigned_demand[nearest] + demand > self.hub_capacities[nearest]:
                    other = hub2 if nearest == hub1 else hub1
                    if assigned_demand[other] + demand <= self.hub_capacities[other]:
                        nearest = other
                    else:
                        nearest = None
                        demand = 0

                if nearest is not None:
                    assigned_demand[nearest] += demand
                
                assigned_hospitals[idx] = nearest

            # Costruisci le connessioni e calcola il tempo totale per questa coppia
            hub_connections, total_time = build_hub_hospital_connections(assigned_hospitals, [hub1, hub2])
            
            clusters[(hub1, hub2)] = {
                'assigned_hospitals': assigned_hospitals,
                'hub_connections': hub_connections,
                'total_travel_time': total_time
            }

        # 4. Trova la coppia con somma minima dei tempi di viaggio
        min_sum = float('inf')
        best_combo = None

        for combo, data in clusters.items():
            total_travel_time = data['total_travel_time']
            if total_travel_time < min_sum:
                min_sum = total_travel_time
                best_combo = combo

        # 5. Restituisci la migliore coppia e le sue connessioni
        if best_combo is not None:
            hub_hospital_connections = clusters[best_combo]['hub_connections']
            return best_combo, hub_hospital_connections
        else:
            return None, {}

    def handle_facility_budget_constraint(self, P_u: Dict, activated_hubs: tuple) -> Dict:
        """
        Gestisce il vincolo di budget sulle facility seguendo l'algoritmo del documento.
        
        Args:
            P_u: Dictionary dei path hub->ospedale dal clustering
            activated_hubs: Tupla degli hub attivati (es. ('HUB_1', 'HUB_2'))
        """
        # Converti la tupla in set per facilità d'uso
        activated_hubs_set = set(activated_hubs)
        
        # Facility già usate negli ESP finali
        final_paths_facilities = self._get_facilities_paths(self.final_paths)
        budget_facility = math.floor(self.inst['constants']['b_F'] / self.inst['costs']['facility_costs']['F_6'])
        
        # CALCOLO CORRETTO: Considera TUTTI i path hub insieme
        all_hub_facilities = set()
        for path in P_u.values():
            hub_path_facilities = self._get_facilities(path)
            all_hub_facilities.update(hub_path_facilities)
        
        # Verifica se TUTTI i path hub insieme rispettano il budget
        total_facilities_needed = len(set(all_hub_facilities).union(final_paths_facilities))
        
        if total_facilities_needed <= budget_facility:
            # Tutti i path rispettano il budget
            return P_u.copy()
        
        # Se il budget viene violato, applica l'algoritmo di riduzione
        print(f"Budget facility violato: {total_facilities_needed} > {budget_facility}")
        print(f"Facility negli ESP: {len(final_paths_facilities)}")
        print(f"Facility negli hub paths: {len(all_hub_facilities)}")
        print(f"Facility nuove da hub paths: {len(all_hub_facilities - final_paths_facilities)}")
        
        # Inizializza strutture dati
        P_U_final = {}  # P(Ū) - path ammissibili
        P_INF_U_original = []  # Salva info originali per ricostruzione finale
        
        # STRATEGIA: Ordina i path per numero di facility e rimuovi progressivamente
        # quelli che contribuiscono di più al superamento del budget
        
        # Calcola il contributo di ogni path al numero totale di facility
        path_contributions = []
        for k, path in P_u.items():
            path_facilities = set(self._get_facilities(path))
            # Facility nuove che questo path introdurrebbe
            new_facilities = path_facilities - final_paths_facilities
            # Facility già esistenti che riutilizza
            existing_facilities = path_facilities - new_facilities
            
            path_contributions.append({
                'commodity': k,
                'path': path,
                'new_facilities': new_facilities,
                'existing_facilities': existing_facilities,
                'new_facility_count': len(new_facilities),
                'total_facility_count': len(path_facilities)
            })
        
        # Ordina per numero di facility nuove (decrescente) e poi per totale
        path_contributions.sort(key=lambda x: (x['new_facility_count'], x['total_facility_count']), reverse=True)
        
        # Calcola quante facility in eccesso abbiamo
        excess_facilities = total_facilities_needed - budget_facility
        print(f"Facility in eccesso: {excess_facilities}")
        
        # FASE 1: Prova a rimuovere completamente i path che introducono più facility nuove
        current_new_facilities = set()
        removed_paths = []
        
        for path_info in path_contributions:
            # Simula l'aggiunta di questo path
            candidate_new_facilities = current_new_facilities.union(path_info['new_facilities'])
            candidate_total = len(final_paths_facilities) + len(candidate_new_facilities)
            
            if candidate_total <= budget_facility:
                # Path può essere incluso
                P_U_final[path_info['commodity']] = path_info['path']
                current_new_facilities.update(path_info['new_facilities'])
            else:
                # Path deve essere rimosso o modificato
                removed_paths.append(path_info)
                print(f"Path commodity {path_info['commodity']} rimosso/da modificare: "
                    f"{len(path_info['new_facilities'])} nuove facility")
        
        # FASE 2: Per i path rimossi, trova ESP alternativi in G^ESP
        print(f"Cerco ESP alternativi per {len(removed_paths)} commodities...")
        
        # Calcola F(Ū) - facility utilizzabili
        F_U = final_paths_facilities.copy()
        F_U.update(current_new_facilities)
        
        # Costruisci V̄ = F(Ū) ∪ (⋃_{p ∈ P̄_{|K|}}{p})
        V_bar = F_U.copy()
        for path in self.final_paths.values():
            V_bar.update(path)
        
        for path_info in removed_paths:
            k = path_info['commodity']
            hub = path_info['path'][0]
            hospital = path_info['path'][-1]
            
            print(f"  Cerco ESP per commodity {k}: {hub} -> {hospital}")
            
            # Trova nuovo ESP in G^ESP usando solo le facility ammissibili
            new_esp = self._find_esp_in_gesp(hub, hospital, activated_hubs_set, 
                                                F_U, budget_facility)
            
            if new_esp:
                # Verifica che il nuovo path non introduca facility aggiuntive
                new_esp_facilities = set(self._get_facilities(new_esp))
                if new_esp_facilities.issubset(F_U):
                    P_U_final[k] = new_esp
                    print(f"    Trovato ESP valido: {new_esp}")
                else:
                    print(f"    ESP trovato introduce nuove facility: {new_esp_facilities - F_U}")
                    # Prova path diretto se possibile
                    if self.graph.has_edge(hub, hospital):
                        P_U_final[k] = [hub, hospital]
                        print(f"    Uso path diretto: {[hub, hospital]}")
            else:
                print(f"    Nessun ESP trovato, uso path diretto se possibile")
                # Ultima risorsa: path diretto se esiste
                if self.graph.has_edge(hub, hospital):
                    P_U_final[k] = [hub, hospital]
        
        # Verifica finale del budget
        final_hub_facilities = set()
        for path in P_U_final.values():
            final_hub_facilities.update(self._get_facilities(path))
        
        final_total = len(set(final_paths_facilities).union(final_hub_facilities))
        print(f"Verifica finale: {final_total} facility totali (budget: {budget_facility})")
        
        if final_total > budget_facility:
            print(f"ATTENZIONE: Budget ancora violato dopo ottimizzazione!")
        
        return P_U_final

    def debug_facility_budget(self, P_u: Dict) -> None:
        """Metodo di debugging per analizzare l'utilizzo delle facility"""
        final_paths_facilities = self._get_facilities_paths(self.final_paths)
        budget_facility = math.floor(self.inst['constants']['b_F'] / self.inst['costs']['facility_costs']['F_6'])
        
        print(f"\n=== DEBUG FACILITY BUDGET ===")
        print(f"Budget totale facility: {budget_facility}")
        print(f"Facility negli ESP esistenti: {len(final_paths_facilities)}")
        print(f"Facility ESP: {sorted(final_paths_facilities)}")
        
        all_hub_facilities = set()
        print(f"\nPath hub:")
        for k, path in P_u.items():
            path_facilities = set(self._get_facilities(path))
            new_facilities = path_facilities - final_paths_facilities
            print(f"  Commodity {k}: {path}")
            print(f"    Facility nel path: {sorted(path_facilities)}")
            print(f"    Facility nuove: {sorted(new_facilities)}")
            all_hub_facilities.update(path_facilities)
        
        total_facilities = len(set(all_hub_facilities).union(final_paths_facilities))
        new_from_hubs = len(all_hub_facilities - final_paths_facilities)
        
        print(f"\nRiepilogo:")
        print(f"  ESP facility: {len(final_paths_facilities)}")
        print(f"  Hub facility totali: {len(all_hub_facilities)}")
        print(f"  Hub facility nuove: {new_from_hubs}")
        print(f"  Totale combinato: {total_facilities}")
        print(f"  Budget rispettato: {'✓' if total_facilities <= budget_facility else '❌'}")
        
        if total_facilities > budget_facility:
            excess = total_facilities - budget_facility
            print(f"  Eccesso: {excess} facility")
        print(f"=================================\n")

    def _find_esp_in_gesp_strict(self, hub: str, hospital: str, activated_hubs_set: Set[str], 
                            allowed_facilities: Set[str], budget_facility: int) -> List[str]:
        """
        Trova ESP in G^ESP utilizzando SOLO le facility già ammissibili nel budget.
        
        Args:
            hub: Hub di partenza
            hospital: Ospedale di destinazione  
            activated_hubs_set: Hub attivati
            allowed_facilities: Facility che possono essere utilizzate senza violare il budget
            budget_facility: Budget totale (per logging)
            
        Returns:
            Shortest path in G^ESP che utilizza solo facility ammissibili
        """
        # V^ESP = Ū ∪ facility_ammissibili ∪ nodi_ESP_esistenti
        V_ESP = activated_hubs_set.copy()
        V_ESP.update(allowed_facilities)
        V_ESP.add(hub)
        V_ESP.add(hospital)
        
        # Aggiungi nodi dagli ESP esistenti
        for path in self.final_paths.values():
            V_ESP.update(path)
        
        # Costruisci G^ESP
        G_ESP = nx.DiGraph()
        G_ESP.add_nodes_from(V_ESP)
        
        A_U = self.inst['arcs']['A_U']
        for (i, j) in A_U:
            # Applica filtri per archi validi
            if i in self.inst['nodes']['hubs'] and j in self.inst['nodes']['hospitals'] and j != hospital:
                continue
            if i in self.inst['nodes']['facilities'] and j in self.inst['nodes']['hospitals'] and j != hospital:
                continue
            if i in self.inst['nodes']['hospitals'] and j in self.inst['nodes']['hospitals']:
                continue
                
            if i in V_ESP and j in V_ESP:
                weight = self._calculate_drone_travel_time(i, j)
                G_ESP.add_edge(i, j, weight=weight)
        
        # Trova shortest path
        try:
            esp = nx.shortest_path(G_ESP, source=hub, target=hospital, weight='weight')
            return esp
            
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def _get_facilities(self, path: List[str]) -> List[str]:
        """Estrae facility da un singolo path"""
        return [node for node in path if node.startswith('F_')]

    def _get_facilities_paths(self, paths: Dict) -> Set[str]:
        """Estrae tutte le facility uniche da un dictionary di paths"""
        all_facilities = set()
        for path in paths.values():
            facilities = self._get_facilities(path)
            all_facilities.update(facilities)
        return all_facilities

    def _count_facilities_in_pinf(self, P_INF_U: List[Dict]) -> int:
        """Conta facility totali nei path rimanenti in P^INF(Ū)"""
        total_facilities = set()
        for path_info in P_INF_U:
            path_facilities = self._get_facilities(path_info['path'])
            total_facilities.update(path_facilities)
        return len(total_facilities)

    def _remove_highest_traversal_time_facility(self, path: List[str], commodity_k: int) -> List[str]:
        """
        Rimuove la facility con τ^p''_c(j) più alto dal path.
        
        Args:
            path: Path corrente
            commodity_k: Indice della commodity per calcolare i tempi di attraversamento
            
        Returns:
            Nuovo path senza la facility con tempo più alto
        """
        facilities_in_path = self._get_facilities(path)
        
        if not facilities_in_path:
            return path.copy()
        
        # Calcola τ^p''_c(j) per ogni facility j nel path
        facility_traversal_times = {}
        for facility in facilities_in_path:
            traversal_time = self._calculate_facility_traversal_time(facility, path)
            facility_traversal_times[facility] = traversal_time
        
        # Trova facility con tempo più alto (ordinamento non crescente)
        facility_to_remove = max(facility_traversal_times.items(), key=lambda x: x[1])[0]
        
        # Crea nuovo path senza la facility selezionata
        new_path = [node for node in path if node != facility_to_remove]
        
        return new_path

    def _calculate_facility_traversal_time(self, facility: str, path: List[str]) -> float:
        """
        Calcola il tempo di attraversamento τ^p''_c(j) della facility j nel path p''.
        
        Args:
            facility: Nodo facility
            path: Path completo
            
        Returns:
            Tempo di attraversamento della facility
        """
        try:
            facility_idx = path.index(facility)
            traversal_time = 0.0
            
            # Tempo di setup facility: τ_i
            traversal_time += self.inst['constants']['tau_i']
            
            # Tempo arco entrante (se esiste)
            if facility_idx > 0:
                prev_node = path[facility_idx - 1]
                if self.graph.has_edge(prev_node, facility):
                    distance = self.inst['matrices']['distance'].loc[prev_node, facility]
                    traversal_time += distance / self.inst['constants']['v']
            
            # Tempo arco uscente (se esiste)
            if facility_idx < len(path) - 1:
                next_node = path[facility_idx + 1]
                if self.graph.has_edge(facility, next_node):
                    distance = self.inst['matrices']['distance'].loc[facility, next_node]
                    traversal_time += distance / self.inst['constants']['v']
            
            return traversal_time
            
        except ValueError:
            # Facility non trovata nel path
            return 0.0
    
    def _find_esp_in_gesp(self, hub: str, hospital: str, activated_hubs: Set[str], 
                        current_facilities: Set[str], budget_facility: int) -> List[str]:
        """
        Trova ESP in G^ESP per collegare hub a hospital utilizzando facility ammissibili.
        
        Il punto chiave è che G^ESP include TUTTE le facility che possono essere utilizzate
        rispettando il budget, non solo quelle già utilizzate.
        
        Args:
            hub: Hub di partenza
            hospital: Ospedale di destinazione  
            activated_hubs: Hub attivati
            current_facilities: Facility già utilizzate
            budget_facility: Budget totale facility disponibile
            
        Returns:
            Shortest path in G^ESP che utilizza facility ammissibili
        """
        # V^ESP = Ū ∪ V̄
        V_ESP = set(activated_hubs)  # Hub attivati
        
        # Aggiungi facility già utilizzate (sempre ammissibili)
        V_ESP.update(current_facilities)
        
        # Calcola facility aggiuntive utilizzabili rispettando il budget
        facility_slots_available = budget_facility - len(current_facilities)
        
        if facility_slots_available > 0:
            # Trova TUTTE le facility non ancora utilizzate
            all_facilities = [node for node in self.graph.nodes() if node.startswith('F_')]
            unused_facilities = [f for f in all_facilities if f not in current_facilities]
            
            # IMPORTANTE: Aggiungi TUTTE le facility non utilizzate al grafo
            # L'algoritmo ESP sceglierà quelle ottimali rispettando il budget
            V_ESP.update(unused_facilities)
        
        # Aggiungi tutti i nodi degli ESP esistenti (ospedali, etc.)
        for path in self.final_paths.values():
            V_ESP.update(path)
        
        # Aggiungi hub e hospital target
        V_ESP.add(hub)
        V_ESP.add(hospital)
        
        # Costruisci G^ESP con archi da A_U
        G_ESP = nx.DiGraph()
        G_ESP.add_nodes_from(V_ESP)
        
        A_U = self.inst['arcs']['A_U']
        for (i, j) in A_U:
            if i in self.inst['nodes']['hubs'] and j in self.inst['nodes']['hospitals'] and j != hospital:
                continue
            if i in self.inst['nodes']['facilities'] and j in self.inst['nodes']['hospitals'] and j != hospital:
                continue
            if i in self.inst['nodes']['hospitals'] and j in self.inst['nodes']['hospitals']:
                continue
            if i in V_ESP and j in V_ESP:
                weight = self._calculate_drone_travel_time(i, j)
                G_ESP.add_edge(i, j, weight=weight)
        

        
        # Trova shortest path in G^ESP
        try:
            esp = nx.shortest_path(G_ESP, source=hub, target=hospital, weight='weight')
            
            # Verifica che il path rispetti il budget
            esp_facilities = self._get_facilities(esp)
            total_facilities_needed = len(set(esp_facilities).intersection(current_facilities))
            
            if total_facilities_needed <= budget_facility:
                return esp
            else:
                # Se il path ottimale viola ancora il budget, prova senza facility aggiuntive
                return self._find_esp_in_gesp_strict(hub, hospital, activated_hubs, current_facilities, budget_facility)
                
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None
            
    

    def _calculate_drone_travel_time(self, node_i: str, node_j: str) -> float:
        """
        Calcola tempo di viaggio drone tra due nodi.
        
        Args:
            node_i: Nodo di partenza
            node_j: Nodo di arrivo
            
        Returns:
            Tempo di viaggio = τ_i * I_facility(j) + distance(i,j)/v
        """
        # Tempo di setup se j è una facility
        setup_time = (self.inst['constants']['tau_i'] * 
                    self.inst['facility_indicators'].get(node_j, 0))
        
        # Tempo di viaggio
        distance = self.inst['matrices']['distance'].loc[node_i, node_j]
        travel_time = distance / self.inst['constants']['v']
        
        return setup_time + travel_time
    
    def activate_hubs(self) -> Dict:
        
        # 1. Costruisci sottografo
        subgraph, hospitals_deficit = self.build_hub_subgraph()
        
        # 2. Applica clustering
        best_combo, P_u = self.capacitated_kmeans_clustering(subgraph, hospitals_deficit)
        #self.debug_facility_budget(P_u)
        
        P_U_final = self.handle_facility_budget_constraint(P_u, best_combo)
        
        return best_combo, P_U_final

class Heuristic:
    """
    Implementazione completa dell'algoritmo euristico per la progettazione 
    della rete logistica dei droni secondo l'Algoritmo 1 del paper.
    """
    
    def __init__(self, instance_data: Dict, s_max: int = 3):
        """
        Inizializza l'euristica.
        
        Args:
            graph: Grafo G = (V, A)
            commodities: Dictionary delle commodity K {k: (i^k, j^k, ...)}
            instance_data: Dati dell'istanza (budget, costi, etc.)
            s_max: Dimensione massima dei vicinati estesi
        """
        #self.graph = None
        self.commodities = instance_data['commodities']
        self.instance_data = instance_data
        self.s_max = s_max
        
        # Strutture dati principali
        self.graph = None
        self.paths = None  # P_{|K|}: ESPs correnti
        self.F = set()   # Insieme facility correnti
        self.D = None    # Matrice distanze
        self.solution = {}
        
        # Parametri budget e costi
        self.budget_facilities = instance_data['constants']['b_F']
        self.facility_cost = instance_data['costs']['facility_costs']['F_6']
        
    
    def initialize_components(self):
        self.graph = nx.DiGraph()
        A = self.instance_data['arcs']['A']
        dist = self.instance_data['matrices']['distance']
        hospitals = set(self.instance_data['nodes']['hospitals'])
        for (i,j) in A:
            if i in hospitals and j in hospitals:
                continue
            weight = (self.instance_data['constants']['tau_i'] * self.instance_data['facility_indicators'].get(j,0) 
                    + dist.loc[i,j] / self.instance_data['constants']['v'])
            self.graph.add_edge(i, j, weight=weight)
        paths = {}
        for k in self.instance_data['constants']['K']:
            i_k = self.instance_data['commodities'][k][0]
            j_k = self.instance_data['commodities'][k][1]
            #other_hospitals = [i for i in inst['nodes']['hospitals'] if i != i_k]
            
            if (i_k, j_k) in A:
                ESP = [i_k, j_k]
            else:
                ESP = nx.shortest_path(self.graph,i_k,j_k,weight='weight')
            paths[k] = ESP
        self.paths = paths
        
        
    
    def extract_facilities_from_paths(self, paths: Dict[int, List[str]]) -> Set[str]:
        """
        PASSO 4: Definisce F = U_{p ∈ P_{|K|}} F(p).
        
        Args:
            paths: Dictionary con gli ESPs
            
        Returns:
            Set di tutte le facility negli ESPs
        """
        facilities = []
        for k, path, in paths.items():
            #print(path)
            facility = [node for node in path if node[0] == 'F']
            facilities.append(facility)

        all_facilities = [f for sublist in facilities for f in sublist]
        # Ottieni solo i valori unici mantenendo l'ordine
        unique_facilities = list(dict.fromkeys(all_facilities))
        #print(len(unique_facilities))
        
        return unique_facilities

    
    def build_distance_matrix(self, paths: Dict[int, List[str]]) -> pd.DataFrame:
        """
        PASSO 5: Calcola la matrice delle distanze D.
        
        Args:
            paths: Dictionary con gli ESPs
            
        Returns:
            DataFrame con la matrice delle distanze
        """
        _, distance_df = PathDistanceMetric(self.graph).build_distance_matrix(paths)
        return distance_df
    
    def check_budget_constraint(self, facilities: List[str]) -> bool:
        """
        Verifica se il vincolo di budget è soddisfatto: |F| ≤ ⌈b^F/c⌉.
        
        Args:
            facilities: Set di facility
            
        Returns:
            True se il vincolo è soddisfatto
        """
        max_facilities = math.floor(self.budget_facilities / self.facility_cost) #- 1
        return len(facilities) > max_facilities 
    
    def update_paths_and_distance_matrix(self, modified_paths: Dict[int, List[str]]):
        """
        Aggiorna gli ESPs e ricalcola la matrice delle distanze dopo l'exploration.
        
        Args:
            modified_paths: Nuovi ESPs modificati P̄(N*_s(p))
        """
        # Aggiorna i paths: sostituisce i paths modificati
        for k, new_path in modified_paths.items():
            self.paths[k] = new_path

        
        # Ricalcola le distanze che coinvolgono i paths modificati
        commodities_to_update = set(modified_paths.keys())
        all_commodities = set(self.paths.keys())
        
        for k1 in commodities_to_update:
            for k2 in all_commodities:
                if k1 != k2:
                    # Ricalcola d(p1, p2) e d(p2, p1)
                    distance_12 = PathDistanceMetric(self.graph).calculate_metric(self.paths[k1], self.paths[k2])
                    distance_21 = PathDistanceMetric(self.graph).calculate_metric(self.paths[k2], self.paths[k1])
                    
                    self.D.loc[k1, k2] = distance_12
                    self.D.loc[k2, k1] = distance_21
    
    
    def run_heuristic(self) -> Dict:
        """
        Implementa l'Algoritmo 1 completo dell'euristica.
        
        Returns:
            Dictionary con la soluzione finale o vuoto se non trovata
        """
        #print("=== AVVIO ALGORITMO EURISTICO ===")
        start_time = time.time()
        # PASSO 1: Inizializzazione
        #print("PASSO 1: Inizializzazione")
        self.solution = {}
        s = 1  # Dimensione vicinati iniziale
        self.initialize_components()
        # PASSO 2: Calcola ESPs iniziali
        #print("PASSO 2: Calcolo ESPs iniziali")
        #print(f"ESPs calcolati per {len(self.paths)} commodity")
        
        # PASSO 3: Popola P_{|K|}
        #print("PASSO 3: Popolazione P_{|K|}")
        # (già fatto nel passo 2)
        
        # PASSO 4: Definisci F
        #print("PASSO 4: Estrazione facility")
        self.F = self.extract_facilities_from_paths(self.paths)
        #print(f"Facility estratte: {len(self.F)}")
        
        # PASSO 5: Calcola matrice distanze
        #print("PASSO 5: Calcolo matrice distanze")
        self.D = self.build_distance_matrix(self.paths)
        
        # PASSO 6: Dimensione vicinati iniziale
        #print("PASSO 6: Impostazione dimensione vicinati")
        # (s già impostato)
        
        # Inizializza constructor vicinati
        neighborhood_constructor = NeighborhoodConstructor(self.paths, self.D)
        
        # PASSO 9: Ordine iniziale
        order = 1
        
        # CICLO PRINCIPALE
        iteration = 0
        #print(f"\\n=== INIZIO CICLO PRINCIPALE ===")
        #print(f"Budget constraint: |F| = {len(self.F)} <= {math.ceil(self.budget_facilities / self.facility_cost)}")
        while (self.check_budget_constraint(self.F)) and s <= self.s_max:
            iteration += 1
            #print(f"\n--- Iteration {iteration} ---")
            #print(f" Current |F| = {len(self.F)} with s={s} and order={order}")       
            # PASSO 7: Definisci vicinati N_s(p)
            use_extended = (s > 1)
            
            # PASSO 11: Best neighbour
            commodity_id, best_neighborhood, min_distance = neighborhood_constructor.get_best_neighborhood_by_order(
                s, order, use_extended
            )
            
            if commodity_id is None:
                break

            
            explorer = NeighborhoodExplorer(self.graph, self.paths)
            F_bar_vicinato, original_paths, modified_paths = explorer.explore_neighborhood(best_neighborhood, commodity_id)

            # PASSO 13: Inizializza temp_paths
            temp_paths = self.paths.copy()
            for k, path in modified_paths.items():
                temp_paths[k] = path

            #print(f" Iteration {iteration}, s={s}, order={order} - Explored neighborhood for commodity {commodity_id} with distance {min_distance:.2f}")
            # *** Calcola F_bar da temp_paths ***
            F_bar = self.extract_facilities_from_paths(temp_paths)
            num_neighborhoods = len(self.paths)  # O calcola il numero effettivo di vicinati
            # CICLO INTERNO - usa F_bar calcolato sopra!
            while len(F_bar) >= len(self.F) and order < num_neighborhoods:
                order += 1
                
                commodity_id, next_neighborhood, next_distance = neighborhood_constructor.get_best_neighborhood_by_order(
                    s, order, use_extended
                )
                #print(f" Iteration {iteration}, s={s}, order={order} - Next best neighborhood for commodity {commodity_id} with neighboors {next_neighborhood} with distance {next_distance:.2f}")
    
                if commodity_id is None:
                    break
                
                # Esplora vicinato successivo
                explorer_temp = NeighborhoodExplorer(self.graph, temp_paths)
                _, _, modified_paths_new = explorer_temp.explore_neighborhood(next_neighborhood, commodity_id)
                #print(f"  Explored neighborhood for commodity {commodity_id} with distance {next_distance:.2f}")
                # Aggiorna temp_paths



                for k, path in modified_paths_new.items():
                    temp_paths[k] = path
                
                # *** FIX CRITICO: Ricalcola F_bar dopo ogni modifica! ***
                F_bar = self.extract_facilities_from_paths(temp_paths)
                
                # Se hai trovato un miglioramento, esci dal ciclo
                if len(F_bar) < len(self.F):
                    #print(f"  Found improvement: |F| {len(self.F)} -> {len(F_bar)}")
                    break


            # PASSO 17: Check miglioramento
            if len(F_bar) < len(self.F):
                #print(f"\n*** Improvement accepted! ***")
                self.F = F_bar
                self.paths = temp_paths  # temp_paths ha già tutte le commodity
                self.D = self.build_distance_matrix(self.paths)
                #self.update_paths_and_distance_matrix(self.paths)
                
                s = 1
                order = 1
                neighborhood_constructor = NeighborhoodConstructor(self.paths, self.D)
            else:
                s += 1
                order = 1
                # *** CRITICO: Ricrea vicinati con nuovo s ***
                neighborhood_constructor = NeighborhoodConstructor(self.paths, self.D)
                commodity_id, best_neighborhood, min_distance = neighborhood_constructor.get_best_neighborhood_by_order(
                s, order, use_extended
                )
            
        
        # VERIFICA FINALE BUDGET
        #print(f"\\n=== VERIFICA FINALE ===")
        max_facilities = math.floor(self.budget_facilities / self.facility_cost) 
        #return len(facilities) <= max_facilities 
        if len(self.F) <= max_facilities:
            #print(f"VINCOLO BUDGET SODDISFATTO: |F| = {len(self.F)}")
            #print(f"Facility finali: {self.F}")
            #print(self.paths)
            
            # PASSO 26: Attivazione hub
            #print("PASSO 26: Attivazione hub")
            hub_activator = HubActivation(self.graph, self.instance_data, self.paths)
            activated_hubs, hub_paths = hub_activator.activate_hubs()
            
            #print(f"Hub attivati: {activated_hubs}")
            
            #print("PASSO 27: Collegamento percorsi hub con percorsi principali")
            connected_paths = self.connect_hub_paths_to_main_paths(self.paths, hub_paths)

            t_p = self.calculate_path_times(connected_paths, self.instance_data)

            #
            
            #return self.solution
            end_time = time.time()
            run_time = end_time - start_time
            # PASSO 27: Costruzione soluzione finale
            self.solution = {
                'paths': self.paths.copy(),
                'facilities': self.F.copy(),
                'OBJ': sum(t_p.values()),
                'connected_paths': connected_paths,
                'activated_hubs': activated_hubs,
                'hub_paths': hub_paths,
                'total_facilities': len(self.F),
                'budget_satisfied': True,
                'iterations': iteration,
                'TIME(s)': run_time
            }
            
            #print("SOLUZIONE TROVATA!")
           # print(connected_paths)
            #print()
            #print(f"OBJ: {sum(t_p.values())}")


            #if iteration >= 1:
                #print("\n=== APPLICAZIONE LOCAL SEARCH ===")
            local_search = LocalSearch(self.solution, self.instance_data, self.graph)
            improved_solution = local_search.run()
            instance_name = f"Instance_{self.instance_data.get('instance_id', 'unknown')}"
            local_search.save_move_statistics(
                instance_name=instance_name,
                initial_obj=self.solution['OBJ'],  # Passa OBJ euristica
                results_file='local_search_stats.csv'
                )
    
            self.solution = improved_solution
                
                # Aggiorna risultati
                #print(f"Obiettivo finale dopo LS: {self.solution['OBJ']:.2f}")

            results = {
                "N.H. o-d": len(set((k[0], k[1]) for k in self.instance_data['commodities'].values())),
                "N.C.": len(self.instance_data['constants']['K']),
                "N.H.": len(self.instance_data['nodes']['hospitals']),
                "N.F.": len(self.instance_data['nodes']['facilities']),
                "N. HUBS": len(self.instance_data['nodes']['hubs']),
                #"N.A.": len(active_paths),
                "N.D.H": len(self.instance_data['drones']['hospital_drones']),
                "N.D.HUBS": len(self.instance_data['drones']['hub_drones']),
                "N.D.": len(self.instance_data['drones']['hospital_drones']) + len(self.instance_data['drones']['hub_drones']), 
                "N.D.R": sum(self.instance_data['commodities'][k][4] for k in self.instance_data['constants']['K']),
                "B.U.": self.instance_data['constants']['b_U'],
                "B.F.": self.instance_data['constants']['b_F'],
                "OBJ": sum(t_p.values()),
                'OBJ-LS': self.solution['OBJ'], #sum(t_p.values()),
                'total_facilities': len(self.solution['facilities']), #len(self.F),
                'budget_satisfied': True,
                'iterations': iteration,
                'TIME(s)': run_time
            }

            data = {**results}
            df_results = pd.DataFrame([data])

            results_filename = 'DMNDP-VNS-rome.csv'
            df_results.to_csv(results_filename, mode='a', 
                            header=not pd.io.common.file_exists(results_filename),
                            index=False)
            print(f"Results saved to {results_filename}")

            
        else:
            #print(f"VINCOLO BUDGET NON SODDISFATTO: |F| = {len(self.F)} > {math.ceil(self.budget_facilities / self.facility_cost)}")
            #print(f"Parametro s_max = {self.s_max} potrebbe essere insufficiente")
            end_time = time.time()
            run_time = end_time - start_time
            self.solution = {
                'budget_satisfied': False,
                'final_facilities': len(self.F),
                'iterations': iteration,
                'reason': f'Budget constraint not satisfied with s_max={self.s_max}'
            }

            results = {
                "N.H. o-d": len(set((k[0], k[1]) for k in self.instance_data['commodities'].values())),
                "N.C.": len(self.instance_data['constants']['K']),
                "N.H.": len(self.instance_data['nodes']['hospitals']),
                "N.F.": len(self.instance_data['nodes']['facilities']),
                "N. HUBS": len(self.instance_data['nodes']['hubs']),
                #"N.A.": len(active_paths),
                "N.D.H": len(self.instance_data['drones']['hospital_drones']),
                "N.D.HUBS": len(self.instance_data['drones']['hub_drones']),
                "N.D.": len(self.instance_data['drones']['hospital_drones']) + len(self.instance_data['drones']['hub_drones']), 
                "N.D.R": sum(self.instance_data['commodities'][k][4] for k in self.instance_data['constants']['K']),
                "B.U.": self.instance_data['constants']['b_U'],
                "B.F.": self.instance_data['constants']['b_F'],
                'OBJ': None,#sum(t_p.values()),
                'total_facilities': len(self.F),
                'budget_satisfied': True,
                'iterations': iteration,
                'TIME(s)': run_time
            }

            data = {**results}
            df_results = pd.DataFrame([data])

            results_filename = 'DMNDP-VNS-rome.csv'
            df_results.to_csv(results_filename, mode='a', 
                            header=not pd.io.common.file_exists(results_filename),
                            index=False)
            print(f"Results saved to {results_filename}")
        
        return self.solution
    
    def print_solution_summary(self):
        """Stampa un riassunto della soluzione."""
        if not self.solution:
            print("Nessuna soluzione disponibile")
            return
        
        print("\n" + "="*50)
        print("RIASSUNTO SOLUZIONE")
        print("="*50)
        
        if self.solution.get('budget_satisfied', False):
            print(f"✓ Soluzione ammissibile trovata")
            print(f"✓ Facility attivate: {self.solution['total_facilities']}")
            print(f"✓ Hub attivati: {self.solution['activated_hubs']}")
            print(f"✓ Iterazioni: {self.solution['iterations']}")
            print(f"✓ ESPs finali: {len(self.solution['paths'])}")
        else:
            print(f"✗ Soluzione non ammissibile")
            print(f"✗ Motivo: {self.solution.get('reason', 'Sconosciuto')}")
            print(f"✗ Facility finali: {self.solution.get('final_facilities', 'N/A')}")
            print(f"✗ Iterazioni: {self.solution.get('iterations', 'N/A')}")

    def connect_hub_paths_to_main_paths(self, main_paths: Dict[int, List[str]], 
                                       hub_paths: Dict[int, List[str]]) -> Dict[int, List[str]]:
        """
        Collega i percorsi degli hub ai percorsi principali per creare la rete completa.
        
        Args:
            main_paths: Percorsi principali {commodity_id: path}
            hub_paths: Percorsi hub->ospedale {commodity_id: hub_path}
            
        Returns:
            Dictionary con i percorsi completi collegati
        """
        connected_paths = {}
        
        for k, main_path in main_paths.items():
            if k in hub_paths:
                hub_path = hub_paths[k]
                origin_hospital = main_path[0]
                
                # Verifica che l'hub_path termini nell'ospedale origine del main_path
                if hub_path[-1] == origin_hospital:
                    # Collega: hub_path + main_path (rimuovendo il nodo duplicato)
                    connected_path = hub_path[:-1] + main_path
                    connected_paths[k] = connected_path
                    #print(f"  DEBUG: Connected commodity {k}: HUB->hospital ({len(hub_path)}) + main_path ({len(main_path)}) = {len(connected_path)} nodes")
                else:
                    # Fallback: mantieni solo il percorso principale
                    connected_paths[k] = main_path
                    #print(f"  WARNING: Hub path for commodity {k} doesn't connect properly (hub ends at {hub_path[-1]}, main starts at {origin_hospital})")
            else:
                # Nessun hub path per questa commodity
                connected_paths[k] = main_path
                
        #print(f"  DEBUG: Connected {len([k for k in connected_paths.keys() if k in hub_paths])} commodity paths with hub connections")
        return connected_paths
    
    def extract_complete_facilities_from_connected_paths(self, connected_paths: Dict[int, List[str]]) -> Set[str]:
        """
        Estrae tutte le facility dai percorsi collegati (inclusi quelli degli hub).
        
        Args:
            connected_paths: Percorsi completi con collegamenti hub
            
        Returns:
            Set completo di facility necessarie
        """
        facilities = set()
        
        for k, path in connected_paths.items():
            path_facilities = [node for node in path if node.startswith('F_')]
            facilities.update(path_facilities)
        
        #print(f"  DEBUG: Total facilities in connected paths: {len(facilities)}")
        return facilities
    

    def calculate_path_times(self, connected_paths: Dict[int, List[str]], inst: Dict) -> Dict[int, float]:
        """
        Calcola il tempo t_p per ogni percorso collegato secondo la formulazione del problema.
        
        Args:
            connected_paths: Dictionary dei percorsi collegati {commodity_id: path}
            inst: Dati dell'istanza contenenti parametri e matrici
            
        Returns:
            Dictionary {commodity_id: t_p} con i tempi calcolati per ogni percorso
        """
        t_p = {}
        facilities = set()
        
        # Estrai tutte le facility per la verifica
        for k ,path in connected_paths.items():
            path_facilities = [node for node in path if node[0] == 'F']
            #print(path_facilities)

            facilities.update(path_facilities)
        
        for k, path in connected_paths.items():
            try:
                # Parametri della commodity
                commodity_data = inst['commodities'][k]
                i_k = commodity_data[0]  # origine
                j_k = commodity_data[1]  # destinazione  
                e_k = commodity_data[2]  # tempo emergenza
                l_k = commodity_data[3]  # limite tempo
                s_k = commodity_data[4]  # droni richiesti
                pi_k = commodity_data[5]  # penalità
                
                n_ik = inst['drones']['available_drones'].get(i_k, 0)  # droni disponibili all'origine
                
                # Determina tipo di percorso (hub se inizia con HUB_)
                path_type = 'hub' if path[0].startswith('HUB_') else 'direct'
                
                if path_type == 'direct':
                    # Percorso diretto: i_k -> j_k
                    facilities_in_path = [node for node in path if node in facilities]
                    facility_time = inst['constants']['tau_i'] * len(facilities_in_path)
                    travel_time = 0
                    for i in range(len(path) - 1):
                        from_node, to_node = path[i], path[i + 1]
                        travel_time += inst['matrices']['distance'].loc[from_node, to_node] / inst['constants']['v']
                    
                    # Tempo totale arrivo
                    t_jk = e_k + facility_time + travel_time
                    delay = max(0, t_jk - l_k)
                    penalty = pi_k * delay
                    
                    # Costo totale
                    t_p[k] = s_k * (facility_time + travel_time + penalty)
                
                else:  # path_type == 'hub'
                    # Percorso con hub: HUB -> i_k -> j_k
                    
                    # Trova l'indice dell'ospedale origine nel percorso
                    try:
                        ik_index = path.index(i_k)
                    except ValueError:
                        ik_index = len([node for node in path if node.startswith('HUB_')])
                    
                    # Segmenti del percorso
                    ik_to_jk = path[ik_index:]  # Da i_k a j_k
                    
                    # CALCOLO PERCORSO COMPLETO (HUB -> j_k)
                    facilities_complete = [node for node in path if node in facilities]
                    #print(facilities_complete)
                #print(path,facilities_complete)
                    facility_time_complete = inst['constants']['tau_i'] * len(facilities_complete)
                    #print(facility_time_complete)

                    travel_time_complete = 0
                    for i, j in zip(path[:-1], path[1:]):
                        travel_time_complete += inst['matrices']['distance'].loc[i, j] / inst['constants']['v']
                    #print(travel_time_complete)
                    t_jk_complete = e_k + facility_time_complete + travel_time_complete
                    delay_complete = max(0, t_jk_complete - l_k)
                    
                    # Costo per droni dal hub
                    s_k_p_complete = s_k - n_ik  # Droni forniti dal hub
                    cost_complete = 0
                    if s_k_p_complete > 0:
                        cost_complete = s_k_p_complete * (facility_time_complete + travel_time_complete)
                    
                    # CALCOLO PERCORSO PARZIALE (i_k -> j_k)
                    facilities_partial = [node for node in ik_to_jk if node in facilities]
                    facility_time_partial = inst['constants']['tau_i'] * len(facilities_partial)

                    travel_time_partial = 0
                    for i, j in zip(ik_to_jk[:-1], ik_to_jk[1:]):
                        travel_time_partial += inst['matrices']['distance'].loc[i, j] / inst['constants']['v']
                
                    t_jk_partial = e_k + facility_time_partial + travel_time_partial
                    delay_partial = max(0, t_jk_partial - l_k)
                    
                    
                    # Costo per droni dall'ospedale origine
                    s_k_p_partial = n_ik  # Droni disponibili localmente
                    cost_partial = 0
                    if s_k_p_partial > 0:
                        cost_partial = s_k_p_partial * (facility_time_partial + travel_time_partial)
                    
                    # PENALITÀ COMBINATA
                    penalty = (n_ik * pi_k * delay_partial) + ((s_k - n_ik) * pi_k * delay_complete)
                    #print(cost_complete, cost_partial, penalty)
                    # COSTO TOTALE
                    t_p[k] = cost_complete + cost_partial + penalty
                    
            except Exception as e:
                #print(f"Error calculating time for commodity {k}: {e}")
                t_p[k] = float('inf')  # Assegna valore infinito in caso di errore
        
        return t_p
    
class LocalSearch:
    """
    Local search procedure to improve DMNDP solution by minimizing sum(t_p).
    Explores multiple neighborhood structures while maintaining feasibility.
    """
    
    def __init__(self, heuristic_solution: Dict, instance_data: Dict, graph: nx.DiGraph):
        """
        Initialize local search with heuristic solution.
        
        Args:
            heuristic_solution: Solution from Heuristic class
            instance_data: Instance data
            graph: Original graph
        """
        self.inst = instance_data
        self.graph = graph
        
        # Current solution components
        self.connected_paths = copy.deepcopy(heuristic_solution['connected_paths'])
        self.facilities = copy.deepcopy(heuristic_solution['facilities'])
        self.activated_hubs = copy.deepcopy(heuristic_solution['activated_hubs'])
        self.hub_paths = copy.deepcopy(heuristic_solution['hub_paths'])
        
        # Budget parameters
        self.budget_facilities = instance_data['constants']['b_F']
        self.facility_cost = instance_data['costs']['facility_costs']['F_6']
        self.max_facilities = math.floor(self.budget_facilities/self.facility_cost)
        
        # Current objective value
        self.current_obj = heuristic_solution['OBJ']
        self.best_obj = self.current_obj
        
        # Build graphs A and A_U
        self.graph_A = self._build_graph_A()
        self.graph_A_U = self._build_graph_A_U()

        self.move_stats = {
            'facility_swap': {'attempts': 0, 'successes': 0, 'total_improvement': 0.0},
            'reroute': {'attempts': 0, 'successes': 0, 'total_improvement': 0.0},
            'remove_facility': {'attempts': 0, 'successes': 0, 'total_improvement': 0.0},
            'hub_reassignment': {'attempts': 0, 'successes': 0, 'total_improvement': 0.0}
        }
        self.improvement_history = []  # Lista di (iteration, move_type, delta_obj)
    
        
    def _build_graph_A(self) -> nx.DiGraph:
        """Build graph A (for paths i_k -> j_k)"""
        G_A = nx.DiGraph()
        hospitals = set(self.inst['nodes']['hospitals'])
        
        for (i, j) in self.inst['arcs']['A']:
            # Exclude hospital-to-hospital arcs
            if i in hospitals and j in hospitals:
                continue
            weight = (self.inst['constants']['tau_i'] * self.inst['facility_indicators'].get(j, 0) 
                     + self.inst['matrices']['distance'].loc[i, j] / self.inst['constants']['v'])
            G_A.add_edge(i, j, weight=weight)
        
        return G_A
    
    def _build_graph_A_U(self) -> nx.DiGraph:
        """Build graph A_U (for hub paths hub -> i_k)"""
        G_A_U = nx.DiGraph()
        hospitals = set(self.inst['nodes']['hospitals'])
        hubs = set(self.inst['nodes']['hubs'])
        
        for (i, j) in self.inst['arcs']['A_U']:
            # Exclude certain hospital connections
            if i in hubs and j in hospitals:
                pass  # Allow hub -> hospital
            elif i in self.inst['nodes']['facilities'] and j in hospitals:
                pass  # Allow facility -> hospital  
            elif i in hospitals and j in hospitals:
                continue  # Block hospital -> hospital
            
            weight = (self.inst['constants']['tau_i'] * self.inst['facility_indicators'].get(j, 0) 
                     + self.inst['matrices']['distance'].loc[i, j] / self.inst['constants']['v'])
            G_A_U.add_edge(i, j, weight=weight)
        
        return G_A_U
    
    def is_path_valid(self, path: List[str], path_type: str = 'main') -> bool:
        """
        Check if a path is valid (no intermediate hospitals, feasible on correct graph).
        
        Args:
            path: Path to validate
            path_type: 'main' for i_k->j_k, 'hub' for hub->i_k
        """
        if len(path) < 2:
            return False
        
        hospitals = set(self.inst['nodes']['hospitals'])
        
        # Check no intermediate hospitals
        for i in range(1, len(path) - 1):
            if path[i] in hospitals:
                return False
        
        # Check path feasibility on correct graph
        graph = self.graph_A_U if path_type == 'hub' else self.graph_A
        
        for i in range(len(path) - 1):
            if not graph.has_edge(path[i], path[i+1]):
                return False
        
        return True
    
    def calculate_path_time(self, k: int, connected_path: List[str]) -> float:
        """Calculate t_p for a single commodity path."""
        try:
            commodity_data = self.inst['commodities'][k]
            i_k = commodity_data[0]
            j_k = commodity_data[1]
            e_k = commodity_data[2]
            l_k = commodity_data[3]
            s_k = commodity_data[4]
            pi_k = commodity_data[5]
            
            n_ik = self.inst['drones']['available_drones'].get(i_k, 0)
            path = connected_path
            
            # Determine if hub path
            path_type = 'hub' if path[0].startswith('HUB_') else 'direct'
            
            facilities_in_path = [node for node in path if node.startswith('F_')]
            facility_time = self.inst['constants']['tau_i'] * len(facilities_in_path)
            
            travel_time = 0
            for i in range(len(path) - 1):
                from_node, to_node = path[i], path[i + 1]
                travel_time += self.inst['matrices']['distance'].loc[from_node, to_node] / self.inst['constants']['v']
            
            if path_type == 'direct':
                t_jk = e_k + facility_time + travel_time
                delay = max(0, t_jk - l_k)
                penalty = pi_k * delay
                return s_k * (facility_time + travel_time + penalty)
            
            else:  # hub path
                try:
                    ik_index = path.index(i_k)
                except ValueError:
                    ik_index = len([node for node in path if node.startswith('HUB_')])
                
                ik_to_jk = path[ik_index:]
                
                # Complete path (hub -> j_k)
                t_jk_complete = e_k + facility_time + travel_time
                delay_complete = max(0, t_jk_complete - l_k)
                
                s_k_p_complete = s_k - n_ik
                cost_complete = 0
                if s_k_p_complete > 0:
                    cost_complete = s_k_p_complete * (facility_time + travel_time)
                
                # Partial path (i_k -> j_k)
                facilities_partial = [node for node in ik_to_jk if node.startswith('F_')]
                facility_time_partial = self.inst['constants']['tau_i'] * len(facilities_partial)
                
                travel_time_partial = 0
                for i, j in zip(ik_to_jk[:-1], ik_to_jk[1:]):
                    travel_time_partial += self.inst['matrices']['distance'].loc[i, j] / self.inst['constants']['v']
                
                t_jk_partial = e_k + facility_time_partial + travel_time_partial
                delay_partial = max(0, t_jk_partial - l_k)
                
                s_k_p_partial = n_ik
                cost_partial = 0
                if s_k_p_partial > 0:
                    cost_partial = s_k_p_partial * (facility_time_partial + travel_time_partial)
                
                penalty = (n_ik * pi_k * delay_partial) + ((s_k - n_ik) * pi_k * delay_complete)
                
                return cost_complete + cost_partial + penalty
                
        except Exception as e:
            return float('inf')
    
    def calculate_total_objective(self, paths: Dict[int, List[str]]) -> float:
        """Calculate sum(t_p) for all paths."""
        total = 0
        for k, path in paths.items():
            total += self.calculate_path_time(k, path)
        return total
    
    def extract_facilities_from_paths(self, paths: Dict[int, List[str]]) -> Set[str]:
        """Extract all facilities from paths."""
        facilities = set()
        for path in paths.values():
            for node in path:
                if node.startswith('F_'):
                    facilities.add(node)
        return facilities
    
    # ============ NEIGHBORHOOD MOVES ============
    
    def move_facility_swap_in_path(self, k: int) -> Tuple[bool, Dict, float]:
        """
        Try swapping one facility with another in a path.
        """
        path = self.connected_paths[k]
        facilities_in_path = [node for node in path if node.startswith('F_')]
        
        if not facilities_in_path:
            return False, None, None
        
        # Try each facility swap
        all_facilities = list(self.inst['nodes']['facilities'])
        
        for old_facility in facilities_in_path:
            for new_facility in all_facilities:
                if new_facility == old_facility or new_facility in facilities_in_path:
                    continue
                
                # Create new path with swapped facility
                new_path = [new_facility if node == old_facility else node for node in path]
                
                # Split into hub and main parts
                i_k = self.inst['commodities'][k][0]
                
                if new_path[0].startswith('HUB_'):
                    try:
                        ik_index = new_path.index(i_k)
                    except ValueError:
                        continue
                    
                    hub_part = new_path[:ik_index+1]
                    
                    main_part = new_path[ik_index:]
                    
                    if not self.is_path_valid(hub_part, 'hub') or not self.is_path_valid(main_part, 'main'):
                        continue
                else:
                    if not self.is_path_valid(new_path, 'main'):
                        continue
                
                # Check facility budget
                new_paths = self.connected_paths.copy()
                new_paths[k] = new_path
                new_facilities = self.extract_facilities_from_paths(new_paths)
                
                if len(new_facilities) > self.max_facilities:
                    continue
                
                # Calculate new objective
                new_obj = self.calculate_total_objective(new_paths)
                
                if new_obj < self.current_obj:
                    return True, new_paths, new_obj
        
        return False, None, None
    
    def move_reroute_path(self, k: int) -> Tuple[bool, Dict, float]:
        """
        Find alternative ESP for a commodity using shortest path.
        """
        commodity_data = self.inst['commodities'][k]
        i_k = commodity_data[0]
        j_k = commodity_data[1]
        
        current_path = self.connected_paths[k]
        is_hub_path = current_path[0].startswith('HUB_')

        #extract subgraph with current facilities
        subgraph_A = self.graph_A.subgraph(set(self.facilities).union(set(self.inst['nodes']['hospitals']))).copy()
        subgraph_A_U = self.graph_A_U.subgraph(set(self.facilities).union(set(self.inst['nodes']['hospitals'])).union(set(self.inst['nodes']['hubs']))).copy()   

        #
        
        if is_hub_path:
            # Find hub and split path
            hub = current_path[0]
            try:
                ik_index = current_path.index(i_k)
            except ValueError:
                return False, None, None
            
            # Try new hub -> i_k path
            try:
                new_hub_part = nx.shortest_path(subgraph_A_U, hub, i_k, weight='weight')
                if not self.is_path_valid(new_hub_part, 'hub'):
                    return False, None, None
            except nx.NetworkXNoPath:
                return False, None, None
            
            # Try new i_k -> j_k path
            try:
                if (i_k, j_k) in self.inst['arcs']['A']:
                    new_main_part = [i_k, j_k]
                else:
                    new_main_part = nx.shortest_path(subgraph_A, i_k, j_k, weight='weight')
                if not self.is_path_valid(new_main_part, 'main'):
                    return False, None, None
            except nx.NetworkXNoPath:
                return False, None, None
            
            new_path = new_hub_part[:-1] + new_main_part
            
        else:
            # Try new direct path
            try:
                if (i_k, j_k) in self.inst['arcs']['A']:
                    new_path = [i_k, j_k]
                else:
                    new_path = nx.shortest_path(subgraph_A, i_k, j_k, weight='weight')
                if not self.is_path_valid(new_path, 'main'):
                    return False, None, None
            except nx.NetworkXNoPath:
                return False, None, None
        
        # Check facility budget
        new_paths = self.connected_paths.copy()
        new_paths[k] = new_path
        new_facilities = self.extract_facilities_from_paths(new_paths)
        
        if len(new_facilities) > self.max_facilities:
            return False, None, None
        
        # Calculate new objective
        new_obj = self.calculate_total_objective(new_paths)
        
        if new_obj < self.current_obj:
            return True, new_paths, new_obj
        
        return False, None, None
    
    def move_remove_facility(self) -> Tuple[bool, Dict, float]:
        """
        Try removing an underutilized facility and rerouting affected paths.
        """
        # Count facility usage
        facility_usage = {}
        for facility in self.facilities:
            count = sum(1 for path in self.connected_paths.values() if facility in path)
            facility_usage[facility] = count
        
        # Try removing least used facilities
        for facility, usage in sorted(facility_usage.items(), key=lambda x: x[1]):
            # Find paths using this facility
            affected_commodities = [k for k, path in self.connected_paths.items() if facility in path]
            
            if not affected_commodities:
                continue
            
            # Try rerouting all affected paths without this facility
            new_paths = self.connected_paths.copy()
            all_valid = True
            
            for k in affected_commodities:
                commodity_data = self.inst['commodities'][k]
                i_k = commodity_data[0]
                j_k = commodity_data[1]
                
                current_path = new_paths[k]
                is_hub_path = current_path[0].startswith('HUB_')
                
                # Create temporary graph without this facility
                temp_graph_A = self.graph_A.copy()
                temp_graph_A_U = self.graph_A_U.copy()
                
                if facility in temp_graph_A:
                    temp_graph_A.remove_node(facility)
                if facility in temp_graph_A_U:
                    temp_graph_A_U.remove_node(facility)
                
                try:
                    if is_hub_path:
                        hub = current_path[0]
                        new_hub_part = nx.shortest_path(temp_graph_A_U, hub, i_k, weight='weight')
                        if (i_k, j_k) in self.inst['arcs']['A']:
                            new_main_part = [i_k, j_k]
                        else:
                            new_main_part = nx.shortest_path(temp_graph_A, i_k, j_k, weight='weight')
                        
                        if not self.is_path_valid(new_hub_part, 'hub') or not self.is_path_valid(new_main_part, 'main'):
                            all_valid = False
                            break
                        
                        new_paths[k] = new_hub_part[:-1] + new_main_part
                    else:
                        if (i_k, j_k) in self.inst['arcs']['A']:
                            new_path = [i_k, j_k]
                        else:
                            new_path = nx.shortest_path(temp_graph_A, i_k, j_k, weight='weight')
                        
                        if not self.is_path_valid(new_path, 'main'):
                            all_valid = False
                            break
                        
                        new_paths[k] = new_path
                        
                except nx.NetworkXNoPath:
                    all_valid = False
                    break
            
            if not all_valid:
                continue

            # Check facility budget
            new_facilities = self.extract_facilities_from_paths(new_paths)
            if len(new_facilities) > self.max_facilities:
                continue
            
            # Calculate new objective
            new_obj = self.calculate_total_objective(new_paths)
            
            if new_obj < self.current_obj:
                return True, new_paths, new_obj
        
        return False, None, None


    def move_hub_reassignment(self) -> Tuple[bool, Dict, float]:
        """
        Try swapping activated hubs and optimizing facilities in affected paths.
        """
        current_hubs = list(self.activated_hubs)
        all_hubs = list(self.inst['nodes']['hubs'])
        
        # Find hospitals with deficit (those using hub paths)
        hospitals_with_deficit = {}
        hub_commodities = []  # Track which commodities use hub paths
        
        for k, path in self.connected_paths.items():
            if path[0].startswith('HUB_'):
                hub_commodities.append(k)
                i_k = self.inst['commodities'][k][0]
                s_k = self.inst['commodities'][k][4]
                n_ik = self.inst['drones']['available_drones'].get(i_k, 0)
                deficit = s_k - n_ik
                if deficit > 0:
                    if i_k not in hospitals_with_deficit:
                        hospitals_with_deficit[i_k] = deficit
        
        if not hospitals_with_deficit:
            return False, None, None
        

        
        # Try swapping one hub with another
        for hub_to_remove in current_hubs:
            for new_hub in all_hubs:
                if new_hub in current_hubs:
                    continue
                
                # Create new hub configuration
                new_hubs = [h if h != hub_to_remove else new_hub for h in current_hubs]
                new_hubs_set = set(new_hubs)
                
                # Try reassigning all hospitals to new hub configuration
                new_paths = {}
                all_valid = True
                
                for k, path in self.connected_paths.items():
                    if not path[0].startswith('HUB_'):
                        new_paths[k] = path
                        continue
                    
                    # This is a hub path - need to reassign and optimize
                    i_k = self.inst['commodities'][k][0]
                    j_k = self.inst['commodities'][k][1]
                    s_k = self.inst['commodities'][k][4]
                    n_ik = self.inst['drones']['available_drones'].get(i_k, 0)
                    deficit = s_k - n_ik
                    
                    # Find best hub from new configuration
                    best_hub_path = None
                    best_time = float('inf')
                    
                    for hub in new_hubs_set:
                        # Check hub capacity
                        hub_capacity = len(self.inst['drones']['hub_drones_sets'][hub])
                        if deficit > hub_capacity:
                            continue
                        
                        # Try multiple path options with different facility combinations
                        path_candidates = []
                        
                        # Option 1: Direct shortest path (may include facilities)
                        try:
                            if (hub, i_k) in self.inst['arcs']['A_U']:
                                hub_part = [hub, i_k]
                            else:
                                hub_part = nx.shortest_path(self.graph_A_U, hub, i_k, weight='weight')
                            
                            if (i_k, j_k) in self.inst['arcs']['A']:
                                main_part = [i_k, j_k]
                            else:
                                main_part = nx.shortest_path(self.graph_A, i_k, j_k, weight='weight')
                            
                            if self.is_path_valid(hub_part, 'hub') and self.is_path_valid(main_part, 'main'):
                                candidate_path = hub_part[:-1] + main_part
                                path_candidates.append(candidate_path)
                        except nx.NetworkXNoPath:
                            pass
                        
                        # Option 2: Try paths with specific facilities
                        all_facilities = list(self.inst['nodes']['facilities'])
                        
                        # Sample some facilities to try (to avoid combinatorial explosion)
                        sample_size = min(5, len(all_facilities))
                        sampled_facilities = random.sample(all_facilities, sample_size)

                        # Avoid add facility if exist direct path
                        #if (hub, i_k) in self.inst['arcs']['A_U'] and (i_k, j_k) in self.inst['arcs']['A']:
                        #    sampled_facilities = []

                        for facility in sampled_facilities:
                            # Try hub -> facility -> i_k
                            try:
                                if self.graph_A_U.has_edge(hub, facility) and self.graph_A_U.has_edge(facility, i_k):
                                    hub_with_facility = [hub, facility, i_k]

                                    if (i_k, j_k) in self.inst['arcs']['A']:
                                        main_part = [i_k, j_k]
                                    else:
                                        main_part = nx.shortest_path(self.graph_A, i_k, j_k, weight='weight')

                                    if self.is_path_valid(hub_with_facility, 'hub') and self.is_path_valid(main_part, 'main'):
                                        candidate_path = hub_with_facility[:-1] + main_part
                                        path_candidates.append(candidate_path)
                            except:
                                pass

                            # Try i_k -> facility -> j_k for main part
                            try:

                                if self.graph_A.has_edge(i_k, facility) and self.graph_A.has_edge(facility, j_k):
                                    if (hub, i_k) in self.inst['arcs']['A_U']:
                                        hub_part = [hub, i_k]
                                    else:
                                        hub_part = nx.shortest_path(self.graph_A_U, hub, i_k, weight='weight')

                                    main_with_facility = [i_k, facility, j_k]

                                    if self.is_path_valid(hub_part, 'hub') and self.is_path_valid(main_with_facility, 'main'):
                                        candidate_path = hub_part[:-1] + main_with_facility
                                        path_candidates.append(candidate_path)
                            except:
                                pass
                        


                    
                        # Option 3: Try removing facilities from original path if hub changed
                        if path[0] == hub_to_remove and hub == new_hub:
                            # Build path without facilities
                            try:
                                # Create temporary graphs without facilities
                                temp_A_U = self.graph_A_U.copy()
                                temp_A = self.graph_A.copy()
                                
                                for f in all_facilities:
                                    if f in temp_A_U:
                                        # Keep edges but increase weight to discourage use
                                        for u, v in list(temp_A_U.edges(f)):
                                            temp_A_U[u][v]['weight'] *= 100
                                        for u, v in list(temp_A_U.in_edges(f)):
                                            temp_A_U[u][v]['weight'] *= 100
                                    if f in temp_A:
                                        for u, v in list(temp_A.edges(f)):
                                            temp_A[u][v]['weight'] *= 100
                                        for u, v in list(temp_A.in_edges(f)):
                                            temp_A[u][v]['weight'] *= 100
                                
                                if (hub, i_k) in self.inst['arcs']['A_U']:
                                    hub_no_facility = [hub, i_k]
                                else:

                                    hub_no_facility = nx.shortest_path(temp_A_U, hub, i_k, weight='weight')
                                if (i_k, j_k) in self.inst['arcs']['A']:
                                    main_no_facility = [i_k, j_k]
                                else:
                                    main_no_facility = nx.shortest_path(temp_A, i_k, j_k, weight='weight')
                                
                                if self.is_path_valid(hub_no_facility, 'hub') and self.is_path_valid(main_no_facility, 'main'):
                                    candidate_path = hub_no_facility[:-1] + main_no_facility
                                    path_candidates.append(candidate_path)
                            except:
                                pass
                        
                        # Evaluate all candidate paths for this hub
                        for candidate_path in path_candidates:
                            # Check facility budget with this path
                            temp_paths = new_paths.copy()
                            temp_paths[k] = candidate_path
                            temp_facilities = self.extract_facilities_from_paths(temp_paths)
                            
                            if len(temp_facilities) > self.max_facilities:
                                continue
                            
                            candidate_time = self.calculate_path_time(k, candidate_path)
                            
                            if candidate_time < best_time:
                                best_time = candidate_time
                                best_hub_path = candidate_path
                    
                    if best_hub_path is None:
                        all_valid = False
                        break
                    
                    new_paths[k] = best_hub_path
                
                if not all_valid:
                    continue
                
                # Final facility budget check
                new_facilities = self.extract_facilities_from_paths(new_paths)
                if len(new_facilities) > self.max_facilities:
                    continue
                
                # Calculate new objective
                new_obj = self.calculate_total_objective(new_paths)
                
                if new_obj < self.current_obj:
                    self.activated_hubs = new_hubs_set
                    return True, new_paths, new_obj
        
        # Try optimizing facilities for existing hub paths without changing hubs
        if len(hub_commodities) > 0:
            new_paths = self.connected_paths.copy()
            
            for k in hub_commodities:
                current_path = self.connected_paths[k]
                hub = current_path[0]
                i_k = self.inst['commodities'][k][0]
                j_k = self.inst['commodities'][k][1]
                
                best_path = current_path
                best_time = self.calculate_path_time(k, current_path)
                
                # Try adding/removing/swapping facilities
                all_facilities = list(self.inst['nodes']['facilities'])
                facilities_in_path = [node for node in current_path if node.startswith('F_')]
                
                # Try removing each facility
                for facility_to_remove in facilities_in_path:
                    try:
                        # Create path avoiding this facility
                        temp_A_U = self.graph_A_U.copy()
                        temp_A = self.graph_A.copy()
                        
                        if facility_to_remove in temp_A_U:
                            temp_A_U.remove_node(facility_to_remove)
                        if facility_to_remove in temp_A:
                            temp_A.remove_node(facility_to_remove)
                        
                        if (hub, i_k) in self.inst['arcs']['A_U']:
                            new_hub_part = [hub, i_k]
                        else:
                            new_hub_part = nx.shortest_path(temp_A_U, hub, i_k, weight='weight')
                        if (i_k, j_k) in self.inst['arcs']['A']:
                            new_main_part = [i_k, j_k]
                        else:

                            new_main_part = nx.shortest_path(temp_A, i_k, j_k, weight='weight')
                        
                        if self.is_path_valid(new_hub_part, 'hub') and self.is_path_valid(new_main_part, 'main'):
                            candidate_path = new_hub_part[:-1] + new_main_part
                            
                            temp_paths = new_paths.copy()
                            temp_paths[k] = candidate_path
                            temp_facilities = self.extract_facilities_from_paths(temp_paths)
                            
                            if len(temp_facilities) <= self.max_facilities:
                                candidate_time = self.calculate_path_time(k, candidate_path)
                                if candidate_time < best_time:
                                    best_time = candidate_time
                                    best_path = candidate_path
                    except:
                        pass
               
                if best_path != current_path:
                    new_paths[k] = best_path

            


            
            # Check if we found improvements
            new_obj = self.calculate_total_objective(new_paths)
            if new_obj < self.current_obj:
                return True, new_paths, new_obj
        
        return False, None, None

    def postprocess_paths(self) -> None:
        """
        Replace paths with direct arcs when they exist.
        """
        print("\n=== POST-PROCESSING PATHS ===")
        changes = 0
        
        for k, path in list(self.connected_paths.items()):
            i_k = self.inst['commodities'][k][0]
            j_k = self.inst['commodities'][k][1]
            
            modified = False
            
            # Check if hub path
            if path[0].startswith('HUB_'):
                try:
                    ik_index = path.index(i_k)
                except ValueError:
                    continue
                
                hub = path[0]
                hub_part = path[:ik_index+1]
                main_part = path[ik_index:]
                
                # Replace hub part if direct arc exists
                if (hub, i_k) in self.inst['arcs']['A_U'] and len(hub_part) > 2:
                    hub_part = [hub, i_k]
                    modified = True
                
                # Replace main part if direct arc exists
                if (i_k, j_k) in self.inst['arcs']['A'] and len(main_part) > 2:
                    main_part = [i_k, j_k]
                    modified = True
                
                if modified:
                    new_path = hub_part[:-1] + main_part
                    self.connected_paths[k] = new_path
                    changes += 1
                    print(f"  Commodity {k}: {len(path)} → {len(new_path)} nodes")
            
            else:
                # Direct path - replace if direct arc exists
                if (i_k, j_k) in self.inst['arcs']['A'] and len(path) > 2:
                    new_path = [i_k, j_k]
                    self.connected_paths[k] = new_path
                    changes += 1
                    print(f"  Commodity {k}: {len(path)} → {len(new_path)} nodes")
        
        if changes > 0:
            # Recalculate objective and facilities
            old_facilities = len(self.facilities)
            self.facilities = self.extract_facilities_from_paths(self.connected_paths)
            old_obj = self.current_obj
            self.current_obj = self.calculate_total_objective(self.connected_paths)
            self.best_obj = self.current_obj
            
            print(f"\nPost-processing complete:")
            print(f"  Paths cleaned: {changes}")
            print(f"  Facilities: {old_facilities} → {len(self.facilities)}")
            print(f"  Objective: {old_obj:.2f} → {self.current_obj:.2f} (Δ {self.current_obj - old_obj:+.2f})")
        else:
            print("No changes needed - all paths already optimal")
        
    # ============ MAIN LOCAL SEARCH ============

    
    def run(self, max_iterations: int = 100, max_no_improve: int = 20) -> Dict:
        """Run with detailed tracking"""
        
        iteration = 0
        no_improve_count = 0
        
        while iteration < max_iterations and no_improve_count < max_no_improve:
            iteration += 1
            improved = False
            
            # Hub reassignment (ogni 5 iterazioni)
            if iteration % 5 == 0:
                self.move_stats['hub_reassignment']['attempts'] += 1
                success, new_paths, new_obj = self.move_hub_reassignment()
                if success:
                    improvement = self.current_obj - new_obj
                    self.move_stats['hub_reassignment']['successes'] += 1
                    self.move_stats['hub_reassignment']['total_improvement'] += improvement
                    self.improvement_history.append((iteration, 'hub_reassignment', improvement))
                    
                    self.connected_paths = new_paths
                    self.facilities = self.extract_facilities_from_paths(new_paths)
                    self.current_obj = new_obj
                    improved = True
                    no_improve_count = 0
                    continue
            
            # Reroute e facility swap
            commodities = list(self.connected_paths.keys())
            random.shuffle(commodities)
            
            for k in commodities:
                # Reroute
                self.move_stats['reroute']['attempts'] += 1
                success, new_paths, new_obj = self.move_reroute_path(k)
                if success:
                    improvement = self.current_obj - new_obj
                    self.move_stats['reroute']['successes'] += 1
                    self.move_stats['reroute']['total_improvement'] += improvement
                    self.improvement_history.append((iteration, 'reroute', improvement))
                    
                    self.connected_paths = new_paths
                    self.facilities = self.extract_facilities_from_paths(new_paths)
                    self.current_obj = new_obj
                    improved = True
                    break
                
                # Facility swap
                #self.move_stats['facility_swap']['attempts'] += 1
                #success, new_paths, new_obj = self.move_facility_swap_in_path(k)
                #if success:
                #    improvement = self.current_obj - new_obj
                #    self.move_stats['facility_swap']['successes'] += 1
                #    self.move_stats['facility_swap']['total_improvement'] += improvement
                #    self.improvement_history.append((iteration, 'facility_swap', improvement))
                #    
                #    self.connected_paths = new_paths
                #    self.facilities = self.extract_facilities_from_paths(new_paths)
                #    self.current_obj = new_obj
                #    improved = True
                #    break
            
            # Remove facility
            if not improved and iteration % 3 == 0:
                self.move_stats['remove_facility']['attempts'] += 1
                success, new_paths, new_obj = self.move_remove_facility()
                if success:
                    improvement = self.current_obj - new_obj
                    self.move_stats['remove_facility']['successes'] += 1
                    self.move_stats['remove_facility']['total_improvement'] += improvement
                    self.improvement_history.append((iteration, 'remove_facility', improvement))
                    
                    self.connected_paths = new_paths
                    self.facilities = self.extract_facilities_from_paths(new_paths)
                    self.current_obj = new_obj
                    improved = True
            
            if improved:
                no_improve_count = 0
                if self.current_obj < self.best_obj:
                    self.best_obj = self.current_obj
            else:
                no_improve_count += 1
        
        # Post-processing
        self.postprocess_paths()
        
        return {
            'connected_paths': self.connected_paths,
            'facilities': self.facilities,
            'OBJ': self.current_obj,
            'activated_hubs': self.activated_hubs,
            'iterations': iteration,
            'move_stats': self.move_stats,
            'improvement_history': self.improvement_history
        }
    
    def save_move_statistics(self, instance_name: str, initial_obj: float, results_file: str = 'local_search_stats.csv'):
            """Salva statistiche dettagliate delle mosse"""
            
            stats_data = {
                'instance': instance_name,
                'initial_obj': initial_obj,  # Passato come parametro
                'final_obj': self.current_obj,
                'total_improvement': initial_obj - self.current_obj,
                'improvement_pct': 100 * (initial_obj - self.current_obj) / initial_obj,
                'iterations': len(self.improvement_history)
            }
            
            mapping = {
                'hub_reassignment': 'M4',
                'reroute': 'M2', 
                'facility_swap': 'M1',
                'remove_facility': 'M3'
            }

            # Aggiungi statistiche per ogni mossa
            for move_name, move_data in self.move_stats.items():
                move_code = mapping[move_name]  # Get mapped code (M1, M2, etc)
                stats_data[f'{move_code}_attempts'] = move_data['attempts']
                stats_data[f'{move_code}_successes'] = move_data['successes'] 
                stats_data[f'{move_code}_success_rate'] = (
                    move_data['successes'] / move_data['attempts']
                    if move_data['attempts'] > 0 else 0
                )
                stats_data[f'{move_code}_total_improvement'] = move_data['total_improvement']
                stats_data[f'{move_code}_avg_improvement'] = (
                    move_data['total_improvement'] / move_data['successes']
                    if move_data['successes'] > 0 else 0
                )
                        
            
            df = pd.DataFrame([stats_data])
            
            df.to_csv(results_file, mode='a', 
                    header=not pd.io.common.file_exists(results_file),
                    index=False)


