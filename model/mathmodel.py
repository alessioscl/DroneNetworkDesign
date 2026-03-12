"""
Drone Distribution System for Medical Transport
Refactored version with clear separation of concerns
"""

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
from model.instancegenerator import InstanceGenerator

class MathematicalModel:
    """
    Mathematical model for the drone distribution optimization problem.
    Handles model building, solving, and solution analysis.
    """
    
    def __init__(self, instance: Dict, nodes_df):
        """Initialize model with instance data."""
        self.instance = instance
        self.model = None
        self.time_obj = None
        self.callback = None
        self.nodes_df = nodes_df
        

    '''
    def _create_cut_callback(self):
        """Create cutting plane callback for relaxed model."""
        def cut_callback(model, where):
            if where != GRB.Callback.MIPSOL:
                return

            VIOL_TOL = 1e-6
            
            # Get solution values
            z_val = {
                i: model.cbGetSolution(model._z[i])
                for i in model._F.union(model._U)
            }

            f_val = {
                (k, delta, i, j): model.cbGetSolution(model._f[k, delta, i, j])
                for k in model._K
                for delta in model._DELTA_H | model._delta_u
                for (i, j) in model._A
            }

            # Add cuts for facility/hub activation
            for i in model._F.union(model._U):
                for k in model._K:
                    for delta in model._DELTA_H | model._delta_u:
                        for j in set(model._O_k_H.values()) | set(model._D_k_H.values()) | model._F:
                            if (i, j) in model._A:
                                f = f_val[k, delta, i, j]
                                z = z_val[i]
                                if f > 0.5 and z < 1.0 - VIOL_TOL:
                                    model.cbLazy(model._f[k, delta, i, j] <= model._z[i])

        return cut_callback
    '''
    def _create_cut_callback(self):
        """Create optimized cutting plane callback for relaxed model."""
        def cut_callback(model, where):
            if where != GRB.Callback.MIPSOL:
                return

            VIOL_TOL = 1e-6
            cuts_added = 0
            max_cuts = 100  # Limite massimo di tagli per iterazione
            
            # Get solution values solo quando necessario
            z_val = {
                i: model.cbGetSolution(model._z[i])
                for i in model._F.union(model._U)
            }

            # Pre-filtra i nodi che potrebbero essere violati
            candidate_nodes = {i for i, z in z_val.items() if z < 1.0 - VIOL_TOL}
            
            if not candidate_nodes:
                return  # Nessuna violazione possibile
                
            # Get f values solo per i nodi candidati
            f_val = {}
            for i in candidate_nodes:
                for k in model._K:
                    for delta in model._DELTA_H | model._delta_u:
                        for j in set(model._O_k_H.values()) | set(model._D_k_H.values()) | model._F:
                            if (i, j) in model._A:
                                f_val[k, delta, i, j] = model.cbGetSolution(model._f[k, delta, i, j])

            # Aggiungi tagli solo dove necessario
            for i in candidate_nodes:
                z = z_val[i]
                
                # Trova violazioni per questo nodo
                violations = []
                for k in model._K:
                    for delta in model._DELTA_H | model._delta_u:
                        for j in set(model._O_k_H.values()) | set(model._D_k_H.values()) | model._F:
                            if (i, j) in model._A:
                                f = f_val.get((k, delta, i, j), 0)
                                if f > VIOL_TOL:  # Solo se f > 0
                                    violations.append((k, delta, i, j, f))
                
                # Ordina per violazione più grande
                violations.sort(key=lambda x: x[4], reverse=True)
                
                # Aggiungi solo i tagli più violati
                for k, delta, i, j, f_val_curr in violations[:10]:  # Max 10 tagli per nodo
                    if cuts_added >= max_cuts:
                        return
                        
                    model.cbLazy(model._f[k, delta, i, j] <= model._z[i])
                    cuts_added += 1

            # Optional: log per debugging
            #if cuts_added > 0:
            #    print(f"Added {cuts_added} lazy cuts")

        return cut_callback
    

    def build_model(self, model_type: str = 'binary') -> Tuple[gp.Model, gp.LinExpr]:
        """Build the complete Gurobi model."""
        if model_type not in ['binary', 'relaxed']:
            raise ValueError("model_type must be either 'binary' or 'relaxed'")
        
        K = self.instance['constants']['K']
        commodities = self.instance['commodities']
        H = self.instance['nodes']['hospitals']
        F = self.instance['nodes']['facilities']
        U = self.instance['nodes']['hubs']
        V = self.instance['nodes']['all']
        A_U = self.instance['arcs']['A_U'] 
        A_0 = self.instance['arcs']['A_0'] 
        A_1 = self.instance['arcs']['A_1']
        A_2 = self.instance['arcs']['A_2'] 
        A_3 = self.instance['arcs']['A_3']
        A = self.instance['arcs']['A']
        F_ij = self.instance['arcs']['F_ij']
        DELTA_H = self.instance['drones']['hospital_drones']
        drone_sets = self.instance['drones']['hospital_drones_sets']
        delta_u = self.instance['drones']['hub_drones']
        delta_hub = self.instance['drones']['hub_drones_sets']
        n_i = self.instance['drones']['available_drones']
        a_F_i = self.instance['facility_indicators']
        O_k_H = self.instance['origins']
        D_k_H = self.instance['destinations']
        distance_matrix = self.instance['matrices']['distance']
        housing_matrix = self.instance['matrices']['housing']
        c_F_i = self.instance['costs']['facility_costs']
        c_U_i = self.instance['costs']['hub_costs']
        b_F = self.instance['constants']['b_F']
        b_U = self.instance['constants']['b_U']
        pi_k = self.instance['penalties']
        tau_i = self.instance['constants']['tau_i']
        v = self.instance['constants']['v']
        
         # Creazione del modello
        m = gp.Model("Drone_Distribution")
            
        # VARIABILI BINARIE TATTICHE
        z ={}
        if model_type == 'binary':
            for i in F.union(U):
                z[i] = m.addVar(vtype=GRB.BINARY, name=f'z-{i}')
        else:
            for i in F.union(U):
                z[i] = m.addVar(vtype=GRB.CONTINUOUS, lb=0,ub = 1, name=f'z-{i}')
            #z = m.addVars(F.union(U), vtype=GRB.BINARY, name='z')
            
        # VARIABILI OPERATIVE
        f = {}
        for k in K:
            for delta in DELTA_H | delta_u:
                for (i,j) in A:
                    f[(k, delta, i, j)] = m.addVar(vtype=GRB.BINARY, name=f'f-{k}-{delta}-{i}-{j}')
        #f = m.addVars(k,DELTA_H|delta_u,A, vtype=GRB.BINARY, name='f')
            
        # VARIABILI DI RITARDO
        t = {}
        for k in K:
            for delta in DELTA_H | delta_u:
                t[(k, delta)] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f't-{k}-{delta}')
        #t = m.addVars(K,DELTA, vtype=GRB.BINARY, name='z')
        
               
        # VINCOLI
        # Vincoli di flusso per i nodi di origine, destinazione e facilities intermedie
        flow_constraints = {}
        for k in K:
            i_k, j_k, s_k = commodities[k][0], commodities[k][1], commodities[k][4]
            steps = min(s_k, n_i[i_k])  # Numero massimo di delta utilizzabili
            for delta in list(drone_sets[i_k])[:steps]:
                for i in V:
                    #inflow = gp.quicksum(f[k, delta, j, i] for j in V - {i} if (j, i) in A)
                    #outflow = gp.quicksum(f[k, delta, i, j] for j in V - {i} if (i, j) in A)
                    if i == i_k:  # Nodo di origine
                        flow_constraints[k, i, delta] = m.addConstr(gp.quicksum(f[k, delta, i, j] for j in V if i!=j if (i, j) in A) - gp.quicksum(f[k, delta, j, i] for j in V if i!=j if (j, i) in A) == 1, name=f"Origin_{k}_{delta}")
                    elif i == j_k:  # Nodo di destinazione
                        flow_constraints[k, i, delta] = m.addConstr(gp.quicksum(f[k, delta, i, j] for j in V if i!=j if (i, j) in A) - gp.quicksum(f[k, delta, j, i] for j in V if i!=j if (j, i) in A) == -1, name=f"Destination_{k}_{delta}")
                    elif i in F:  # Nodo facility intermedia
                        flow_constraints[k, i, delta] = m.addConstr(gp.quicksum(f[k, delta, i, j] for j in V if i!=j if (i, j) in A) - gp.quicksum(f[k, delta, j, i] for j in V if i!=j if (j, i) in A) == 0, name=f"Facility_{k}_{delta}")
                   
        
        for k in K:
            i_k, j_k = commodities[k][0], commodities[k][1]
            #steps = min(s_k, n_i[i_k])  # Numero massimo di delta utilizzabili
            #for delta in list(drone_sets[i_k])[:steps] + list(delta_u):
            for delta in DELTA_H | delta_u:
                for i in V-U:
                    for j in H:
                        if j != j_k and (i,j) in A:
                            m.addConstr(f[k,delta,i,j] == 0)

        
        flow_hub = {}
        for k in K:
            i_k = commodities[k][0]
            j_k = commodities[k][1]
            for delta in set().union(*(delta_hub[j] for j in U)): #delta_u:
                for i in F:
                    inflow = gp.quicksum(f[k,delta,j,i] for j in V-{i} if (j,i) in A )
                    outflow = gp.quicksum(f[k,delta,i,j] for j in V-{i} if (i,j) in A )
                    flow_hub[k,i,j] = m.addConstr(outflow-inflow == 0, name = f'flow_facilitiy_hub_{k}')
                    
        
        delta_union = {}
        for k in K:
            i_k = commodities[k][0]
            s_k = commodities[k][4]
            relevant_nodes = {j for j in U if (j, i_k) in A_2}
            # Unisci gli insiemi Delta^U(j) per i nodi filtrati
            delta_union[k] = set().union(*(delta_hub[j] for j in relevant_nodes))
            if n_i[i_k] < s_k:
                for delta in delta_union[k]:  # Delta^U(j) per (j, i^k) in A^2
                    m.addConstr(
                        gp.quicksum(f[k, delta, i_k, j] for j in V - {i_k} if (i_k,j) in A) -
                        gp.quicksum(f[k, delta, j, i_k] for j in V - {i_k} if (j, i_k) in A) == 0,
                        name=f"flow_balance_{i_k}_{k}_{delta}")
        
        hub_flow = {}
        for k in K:
            i_k = commodities[k][0]
            s_k = commodities[k][4]
            if n_i[i_k] < s_k:
                hub_flow[k] = m.addConstr(gp.quicksum(f[k, d, j,i_k] for j in U if (j, i_k) in A_2 for d in delta_hub[j]) == s_k - n_i[i_k], name = f'hub_recovery_{k}')
        
        for k in K:
            i_k = commodities[k][0]
            j_k = commodities[k][1]
            s_k = commodities[k][4]
            if n_i[i_k] < s_k:
                # Calcola l'unione degli insiemi Delta^U(j) per i nodi j in U tali che (j, i^k) in A^2
                relevant_deltas = set().union(*(delta_hub[j] for j in U if (j, i_k) in A_2))
                m.addConstr(gp.quicksum(-f[k, delta, i, j_k]for i in V if i != j_k for delta in relevant_deltas if (i, j_k) in A) == -(s_k - n_i[i_k]), name=f"flow_balance_{k}")
        
                
        
        
        for h in U:
            for delta in delta_hub[h]:
                m.addConstr(gp.quicksum(f[k,delta,h,i] for k in K for i in O_k_H.values() if (h,i) in A)<=1, name = f'assignment-{delta}-{i}-{j}')    
        
        
        
        for i in O_k_H.values():  # For each origin node
            for delta in drone_sets[i]:              
                m.addConstr(
                    gp.quicksum(
                        f[k,delta,i,j] 
                        for k in K 
                        for j in F.union(D_k_H.values()) 
                        if (i,j) in A
                    ) <= 1
                )
        
        if model_type == 'binary':
            for (j, i_k) in A_3:
                for h in F_ij[(j, i_k)]:
                    m.addConstr(gp.quicksum(f[k,d,j,i_k] for k in K for d in delta_hub[j] if (j,i_k) in A)<=len(K)*len(delta_hub[j])*z[h], name =f'HubFacility_Usage_{j}_{h}')
            
            #for j in U:
            #    result = {O_k_H[k] for k in K if (j, i_k) in A_2}
            #    m.addConstr(gp.quicksum(f[k,d,j,i_k] for k in K for d in delta_hub[j] for i_k in result if (j,i_k) in A) <= len(K)*len(delta_hub[j])*len(A_2)*z[j], name = f'hub{j}_activation')
            for j in U:
                #result = {O_k_H[k] for k in K if (j, O_k_H[k]) in A_2}
                m.addConstr(
                    gp.quicksum(f[k, d, j, O_k_H[k]] 
                                for k in K for d in delta_hub[j] 
                                for i_k in [O_k_H[k]] if (j, i_k) in A)
                    <= len(K) * len(delta_hub[j]) * len(A_2) * z[j]
                )
            
            for i in F:
                #result_set = set().union((O_k_H[k] for k in K))  # Unione di tutti gli O^k(H)
                result_set = set(O_k_H[k] for k in K)
                #result_set = set().union(*[{O_k_H[k]} for k in K])
                #print(f"Initial result_set for facility {i}: {result_set}")
                result_set.update(F)  # Aggiunge gli elementi di F
                result_set.discard(i)
                m.addConstr(gp.quicksum(f[k,d,j,i] for k in K for d in DELTA_H | delta_u for j in result_set if (j,i) in A)<= len(K)*len(DELTA_H | delta_u)*len(result_set)*z[i])
        
        
        shortest_path_costs = {}
        all_origins = set(O_k_H.values())  # Tutti i nodi origine
        for (i,j) in A_3:
            forbidden = all_origins - {j}
            path_arcs = InstanceGenerator.shortest_path(i, j, A = A_U,V = V, distance_matrix = distance_matrix, tau_i = tau_i, a_f_i = a_F_i, v = v, forbidden_nodes=forbidden)[0]
            if path_arcs:
                shortest_path_costs[(i,j)] = sum((tau_i * a_F_i[i] + distance_matrix.loc[i,j] / v) for (i,j) in path_arcs)
        

        for k in K:
            e_k = commodities[k][2]
            l_k = commodities[k][3]
            for delta in DELTA_H | delta_u:
                m.addConstr(e_k + gp.quicksum((tau_i * a_F_i[i] + distance_matrix.loc[i,j] / v) * f[k,delta,i,j] 
                                   for (i,j) in A if (i,j) not in A_3) + gp.quicksum(shortest_path_costs[(i,j)] * f[k,delta,i,j] for (i,j) in A_3 if shortest_path_costs[(i,j)]) - l_k <= t[k,delta],name = f'penalty-{k}-{delta}')    
        
        budget_hub = {}
        budget_hub[i] = m.addConstr(gp.quicksum(c_U_i[i] * z[i] for i in U) <= b_U, name = 'budgethub')
        
        budget_facility = {}
        budget_facility[i] = m.addConstr(gp.quicksum(c_F_i[i] * z[i] for i in F) <= b_F, name = 'budgetfacilities')
        

        
        time_obj = (gp.quicksum((tau_i * a_F_i[i] + distance_matrix.loc[i,j] / v) * f[k,delta,i,j] 
                   for k in K
                   for delta in DELTA_H | delta_u 
                   for (i,j) in A - A_3) + 
                gp.quicksum(shortest_path_costs[(i,j)] * f[k,delta,i,j]
                   for k in K
                   for delta in DELTA_H | delta_u 
                   for (i,j) in A_3 if shortest_path_costs[(i,j)]) + 
                gp.quicksum(pi_k[k] * t[k, delta] 
                   for k in K 
                   for delta in DELTA_H | delta_u))
                #gp.quicksum(pi_k[k] * gp.quicksum(t[k, delta] 
                #   for delta in DELTA_H | delta_u) for k in K))

        #cost_obj = gp.quicksum(c_U_i[i] * z[i] for i in U) + gp.quicksum(c_F_i[i] * z[i] for i in F)

        
        m.setObjective(time_obj, GRB.MINIMIZE)
        
        # Configure model parameters
        if model_type == 'relaxed':
            m.setParam('OutputFlag', 0)
            m.setParam('LazyConstraints', 1)
            self.callback = self._create_cut_callback()
            # Store variables for callback
            m._f = f
            m._z = z
            m._F = self.instance['nodes']['facilities']
            m._U = self.instance['nodes']['hubs']
            m._K = self.instance['constants']['K']
            m._DELTA_H = self.instance['drones']['hospital_drones']
            m._delta_u = self.instance['drones']['hub_drones']
            m._A = self.instance['arcs']['A']
            m._O_k_H = self.instance['origins']
            m._D_k_H = self.instance['destinations']
        else:
            m.setParam('OutputFlag', 0)
            self.callback = None
        
        m.setParam('TimeLimit', 3600)
        m.setParam('Method', 4)
        m.update()
        
        # Write model to file
        #filename = f'{"relaxed_" if model_type == "relaxed" else ""}model.lp'
        #m.write(filename)
        
        return m, time_obj

    def solve(self, model_type: str = 'binary') -> Optional[Dict]:
        """Solve the optimization model."""
        start_build_time = time.time()
        if not hasattr(self, 'model') or self.model is None:
            self.model, self.time_obj = self.build_model(model_type)
        end_build_time = time.time()
        model_build_time = end_build_time - start_build_time

        try:
            start_optimize = time.time()
            if model_type == 'binary':
                self.model.optimize()
            else:
                self.model.optimize(self.callback)
            end_optimize = time.time()
            optimize_time = end_optimize - start_optimize
        except Exception as e:
            print(f"Optimization error: {str(e)}")
            return None
        
        commodities = self.instance['commodities']
        K = self.instance['constants']['K']
        A_U = self.instance['arcs']['A_U'] 
        A_3 = self.instance['arcs']['A_3']
        A = self.instance['arcs']['A']
        DELTA_H = self.instance['drones']['hospital_drones']
        delta_u = self.instance['drones']['hub_drones']
        F = self.instance['nodes']['facilities']
        U = self.instance['nodes']['hubs']
        V = self.instance['nodes']['all']
        pi_k = self.instance['penalties']
        c_F_i = self.instance['costs']['facility_costs']
        c_U_i = self.instance['costs']['hub_costs']
        b_F = self.instance['constants']['b_F']
        b_U = self.instance['constants']['b_U']
        a_F_i = self.instance['facility_indicators']
        distance_matrix = self.instance['matrices']['distance']
        O_k_H = self.instance['origins']
        tau_i = self.instance['constants']['tau_i']
        v = self.instance['constants']['v']

        if self.model.status == GRB.OPTIMAL or self.model.status == GRB.TIME_LIMIT:
            #print("\nOptimal solution found!")
            z = {}
            for i in self.instance['nodes']['facilities'].union(self.instance['nodes']['hubs']):
                var_name = f'z-{i}'
                z[i] = self.model.getVarByName(var_name)
            
            f = {}
            for k in K:
                for delta in DELTA_H | delta_u:
                    for (i, j) in A:
                        var_name = f'f-{k}-{delta}-{i}-{j}'
                        f[(k, delta, i, j)] = self.model.getVarByName(var_name)
            
            t = {}
            for k in K:
                for delta in DELTA_H | delta_u:
                    var_name = f't-{k}-{delta}'
                    t[(k, delta)] = self.model.getVarByName(var_name)
                    
            
            shortest_path_costs = {}
            all_origins = set(O_k_H.values())  # Tutti i nodi origine
            for (i,j) in A_3:
                forbidden = all_origins - {j}
                path_arcs = InstanceGenerator.shortest_path(i, j, A = A_U,V = V, distance_matrix = distance_matrix, tau_i = tau_i, a_f_i = a_F_i, v = v, forbidden_nodes=forbidden)[0]
                if path_arcs:
                    shortest_path_costs[(i,j)] = sum((tau_i * a_F_i[i] + distance_matrix.loc[i,j] / v) for (i,j) in path_arcs)
            
            time_obj = (gp.quicksum((tau_i * a_F_i[i] + distance_matrix.loc[i,j] / v) * f[k,delta,i,j].X
                   for k in K
                   for delta in DELTA_H | delta_u 
                   for (i,j) in A - A_3) + 
                gp.quicksum(shortest_path_costs[(i,j)] * f[k,delta,i,j].X
                   for k in K
                   for delta in DELTA_H | delta_u 
                   for (i,j) in A_3 if shortest_path_costs[(i,j)]) + 
                gp.quicksum(pi_k[k] * t[k, delta].X for k in K for delta in DELTA_H | delta_u))
            

            penalty_cost = sum(self.instance['penalties'][k] * sum(t[k, delta].X for delta in DELTA_H | delta_u) for k in K) 
            operational_cost = time_obj - penalty_cost
        
            
            #print("\n=== PERFORMANCE METRICS ===")
            #print(f"Late Commodities Ratio: {late_ratio:.2%}")
            #print(f"Logistics Ratio: {logistics_ratio:.2f} km/commodity")
            #print(f"Penalty Cost Ratio: {penalty_ratio:.2%}")
            #print(f"Drone Assignment Cost Ratio: {drone_assignment_ratio:.2%}")
            #
            #print("\n=== TIMING BREAKDOWN ===")
            #print(f"Model Build Time: {model_build_time:.3f} s")
            #print(f"Optimization Execution Time: {optimize_time:.3f} s")
            #print(f"Total Execution Time: {model_build_time + optimize_time:.3f} s")
            #
            #print("\n=== OBJ BREAKDOWN ===")
            #print(f"Objective value: {obj:.2f}")
            facilities_cost = sum(self.instance['costs']['facility_costs'][i] * self.model.getVarByName(f'z-{i}').X for i in self.instance['nodes']['facilities'])
            hubs_cost = sum(self.instance['costs']['hub_costs'][i] * self.model.getVarByName(f'z-{i}').X for i in self.instance['nodes']['hubs'])
            #print(f"Penalty costs: {penalty_cost}")
            #print(f"Facilities_costs: {facilities_cost}")
            #print(f"Hubs_costs: {hubs_cost}")
            #print(f"Operational: {operational_cost}")
            #print(f"Budget hub utilizzato: {hubs_cost:.2f}/{self.instance['constants']['b_U']:.2f}")
            #print(f"Budget facility utilizzato: {facilities_cost:.2f}/{self.instance['constants']['b_F']:.2f}")

            
        #elif self.model.status == GRB.TIME_LIMIT:
        #    print("\nTime limit reached. Best solution found:")
        #    print(f"Current objective value: {self.model.objVal:.2f}")
        #    print(f"Optimality gap: {self.model.MIPGap * 100:.2f}%")
        else:
            #print(f"\nOptimization was stopped with status {self.model.status}")
            return None
        
        
        active_edges = [(v.VarName.split('-')[3], v.VarName.split('-')[4]) 
                        for v in self.model.getVars() 
                        if v.VarName.startswith('f-') and v.X > 0.5]
        hub_edges = []
        other_edges = []
        for i, j in active_edges:
            i_norm = i.split('_origin_')[0].split('_dest_')[0]
            j_norm = j.split('_origin_')[0].split('_dest_')[0]
            if i_norm in self.instance['nodes']['hubs']:
                hub_edges.append((i_norm, j_norm))
            else:
                other_edges.append((i_norm, j_norm))
        
                            
        urgent_types = {"organ", "blood"}
        n_urgent = sum(1 for commodity in self.instance['commodities'].values() if commodity[6] in urgent_types)
        n_non_urgent = len(self.instance['commodities']) - n_urgent

        active_facilities = [i for i in self.instance['nodes']['facilities'] if self.model.getVarByName(f'z-{i}').X > 0.5]
        active_hubs = [i for i in self.instance['nodes']['hubs'] if self.model.getVarByName(f'z-{i}').X > 0.5]
        
        # Converti le liste in stringhe separate da punto e virgola per l'output CSV
        active_facilities_str = ';'.join(sorted(active_facilities)) if active_facilities else "None"
        active_hubs_str = ';'.join(sorted(active_hubs)) if active_hubs else "None"


        instance_features = {
            "N.H. o-d": len(set((k[0], k[1]) for k in self.instance['commodities'].values())),
            "N.C.": len(self.instance['constants']['K']),
            "N.H.": len(self.instance['nodes']['hospitals']),
            "N.F.": len(self.instance['nodes']['facilities']),
            "N. HUBS": len(self.instance['nodes']['hubs']),
            "N.A.": len(self.instance['arcs']['A']),
            "N.D.H": len(self.instance['drones']['hospital_drones']),
            "N.D.HUBS": len(self.instance['drones']['hub_drones']),
            "N.D.": len(self.instance['drones']['hospital_drones']) + len(self.instance['drones']['hub_drones']),
            "N.D.R": sum(k[4] for k in self.instance['commodities'].values()),
            "B.U.": self.instance['constants']['b_U'],
            "B.F.": self.instance['constants']['b_F'],
            "N.URGENT": n_urgent,
            "N.NON-URGENT": n_non_urgent
        }
    
        results = {
            "LB": self.model.ObjBound,
            "GAP": self.model.MIPGap,
            "C.OBJ": facilities_cost + hubs_cost, #self.model.ObjVal,
            "T.OBJ": self.model.ObjVal,#time_obj,
            "C.PEN": penalty_cost,
            "C.OPER": operational_cost,
            "L.C.R": self.get_delayed_commodities_ratio(),
            "LOG.R": self.get_logistics_ratio(),
            "C.F.A": facilities_cost,
            "C.HUBS.A": hubs_cost,
            #"A.F.R(ale)": self.active_facility_ratio()[0],
            #"A.F.R(dem)": self.active_facility_ratio()[1],
            "N.F.A ": sum(self.model.getVarByName(f'z-{i}').X for i in self.instance['nodes']['facilities']),
            "N.HUBS.A ": sum(self.model.getVarByName(f'z-{i}').X for i in self.instance['nodes']['hubs']),
            "N. ARCS": len(active_edges),
            "N. HUBS ARCS": len(hub_edges),
            "ACTIVE_FACILITIES": active_facilities_str,
            "ACTIVE_HUBS": active_hubs_str
        }
    
        computational_performance = {
            "Model Build Time": model_build_time,
            "Optimization Execution Time": optimize_time,
            "TIME(s)": model_build_time + optimize_time
        }
    
        data = {**instance_features, **results, **computational_performance}
        df_results = pd.DataFrame([data])
        results_filename = 'DMNDP-MM-rome.csv' if model_type == 'binary' else 'DMNDP-TEST-RELAXED.csv'
        df_results.to_csv(results_filename, mode='a', 
                        header=not pd.io.common.file_exists(results_filename), 
                        index=False)
        print(f"Results saved to {results_filename}")

        
        return self.model

        #if self.model.status == GRB.OPTIMAL or self.model.status == GRB.TIME_LIMIT:
        #    solution_info = self._extract_solution()
        #    solution_info.update({
        #        'model_build_time': model_build_time,
        #        'optimize_time': optimize_time,
        #        'total_time': model_build_time + optimize_time,
        #        'model_type': model_type,
        #        'status': self.model.status
        #    })
        #    return solution_info
        #else:
        #    print(f"Optimization failed with status: {self.model.status}")
        #    return None

    def get_delayed_commodities_ratio(self):
        """Calculate the ratio of commodities delivered with delay."""
        if not hasattr(self, 'model'):# or self.model.status != GRB.OPTIMAL:
            raise ValueError("Model not solved or not optimal.")
        
        K = self.instance['constants']['K']
        DELTA_H = self.instance['drones']['hospital_drones']
        delta_u = self.instance['drones']['hub_drones']
        
        delayed_count = 0
        for k in K:
            for delta in DELTA_H | delta_u:
                t_var = self.model.getVarByName(f't-{k}-{delta}')
                if t_var and t_var.X > 1e-6:
                    delayed_count += 1
                    break
        total_commodities = len(K)
        return delayed_count / total_commodities if total_commodities > 0 else 0.0
    
    def get_logistics_ratio(self):
        """Calculate the logistics ratio (total distance / total commodities)."""
        if not hasattr(self, 'model'):# or self.model.status != GRB.OPTIMAL:
            raise ValueError("Model not solved or not optimal.")
        
        K = self.instance['constants']['K']
        A = self.instance['arcs']['A']
        distance_matrix = self.instance['matrices']['distance']
        DELTA_H = self.instance['drones']['hospital_drones']
        delta_u = self.instance['drones']['hub_drones']
        
        total_distance = 0.0
        for k in K:
            for delta in DELTA_H | delta_u:
                for (i, j) in A:
                    f_var = self.model.getVarByName(f'f-{k}-{delta}-{i}-{j}')
                    if f_var and f_var.X > 0.5:
                        total_distance += distance_matrix.loc[i, j]
        
        total_commodities = len(K)
        return total_distance / total_commodities if total_commodities > 0 else 0.0
    
    def active_facility_ratio(self):
        """
        Calcola il rapporto tra il numero di facility attive nella soluzione ottima
        e facenti parte degli shortest paths e il numero totale di facility presenti negli shortest paths.
        
        Returns:
            float: Il rapporto calcolato.
        """
        if not hasattr(self, 'model'):#  or self.model.status != GRB.OPTIMAL:
            raise ValueError("Il modello non è stato ottimizzato correttamente.")
        
       
        
        # Ottieni tutte le facility presenti negli shortest paths
        facilities_in_shortest_paths = set()
        V = self.instance['nodes']['all']
        distance_matrix = self.instance['matrices']['distance']
        a_F_i = self.instance['facility_indicators']
        F = self.instance['nodes']['facilities']
        U = self.instance['nodes']['hubs']
        tau_i = self.instance['constants']['tau_i']
        v = self.instance['constants']['v']
        
        active_facilities = [i for i in F if self.model.getVarByName(f'z-{i}').X > 0.5]

        for k, commodity in self.instance['commodities'].items():
            i_k, j_k = commodity[0], commodity[1]
            _, _, sp_facilities = InstanceGenerator.shortest_path(
                i_k, j_k, self.instance['arcs']['A'], V, distance_matrix, tau_i,
                a_F_i, v
            )
            facilities_in_shortest_paths.update(sp_facilities)
        
        # Calcola il numero di facility attive che fanno parte degli shortest paths
        active_facilities_in_shortest_paths = set(active_facilities).intersection(facilities_in_shortest_paths)
        
        # Calcola il rapporto
        total_facilities_in_shortest_paths = len(set(facilities_in_shortest_paths))
        if total_facilities_in_shortest_paths == 0:
            return 0.0  # Evita la divisione per zero
        #print(active_facilities)
        #print(active_facilities_in_shortest_paths)
        #print(facilities_in_shortest_paths)
        ratio_ale = len(active_facilities_in_shortest_paths) / len(active_facilities) #total_facilities_in_shortest_paths
        ratio_dem = len(active_facilities_in_shortest_paths) / total_facilities_in_shortest_paths
        return ratio_ale, ratio_dem
    

    def solution_summary(self):
        if not hasattr(self, 'model') or self.model is None:
            print("Model not solved yet.")
            return
        if self.model.status not in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
            print(f"Model status is {self.model.status}. No solution available.")
            return
        commodities = self.instance['commodities']
        K = self.instance['constants']['K']
        A = self.instance['arcs']['A']
        A_3 = self.instance['arcs']['A_3']
        DELTA_H = self.instance['drones']['hospital_drones']
        delta_u = self.instance['drones']['hub_drones']
        F = self.instance['nodes']['facilities']
        U = self.instance['nodes']['hubs']
        V = self.instance['nodes']['all']
        pi_k = self.instance['penalties']
        a_F_i = self.instance['facility_indicators']
        c_F_i = self.instance['costs']['facility_costs']
        c_U_i = self.instance['costs']['hub_costs']
        b_F = self.instance['constants']['b_F']
        b_U = self.instance['constants']['b_U']
        distance_matrix = self.instance['matrices']['distance']
        F_ij = self.instance['arcs']['F_ij']
        tau_i = self.instance['constants']['tau_i']
        v = self.instance['constants']['v']

        # Recupera le variabili di decisione dal modello
        z = {}
        for i in self.instance['nodes']['facilities'].union(self.instance['nodes']['hubs']):
            var_name = f'z-{i}'
            z[i] = self.model.getVarByName(var_name)

        f = {}
        for k in K:
            for delta in DELTA_H | delta_u:
                for (i, j) in A:
                    var_name = f'f-{k}-{delta}-{i}-{j}'
                    f[(k, delta, i, j)] = self.model.getVarByName(var_name)

        t = {}
        for k in K:
            for delta in DELTA_H | delta_u:
                var_name = f't-{k}-{delta}'
                t[(k, delta)] = self.model.getVarByName(var_name)

        active_edges = [(v.VarName.split('-')[3], v.VarName.split('-')[4])
                        for v in self.model.getVars() 
                        if v.VarName.startswith('f-') and v.X > 0.5]
        hub_edges = []
        other_edges = []
        for i, j in active_edges:
            i_norm = i.split('_origin_')[0].split('_dest_')[0]
            j_norm = j.split('_origin_')[0].split('_dest_')[0]
            if i_norm in self.instance['nodes']['hubs']:
                hub_edges.append((i_norm, j_norm))
            else:
                other_edges.append((i_norm, j_norm))

        print("\nSolution Statistics:")
        print(f"Active Facilities: {sum(1 for v in self.model.getVars() if v.VarName.startswith('z-') and v.X > 0.5)}")
        print(f"Total Routes: {len(active_edges)}")
        print(f"Hub Routes: {len(hub_edges)}")
        print(f"Direct Routes: {len(other_edges)}")

        def reconstruct_path(start, end, arcs, F_ij=None, active_hub=None):
            """
            Ricostruisce il percorso considerando sia percorsi diretti che attraverso hub.
            
            Args:
                start: nodo di origine
                end: nodo di destinazione 
                arcs: archi attivi
                F_ij: dizionario delle facility intermedie per percorsi hub-origine
                active_hub: hub attivo per questo percorso
            """
            if active_hub:
                # Caso percorso attraverso hub
                # Prima parte: hub -> facilities -> i_k
                facilities_to_origin = []
                if (active_hub,start) in A_3:
                    facilities_to_origin = list(F_ij[(active_hub, start)])
                first_part = [active_hub] + facilities_to_origin + [start]
                
                # Seconda parte: i_k -> facilities -> j_k
                second_part = []
                current = start
                arcs_remaining = arcs.copy()
                
                while current != end and arcs_remaining:
                    found = False
                    for (i, j) in arcs_remaining:
                        if i == current:
                            second_part.append(j)
                            current = j
                            arcs_remaining.remove((i, j))
                            found = True
                            break
                    if not found:
                        break
                        
                return first_part + second_part if second_part else first_part
            else:
                # Caso percorso diretto
                path = [start]
                current = start
                arcs_remaining = arcs.copy()
                
                while current != end and arcs_remaining:
                    found = False
                    for (i, j) in arcs_remaining:
                        if i == current:
                            path.append(j)
                            current = j
                            arcs_remaining.remove((i, j))
                            found = True
                            break
                    if not found:
                        break
                        
                return path if path[-1] == end else None
        # Ciclo per ogni commodity per stampare il riepilogo dettagliato
        for k in K:
            i_k = commodities[k][0]
            j_k = commodities[k][1]
            s_k = commodities[k][4]
            n_i = commodities[k][8]
            delay = sum(t[k, delta].X for delta in DELTA_H | delta_u)
            #print(f"\nCommodity {k} routing:")
            #print(f"  Origine: {i_k}")
            #print(f"  Destinazione: {j_k}")
            #print(f"  Drone richiesti: {s_k}")
            #print(f"  Ritardo totale: {delay:.2f}")

             # Calcola il tempo di percorrenza effettivo
            total_travel_time = 0
            for delta in DELTA_H | delta_u:
                route_time = 0
                for (i, j) in A:
                    if f[k, delta, i, j].X > 0.5:
                        # Calcola il tempo di percorrenza per questo arco
                        travel_time = (tau_i * a_F_i[i] + distance_matrix.loc[i,j] / v)
                        route_time += travel_time
                if route_time > 0:
                    #print(f"  Tempo di percorrenza (delta {delta}): {route_time:.2f} ore")
                    total_travel_time += route_time

            #print(f"  Tempo di percorrenza totale: {total_travel_time:.2f} ore")
            
            # Estrae il percorso effettivo per la commodity
            effective_arcs = None
            active_hub = None
            for delta in DELTA_H | delta_u:
                candidate_arcs = [(i, j) for i in V for j in V - {i} 
                                if ((i, j) in A or (i,j) in A_3) and f[k, delta, i, j].X > 0.5]
                
                # Cerca se c'è un hub attivo nel percorso
                for (i, j) in candidate_arcs:
                    i_base = i.split('_origin_')[0].split('_dest_')[0]
                    if i_base in U and z[i_base].X > 0.5:
                        active_hub = i_base
                        break
                
                if candidate_arcs:
                    effective_arcs = candidate_arcs
                    break

            if effective_arcs:
                effective_route = reconstruct_path(i_k, j_k, effective_arcs, 
                                                F_ij if active_hub else None, 
                                                active_hub)
            else:
                effective_route = []

            print(f"Commodity {k}: {effective_route if effective_route else 'Non disponibile'}, "
                f"{total_travel_time:.2f}, {delay:.2f}")
            # Calcola lo shortest path (in termini di tempo) usando la funzione self.shortest_path
            sp, sp_time, sp_facilities = InstanceGenerator.shortest_path(
                i_k, j_k, self.instance['arcs']['A'], V, distance_matrix, tau_i,
                a_F_i, v)
            # Ricostruisce lo shortest path in termini di nodi
            sp_nodes = [i_k]
            for arc in sp:
                sp_nodes.append(arc[1])
            #print(f"  Shortest path: {sp_nodes}")
            #print(f"  Tempo di viaggio (shortest path): {sp_time:.2f}")

            # Calcola il numero di coincidenze tra lo shortest path e il percorso effettivo
            if effective_route:
                coincidences = len(set(sp_facilities).intersection(set(effective_route)))
            else:
                coincidences = 0
            #print(f"  Numero di facility attive facenti parte di shortest path: {coincidences}")
            #print(f"  Ratio:{self.active_facility_ratio():.2f}")
            
            # Stampa anche i flussi 
            #print("  Flussi trovati:")
            #for delta in DELTA_H | delta_u:
            #    for i in V:
            #        for j in V - {i}:
            #            if (i, j) in A and f[k, delta, i, j].X > 0.5:
            #                print(f"    Flusso: {i} -> {j} ({delta})")

    
    def visualize_solution(self):
        """
        Visualize the network solution with nodes from DataFrame.
        """
        fig, ax = plt.subplots(figsize=(12,6))
        
        # Create network graph from DataFrame
        G = nx.DiGraph()
        
        # Create mapping for node IDs
        node_mapping = {}
        
        for idx, row in self.nodes_df.iterrows():
            node_id = row['id']
            G.add_node(node_id, 
                    pos=(row['lon'], row['lat']),
                    type=row['type'])
            # Create mappings for different node name formats
            node_mapping[f"H_{node_id}"] = node_id
            node_mapping[f"F_{node_id}"] = node_id
            node_mapping[f"U_{node_id}"] = node_id
            node_mapping[node_id] = node_id

        # Get node lists by type
        hospitals = [n for n in G.nodes if G.nodes[n]['type'] == 'hospital']
        stations = [n for n in G.nodes if G.nodes[n]['type'] == 'facility']
        hubs = [n for n in G.nodes if G.nodes[n]['type'] == 'hub']

        # Get active edges
        active_edges = [(v.VarName.split('-')[3], v.VarName.split('-')[4])
                        for v in self.model.getVars() 
                        if v.VarName.startswith('f-') and v.X > 0.5]

        # Separate hub and direct edges
        hub_edges = []
        direct_edges = []
        
        for i, j in active_edges:
            i_norm = i.split('_origin_')[0].split('_dest_')[0]
            j_norm = j.split('_origin_')[0].split('_dest_')[0]
            if i_norm in G.nodes and j_norm in G.nodes:
                edge = (i_norm, j_norm)
                if i_norm in self.instance['nodes']['hubs']:
                    hub_edges.append(edge)
                else:
                    direct_edges.append(edge)

        G.add_edges_from(hub_edges + direct_edges)
        
        # Get node positions
        pos = nx.get_node_attributes(G, 'pos')
        
        # Draw nodes
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
        
        # Draw edges
        if hub_edges:
            nx.draw_networkx_edges(G, pos, edgelist=hub_edges,
                                edge_color='black', style='dashed',
                                width=0.8, arrowsize=10, arrowstyle='-|>')
        
        if direct_edges:
            nx.draw_networkx_edges(G, pos, edgelist=direct_edges,
                                edge_color='black', style='solid',
                                width=0.8, arrowsize=10, arrowstyle='-|>')
        
        # Add labels
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
                        for s in e if G.nodes[s]['type'] == 'facility')
        
        # Add solution statistics
        stats_text = (
            f"Solution Statistics:\n"
            f"Active paths: {len(active_edges)}\n"
            f"Used hubs: {len(used_hubs)}\n"
            f"Used facilities: {len(used_stations)}\n"
            f"Total facility cost: {sum(self.instance['costs']['facility_costs'][f] for f in used_stations):.2f}\n"
            f"Total hub cost: {sum(self.instance['costs']['hub_costs'][h] for h in used_hubs):.2f}"
        )
        
        fig.text(0.02, 0.02, stats_text, fontsize=10, family='monospace')
        plt.tight_layout()
        plt.savefig('solution_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()

    def solution_summary2(self):
        """Visualizzazione migliorata dei risultati della soluzione."""
        if not hasattr(self, 'model') or self.model is None:
            print("❌ Model not solved yet.")
            return
        if self.model.status not in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
            print(f"❌ Model status is {self.model.status}. No solution available.")
            return
        
        # Recupera dati necessari
        commodities = self.instance['commodities']
        K = self.instance['constants']['K']
        A = self.instance['arcs']['A']
        A_3 = self.instance['arcs']['A_3']
        DELTA_H = self.instance['drones']['hospital_drones']
        delta_u = self.instance['drones']['hub_drones']
        F = self.instance['nodes']['facilities']
        U = self.instance['nodes']['hubs']
        V = self.instance['nodes']['all']
        H = self.instance['nodes']['hospitals']
        F_ij = self.instance['arcs']['F_ij']
        
        # Recupera variabili dal modello
        z, f, t = self._extract_variables()
        
        # === HEADER PRINCIPALE ===
        print("=" * 80)
        print("🎯 DRONE DISTRIBUTION OPTIMIZATION - SOLUTION SUMMARY")
        print("=" * 80)
        
        # === STATISTICHE GENERALI ===
        self._print_general_statistics(z, f)
        
        # === INFRASTRUTTURA ATTIVATA ===
        self._print_active_infrastructure(z)
        

    def _extract_variables(self):
        """Estrae le variabili dal modello Gurobi."""
        z = {}
        for i in self.instance['nodes']['facilities'].union(self.instance['nodes']['hubs']):
            var = self.model.getVarByName(f'z-{i}')
            z[i] = var.X if var else 0
        
        f = {}
        for k in self.instance['constants']['K']:
            for delta in self.instance['drones']['hospital_drones'] | self.instance['drones']['hub_drones']:
                for (i, j) in self.instance['arcs']['A']:
                    var = self.model.getVarByName(f'f-{k}-{delta}-{i}-{j}')
                    f[(k, delta, i, j)] = var.X if var else 0
        
        t = {}
        for k in self.instance['constants']['K']:
            for delta in self.instance['drones']['hospital_drones'] | self.instance['drones']['hub_drones']:
                var = self.model.getVarByName(f't-{k}-{delta}')
                t[(k, delta)] = var.X if var else 0
        
        return z, f, t

    def _print_general_statistics(self, z, f):
        """Stampa le statistiche generali della soluzione."""
        active_facilities = sum(1 for i, val in z.items() if val > 0.5 and i in self.instance['nodes']['facilities'])
        active_hubs = sum(1 for i, val in z.items() if val > 0.5 and i in self.instance['nodes']['hubs'])
        active_routes = sum(1 for val in f.values() if val > 0.5)
        
        print(f"SOLUTION OVERVIEW")
        print(f"   Status: {'✅ Optimal' if self.model.status == GRB.OPTIMAL else '⏱️ Time Limit'}")
        print(f"   Objective Value: {self.model.objVal:.2f}")
        print(f"   Active Facilities: {active_facilities}/{len(self.instance['nodes']['facilities'])}")
        print(f"   Active Hubs: {active_hubs}/{len(self.instance['nodes']['hubs'])}")
        print(f"   Total Active Routes: {active_routes}")

    def _print_active_infrastructure(self, z):
        """Stampa l'infrastruttura attivata."""
        print(f"\nACTIVE INFRASTRUCTURE")
        
        # Facilities attive
        active_facilities = [i for i, val in z.items() 
                            if val > 0.5 and i in self.instance['nodes']['facilities']]
        if active_facilities:
            print(f"   Facilities: {', '.join(sorted(active_facilities))}")
        else:
            print(f"   Facilities: None")
        
        # Hubs attivi
        active_hubs = [i for i, val in z.items() 
                    if val > 0.5 and i in self.instance['nodes']['hubs']]
        if active_hubs:
            print(f"   Hubs: {', '.join(sorted(active_hubs))}")
        else:
            print(f"   Hubs: None")
        
        # Budget utilizzato
        c_F_i = self.instance['costs']['facility_costs']
        c_U_i = self.instance['costs']['hub_costs']
        b_F = self.instance['constants']['b_F']
        b_U = self.instance['constants']['b_U']
        
        facility_cost = sum(c_F_i[i] * z[i] for i in active_facilities if i in c_F_i)
        hub_cost = sum(c_U_i[i] * z[i] for i in active_hubs if i in c_U_i)
        
        print(f"   Budget Used - Facilities: {facility_cost:.0f}/{b_F:.0f} ({facility_cost/b_F*100:.1f}%)")
        print(f"   Budget Used - Hubs: {hub_cost:.0f}/{b_U:.0f} ({hub_cost/b_U*100:.1f}%)")
        print("=" * 80)

    

    #nodes_df = pd.read_csv("crawford/data/ZONE.csv")
    #density_df = pd.read_csv("crawford/data/population_density.csv")
    #
    #generator = InstanceGenerator(nodes_df, density_df)
    #generator.H_max =10000
    #generator.d_max = 30
    ##generator.num_commodity = 3
    #instance = generator.load_commodities('crawford/data/DMNDP-K10-0.csv')
    ##instance = generator.generate(scenario='non-critical', commodity_df)    # or 'critical'
    #
    ## Optionally, print a summary
    #generator.summary()
    #
    ## Build and solve the model
    #model = MathematicalModel(instance, nodes_df)
    #result = model.solve(model_type='relaxed')  # or 'binary' or 'relaxed'
    #if result is not None:
    #    model.solution_summary()
    #    model.visualize_solution()
    #else:
    #    print("No solution found.")                        # or 'relaxed'



