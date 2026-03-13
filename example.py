"""
run_single_instance.py
======================
Esegue i tre algoritmi (Modello Matematico, Matheuristic, Euristica VNS)
su una singola istanza del DMNDP e produce un report visuale comparativo.

Uso:
    python run_single_instance.py                          # default: Roma, 10 commodity, istanza 1, non-critico
    python run_single_instance.py --city milano5 --K 20    # Milano, 20 commodity
    python run_single_instance.py --city napoli5 --seed 3 --scenario c   # Napoli, critico
"""

import argparse
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from matplotlib.lines import Line2D
import os
import sys

from model.instancegenerator import InstanceGenerator
from model.mathmodel import MathematicalModel
from model.matheuristic import MathEuristic
from model.heuristic import Heuristic


# ═══════════════════════════════════════════════════════════
#  CONFIGURAZIONE
# ═══════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description="Esegui DMNDP su singola istanza con visualizzazione")
    parser.add_argument("--city", type=str, default="roma5",
                        choices=["roma5", "milano5", "napoli5"],
                        help="Città (default: roma5)")
    parser.add_argument("--K", type=int, default=10,
                        help="Numero di commodity (default: 10)")
    parser.add_argument("--seed", type=int, default=1,
                        help="Seed dell'istanza (default: 1)")
    parser.add_argument("--scenario", type=str, default="nc",
                        choices=["c", "nc"],
                        help="Scenario: c=critico, nc=non-critico (default: nc)")
    parser.add_argument("--density_file", type=str, default="data/population_density.csv",
                        help="Path al file densità abitativa")
    parser.add_argument("--d_max", type=float, default=15,
                        help="Distanza massima drone in km (default: 15 per urbano)")
    parser.add_argument("--v", type=float, default=30,
                        help="Velocità media drone in km/h (default: 30)")
    parser.add_argument("--K_paths", type=int, default=50,
                        help="Percorsi candidati per la matheuristic (default: 50)")
    parser.add_argument("--s_max", type=int, default=20,
                        help="Shaking massimo per VNS (default: 20)")
    parser.add_argument("--skip_mm", action="store_true",
                        help="Salta il modello matematico esatto (lento su istanze grandi)")
    parser.add_argument("--output_dir", type=str, default="output",
                        help="Cartella di output per i grafici (default: output)")
    return parser.parse_args()


# ═══════════════════════════════════════════════════════════
#  CARICAMENTO ISTANZA
# ═══════════════════════════════════════════════════════════

def load_instance(args):
    """Carica l'istanza specificata e restituisce (instance, nodes_df, density_df, instance_name)."""
    city = args.city
    city_prefix = city.replace("5", "").upper()  # roma5 -> ROMA, milano5 -> MILANO

    nodes_path = f"data/{city}/{city}.csv"
    commodity_path = f"data/{city}/{city_prefix}-{args.K}-{args.seed}-{args.scenario}.csv"

    if not os.path.exists(nodes_path):
        print(f"[ERRORE] File nodi non trovato: {nodes_path}")
        sys.exit(1)
    if not os.path.exists(commodity_path):
        print(f"[ERRORE] File commodity non trovato: {commodity_path}")
        sys.exit(1)

    nodes_df = pd.read_csv(nodes_path)
    density_df = pd.read_csv(args.density_file)

    generator = InstanceGenerator(nodes_df, density_df)
    generator.d_max = args.d_max
    generator.v = args.v
    instance = generator.load_commodities(commodity_path)

    instance_name = f"{city_prefix}-{args.K}-{args.seed}-{args.scenario}"
    return instance, nodes_df, density_df, instance_name


# ═══════════════════════════════════════════════════════════
#  ESECUZIONE ALGORITMI
# ═══════════════════════════════════════════════════════════

def run_mathematical_model(instance, nodes_df):
    """Esegue il modello matematico esatto."""
    print("\n" + "="*60)
    print("  MODELLO MATEMATICO ESATTO (MIP)")
    print("="*60)
    t0 = time.time()
    mm = MathematicalModel(instance, nodes_df)
    result = mm.solve()
    elapsed = time.time() - t0

    if result is not None and mm.model.status in [2, 9]:  # OPTIMAL o TIME_LIMIT
        obj = mm.model.ObjVal
        gap = mm.model.MIPGap
        lb = mm.model.ObjBound
        n_fac = sum(1 for i in instance['nodes']['facilities']
                    if mm.model.getVarByName(f'z-{i}') and mm.model.getVarByName(f'z-{i}').X > 0.5)
        n_hub = sum(1 for i in instance['nodes']['hubs']
                    if mm.model.getVarByName(f'z-{i}') and mm.model.getVarByName(f'z-{i}').X > 0.5)
        lcr = mm.get_delayed_commodities_ratio()
        logr = mm.get_logistics_ratio()

        print(f"  OBJ = {obj:.2f}   GAP = {gap*100:.2f}%   LB = {lb:.2f}")
        print(f"  Facility attive: {n_fac}   Hub attivi: {n_hub}")
        print(f"  Tempo: {elapsed:.1f}s")

        return {
            'name': 'Modello Matematico',
            'obj': obj, 'gap': gap, 'lb': lb,
            'n_facilities': n_fac, 'n_hubs': n_hub,
            'lcr': lcr, 'logr': logr, 'time': elapsed,
            'model_obj': mm
        }
    else:
        print("  Nessuna soluzione trovata.")
        return None


def run_matheuristic(instance, nodes_df, density_df, K_paths):
    """Esegue la matheuristic."""
    print("\n" + "="*60)
    print("  MATHEURISTIC (K-shortest paths MIP)")
    print("="*60)
    t0 = time.time()
    mh = MathEuristic(instance, K_paths=K_paths, nodes_df=nodes_df, density_df=density_df)
    result = mh.solve()
    elapsed = time.time() - t0

    if result is not None and mh.model.status in [2, 9]:
        obj = mh.model.ObjVal
        gap = mh.model.MIPGap

        selected_paths = [p for p in mh.P if mh.x[p].X > 0.5]
        facility_sets = [mh.F_p[p] for p in selected_paths]
        used_facilities = set.union(*facility_sets) if facility_sets else set()
        used_hubs = set()
        for p in selected_paths:
            if 'P_j_ik_jk_' in p:
                j = p.split('_')[5] + "_" + p.split('_')[6]
                used_hubs.add(j)

        print(f"  OBJ = {obj:.2f}   GAP = {gap*100:.2f}%")
        print(f"  Facility attive: {len(used_facilities)}   Hub attivi: {len(used_hubs)}")
        #print(f"  Percorsi selezionati: {len(selected_paths)}")
        print(f"  Tempo: {elapsed:.1f}s")

        return {
            'name': 'Matheuristic',
            'obj': obj, 'gap': gap, 'lb': mh.model.ObjBound,
            'n_facilities': len(used_facilities), 'n_hubs': len(used_hubs),
            'time': elapsed,
            'model_obj': mh
        }
    else:
        print("  Nessuna soluzione trovata.")
        return None


def run_heuristic(instance, s_max):
    """Esegue l'euristica VNS + Local Search."""
    print("\n" + "="*60)
    print("  EURISTICA")
    print("="*60)
    t0 = time.time()
    h = Heuristic(instance, s_max=s_max)
    solution = h.run_heuristic()
    elapsed = time.time() - t0

    if solution and solution.get('budget_satisfied', True):
        obj = solution.get('OBJ', None)
        n_fac = solution.get('total_facilities', len(solution.get('facilities', [])))
        n_hub = len(solution.get('activated_hubs', []))
        iterations = solution.get('iterations', 0)

        print(f"  OBJ = {obj:.2f}" if obj else "  OBJ = N/A")
        print(f"  Facility attive: {n_fac}   Hub attivi: {n_hub}")
        #print(f"  Iterazioni VNS: {iterations}")
        print(f"  Tempo: {elapsed:.1f}s")

        return {
            'name': 'Euristica VNS',
            'obj': obj, 'gap': None, 'lb': None,
            'n_facilities': n_fac, 'n_hubs': n_hub,
            'time': elapsed, 'iterations': iterations,
            'solution': solution
        }
    else:
        reason = solution.get('reason', 'Budget non soddisfatto') if solution else 'Soluzione vuota'
        print(f"  Soluzione non trovata: {reason}")
        return None


# ═══════════════════════════════════════════════════════════
#  VISUALIZZAZIONE
# ═══════════════════════════════════════════════════════════

def plot_network(nodes_df, instance, instance_name, output_dir):
    """Visualizza la rete geografica con ospedali, facility e hub."""
    fig, ax = plt.subplots(figsize=(12, 8))

    hospitals = nodes_df[nodes_df['type'] == 'hospital']
    facilities = nodes_df[nodes_df['type'] == 'facility']
    hubs = nodes_df[nodes_df['type'] == 'hub']

    ax.scatter(facilities['lon'], facilities['lat'], c='#f0f0f0', s=60,
              edgecolors='gray', linewidths=0.5, marker='^', zorder=2, label='Facility')
    ax.scatter(hubs['lon'], hubs['lat'], c='#4a90d9', s=120,
              edgecolors='black', linewidths=0.8, marker='s', zorder=3, label='Hub')
    ax.scatter(hospitals['lon'], hospitals['lat'], c='#2ecc71', s=150,
              edgecolors='black', linewidths=0.8, marker='o', zorder=4, label='Ospedale')

    for _, row in hospitals.iterrows():
        ax.annotate(row['id'], (row['lon'], row['lat']),
                   fontsize=7, ha='center', va='bottom', fontweight='bold',
                   xytext=(0, 6), textcoords='offset points')

    # Disegna archi disponibili (sottili, grigi)
    for (i, j) in list(instance['arcs']['A'])[:200]:  # limita per leggibilità
        i_base = i.split('_origin_')[0].split('_dest_')[0]
        j_base = j.split('_origin_')[0].split('_dest_')[0]
        i_row = nodes_df[nodes_df['id'] == i_base]
        j_row = nodes_df[nodes_df['id'] == j_base]
        if not i_row.empty and not j_row.empty:
            ax.plot([i_row['lon'].values[0], j_row['lon'].values[0]],
                    [i_row['lat'].values[0], j_row['lat'].values[0]],
                    color='#e0e0e0', linewidth=0.3, zorder=1)

    ax.set_xlabel('Longitudine', fontsize=10)
    ax.set_ylabel('Latitudine', fontsize=10)
    ax.set_title(f'Rete DMNDP — {instance_name}', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)

    info_text = (f"Ospedali: {len(hospitals)}  |  Facility: {len(facilities)}  |  Hub: {len(hubs)}\n"
                 f"Commodity: {len(instance['constants']['K'])}  |  "
                 f"Archi: {len(instance['arcs']['A'])}")
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=8,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.4', facecolor='wheat', alpha=0.7))

    plt.tight_layout()
    filepath = os.path.join(output_dir, f'rete_{instance_name}.png')
    plt.savefig(filepath, dpi=200, bbox_inches='tight')
    print(f"  Grafico rete salvato in: {filepath}")
    plt.close()


def plot_comparison(results, instance_name, output_dir):
    """Crea grafici comparativi tra i tre algoritmi."""
    valid = [r for r in results if r is not None]
    if not valid:
        print("  Nessun risultato da visualizzare.")
        return

    names = [r['name'] for r in valid]
    colors = {'Modello Matematico': '#e74c3c', 'Matheuristic': '#3498db', 'Euristica VNS': '#2ecc71'}
    bar_colors = [colors.get(n, '#95a5a6') for n in names]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f'Confronto Algoritmi — {instance_name}', fontsize=14, fontweight='bold')

    # ── 1. Funzione Obiettivo ──
    ax = axes[0]
    objs = [r['obj'] if r['obj'] is not None else 0 for r in valid]
    bars = ax.bar(names, objs, color=bar_colors, edgecolor='black', linewidth=0.5)
    for bar, val in zip(bars, objs):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(objs)*0.01,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_ylabel('Funzione Obiettivo')
    ax.set_title('Valore OBJ')
    ax.tick_params(axis='x', rotation=15)
    ax.grid(axis='y', alpha=0.3)

    # ── 2. Infrastruttura Attivata ──
    ax = axes[1]
    x = np.arange(len(names))
    width = 0.35
    facs = [r['n_facilities'] for r in valid]
    hubs_vals = [r['n_hubs'] for r in valid]
    ax.bar(x - width/2, facs, width, label='Facility', color='#f39c12', edgecolor='black', linewidth=0.5)
    ax.bar(x + width/2, hubs_vals, width, label='Hub', color='#4a90d9', edgecolor='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15)
    ax.set_ylabel('Numero')
    ax.set_title('Infrastruttura Attivata')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # ── 3. Tempo Computazionale ──
    ax = axes[2]
    times = [r['time'] for r in valid]
    bars = ax.bar(names, times, color=bar_colors, edgecolor='black', linewidth=0.5)
    for bar, val in zip(bars, times):
        label = f'{val:.1f}s' if val < 60 else f'{val/60:.1f}m'
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(times)*0.01,
                label, ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_ylabel('Tempo (secondi)')
    ax.set_title('Tempo Computazionale')
    ax.tick_params(axis='x', rotation=15)
    ax.grid(axis='y', alpha=0.3)
    if max(times) / (min(t for t in times if t > 0) + 1e-9) > 10:
        ax.set_yscale('log')

    plt.tight_layout()
    filepath = os.path.join(output_dir, f'confronto_{instance_name}.png')
    plt.savefig(filepath, dpi=200, bbox_inches='tight')
    print(f"  Grafico confronto salvato in: {filepath}")
    plt.close()


def plot_commodity_table(instance, instance_name, output_dir):
    """Visualizza la tabella delle commodity come immagine."""
    commodities = instance['commodities']
    rows = []
    for k, c in commodities.items():
        origin_base = c[0].split('_origin_')[0]
        dest_base = c[1].split('_dest_')[0]
        rows.append({
            'ID': k,
            'Origine': origin_base,
            'Destinazione': dest_base,
            'Tipo': c[6],
            'Pronto (h)': f"{c[2]:.2f}",
            'Scadenza (h)': f"{c[3]:.1f}",
            'Droni': c[4],
            'Penalità': c[5],
            'Quantità': c[7]
        })

    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(14, max(3, 0.4 * len(df) + 1.5)))
    ax.axis('off')
    ax.set_title(f'Commodity — {instance_name}', fontsize=13, fontweight='bold', pad=15)

    # Colori per tipo
    type_colors = {
        'organ': '#e74c3c', 'blood': '#c0392b',
        'fluids': '#3498db', 'lab_sample': '#f39c12', 'medication': '#2ecc71'
    }
    cell_colors = []
    for _, row in df.iterrows():
        row_colors = ['#f9f9f9'] * len(df.columns)
        tipo = row['Tipo']
        if tipo in type_colors:
            row_colors[3] = type_colors[tipo] + '30'  # colore con trasparenza (approx)
        cell_colors.append(row_colors)

    table = ax.table(cellText=df.values, colLabels=df.columns, loc='center',
                     cellLoc='center', cellColours=cell_colors)
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)

    # Stile header
    for j in range(len(df.columns)):
        table[0, j].set_facecolor('#2c3e50')
        table[0, j].set_text_props(color='white', fontweight='bold')

    plt.tight_layout()
    filepath = os.path.join(output_dir, f'commodity_{instance_name}.png')
    plt.savefig(filepath, dpi=200, bbox_inches='tight')
    print(f"  Tabella commodity salvata in: {filepath}")
    plt.close()


def print_summary_table(results):
    """Stampa una tabella riassuntiva su terminale."""
    print("\n" + "="*80)
    print("  RIEPILOGO COMPARATIVO")
    print("="*80)
    header = f"{'Algoritmo':<25} {'OBJ':>12} {'GAP':>8} {'Facility':>10} {'Hub':>6} {'Tempo':>10}"
    print(header)
    print("-"*80)

    for r in results:
        if r is None:
            continue
        obj_str = f"{r['obj']:.2f}" if r['obj'] is not None else "N/A"
        gap_str = f"{r['gap']*100:.2f}%" if r.get('gap') is not None else "-"
        time_str = f"{r['time']:.1f}s" if r['time'] < 60 else f"{r['time']/60:.1f}min"
        print(f"{r['name']:<25} {obj_str:>12} {gap_str:>8} {r['n_facilities']:>10} {r['n_hubs']:>6} {time_str:>10}")

    print("="*80)


# ═══════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════

def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("╔══════════════════════════════════════════════════════════╗")
    print("║    DMNDP — Drone Medical Network Design Problem         ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"  Città: {args.city}   K={args.K}   seed={args.seed}   scenario={args.scenario}")
    print(f"  d_max={args.d_max} km   v={args.v} km/h")
    print()

    # ── Caricamento ──
    instance, nodes_df, density_df, instance_name = load_instance(args)

    # ── Visualizzazione istanza ──
    print("\n[1/5] Visualizzazione rete...")
    plot_network(nodes_df, instance, instance_name, args.output_dir)

    print("\n[2/5] Tabella commodity...")
    plot_commodity_table(instance, instance_name, args.output_dir)

    # ── Esecuzione algoritmi ──
    results = []

    if not args.skip_mm:
        print("\n[3/5] Modello Matematico Esatto...")
        results.append(run_mathematical_model(instance, nodes_df))
    else:
        print("\n[3/5] Modello Matematico — SALTATO (--skip_mm)")
        results.append(None)

    print("\n[4/5] Matheuristic...")
    results.append(run_matheuristic(instance, nodes_df, density_df, args.K_paths))

    print("\n[5/5] Euristica VNS...")
    results.append(run_heuristic(instance, args.s_max))

    # ── Confronto ──
    print_summary_table(results)

    print("\nGenerazione grafici comparativi...")
    plot_comparison(results, instance_name, args.output_dir)

    # ── Visualizzazione soluzione MM (se disponibile) ──
    mm_result = results[0] if not args.skip_mm else None
    if mm_result and mm_result.get('model_obj'):
        print("\nVisualizzazione soluzione Modello Matematico...")
        try:
            mm_result['model_obj'].visualize_solution()
        except Exception as e:
            print(f"  (visualizzazione non disponibile: {e})")

    print(f"\nTutti i grafici salvati in: {args.output_dir}/")
    print("Done!")


if __name__ == "__main__":
    main()