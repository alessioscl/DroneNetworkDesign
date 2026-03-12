from model.instancegenerator import InstanceGenerator
from model.mathmodel import MathematicalModel
from model.matheuristic import MathEuristic
from model.heuristic import Heuristic
import pandas as pd
import glob
import os


input_dir = 'crawford/data/old/'

instances = ['K10', 'K15', 'K20', 'K30', 'K40', 'K50', 'K60', 'K70']

nodes_df = pd.read_csv('crawford/data/old/ZONE.csv')
density_df = pd.read_csv('crawford/data/population_density.csv')

input_files = sorted(glob.glob(os.path.join(input_dir, '*.csv')))
input_files_filtered = []
for x in input_files:
    try:
        if x.split('-')[1] in instances: # and int(x.split('-')[2][0]) == 4:#or x.split('-')[1] == 'K20':
            input_files_filtered.append(x)
    except:
        continue

file_optimized = 0
for filename in input_files_filtered:
    print(f"\nProcessing {filename} ({file_optimized/len(input_files_filtered)*100:.2f}%)")
    print("\n")
    #print(f"Processing file: {filename}")
    generator = InstanceGenerator(nodes_df, density_df)
    instance = generator.load_commodities(filename)
    #
    #model_type = 'binary'  # 'binary' o 'relaxed'
    math_model = MathematicalModel(instance, nodes_df)
    model = math_model.solve()
    #math_model.solution_summary()

    #outputname = filename.replace('data','solution').replace('.csv','-sol.txt')
    #print(f"Saving solution to: {outputname}")
    mth = MathEuristic(instance, K_paths=50, nodes_df=nodes_df, density_df=density_df)
    model = mth.solve() #file_name=outputname

    ## heuristic
    euristic = Heuristic(instance, s_max=20)
    solution = euristic.run_heuristic()

    file_optimized += 1