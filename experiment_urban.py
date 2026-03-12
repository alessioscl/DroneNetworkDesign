from model.instancegenerator import InstanceGenerator
from model.mathmodel import MathematicalModel
from model.matheuristic import MathEuristic
from model.heuristic import Heuristic
import pandas as pd
import glob
import os


input_dir = 'crawford/data/roma5/'

nodes_df = pd.read_csv('crawford/data/roma5/roma5.csv')
density_df = pd.read_csv('crawford/data/population_density.csv')

input_files = sorted(glob.glob(os.path.join(input_dir, '*.csv')))


file_optimized = 0
for filename in input_files:
    print(f"\nProcessing {filename} ({file_optimized/len(input_files)*100:.2f}%)")
    print("\n")
    #print(f"Processing file: {filename}")
    generator = InstanceGenerator(nodes_df, density_df)
    generator.d_max = 15
    generator.v = 30
    instance = generator.load_commodities(filename)
    #
    #model_type = 'binary'  # 'binary' o 'relaxed'
    math_model = MathematicalModel(instance, nodes_df)
    model = math_model.solve()
    #math_model.solution_summary()

    outputname = filename.replace('data','solution').replace('.csv','-sol.txt')
    #print(f"Saving solution to: {outputname}")
    mth = MathEuristic(instance, K_paths=50, nodes_df=nodes_df, density_df=density_df)
    model = mth.solve(file_name=outputname) #file_name=outputname

    ## heuristic
    euristic = Heuristic(instance, s_max=20)
    solution = euristic.run_heuristic()

    file_optimized += 1