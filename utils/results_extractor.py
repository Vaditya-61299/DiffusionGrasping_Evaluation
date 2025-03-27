import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
def process_data(data, num_grasps, dataset_name):
    success_cases = []
    robust_cases_abs = []
    sing_values = []
    smallest_singular_values = []
    wrench_space_volumes = []
    force_closure_abs = []
    emd_samples = []

    for obj_dict in data:
        for values in obj_dict.values():
            success_count = values["Success cases"]
            robust_count = values["Robust cases"]

            emd_samples.append(values.get("EMD-Sample-Dataset", np.nan))
            success_cases.append(success_count)
            robust_cases_abs.append(robust_count)

            stable_grasps = 0
            for grasp in values["Grasps"]:
                singular_values = grasp['Singular Values of Grasp Matrix']

                if singular_values == [-1.0] * 6:
                    smallest_singular_values.append(0)
                    wrench_space_volumes.append(0)
                    continue

                sing_values.append(singular_values)
                sv_min = min(singular_values)
                sv_volume = np.prod(singular_values)

                smallest_singular_values.append(sv_min)
                wrench_space_volumes.append(sv_volume)

                if not grasp["Force Closure"]["Convex_hull"] and grasp["Force Closure"]["is_full_rank"] and grasp["Force Closure"]["random wrench"]:
                    stable_grasps += 1

            force_closure_abs.append(stable_grasps)

    success_rate = sum(success_cases) / num_grasps
    robust_abs_total = sum(robust_cases_abs) / num_grasps
    force_closure_abs_total = sum(force_closure_abs) / num_grasps

    robust_rel_total = sum(robust_cases_abs) / sum(success_cases) if sum(success_cases) > 0 else np.nan
    force_closure_rel_total = sum(force_closure_abs) / sum(success_cases) if sum(success_cases) > 0 else np.nan

    emd_means = [val[0] if isinstance(val, list) and len(val) > 0 else np.nan for val in emd_samples]
    emd_stds = [val[1] if isinstance(val, list) and len(val) > 1 else np.nan for val in emd_samples]

    df_model = pd.DataFrame({
        "Dataset": [dataset_name] * len(success_cases),
        "Success Rate": success_cases,
        "EMD Mean": np.clip(emd_means, 0, 1),
        "EMD Std": emd_stds,
        "Robust Cases (Absolute)": [robust_abs_total] * len(success_cases),
        "Force Closure (Absolute)": [force_closure_abs_total] * len(success_cases),
        "Robust Cases (Relative)": [robust_rel_total] * len(success_cases),
        "Force Closure (Relative)": [force_closure_rel_total] * len(success_cases)
    })
    
    df_sing = pd.DataFrame({
        "Dataset": [dataset_name] * len(smallest_singular_values),
        "Smallest Singular Value": smallest_singular_values,
        "Wrench Space Volume": wrench_space_volumes,
    })
    
    return df_model, df_sing


_dir=os.path.join(os.path.dirname(os.path.dirname(__file__)),"LOGS")
# Load JSON Data
with open(os.path.join(_dir,'Results_fixed.json'), 'r') as file1:
    data_model = json.load(file1)


num_grasps = 3750  # Total grasps for random objects (15 classes, 5 objects each, 50 grasps each object)

#num_grasps= 5050 # Total grasps for Mugs

df_model_model, df_sing_model = process_data(data_model, num_grasps, "Model_name")



# Combine DataFrames
df_model_combined = pd.concat([df_model_model], ignore_index=True)
df_sing_combined = pd.concat([df_sing_model], ignore_index=True)

# Violin Plot for EMD Mean
plt.figure(figsize=(6, 5))
sns.violinplot(x="Dataset", y="EMD Mean", data=df_model_combined, palette="pastel")
plt.title("EMD Mean Distribution")
plt.show()

# Scatter Plot for Smallest Singular Value vs Wrench Space Volume
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x="Smallest Singular Value", 
    y="Wrench Space Volume", 
    hue="Dataset", 
    data=df_sing_combined, 
    alpha=0.6,
    palette="Set1"
)
plt.xscale("log")
plt.yscale("log")
plt.title("Smallest Singular Value vs Wrench Space Volume ")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.show()

