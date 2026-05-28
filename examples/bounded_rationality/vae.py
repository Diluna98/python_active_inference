import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib
import os

def build_meta_hierarchical_model(csv_files, n_contexts=10):
    print("Loading data and extracting interoceptive signatures...")
    
    data_list = []
    for f in csv_files:
        if os.path.exists(f):
            df = pd.read_csv(f)
            data_list.append(df)
    
    if not data_list:
        print("No CSV files found. Please check your file paths.")
        return
        
    full_data = pd.concat(data_list, ignore_index=True)

    # --- 1. DEFINE THE CONTEXT OBSERVATION ---
    # We use internal metrics that signal the 'texture' of the environment.
    # Info_gain (EIG): How much can be learned here?
    # Expected_Qs_entropy (H): How confused is the agent?
    context_features = ['info_gain', 'expected_Qs_entropy']
    
    # Fill any NaNs that might occur in early steps of simulation
    full_data[context_features] = full_data[context_features].fillna(0)

    # --- 2. CLUSTER INTO FUNCTIONAL CONTEXTS ---
    # This identifies the 10 "types of situations" the agent encounters.
    scaler = StandardScaler()
    scaled_signatures = scaler.fit_transform(full_data[context_features])
    
    print(f"Clustering into {n_contexts} functional contexts...")
    kmeans = KMeans(n_clusters=n_contexts, random_state=42, n_init=10)
    full_data['context_id'] = kmeans.fit_predict(scaled_signatures)

    # --- 3. BUILD THE META-A-MATRIX (Likelihood) ---
    # We map [Context ID + Resolution] -> [Performance Outcomes]
    # Note: We exclude position (x, y) to keep the model model-independent.
    performance_metrics = ['expected_accuracy', 'inference_time_ms']
    
    a_matrix = full_data.groupby(['context_id', 'resolution'])[performance_metrics].agg(['mean', 'std']).fillna(0)

    # --- 4. SAVE THE MODEL ---
    model_payload = {
        'a_matrix': a_matrix,
        'feature_scaler': scaler,
        'context_mapper': kmeans,
        'feature_names': context_features
    }
    
    output_file = "meta_hierarchical_a_matrix.joblib"
    joblib.dump(model_payload, output_file)
    
    print(f"Success! Meta-model saved to {output_file}")
    return a_matrix, kmeans.cluster_centers_

def display_numerical_results(a_matrix, centers):
    print("\n" + "="*85)
    print(f"{'Context ID':^10} | {'EIG (Mean)':^12} | {'H (Mean)':^12} | {'Res':^5} | {'Acc (Mean)':^10} | {'Time (ms)':^10}")
    print("="*85)
    
    for c_id in range(len(centers)):
        # Extract the signature center for this context
        eig_c, h_c = centers[c_id]
        
        ctx_data = a_matrix.loc[c_id]
        for res in [2, 5, 10, 20]:
            if res in ctx_data.index:
                acc = ctx_data.loc[res, ('expected_accuracy', 'mean')]
                time = ctx_data.loc[res, ('inference_time_ms', 'mean')]
                print(f"{c_id:^10} | {eig_c:^12.4f} | {h_c:^12.4f} | {res:^5} | {acc:^10.4f} | {time:^10.2f}")
        print("-" * 85)

if __name__ == "__main__":
    # Ensure these match your 50-run output filenames
    files = ['results_res_2.csv', 'results_res_5.csv', 'results_res_10.csv', 'results_res_20.csv']
    
    a_matrix, centroids = build_meta_hierarchical_model(files)
    
    if a_matrix is not None:
        display_numerical_results(a_matrix, centroids)