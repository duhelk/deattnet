
import math
import pandas as pd
import os

def get_data_df(dir):
    df_post = pd.read_csv(os.path.join(dir,"postprocessed.csv"))
    df_raw = pd.read_csv(os.path.join(dir,"raw.csv"))
    df_merged = pd.merge(df_post, df_raw, on="key")
    return df_merged

def get_en(df, key):
        e = df.loc[df['key']==key]["easting"].values[0]
        n = df.loc[df['key']==key]["northing"].values[0]
        return e, n

def get_ca(df, key):
    ca = df.loc[df['key']==key]["ca"].values[0]
    return ca


def calculate_err(qid, pred_key, df_query, df_db):
    qe, qn = get_en(df_query, qid)
    qca = get_ca(df_query, qid)

    pe, pn = get_en(df_db, pred_key)
    dist = math.sqrt((qe-pe)**2 + (qn-pn)**2)
    ca = get_ca(df_db, pred_key)
    theta = abs(qca-ca)
    return dist, theta

if __name__ == "__main__":
    
    city = "tokyo"
    root_dir = f"/media/HD2/Workspace/datasets/MSLS/train_val/{city}"
    query_dir = os.path.join(root_dir, "query")
    db_dir = os.path.join(root_dir, "database")

    df_query = get_data_df(query_dir)
    df_db = get_data_df(db_dir)
    
    K_values = [1,3,5,10] 
    methods = ["datt", "mac", "gem",  "netvlad"]
    threshold = (10,30)

    results_by_method = {meth:{} for meth in methods}
    pos_results_by_method = {meth:{} for meth in methods}
    for method in methods:
        print("Processing", method)
        # Read retrieval results
        result_file = f"./results/{city}_{method}_results.txt"
        if not os.path.exists(result_file):
            print(result_file, "does not exist")
            continue

        with open(result_file, "r") as f:
            results ={row.split()[0]:row.split()[1:] for row in f.readlines()}

        results_by_query = {}
        query_count = 0
        # calculate positioning error
        for query, predictions in results.items():
            results_by_query[query] = []
            for pred in predictions[:max(K_values)]:
                pos_err, ang_err = calculate_err(query, pred, df_query, df_db)
                results_by_query[query].append(pos_err)
            query_count += 1
        

        # calculate recall and avg.pos.err
        results_by_K = {K:0 for K in K_values}
        pos_results_by_K = {K:[] for K in K_values}
        for K_val in K_values:
            for query, pos_errors in results_by_query.items():
                pos_results_by_K[K_val].append(min(pos_errors[:K_val]))
                if min(pos_errors[:K_val]) <= threshold[0]:
                    results_by_K[K_val] += 1


        results_by_method[method] = {k:round(v*100/query_count, 2) for k, v in results_by_K.items()}
        pos_results_by_method[method] = {k:round(sum(v)/len(v), 2) for k, v in pos_results_by_K.items()}

    
     #Print recall results
    print("Recall@K")
    for method in methods:
        line = f"{method:{10}}"
        for k, v in results_by_method[method].items():
            line += f"{v:{10}}"
        print(line)

    # Print average positioning error
    print("\nAverage Positioing Error @ K")
    for method in methods:
        line = f"{method:{10}}"
        for k, v in pos_results_by_method[method].items():
            line += f"{v:{10}}"
        print(line)