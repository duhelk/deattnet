

import os
import math
import numpy as np

from scipy.spatial.transform import Rotation


def min_gt_error(test_rows, train_rows):
    min_errrors = []
    for query, gt in test_rows.items():
        gt = np.array([float(x) for x in gt])
        min_err = 99999
        for ref, ref_pose in train_rows.items():
            ref_pose = np.array([float(x) for x in ref_pose])
            err = calculate_err(gt, ref_pose)[0]
            if err < min_err:
                min_err = err
        min_errrors.append(min_err)
    print(sum(min_errrors)/len(min_errrors))
        

def calculate_err(gt_pose, predicted_pose):
    gt_x = gt_pose[:3]
    gt_q = gt_pose[3:]

    predicted_x = predicted_pose[:3]
    predicted_q = predicted_pose[3:]

    q1 = gt_q / np.linalg.norm(gt_q)
    q2 = predicted_q /np.linalg.norm(predicted_q)
    d = abs(np.sum(np.multiply(q1,q2)))
    theta = 2 * np.arccos(d) * 180/math.pi

    if np.isnan(theta):
        print(gt_q, predicted_q)
    
    error_x = np.linalg.norm(gt_x-predicted_x)

    return error_x, theta


def pose_interpolation(ref_poses):
    reference_poses = [ (np.array(pose[:3]), Rotation.from_quat(pose[3:])) for pose in ref_poses]
    reference_rotations = [pose[1].as_quat() for pose in reference_poses]
    average_quaternion = Rotation.from_quat(reference_rotations).mean().as_quat()
    average_translation = np.mean([pose[0] for pose in reference_poses], axis=0)
    final_estimated_pose = (average_translation, Rotation.from_quat(average_quaternion))
    print("Estimated Pose Translation:", final_estimated_pose[0])
    print("Estimated Pose Rotation (Quaternion):", final_estimated_pose[1].as_quat())
    return final_estimated_pose[0].tolist() + final_estimated_pose[1].as_quat().tolist()



if __name__ == "__main__":
    subsets = ["KingsCollege","OldHospital", "StMarysChurch"]
    methods = ["datt","gem"]#, "netvlad", "mac"]
    thresholds = (3,30)# , (5,30)
    K_values = [1,3,5,10,20]
    width = max(len(m) for m in methods)

    data_root = "/media/HD2/Workspace/datasets/Cambridge Landmarks"


    full_results = {subset: {method: "" for method in methods} for subset in subsets}
    avg_err_results = {subset: {method: "" for method in methods} for subset in subsets}


    results_by_method = {meth:{} for meth in methods}
    pos_results_by_method = {meth:{} for meth in methods}
    for method in methods:
        results_by_subset ={subset:{} for subset in subsets}
        pos_results_by_subset ={subset:{} for subset in subsets}
        for subset in subsets:
            # Read dataset
            root_dir = f"{data_root}/{subset}"
            with open (os.path.join(root_dir, "dataset_test.txt"), "r") as f:
                test_rows = {row.split()[0]:row.split()[1:] for row in f.readlines()[3:]}
            with open (os.path.join(root_dir, "dataset_train.txt"), "r") as f:
                train_rows = {row.split()[0]:row.split()[1:] for row in f.readlines()[3:]}

            # Read retrieval results
            result_file = f"./results/{subset}_{method}_all_results.txt"
            if not os.path.exists(result_file):
                print(result_file, "does not exist")
                continue

            with open(result_file, "r") as f:
                results ={row.split()[0]:row.split()[1:] for row in f.readlines()}

            results_by_query = {}
            subset_count = 0
            # calculate positioning error
            for query, predictions in results.items():
                results_by_query[query] = []
                gt = np.array([float(x) for x in test_rows[query]])
                for pred in predictions[:max(K_values)]:
                    res = np.array([float(x) for x in train_rows[pred]])
                    pos_err, ang_err = calculate_err(gt, res)
                    results_by_query[query].append(pos_err)
                subset_count += 1
            
            # calculate recall and avg.pos.err
            results_by_K = {K:0 for K in K_values}
            pos_results_by_K = {K:[] for K in K_values}
            for K_val in K_values:
                for query, pos_errors in results_by_query.items():
                    pos_results_by_K[K_val].append(min(pos_errors[:K_val]))
                    if min(pos_errors[:K_val]) <= thresholds[0]:
                        results_by_K[K_val] += 1

            results_by_subset[subset] = {k:round(v*100/subset_count, 2) for k, v in results_by_K.items()}
            pos_results_by_subset[subset] = {k:round(sum(v)/len(v), 2) for k, v in pos_results_by_K.items()}

        
        
        results_by_method[method] = results_by_subset
        pos_results_by_method[method] = pos_results_by_subset


    #Print recall results
    print("Recall@K")
    for subset in subsets:
        print(f"==={subset}===")
        for method in methods:
            line = f"{method:{10}}"
            for k, v in results_by_method[method][subset].items():
                line += f"{v:{10}}"
            print(line)

    # Print average positioning error
    print("\nAverage Positioing Error @ K")
    for subset in subsets:
        print(f"==={subset}===")
        for method in methods:
            line = f"{method:{10}}"
            for k, v in pos_results_by_method[method][subset].items():
                line += f"{v:{10}}"
            print(line)
