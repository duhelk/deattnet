# DeAttNet
DeAttNet: Depth Attention Network

1. Depth image rendering
    1. Using data_prep/gen_depthmaps.py 
    2. Zoedepth is available on torch.hub, So you can either directly use it or use via a local repository 
    3. Refer to https://github.com/isl-org/ZoeDepth , for any zoedepth loading issues
2. Training
    1. train.py
    2. Configurations defined in config/train.ini
    3. Provide command line arguments accordingly specially dataset_root_dir
    4. Training supports both Cambridge Landmarks and Mapillary Street-level Sequences datasets
    5. Supports netvlad, max, gem descriptors/pooling.
3. Evaluation
    1. xx_retrieval.py perform the retrieval and create the result file
    2. xx_pos_eval.py gives recall@k and avg.pos.err @k results
    
