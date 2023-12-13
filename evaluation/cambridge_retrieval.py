from deattnet.datasets.data_util import input_transform, input_transform_1d
from deattnet.models.models_generic import get_backend, get_model, get_DAttNet
from deattnet.datasets.cambridge import ImagesFromList, CambridgeLandmarks

import os
import configparser
import torch
import numpy as np
import sys
import torch
import faiss
from tqdm.auto import tqdm
from torch.utils.data import DataLoader


def evaluate(eval_set, eval_set_queries, eval_set_dbs, model, encoder_dim, device, config, out_file="netvlad_results.txt", channels='rgb', top_K=100,pbar_position=1):
    if os.path.exists(out_file):
        print("Already evaluated", out_file)
        return
    cuda = True if device.type == 'cuda' else False
    test_data_loader_queries = DataLoader(dataset=eval_set_queries,
                                          num_workers=1, batch_size=int(config['train']['cachebatchsize']),
                                          shuffle=False, pin_memory=cuda)
    test_data_loader_dbs = DataLoader(dataset=eval_set_dbs,
                                      num_workers=1, batch_size=int(config['train']['cachebatchsize']),
                                      shuffle=False, pin_memory=cuda)

    model.eval()
    with torch.no_grad():
        tqdm.write('====> Extracting Features')
        pool_size = encoder_dim
        if config['global_params']['pooling'].lower() == 'netvlad':
            pool_size *= int(config['global_params']['num_clusters'])
        qFeat = np.empty((len(eval_set_queries), pool_size), dtype=np.float32)
        dbFeat = np.empty((len(eval_set_dbs), pool_size), dtype=np.float32)

        for feat, test_data_loader in zip([qFeat, dbFeat], [test_data_loader_queries, test_data_loader_dbs]):
            for iteration, (input_data, indices) in \
                    enumerate(tqdm(test_data_loader, position=pbar_position, leave=False, desc='Test Iter'.rjust(15)), 1):
                input_data = input_data.to(device)
                if channels == "rgbd":
                    image_encoding = model.encoder(input_data)
                else:
                    image_encoding  = model.encoder(input_data)

                vlad_encoding = model.pool(image_encoding)
                feat[indices.detach().numpy(), :] = vlad_encoding.detach().cpu().numpy()

                del input_data, image_encoding, vlad_encoding

    del test_data_loader_queries, test_data_loader_dbs

    tqdm.write('====> Building faiss index')
    faiss_index = faiss.IndexFlatL2(pool_size)
    faiss_index.add(dbFeat)
    _, predictions = faiss_index.search(qFeat[:, :], top_K)


    out_dir = os.path.dirname(out_file)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    with open(out_file, "w") as f:
        for query, preds in zip(eval_set.qImages, predictions):
            qid = "/".join(query.split("/")[-2:])
            results = ["/".join(eval_set.dbImages[p].split("/")[-2:]) for p in preds]
            res_line = " ".join([qid]+results)
            f.writelines(res_line + "\n")


def load_model(checkpoint_path, method, device, config, datt=False):

    if "deatt" in method:
        encoder_dim, encoder = get_DAttNet()
    else:
        encoder_dim, encoder = get_backend(in_channels=3)

    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)


    print(f"Loading checkpoint {checkpoint_path}")
    print("Pooling", config['global_params']['pooling'])
        
    model = get_model(encoder, encoder_dim, config['global_params'], append_pca_layer=False)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    model.eval()
    return encoder_dim, model




if __name__ == "__main__":

    device = torch.device("cuda:0")

    configfile = 'configs/eval.ini'
    assert os.path.isfile(configfile)
    config = configparser.ConfigParser()
    config.read(configfile)

    checkpoint_path = config['eval_params']['checkpoint']
    assert os.path.isfile(checkpoint_path)

    channels = config['global_params']['channels']
    method =  config['eval_params']['descriptor']
    dataset_root_dir = config['eval_params']['data_root']
    subsets = config['eval_params']['subsets'].split(',')

    encoder_dim, model = load_model(checkpoint_path, method, device, config)

    
    for subset in subsets:
        eval_set = CambridgeLandmarks(root_dir=dataset_root_dir, subset=subset, mode="eval")
        eval_set_queries = ImagesFromList(dataset_root_dir, subset, eval_set.qImages, transform=input_transform(), transform_1d=input_transform_1d(), channels=channels)
        eval_set_dbs = ImagesFromList(dataset_root_dir, subset, eval_set.dbImages, transform=input_transform(), transform_1d=input_transform_1d(), channels=channels)
        print('===> Evaluating on val set, query count:', subset, len(eval_set.qImages))
        recalls = evaluate(eval_set, eval_set_queries, eval_set_dbs, 
                            model, encoder_dim, device, config, 
                            channels=channels, 
                            out_file=f"./results/{subset}_{method}_results.txt")
        print(recalls)


