from deattnet.datasets.msls import MSLS, ImagesFromList
from deattnet.datasets.data_util import input_transform, input_transform_1d
from deattnet.models.models_generic import get_backend, get_model, get_DAttNet

import os
import configparser
import torch
import numpy as np
import torch
import faiss
from tqdm.auto import tqdm
from torch.utils.data import DataLoader


def val(eval_set, model, encoder_dim, device, config, out_file="netvlad_results.txt", channels='rgb',pbar_position=0):
    cuda= True if device.type == 'cuda' else False

    eval_set_queries = ImagesFromList(eval_set.qImages, transform=input_transform(), transform_1d=input_transform_1d(), channels=channels)
    eval_set_dbs = ImagesFromList(eval_set.dbImages, transform=input_transform(), transform_1d=input_transform_1d(), channels=channels)
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
                image_encoding = model.encoder(input_data)

                vlad_encoding = model.pool(image_encoding)
                feat[indices.detach().numpy(), :] = vlad_encoding.detach().cpu().numpy()

                del input_data, image_encoding, vlad_encoding

    del test_data_loader_queries, test_data_loader_dbs

    tqdm.write('====> Building faiss index')
    faiss_index = faiss.IndexFlatL2(pool_size)
    # noinspection PyArgumentList
    faiss_index.add(dbFeat)

    tqdm.write('====> Calculating recall @ N')
    n_values = [1, 3, 5, 10]

    # for each query get those within threshold distance
    gt = eval_set.all_pos_indices

    # any combination of mapillary cities will work as a val set
    qEndPosTot = 0
    dbEndPosTot = 0
    for cityNum, (qEndPos, dbEndPos) in enumerate(zip(eval_set.qEndPosList, eval_set.dbEndPosList)):
        faiss_index = faiss.IndexFlatL2(pool_size)
        faiss_index.add(dbFeat[dbEndPosTot:dbEndPosTot+dbEndPos, :])
        _, preds = faiss_index.search(qFeat[qEndPosTot:qEndPosTot+qEndPos, :], max(n_values))
        if cityNum == 0:
            predictions = preds
        else:
            predictions = np.vstack((predictions, preds))
        qEndPosTot += qEndPos
        dbEndPosTot += dbEndPos

    with open(out_file, "w") as f:
        for query, preds in zip(eval_set.qImages, predictions):
            qid = query.split("/")[-1].split(".")[0]
            results = [eval_set.dbImages[p].split("/")[-1].split(".")[0] for p in preds]
            res_line = " ".join([qid]+results)
            f.writelines(res_line + "\n")


    correct_at_n = np.zeros(len(n_values))
    # TODO can we do this on the matrix in one go?
    for qIx, pred in enumerate(predictions):
        for i, n in enumerate(n_values):
            # if in top N then also in top NN, where NN > N
            if np.any(np.in1d(pred[:n], gt[qIx])):
                correct_at_n[i:] += 1
                break
    recall_at_n = correct_at_n / len(eval_set.qIdx)

    all_recalls = {}  # make dict for output
    for i, n in enumerate(n_values):
        all_recalls[n] = recall_at_n[i]
        tqdm.write("====> Recall@{}: {:.4f}".format(n, recall_at_n[i]))

    return all_recalls

def load_model(checkpoint_path, method, device, config):
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
    city = "tokyo"

    encoder_dim, model = load_model(checkpoint_path, method, device, config)
    validation_dataset = MSLS(dataset_root_dir, cities=city, mode='val', subtask="all")
    print('===> Evaluating on val set, query count:', len(validation_dataset.qImages))
    recalls = val(validation_dataset, model, encoder_dim, device, config, channels=channels, out_file=f"./results/{city}_{method}_results.txt", pbar_position=1)
    print(recalls)

