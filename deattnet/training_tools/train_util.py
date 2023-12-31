import torch
import shutil
from os.path import join


def humanbytes(B):
    """Return the given bytes as a human friendly KB, MB, GB, or TB string"""
    B = float(B)
    KB = float(1024)
    MB = float(KB ** 2)  # 1,048,576
    GB = float(KB ** 3)  # 1,073,741,824
    TB = float(KB ** 4)  # 1,099,511,627,776

    if B < KB:
        return '{0} {1}'.format(B,'Bytes' if 0 == B > 1 else 'Byte')
    elif KB <= B < MB:
        return '{0:.2f} KB'.format(B/KB)
    elif MB <= B < GB:
        return '{0:.2f} MB'.format(B/MB)
    elif GB <= B < TB:
        return '{0:.2f} GB'.format(B/GB)
    elif TB <= B:
        return '{0:.2f} TB'.format(B/TB)


def save_checkpoint(state, opt, is_best_sofar, filename='checkpoint.pth.tar'):
    if opt.save_every_epoch:
        model_out_path = join(opt.save_file_path, 'checkpoint_epoch' + str(state['epoch']) + '.pth.tar')
    else:
        model_out_path = join(opt.save_file_path, filename)
    torch.save(state, model_out_path)
    if is_best_sofar:
        shutil.copyfile(model_out_path, join(opt.save_file_path, 'model_best.pth.tar'))