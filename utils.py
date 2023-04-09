import os
import torch
import yaml
import shutil

def save_check(state, is_best: bool, filename = "checkpoint.pth.tar"):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "best_model.pth.tar")

def save_configuration_file(checkpt_folder, args):
    with open(os.path.join(checkpt_folder, 'config.yml'), 'w') as outfile:
        yaml.dump(args, outfile, default_flow_style = False)

def accuracy(output, target, topk = (1,)):
    with torch.no_grad():
        max_k = max(topk)
        batch_size = target.size(0)
        _, predictions = output.topk(max_k, 1, True, True)
        predictions = predictions.t()
        correct = predictions.eq(target.view(1, -1).expand_as(predictions))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim = True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
