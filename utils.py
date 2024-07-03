import os
import json
import torch

def save_checkpoint(model, epoch, dirpath, fname="checkpoint.pt"):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
    }, os.path.join(dirpath, fname))

# def load_checkpoint(model, device, fname="checkpoint.pt"):
#     checkpoint = torch.load(os.path.join(".", fname), map_location=device)
#     model.load_state_dict(checkpoint['model_state_dict'])
#     return model

def log_model_arch(model, dirpath, fname="model.txt"):
    with open(os.path.join(dirpath, fname), 'w') as f:
        print(model, file=f)

def log_run_args(args, dirpath, fname="args.txt"):
    with open(os.path.join(dirpath, fname), 'w') as f:
        f.write(json.dumps(vars(args), indent=4))

