import os
import json
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from tqdm import tqdm

from model import ModelType
from dataset import DatasetType
from utils import *



def get_inline_args():
    parser = argparse.ArgumentParser(description="Short training script.")
    # required things
    parser.add_argument(
        '--dataset-root',
        type=str,
        required=True,
        help="The path to the output checkpoint.")

    # optional things
    parser.add_argument(
        '--run-name',
        type=str,
        default=datetime.strftime(datetime.now(), "%Y%m%d_%H%M%S"),
        help="The run name to use as a label in file names and in W&B.")
    parser.add_argument(
        '--output-dir',
        type=str,
        default="runs",
        help="The path to output directory for logs and checkpoints.")
    parser.add_argument(
        '--eval-every',
        type=int,
        default=2,
        help="The interval at which to run evaluation.")
    parser.add_argument(
        '--checkpoint-every',
        type=int,
        default=10,
        help="The interval at which to store a checkpoint.")
    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        help="The training batch size.")
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help="The training batch size.")
    parser.add_argument(
        '--num-epochs',
        type=int,
        default=60,
        help="The number of training epochs.")
    parser.add_argument(
        '--device',
        type=str,
        help="The number of training epochs.")
    parser.add_argument(
        '--seed',
        type=int,
        help="Seed value.")
    parser.add_argument(
        '--use-tb',
        action="store_true",
        help="Use tensorboard if this parameter is given.")
    return parser.parse_args()



def main(args):

    assert args.run_name.strip() != "", "Invalid run name!"
    OUTPUT_PATH = os.path.join(args.output_dir, args.run_name)
    os.makedirs(OUTPUT_PATH, exist_ok=False)
    log_run_args(args, OUTPUT_PATH)
    print("Running with the following arguments:\n{}".format(args))

    if args.device:
        DEVICE = args.device
    else:
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    if args.use_tb:
        from torch.utils.tensorboard import SummaryWriter
        tb_writer = SummaryWriter(
            log_dir=OUTPUT_PATH,
            comment=args.run_name
        )
        tb_writer.add_text("Configuration", f"{json.dumps(vars(args), indent=4)}")

    # set the seed
    if args.seed:
        np.random.seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        # probably the seed should be set in other libraries too

    # define a model
    model = ModelType()
    model = model.to(DEVICE)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("The model has {:,} trainable parameters.".format(num_params))
    print("The model has the following structure:\n{}".format(model))
    log_model_arch(model, OUTPUT_PATH)

    # define a loss function
    criterion = nn.MSELoss()

    # define an  optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # define the dataloader
    train_ds = DatasetType(args.dataset_root, "train")
    train_loader = torch.utils.data.DataLoader(dataset=train_ds, batch_size=args.batch_size, shuffle=True)
    eval_ds = DatasetType(args.dataset_root, "eval")
    eval_loader = torch.utils.data.DataLoader(dataset=eval_ds, batch_size=args.batch_size, shuffle=True)
    print("Training dataset has {:,} samples.".format(len(train_ds)))
    print("Evaluation dataset has {:,} samples.".format(len(eval_ds)))


    # train
    # (might need modification based on the application)
    best_eval_loss = float('inf')
    for epoch in range(args.num_epochs):
        for inputs, targets in tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch+1}/{args.num_epochs}"):
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)

            outputs = model(inputs)
            train_loss = criterion(outputs, inputs)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

        if args.use_tb:
            tb_writer.add_scalar("train_loss", train_loss, epoch)

        # save checkpoint at the stated intervals
        if (epoch+1) % args.checkpoint_every == 0 or (epoch+1) == args.num_epochs:
            save_checkpoint(model, epoch, OUTPUT_PATH, fname=f"chkpt_ep{epoch+1}.pt")
            print(f"Checkpoint saved at epoch {epoch+1}!")


        # evaluate at the stated intervals
        if (epoch+1) % args.eval_every == 0 or (epoch+1) == args.num_epochs:
            losses_sum = 0
            with torch.no_grad():
                for eval_inputs, eval_targets in tqdm(eval_loader, total=len(eval_loader), desc="Evaluate"):
                    eval_inputs = eval_inputs.to(DEVICE)
                    eval_targets = eval_targets.to(DEVICE)

                    eval_outputs = model(eval_inputs)
                    loss = criterion(eval_outputs, eval_inputs)
                    losses_sum += loss.item()
            eval_loss = losses_sum / len(eval_loader)

            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                save_checkpoint(model, epoch, OUTPUT_PATH, fname="best_chkpt.pt")
                print(f"Best checkpoint saved at epoch {epoch+1}!")

            if args.use_tb:
                tb_writer.add_scalar("eval_loss", eval_loss, epoch)

            print("Eval loss: {:.8f}; Best eval loss: {:.8f}.".format(eval_loss, best_eval_loss))

    if args.use_tb:
        tb_writer.flush()
        tb_writer.close()




if __name__ == "__main__":
    args = get_inline_args()
    main(args)
