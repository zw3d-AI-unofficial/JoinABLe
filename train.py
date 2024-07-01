"""

JoinABLe Joint Axis Prediction Network

"""


import os
import sys
import json
import wandb
import random
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from utils import metrics
from utils import util
from datasets.joint_graph_dataset import JointGraphDataset
from args import args_train
from models.joinable import JoinABLe


class JointPrediction(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.model = JoinABLe(args)
        self.save_hyperparameters()
        self.args = args
        # self.test_iou = torchmetrics.IoU(
        #     threshold=args.threshold,
        #     num_classes=2,
        #     compute_on_step=False,
        #     ignore_index=0,
        # )
        # self.test_accuracy = torchmetrics.Accuracy(
        #     threshold=args.threshold,
        #     num_classes=2,
        #     compute_on_step=False,
        #     # ignore_index=0,
        #     multiclass=True
        # )
        self.test_step_outputs = []

    def training_step(self, batch, batch_idx):
        g1, g2, jg = batch
        jg.edge_attr = jg.edge_attr.long()
        graph_one_unbatched = g1.to_data_list()
        graph_two_unbatched = g2.to_data_list()
        joint_graph_unbatched = jg.to_data_list()
        batch_size = len(joint_graph_unbatched)
        
        x = self.model(g1, g2, jg)
        loss = self.model.compute_loss(self.args, x, jg)
        if self.args.with_type:
            x, x_type = x
            type_acc = self.model.precision_type(x_type, jg.joint_type_list)
            self.log("train_type_acc", type_acc, on_step=False, on_epoch=True, logger=True, batch_size=batch_size)

        top_k = [0, 0, 0]
        start = 0
        for i, item in enumerate(joint_graph_unbatched):
            n_nodes1 = graph_one_unbatched[i].num_nodes
            n_nodes2 = graph_two_unbatched[i].num_nodes
            size_i = n_nodes1 * n_nodes2
            end = start + size_i
            top_k_i = self.model.precision_at_top_k(
                x[start: end], 
                item.edge_attr, 
                n_nodes1, 
                n_nodes2, 
                k=[1,5,50]
            )
            for j in range(len(top_k)):
                top_k[j] += top_k_i[j]
            start = end
        # Log the run at every epoch, although this gets reduced via mean to a float
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
        self.log("train_top_1", top_k[0] / batch_size, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
        self.log("train_top_5", top_k[1] / batch_size, on_step=False, on_epoch=True, logger=True, batch_size=batch_size)
        self.log("train_top_50", top_k[2] / batch_size, on_step=False, on_epoch=True, logger=True, batch_size=batch_size)
        return loss

    def training_step_end(self):
        self.check_gradient_norm()

    # def training_epoch_end(self, outputs):
    #     if self.current_epoch % 1 == 0:
    #         self.check_gradient_norm()

    def check_gradient_norm(self):
        total_norm = torch.tensor(0.0)
        for param in self.model.parameters():
            if param.grad is not None:
                param_norm = torch.norm(param.grad.data)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        print("Gradient norm:", total_norm)
        if torch.isnan(total_norm) or torch.isinf(total_norm):
            print("Gradient has NaN or Inf values!")

    def validation_step(self, batch, batch_idx):
        g1, g2, jg = batch
        jg.edge_attr = jg.edge_attr.long()
        x = self.model(g1, g2, jg)
        loss = self.model.compute_loss(self.args, x, jg)
        if self.args.with_type:
            x, x_type = x
            precision_type = self.model.precision_type(x_type, jg.joint_type_list)
            self.log("val_type_acc", precision_type, on_step=False, on_epoch=True, logger=True, batch_size=1)
        top_k = self.model.precision_at_top_k(x, jg.edge_attr, g1.num_nodes, g2.num_nodes, k=[1,5,50])
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=False, logger=True, batch_size=1)
        self.log("val_top_1", top_k[0], on_step=False, on_epoch=True, prog_bar=False, logger=True, batch_size=1)
        self.log("val_top_5", top_k[1], on_step=False, on_epoch=True, logger=True, batch_size=1)
        self.log("val_top_50", top_k[2], on_step=False, on_epoch=True, logger=True, batch_size=1)
        return {"loss": loss, "top_1": top_k[0]}

    def test_step(self, batch, batch_idx):
        # Get the split we are using from the dataset
        split = self._trainer.test_dataloaders.dataset.split
        # Inference
        g1, g2, jg = batch
        jg.edge_attr = jg.edge_attr.long()
        x = self.model(g1, g2, jg)
        loss = self.model.compute_loss(self.args, x, jg)
        if self.args.with_type:
            x, x_type = x
            precision_type = self.model.precision_type(x_type, jg.joint_type_list)
            self.log("test_type_acc", precision_type, on_step=False, on_epoch=True, logger=True, batch_size=1)

        # Get the probabilities and calculate metrics
        # prob = F.softmax(x, dim=0)
        # self.test_iou.update(prob, jg.edge_attr)
        # self.test_accuracy.update(prob, jg.edge_attr)
        # Calculate the precision at k with a default sequence of k
        top_k = self.model.precision_at_top_k(x, jg.edge_attr, jg.num_nodes_graph1, jg.num_nodes_graph2)
        self.log(f"eval_{split}_loss", loss, on_step=False, on_epoch=True, logger=True, batch_size=1)
        self.log(f"eval_{split}_top_1", top_k[0], on_step=False, on_epoch=True, logger=True, batch_size=1)
        self.log(f"eval_{split}_top_5", top_k[4], on_step=False, on_epoch=True, logger=True, batch_size=1)
        self.log(f"eval_{split}_top_50", top_k[9], on_step=False, on_epoch=True, logger=True, batch_size=1)
        # Log evaluation based on if there are holes or not
        # Batch size 1 and no shuffle lets us use the batch index
        has_holes = self._trainer.test_dataloaders.dataset.has_holes[batch_idx]
        top_1_holes = None
        top_1_no_holes = None
        # if has_holes:
        #     self.log(f"eval_{split}_top_1_holes", top_1, on_step=False, on_epoch=True, logger=True)
        #     top_1_holes = top_1
        # else:
        #     self.log(f"eval_{split}_top_1_no_holes", top_1, on_step=False, on_epoch=True, logger=True)
        #     top_1_no_holes = top_1

        for key, value in JointGraphDataset.joint_type_map.items():
            if value in jg.joint_type_list:
                self.log(f"eval_{split}_top_1_{key}", top_k[0], on_epoch=True, logger=True, batch_size=1)

        self.test_step_outputs.append({
            "loss": loss,
            "top_k": top_k,
            "top_1_holes": top_1_holes,
            "top_1_no_holes": top_1_no_holes
        })
        return {
            "loss": loss,
            "top_k": top_k,
            "top_1_holes": top_1_holes,
            "top_1_no_holes": top_1_no_holes
        }

    def on_test_epoch_end(self):
        # Get the split we are using from the dataset
        split = self._trainer.test_dataloaders.dataset.split
        # test_iou = self.test_iou.compute()
        # test_accuracy = self.test_accuracy.compute()
        # self.log(f"eval_{split}_iou", test_iou)
        # self.log(f"eval_{split}_accuracy", test_accuracy)
        all_top_k = np.stack([x["top_k"] for x in self.test_step_outputs])
        all_top_1_holes = np.array([x["top_1_holes"] for x in self.test_step_outputs if x["top_1_holes"] is not None])
        all_top_1_no_holes = np.array([x["top_1_no_holes"] for x in self.test_step_outputs if x["top_1_no_holes"] is not None])
        top_1_holes = "--"
        top_1_no_holes = "--"
        # # All samples should be either holes or no holes, so check the counts add up to the total
        # assert len(all_top_1_holes) + len(all_top_1_no_holes) == all_top_k.shape[0]
        # if len(all_top_1_holes) > 0:
        #     top_1_holes = all_top_1_holes.mean()
        # if len(all_top_1_no_holes) > 0:
        #     top_1_no_holes = all_top_1_no_holes.mean()

        k_seq = metrics.get_k_sequence()
        top_k = metrics.calculate_precision_at_k_from_sequence(all_top_k, use_percent=False)
        top_k_results = ""
        for k, result in zip(k_seq, top_k):
            top_k_results += f"{k} {result:.4f}%\n"
        self.print(f"Eval top-k results on {split} set:\n{top_k_results[:-2]}")
        for logger in self._trainer.loggers:
            if isinstance(logger, pl.loggers.CometLogger):
                logger.experiment.log_curve(
                    f"eval_{split}_top_k",
                    x=k_seq,
                    y=top_k.tolist(),
                    overwrite=True
                )

        # Log true positive sample id to a txt file
        # with open(self.args.exp_name + '.txt', 'w') as file:
        #     for i, one_top_k in enumerate(all_top_k):
        #         if one_top_k[0]:   
        #             file.write(str(i) + '\n')
        return {
            # "iou": test_iou,
            # "accuracy": test_accuracy,
            "top_1": top_k[0],
            "top_1_holes": top_1_holes,
            "top_1_no_holes": top_1_no_holes
        }

    def forward(self, batch):
        # Used for inference
        g1, g2, jg = batch
        jg.edge_attr = jg.edge_attr.long()
        return self.model(g1, g2, jg)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)
        scheduler = ReduceLROnPlateau(optimizer, "min")
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }


def load_dataset(args, split="train", random_rotate=False, label_scheme="Joint", max_node_count=0):
    return JointGraphDataset(
        root_dir=args.dataset,
        split=split,
        center_and_scale=True,
        random_rotate=random_rotate,
        delete_cache=args.delete_cache,
        limit=args.limit,
        threads=args.threads,
        label_scheme=label_scheme,
        max_node_count=max_node_count,
        input_features=args.input_features,
        skip_far=args.skip_far,
        skip_interference=args.skip_interference,
        skip_nurbs=args.skip_nurbs,
        skip_synthetic=args.skip_synthetic,
        joint_type=args.joint_type,
        quantize=args.quantize,
        n_bits=args.n_bits
    )


def get_trainer(args, loggers, callbacks=None, mode="train"):
    """Get the PyTorch Lightning Trainer"""
    log_every_n_steps = 100
    if mode == "train":
        # Distributed training
        if torch.cuda.device_count() > 1 and args.accelerator != "None":
            trainer = Trainer(
                max_epochs=args.epochs,
                devices=args.gpus,
                strategy=args.accelerator,
                log_every_n_steps=log_every_n_steps,
                callbacks=callbacks,
                logger=loggers
            )
        # Single GPU training
        else:
            trainer = Trainer(
                max_epochs=args.epochs,
                devices=args.gpus,
                log_every_n_steps=log_every_n_steps,
                callbacks=callbacks,
                logger=loggers
            )
    elif mode == "evaluation":
        trainer = Trainer(
            accelerator="cpu",
            logger=loggers,
            log_every_n_steps=log_every_n_steps
        )
    return trainer


def train_once(args, exp_name_dir, loggers, train_dataset, val_dataset):
    """Train once for multiple run training"""
    checkpoint_file = exp_name_dir / f"{args.checkpoint}.ckpt"
    if args.resume and os.path.exists(checkpoint_file):
        model = JointPrediction.load_from_checkpoint(
            checkpoint_file
        )
        print("Resuming existing checkpoint from:", checkpoint_file)
    else:
        model = JointPrediction(args)
        
    # Save in the main experiment directory
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        dirpath=exp_name_dir,
        filename="best",
        save_last=True,
    )
    checkpoint_callback.CHECKPOINT_NAME_LAST = "last"
    callbacks = [checkpoint_callback]

    trainer = get_trainer(
        args,
        loggers,
        callbacks=callbacks,
        mode="train"
    )

    train_loader = train_dataset.get_train_dataloader(
        max_nodes_per_batch=args.max_nodes_per_batch,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    val_loader = val_dataset.get_test_dataloader(batch_size=1, num_workers=args.num_workers)
    trainer.fit(model, train_loader, val_loader)
    if trainer.global_rank == 0:
        print("--------------------------------------------------------------------------------")
        print("TRAINING RESULTS")
        for key, val in trainer.logged_metrics.items():
            print(f"{key}: {val}")
        print("--------------------------------------------------------------------------------")
    return trainer.global_rank


def evaluate_once(args, exp_name_dir, loggers, split, random_test=False):
    """Evaluate once after a multiple run training"""
    # pl.utilities.seed.seed_everything(args.seed)
    # Load the model again as if sync_batchnorm is on it gets modified
    model = JointPrediction(args)
    # print(model)
    if random_test:
        print(f"Evaluating random on {split} split")
    else:
        checkpoint_file = exp_name_dir / f"{args.checkpoint}.ckpt"
        model = JointPrediction.load_from_checkpoint(
            checkpoint_file,
            map_location=torch.device("cpu")
        )
        print(f"Evaluating checkpoint {checkpoint_file} on {split} split")
    trainer = get_trainer(args, loggers, mode="evaluation")
    test_dataset = load_dataset(args, split=split, label_scheme=args.test_label_scheme, max_node_count=2*args.max_node_count)
    test_loader = test_dataset.get_test_dataloader(batch_size=1, num_workers=args.num_workers)
    trainer.test(model, test_loader)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(args):
    """Main entry point for our training script"""

    os.environ['HTTP_PROXY'] = 'http://127.0.0.1:2080'
    os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:2080'
    os.environ['ALL_PROXY'] = 'socks5://127.0.0.1:2080'
    torch.set_float32_matmul_precision('high')
    seed_everything(args.seed)
    
    exp_dir = Path(args.exp_dir)
    exp_name_dir = exp_dir / args.exp_name
    if not exp_name_dir.exists():
        exp_name_dir.mkdir(parents=True)
    if not exp_name_dir.exists():
        exp_name_dir.mkdir(parents=True)

    # We save the logs to the experiment directory
    loggers = []
    if args.traintest == "train" or args.traintest == "traintest":
        loggers = util.get_loggers(args, exp_name_dir)

    # TRAINING
    trainer_global_rank = None
    if args.traintest == "train" or args.traintest == "traintest":
        train_dataset = load_dataset(
            args, split="train",
            random_rotate=args.random_rotate,
            label_scheme=args.train_label_scheme,
            max_node_count=args.max_node_count
        )
        val_dataset = load_dataset(
            args,
            split="val",
            label_scheme=args.train_label_scheme,
            max_node_count=args.max_node_count
        )
        trainer_global_rank = train_once(
            args,
            exp_name_dir,
            loggers,
            train_dataset,
            val_dataset
        )

    # EVALUATION
    # Evaluate on a single CPU to handle very large graphs
    if args.traintest == "test" or args.traintest == "traintest":
        if trainer_global_rank is not None:
            # If we are doing distributed training
            # we need to destroy the process group and intialize a cpu based trainer
            # https://github.com/PyTorchLightning/pytorch-lightning/issues/8375#issuecomment-878739663
            if torch.cuda.device_count() > 1 and args.accelerator == "ddp":
                torch.distributed.destroy_process_group()

        if trainer_global_rank is None or trainer_global_rank == 0:
            evaluate_once(args, exp_name_dir, loggers, args.test_split)

    if args.traintest == "randomtest":
        evaluate_once(args, exp_name_dir, loggers, args.test_split, random_test=True)


if __name__ == "__main__":
    args = args_train.get_args()
    main(args)
