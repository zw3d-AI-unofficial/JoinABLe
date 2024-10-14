"""

JoinABLe Joint Axis Prediction Network

"""


import os
import sys
import json
import wandb
import random
import numpy as np
import csv
from pathlib import Path
import matplotlib.pyplot as plt
# from mplcursors import cursor

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.classification import AUROC, ROC, F1Score
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from utils import metrics
from utils import util
from datasets.joint_graph_dataset import JointGraphDataset
from args import args_train
from models.joinable import JoinABLe
import logging

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
        self.validation_step_outputs = []
        
        self.criterion = nn.BCELoss(reduction='mean')
        # self.criterion = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=torch.tensor([0.5]))
        self.train_accuracy = torchmetrics.Accuracy(task="binary")
        self.val_accuracy = torchmetrics.Accuracy(task="binary")
        self.test_accuracy = torchmetrics.Accuracy(task="binary")
        # self.train_confusion_matrix = torchmetrics.ConfusionMatrix(task='binary',num_classes=2)  #混淆矩阵
        # self.val_confusion_matrix = torchmetrics.ConfusionMatrix(task='binary',num_classes=2)
        # self.train_auroc = AUROC(task='binary')
        # self.val_auroc = AUROC(task='binary')
        # self.test_auroc = AUROC(task='binary')
        # self.train_roc = ROC(task='binary')
        # self.val_roc = ROC(task='binary')
        # self.test_roc = ROC(task='binary')
        # self.f1 = F1Score(task='binary',threshold=args.binary_threshold)   # 初始化 F1 Score 计算器

        # # 设置日志记录器
        # logging.basicConfig(filename='training.log', level=logging.INFO,
        #                     format='%(message)s')
        # self._logger = logging.getLogger(__name__)

    def training_step(self, batch, batch_idx):
        g1, g2, jg = batch
        x = self.model(g1, g2, jg)
        # 修改
        # 记录roc预测概率和标签
        # self.train_auroc.update(x, jg.joint_judge.int())
        # self.train_roc.update(x, jg.joint_judge.int())

        loss = self.criterion(x, jg.joint_judge.float())                                         
        
        joint_graph_unbatched = jg.to_data_list()
        batch_size = len(joint_graph_unbatched)
        self.log("train_loss", loss, on_step=False, on_epoch=True, batch_size=batch_size)

        train_acc = self.train_accuracy(x, jg.joint_judge.float())
        self.log("train_acc", train_acc, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        # binary_predictions = (x > args.binary_threshold).float()                                          # 将预测值转换为二进制类别 
        # self.train_confusion_matrix.update(binary_predictions, jg.joint_judge.float())  # 更新混淆矩阵
        return loss

    def on_train_step_end(self):
        self.check_gradient_norm()

    # def on_train_epoch_end(self):
        # # 计算训练集上的混淆矩阵
        # train_conf_matrix = self.train_confusion_matrix.compute()
        # self.train_confusion_matrix.reset()       
        # # 从混淆矩阵中提取 TP, FP, FN, TN
        # tn, fp, fn, tp = train_conf_matrix.flatten()        
        # tn, fp, fn, tp = tn.float(), fp.float(), fn.float(), tp.float()
        # # 记录混淆矩阵指标
        # self.log("train_TN", tn, on_epoch=True, prog_bar=False)
        # self.log("train_FP", fp, on_epoch=True, prog_bar=False)
        # self.log("train_FN", fn, on_epoch=True, prog_bar=False)
        # self.log("train_TP", tp, on_epoch=True, prog_bar=False)        

        # # 计算训练集上的 ROC 曲线
        # train_fpr, train_tpr, _ = self.train_roc.compute()
        # self.train_roc.reset()       
        # train_auroc = self.train_auroc.compute().cpu() 
        # self.train_auroc.reset()
        # # 绘制 ROC 曲线
        # plt.figure(figsize=(8, 8))
        # plt.plot(train_fpr.cpu().numpy(), train_tpr.cpu().numpy(), label=f'Training ROC (AUC = {train_auroc:.2f})')

    # def training_epoch_end(self, outputs):
    #     if self.current_epoch % 1 == 0:
    #         self.check_gradient_norm()

    def check_gradient_norm(self):                            # 用于检查模型参数的梯度范数，并判断梯度是否包含非数值（NaN）或无穷大（Inf）值
        total_norm = torch.tensor(0.0)                        # 初始化一个 total_norm 变量为 0.0，用于累积所有参数的梯度范数的平方。
        for param in self.model.parameters():                 # 遍历模型的所有可训练参数（通过 self.model.parameters()）
            if param.grad is not None:
                param_norm = torch.norm(param.grad.data)      # 梯度存在，使用 torch.norm 函数计算梯度的 L2 范数
                total_norm += param_norm.item() ** 2          # 将每个参数的梯度范数平方后累加到 total_norm 变量中
        total_norm = total_norm ** (1. / 2)
        print("Gradient norm:", total_norm)
        if torch.isnan(total_norm) or torch.isinf(total_norm):
            print("Gradient has NaN or Inf values!")

    def validation_step(self, batch, batch_idx):
        g1, g2, jg = batch
        x = self.model(g1, g2, jg)

        # # 修改
        # # 记录预测概率和标签
        # self.val_auroc.update(x, jg.joint_judge.int()) #标签是整数类型
        # self.val_roc.update(x, jg.joint_judge.int())

        loss = self.criterion(x, jg.joint_judge.float())
        self.log("val_loss", loss, on_step=False, on_epoch=True, batch_size=1)
        self.val_accuracy.update(x, jg.joint_judge.int())
        self.log("val_acc", self.val_accuracy, on_epoch=True, prog_bar=True, batch_size=1)

        # binary_predictions = (x >args.binary_threshold).float()   # 将预测值转换为二进制类别 
        # self.val_confusion_matrix.update(binary_predictions, jg.joint_judge.float())
        # self.validation_step_outputs.append({
        #     "preds": x ,
        #     "target": jg.joint_judge.int()
        # })          

        # # 将预测错误或不准确的打印出来       
        # is_middle = (x > 0.4) & (x < 0.6)            # 预测值未能很好地区分的；布尔条件组合的张量用&（按位与操作符）而不是 and（逻辑与关键字）                   
        # if (binary_predictions != jg.joint_judge.float()) or is_middle:
        #     self._logger.info(
        #     f"{jg.joint_file_name[0]},{x.item():.2f},{jg.joint_judge[0]}")  # 记录 joint文件名，预测值，标签值      
        return {"preds": x ,"target": jg.joint_judge.int()}
            
    def on_validation_epoch_start(self):
        # 在每个 epoch 开始时释放 GPU 缓存
        torch.cuda.empty_cache()
        # epoch = self.current_epoch
        # self._logger.info( f"validate:") 
        # self._logger.info( f"validate:{epoch}")  # 记录当前epoch
    
    # def on_validation_epoch_end(self):
    #     if len(self.validation_step_outputs) > 0:
    #         # 获取所有批次的预测结果和目标值
    #         val_preds = torch.cat([out["preds"] for out in self.validation_step_outputs]).detach().cpu()
    #         val_target = torch.cat([out["target"] for out in self.validation_step_outputs]).detach().cpu()        
    #         f1_val = self.f1(val_preds, val_target) # 计算 F1 分数
    #         self.log('val_f1', f1_val, on_epoch=True, prog_bar=False)
    #         self.validation_step_outputs.clear()    #清空

    #     # 计算验证集上的混淆矩阵
    #     val_conf_matrix = self.val_confusion_matrix.compute()
    #     self.val_confusion_matrix.reset()        
    #     # 从混淆矩阵中提取 TP, FP, FN, TN
    #     tn, fp, fn, tp = val_conf_matrix.flatten()      
    #     tn, fp, fn, tp = tn.float(), fp.float(), fn.float(), tp.float()  
    #     val_acc = (tp+tn)/(tp+fp+fn+tn)
    #     # 记录混淆矩阵指标
    #     self.log("val_TN", tn, on_epoch=True, prog_bar=False)
    #     self.log("val_FP", fp, on_epoch=True, prog_bar=False)
    #     self.log("val_FN", fn, on_epoch=True, prog_bar=False)
    #     self.log("val_TP", tp, on_epoch=True, prog_bar=False)
    #     self.log("val_acc", val_acc, on_epoch=True, prog_bar=True)

    #     # 计算验证集上的 ROC 曲线
    #     val_fpr, val_tpr, thresholds = self.val_roc.compute() #假阳性率（False Positive Rate, FPR）真阳性率（True Positive Rate, TPR）阈值
    #     # 寻找最佳阈值
    #     distances = torch.sqrt((1 - val_tpr)**2 + val_fpr**2)
    #     best_threshold_index = torch.argmin(distances)           #最小距离
    #     best_threshold = thresholds[best_threshold_index].item() #对应的阈值     
    #     self.log("best_threshold", best_threshold, on_epoch=True, prog_bar=True)
    #     self.log("best_threshold_fpr", val_fpr[best_threshold_index].item(), on_epoch=True, prog_bar=False)
    #     self.log("best_threshold_fpr", val_tpr[best_threshold_index].item(), on_epoch=True, prog_bar=False)

    #     self.val_roc.reset()  
    #     val_auroc = self.val_auroc.compute().cpu()         #ROC曲线下面积
    #     self.log("val_auroc", val_auroc, on_epoch=True, prog_bar=False)
    #     self.val_auroc.reset()

    #     # 绘制 ROC 曲线并添加标记点        
    #     val_fpr = val_fpr.cpu().numpy() # 将张量移动到CPU并转换为NumPy数组
    #     val_tpr = val_tpr.cpu().numpy()
    #     val_thresholds = thresholds.cpu().numpy()
    #     line_val = plt.plot(val_fpr, val_tpr, label=f'Validation ROC (AUC = {val_auroc:.2f})')         # 绘制ROC曲线
    #     # 标出特定点
    #     fractions = [1/5, 1/4, 1/3, 1/2, 3/4]
    #     for fraction in fractions:
    #         index = int(len(val_fpr) * fraction)  # 计算索引位置
    #         fpr_at_fraction = val_fpr[index]
    #         tpr_at_fraction = val_tpr[index]    
    #         thresholds_at_fraction = val_thresholds[index]    
    #         # 在图上标出该点
    #         plt.scatter(fpr_at_fraction, tpr_at_fraction, color='red', zorder=5) #指定x，y
    #         plt.text(fpr_at_fraction, tpr_at_fraction, f'{fraction:.2f}({fpr_at_fraction:.3f}, {tpr_at_fraction:.3f}), {thresholds_at_fraction:.3f}', fontsize=9, ha='right', va='bottom')

    #     # # 添加鼠标悬停提示
    #     # crs = cursor(line_val, hover=True)
    #     # crs.connect("add", lambda sel: sel.annotation.set_text(f"Threshold: {thresholds[int(sel.target.index)]:.2f}"))
    #     # # lambda 函数，它接受一个参数 sel，代表悬停事件的选中对象（selection object）。 set_text 方法用于设置文本内容.

    #     # 添加标题和标签
    #     plt.title('ROC Curve')
    #     plt.xlabel('False Positive Rate')
    #     plt.ylabel('True Positive Rate')
    #     plt.legend(loc='lower right')    
    #     # plt.savefig(f'figs/roc_curve_epoch_{self.current_epoch}.png', dpi=300)    #存储图片        
    #     plt.show() # 显示图形
    #     # plt.close()
        
    def test_step(self, batch, batch_idx):
        # Get the split we are using from the dataset
        split = self._trainer.test_dataloaders.dataset.split

        # Inference
        g1, g2, jg = batch
        x = self.model(g1, g2, jg)

        loss = self.criterion(x, jg.joint_judge.float())
        test_acc = self.test_accuracy(x, jg.joint_judge.int()) 
        self.log(f"eval_{split}_acc", test_acc, on_step=False, on_epoch=True, batch_size=1)

        self.test_step_outputs.append({
            "file_names": jg.joint_file_name[0],
            "loss": loss.item(),
            "pred": x.item(),
            "label": jg.joint_judge.item()
        })
        return {"preds": x ,"target": jg.joint_judge.int()}
    
    def on_test_epoch_end(self):
        # Log results to a csv file
        result = []
        for x in self.test_step_outputs:
            result.append((x["file_names"], f"{x['loss']:.2f}", f"{x['pred']:.2f}", x["label"]))
        result.sort(key=lambda x: -float(x[1]))

        # dump result
        exp_dir = Path(args.exp_dir)
        csv_file = exp_dir / args.exp_name / (args.checkpoint + "_" + args.test_split + ".csv")
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["file_name", "loss", "pred", "label"])
            for row in result:
                writer.writerow(row)

    def forward(self, batch):
        # Used for inference
        g1, g2, jg = batch
        jg.edge_attr = jg.edge_attr.long()
        return self.model(g1, g2, jg)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr) #优化器定义，可训练参数；学习率
        scheduler = ReduceLROnPlateau(optimizer, "min") #学习率调度器定义；学习率将在验证损失不再减小时降低
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",                   #修改2
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

# def train_once(args, exp_name_dir, loggers, train_dataset):                     #修改2
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
        
    # Save in the main experiment directory 模型保存
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
    # trainer.fit(model, train_loader)                                 #修改2
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
        checkpoint_file = exp_name_dir / f"{args.checkpoint}.ckpt"     # 'results/my_experiment/last.ckpt'
        model = JointPrediction.load_from_checkpoint(
            checkpoint_file,
            map_location=torch.device("cpu")
        )
        print(f"Evaluating checkpoint {checkpoint_file} on {split} split")
    trainer = get_trainer(args, loggers, mode="evaluation")    
    test_dataset = load_dataset(args, split=split, label_scheme=args.test_label_scheme, max_node_count=2*args.max_node_count) #从图形数据中加载关节数据
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

    # os.environ['HTTP_PROXY'] = 'http://127.0.0.1:2080'
    # os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:2080'
    # os.environ['ALL_PROXY'] = 'socks5://127.0.0.1:2080'
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'           #训练或者debug先用编号为1的卡
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
        )                                    #修改2
        trainer_global_rank = train_once(
            args,
            exp_name_dir,
            loggers,
            train_dataset,
            val_dataset                       #修改2
        )

    # EVALUATION
    # Evaluate on a single CPU to handle very large graphs
    if args.traintest == "test" or args.traintest == "traintest":
        if trainer_global_rank is not None:                    
            # If we are doing distributed training                           如果正在进行分布式训练
            # we need to destroy the process group and intialize a cpu based trainer
            # https://github.com/PyTorchLightning/pytorch-lightning/issues/8375#issuecomment-878739663
            if torch.cuda.device_count() > 1 and args.accelerator == "ddp":
                torch.distributed.destroy_process_group()                  # 代码会销毁当前的进程组

        if trainer_global_rank is None or trainer_global_rank == 0:
            evaluate_once(args, exp_name_dir, loggers, args.test_split)    # 分布式训练中只在主节点上执行

    if args.traintest == "randomtest":
        evaluate_once(args, exp_name_dir, loggers, args.test_split, random_test=True) # 随机的数据划分


if __name__ == "__main__":
    args = args_train.get_args()
    main(args)
