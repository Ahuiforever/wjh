# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/7/29 13:51
# @Author  : Ahuiforever
# @File    : utils.py
# @Software: PyCharm

import datetime
import glob
import os
import shutil
import inspect

import torch
import torchvision


class PathChecker:
    def __init__(self):
        pass

    def __call__(self, path: str, del_: bool = True):
        if del_:
            shutil.rmtree(path, ignore_errors=True)
        os.makedirs(path, exist_ok=True)


class ModelSaver:
    """
    Save model checkpoints every `checkpoint_interval` epochs
    and keep a maximum of `max_checkpoints_to_keep` checkpoints.
    """

    def __init__(
        self,
        model: torchvision.models,
        optimizer: torch.optim,
        scheduler: torch.optim.lr_scheduler,
        checkpoint_interval: int,
        max_checkpoints_to_keep: int,
        checkpoint_dir: str,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.checkpoint_interval = checkpoint_interval
        self.max_checkpoints_to_keep = max_checkpoints_to_keep
        self.checkpoint_dir = checkpoint_dir
        self.last_checkpoint_path = os.path.join(self.checkpoint_dir, "last.pth")
        self.best_checkpoint_path = os.path.join(self.checkpoint_dir, "best.pth")
        self.lowest_val_loss = float("inf")
        self.pc = PathChecker()

    def __call__(self, **kwargs: any) -> None:
        epoch = kwargs["epoch"]
        val_loss = kwargs["val_loss"]
        val_accuracy = kwargs["val_accuracy"]

        # Save checkpoint every `checkpoint_interval` epochs
        if epoch % self.checkpoint_interval == 0:
            # Delete the existing checkpoint directory if it's the first epoch
            self.pc(path=self.checkpoint_dir, del_=True if epoch == 0 else False)

            # Construct the checkpoint name
            checkpoint_name = (
                f"{self.model.__class__.__name__}_"
                f"{epoch + 1}_"
                # f"{round(float(loss), 4)}_"
                f"{round(float(val_loss), 4)}_"
                f"{round(float(val_accuracy), 4)}.pth"
            )

            checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            }
            # torch.save(checkpoint, checkpoint_path)

            # Update the 'last.pth' checkpoint
            # try:
            #     os.remove(self.last_checkpoint_path)
            # except FileNotFoundError:
            #     try:
            #         shutil.copyfile(checkpoint_path, self.last_checkpoint_path)
            #         # * The checkpoint_path might be occupied.
            #     except PermissionError:
            #         pass

            # Save the checkpoint with the lowest val_loss as 'best.pth'
            if val_loss < self.lowest_val_loss:
                self.lowest_val_loss = val_loss
                torch.save(checkpoint, checkpoint_path)
                # try:
                #     os.remove(self.best_checkpoint_path)
                # except FileNotFoundError:
                #     shutil.copyfile(checkpoint_path, self.best_checkpoint_path)
                # except PermissionError:
                #     pass

            # Delete older checkpoints if more than max_checkpoints_to_keep but keep the best.pth
            if epoch >= self.max_checkpoints_to_keep:
                self._delete_old_checkpoints(self.checkpoint_dir)

    def _delete_old_checkpoints(self, checkpoint_dir: str) -> None:
        checkpoints = glob.glob(os.path.join(checkpoint_dir, "*.pth"))
        checkpoints = sorted(checkpoints, key=os.path.getmtime, reverse=True)
        checkpoints_to_delete = checkpoints[self.max_checkpoints_to_keep :]
        for checkpoint in checkpoints_to_delete:
            if (
                checkpoint != self.best_checkpoint_path
                and checkpoint != self.last_checkpoint_path
            ):
                os.remove(checkpoint)


class LogWriter:
    def __init__(self, log_file: str):
        self.log_file = log_file

    def __call__(self, *messages: any, printf: bool = False) -> None:
        time_stamp = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        caller_frame = inspect.stack()[1]
        caller_code = inspect.getframeinfo(caller_frame[0])
        # file_name = os.path.basename(__file__)

        log_message = f"[{time_stamp}-{os.path.split(caller_code.filename)[-1]}:{caller_code.lineno}]"
        log_message += ", ".join(str(msg) for msg in messages)

        print(log_message) if printf else None

        with open(self.log_file, "a+") as file:
            file.seek(0, 0)  # Move the file pointer to the beginning
            file.write(log_message + "\n")
