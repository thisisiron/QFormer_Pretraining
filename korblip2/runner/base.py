import os
import logging

import torch


class Runner:
    def __init__(self, **kwargs):
        pass

    def build_model(self, conf):
        pass
    
    def build_datasets(self, conf):
        for dataset_name in conf.datasets:
            dataset_config = conf.datasets[dataset_name]

    def train_step(self, model, inputs):
        output = model(inputs)
        loss_dict = {}
        for k, v in output.items():
            if "loss" in k:
                loss_dict[k] = v
        return output["loss"], loss_dict

    def train_epoch(
        self,
        model,
        dataloader,
        optimizer,
        lr_scheduler,
        total_epoch,
        scaler=None,
        start_epoch=0,
        log_freq=50,
        gradient_accum_steps=1,
    ):
        use_amp = scaler is not None

        logging.info(f"Start training for {total_epoch} epochs.")
        total_loss = 0.

        for epoch in range(start_epoch, total_epoch):
            logging.info(f"Start training epoch {epoch}.")

            lr_scheduler.step(epoch)
            
            for i, inputs in enumerate(dataloader):
                ### inputs


                with torch.cuda.amp.autocast(enabled=use_amp):
                    loss, loss_dict = self.train_step(model, inputs)
                    loss /= gradient_accum_steps

                # Perform backpropagation
                if use_amp:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                # Update optimizer every gradient_accum_steps iterations
                if (i + 1) % gradient_accum_steps == 0:
                    if use_amp:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad()

                total_loss += loss.item()

            avg_loss = total_loss / num_batches
            logging.info(f"Epoch [{epoch}/{total_epochs}] completed. Average Loss: {avg_loss:.4f}")

        logging.info("Training completed.")
        return
