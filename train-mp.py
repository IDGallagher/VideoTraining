import os
import math
import torch.distributed
import wandb
import random
import logging
import inspect
import argparse
import datetime
import subprocess
import sys

from pathlib import Path
from tqdm.auto import tqdm
from einops import rearrange
from omegaconf import OmegaConf
from safetensors import safe_open
from typing import Dict, Optional, Tuple

import torch
import torchvision
import torch.nn.functional as F
import torch.distributed as dist

from torch.optim.swa_utils import AveragedModel
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel

import diffusers
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.models import UNet2DConditionModel
from diffusers.pipelines import StableDiffusionPipeline
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available

import transformers
from transformers import CLIPTextModel, CLIPTokenizer
from datasets import load_dataset

from dataset import make_dataset, make_dataloader
sys.path.append("./AnimateDiff/")
from animatediff.models.unet import UNet3DConditionModel
from animatediff.pipelines.pipeline_animation import AnimationPipeline
from animatediff.utils.util import save_videos_grid, zero_rank_print

# helpers

import time

def enumerate_report(seq, delta, growth=1.0):
    last = 0
    count = 0
    for count, item in enumerate(seq):
        now = time.time()
        if now - last > delta:
            last = now
            yield count, item, True
        else:
            yield count, item, False
        delta *= growth

def init_dist(launcher="slurm", backend='nccl', port=29500, **kwargs):
    """Initializes distributed environment."""
    if launcher == 'pytorch':

        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['WORLD_SIZE'] = '1'
        os.environ['RANK'] = '0'
        os.environ['MASTER_PORT'] = str(port)

        rank = int(os.environ['RANK'])
        num_gpus = torch.cuda.device_count()
        print(f"{num_gpus} GPUs found")
        # local_rank = rank % num_gpus
        local_rank = 0
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='gloo', **kwargs)
        
    elif launcher == 'slurm':
        proc_id = int(os.environ['SLURM_PROCID'])
        ntasks = int(os.environ['SLURM_NTASKS'])
        node_list = os.environ['SLURM_NODELIST']
        num_gpus = torch.cuda.device_count()
        local_rank = proc_id % num_gpus
        torch.cuda.set_device(local_rank)
        addr = subprocess.getoutput(
            f'scontrol show hostname {node_list} | head -n1')
        os.environ['MASTER_ADDR'] = addr
        os.environ['WORLD_SIZE'] = str(ntasks)
        os.environ['RANK'] = str(proc_id)
        port = os.environ.get('PORT', port)
        os.environ['MASTER_PORT'] = str(port)
        dist.init_process_group(backend=backend)
        zero_rank_print(f"proc_id: {proc_id}; local_rank: {local_rank}; ntasks: {ntasks}; node_list: {node_list}; num_gpus: {num_gpus}; addr: {addr}; port: {port}")
        
    else:
        raise NotImplementedError(f'Not implemented launcher type: `{launcher}`!')
    
    return local_rank



def main(
    name: str,
    use_wandb: bool,
    launcher: str,
    
    output_dir: str,
    pretrained_model_path: str,

    train_data: Dict,
    cfg_random_null_text: bool = True,
    cfg_random_null_text_ratio: float = 0.1,
    
    unet_checkpoint_path: str = "",
    unet_additional_kwargs: Dict = {},
    ema_decay: float = 0.9999,
    noise_scheduler_kwargs = None,
    
    max_train_epoch: int = -1,
    max_train_steps: int = 100,
    validation_steps: int = 100,
    validation_steps_tuple: Tuple = (-1,),

    learning_rate: float = 3e-5,
    scale_lr: bool = False,
    lr_warmup_steps: int = 0,
    lr_scheduler: str = "constant",

    trainable_modules: Tuple[str] = (None, ),
    # num_workers: int = 32,
    num_workers: int = 1,
    train_batch_size: int = 1,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_weight_decay: float = 1e-2,
    adam_epsilon: float = 1e-08,
    max_grad_norm: float = 1.0,
    gradient_accumulation_steps: int = 1,
    gradient_checkpointing: bool = False,
    checkpointing_epochs: int = 5,
    checkpointing_steps: int = -1,

    mixed_precision_training: bool = True,
    enable_xformers_memory_efficient_attention: bool = True,

    global_seed: int = 42,
    is_debug: bool = False,
):
    check_min_version("0.10.0.dev0")

    # Initialize distributed training
    local_rank      = init_dist(launcher=launcher)
    global_rank     = dist.get_rank()
    num_processes   = dist.get_world_size()
    is_main_process = global_rank == 0

    seed = global_seed + global_rank
    torch.manual_seed(seed)
    
    # Logging folder
    folder_name = "debug" if is_debug else name + datetime.datetime.now().strftime("-%Y-%m-%dT%H-%M-%S")
    output_dir = os.path.join(output_dir, folder_name)
    # if is_debug and os.path.exists(output_dir):
        # os.system(f"rm -rf {output_dir}")

    *_, config = inspect.getargvalues(inspect.currentframe())

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.DEBUG,
    )

    if is_main_process and (not is_debug) and use_wandb:
        run = wandb.init(project="animatediff", name=folder_name, config=config)

    # Handle the output folder creation
    if is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/samples", exist_ok=True)
        os.makedirs(f"{output_dir}/sanity_check", exist_ok=True)
        os.makedirs(f"{output_dir}/checkpoints", exist_ok=True)
        OmegaConf.save(config, os.path.join(output_dir, 'config.yaml'))
    
    # trainable_params = list(filter(lambda p: p.requires_grad, unet.parameters()))
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon,
    )

    if is_main_process:
        zero_rank_print(f"trainable params number: {len(trainable_params)}")
        zero_rank_print(f"trainable params scale: {sum(p.numel() for p in trainable_params) / 1e6:.3f} M")
    
    # Enable gradient checkpointing
    if gradient_checkpointing:
        # unet.enable_gradient_checkpointing()
        pass

    # Move models to GPU

    # Get the training dataset
    train_dataset = make_dataset(**train_data)
    train_dataloader = make_dataloader(train_dataset, batch_size=train_batch_size)

    # Get the training iteration
    if max_train_steps == -1:
        assert max_train_epoch != -1
        max_train_steps = max_train_epoch * len(train_dataset)
        
    if checkpointing_steps == -1:
        assert checkpointing_epochs != -1
        checkpointing_steps = checkpointing_epochs * len(train_dataset)

    if scale_lr:
        learning_rate = (learning_rate * gradient_accumulation_steps * train_batch_size * num_processes)

    # Scheduler
    lr_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
        num_training_steps=max_train_steps * gradient_accumulation_steps,
    )

    # Validation pipeline

    # validation_pipeline = AnimationPipeline(
    #     unet=unet, vae=vae, tokenizer=tokenizer, text_encoder=text_encoder, scheduler=noise_scheduler,
    # ).to("cuda")
    
    # validation_pipeline = StableDiffusionPipeline.from_pretrained(
    #     pretrained_model_path,
    #     unet=unet, vae=vae, tokenizer=tokenizer, text_encoder=text_encoder, scheduler=noise_scheduler, safety_checker=None,
    # )
    # validation_pipeline.enable_vae_slicing()

    # DistributedDataParallel warpper
    # unet.to(local_rank)
    # unet = DistributedDataParallel(unet, device_ids=[local_rank], output_device=local_rank)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataset) / gradient_accumulation_steps)
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # Train!
    total_batch_size = train_batch_size * num_processes * gradient_accumulation_steps

    if is_main_process:
        logging.info("***** Running training *****")
        logging.info(f"  Num examples = {len(train_dataset)}")
        logging.info(f"  Num Epochs = {num_train_epochs}")
        logging.info(f"  Instantaneous batch size per device = {train_batch_size}")
        logging.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logging.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
        logging.info(f"  Total optimization steps = {max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, max_train_steps), disable=not is_main_process)
    progress_bar.set_description("Steps")

    # Support mixed-precision training
    scaler = torch.cuda.amp.GradScaler() if mixed_precision_training else None

    for epoch in range(first_epoch, num_train_epochs):
        # train_dataloader.sampler.set_epoch(epoch)
        # unet.train()
        
        for step, batch, verbose in enumerate_report(train_dataloader, 5):
            if cfg_random_null_text:
                batch['text'] = [name if random.random() > cfg_random_null_text_ratio else "" for name in batch['text']]
                
            # Data batch sanity check
            if epoch == first_epoch and step == 0:
                pixel_values, texts = batch['pixel_values'].cpu(), batch['text']
                pixel_values = rearrange(pixel_values, "b f c h w -> b c f h w")
                for idx, (pixel_value, text) in enumerate(zip(pixel_values, texts)):
                    pixel_value = pixel_value[None, ...]
                    save_videos_grid(pixel_value, f"{output_dir}/sanity_check/{'-'.join(text.replace('/', '').split()[:10]) if not text == '' else f'{global_rank}-{idx}'}.gif", rescale=True)
                
                    
            ### >>>> Training >>>> ###
            
            # Generate image embeddings from video frames            
            pixel_values = batch["pixel_values"].to(local_rank)
            video_length = pixel_values.shape[1]
            with torch.no_grad():
                pixel_values = rearrange(pixel_values, "b f c h w -> (b f) c h w")
                # latents = vae.encode(pixel_values).latent_dist
                # latents = latents.sample()
                # latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)

            # Get the text embedding for conditioning
            with torch.no_grad():
                # prompt_ids = tokenizer(
                #     batch['text'], max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
                # ).input_ids.to(latents.device)
                # encoder_hidden_states = text_encoder(prompt_ids)[0]
                pass

            # Predict the noise residual and compute loss
            # Mixed-precision training
            with torch.cuda.amp.autocast(enabled=mixed_precision_training):
                # model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                # loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                loss = 0
                pass

            optimizer.zero_grad()

            # Backpropagate
            if mixed_precision_training:
                scaler.scale(loss).backward()
                """ >>> gradient clipping >>> """
                scaler.unscale_(optimizer)
                # torch.nn.utils.clip_grad_norm_(unet.parameters(), max_grad_norm)
                """ <<< gradient clipping <<< """
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                """ >>> gradient clipping >>> """
                # torch.nn.utils.clip_grad_norm_(unet.parameters(), max_grad_norm)
                """ <<< gradient clipping <<< """
                optimizer.step()

            lr_scheduler.step()
            progress_bar.update(1)
            global_step += 1
            
            ### <<<< Training <<<< ###
            
            # Wandb logging
            if is_main_process and (not is_debug) and use_wandb:
                wandb.log({"train_loss": loss.item()}, step=global_step)
                
            # Save checkpoint
            if is_main_process and (global_step % checkpointing_steps == 0 or step == len(train_dataset) - 1):
                save_path = os.path.join(output_dir, f"checkpoints")
                state_dict = {
                    "epoch": epoch,
                    "global_step": global_step,
                    # "state_dict": unet.state_dict(),
                }
                if step == len(train_dataset) - 1:
                    torch.save(state_dict, os.path.join(save_path, f"checkpoint-epoch-{epoch+1}.ckpt"))
                else:
                    torch.save(state_dict, os.path.join(save_path, f"checkpoint.ckpt"))
                logging.info(f"Saved state to {save_path} (global_step: {global_step})")
                
            # Periodically validation
            # if is_main_process and (global_step % validation_steps == 0 or global_step in validation_steps_tuple):
            #     samples = []
                
            #     generator = torch.Generator(device=latents.device)
            #     generator.manual_seed(global_seed)
                
            #     height = train_data.sample_size[0] if not isinstance(train_data.sample_size, int) else train_data.sample_size
            #     width  = train_data.sample_size[1] if not isinstance(train_data.sample_size, int) else train_data.sample_size

            #     prompts = validation_data.prompts[:2] if global_step < 1000 and (not image_finetune) else validation_data.prompts

            #     for idx, prompt in enumerate(prompts):
            #         if not image_finetune:
            #             sample = validation_pipeline(
            #                 prompt,
            #                 generator    = generator,
            #                 video_length = train_data.sample_n_frames,
            #                 height       = height,
            #                 width        = width,
            #                 **validation_data,
            #             ).videos
            #             save_videos_grid(sample, f"{output_dir}/samples/sample-{global_step}/{idx}.gif")
            #             samples.append(sample)
                        
            #         else:
            #             sample = validation_pipeline(
            #                 prompt,
            #                 generator           = generator,
            #                 height              = height,
            #                 width               = width,
            #                 num_inference_steps = validation_data.get("num_inference_steps", 25),
            #                 guidance_scale      = validation_data.get("guidance_scale", 8.),
            #             ).images[0]
            #             sample = torchvision.transforms.functional.to_tensor(sample)
            #             samples.append(sample)
                
            #     if not image_finetune:
            #         samples = torch.concat(samples)
            #         save_path = f"{output_dir}/samples/sample-{global_step}.gif"
            #         save_videos_grid(samples, save_path)
                    
            #     else:
            #         samples = torch.stack(samples)
            #         save_path = f"{output_dir}/samples/sample-{global_step}.png"
            #         torchvision.utils.save_image(samples, save_path, nrow=4)

            #     logging.info(f"Saved samples to {save_path}")
                
            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            
            if global_step >= max_train_steps:
                break
            
    dist.destroy_process_group()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   type=str, required=True)
    parser.add_argument("--launcher", type=str, choices=["pytorch", "slurm"], default="pytorch")
    parser.add_argument("--wandb",    action="store_true")
    # parser.add_argument("--pretrained_model_path", type=str, default="models/StableDiffusion/v1-5-pruned.safetensors",)
    args = parser.parse_args()

    name   = Path(args.config).stem
    config = OmegaConf.load(args.config)

    main(name=name, launcher=args.launcher, use_wandb=args.wandb, **config)
