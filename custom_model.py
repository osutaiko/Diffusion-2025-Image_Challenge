#!/usr/bin/env python3
"""
Template for implementing custom generative models
Students should create their own implementation by inheriting from the base classes.

This file provides skeleton code for implementing generative models.
Students need to implement the TODO sections in their own files.
"""

import torch
import numpy as np
from src.base_model import BaseScheduler, BaseGenerativeModel
from src.network import UNet


# ============================================================================
# GENERATIVE MODEL SKELETON
# ============================================================================

class CustomScheduler(BaseScheduler):
    """
    Custom Scheduler Skeleton
    
    This is a simple linear scheduler, similar to the original DDPM.
    """
    
    def __init__(self, num_train_timesteps: int = 1000, beta_start: float = 0.0001, beta_end: float = 0.02, **kwargs):
        super().__init__(num_train_timesteps, **kwargs)
        betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)

    def sample_timesteps(self, batch_size: int, device: torch.device):
        """
        Sample random timesteps for training.
        
        Returns:
            Tensor of shape (batch_size,) with timestep values
        """
        return torch.randint(0, self.num_train_timesteps, (batch_size,), device=device)
    
    def forward_process(self, data, noise, t):
        """
        Apply forward process to add noise to clean data.
        
        Args:
            data: Clean data tensor
            noise: Noise tensor
            t: Timestep tensor
            
        Returns:
            Noisy data at timestep t
        """
        sqrt_alphas_cumprod = self.alphas_cumprod[t].sqrt()
        sqrt_one_minus_alphas_cumprod = (1. - self.alphas_cumprod[t]).sqrt()
        
        # Reshape to (batch_size, 1, 1, 1) for broadcasting
        sqrt_alphas_cumprod = sqrt_alphas_cumprod.view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.view(-1, 1, 1, 1)
        
        return sqrt_alphas_cumprod * data + sqrt_one_minus_alphas_cumprod * noise
    
    def reverse_process_step(self, xt, pred, t, t_next):
        """
        Perform one step of the reverse (denoising) process.
        
        Args:
            xt: Current noisy data
            pred: Model prediction (predicted noise)
            t: Current timestep
            t_next: Next timestep
            
        Returns:
            Updated data at timestep t_next
        """
        alpha_t = self.alphas[t]
        alpha_cumprod_t = self.alphas_cumprod[t]
        beta_t = self.betas[t]

        # Reshape for broadcasting
        alpha_t = alpha_t.view(-1, 1, 1, 1)
        alpha_cumprod_t = alpha_cumprod_t.view(-1, 1, 1, 1)
        beta_t = beta_t.view(-1, 1, 1, 1)

        # Simple DDPM reverse step
        pred_x0 = (xt - torch.sqrt(1. - alpha_cumprod_t) * pred) / torch.sqrt(alpha_cumprod_t)
        
        if t_next[0] < 0:
            return pred_x0

        alpha_cumprod_t_next = self.alphas_cumprod[t_next].view(-1, 1, 1, 1)
        
        mu = (torch.sqrt(alpha_cumprod_t_next) * beta_t * pred_x0 + torch.sqrt(alpha_t) * (1. - alpha_cumprod_t_next) * xt) / (1. - alpha_cumprod_t)
        
        if t[0] > 0:
            variance = (1. - alpha_cumprod_t_next) / (1. - alpha_cumprod_t) * beta_t
            noise = torch.randn_like(xt)
            return mu + torch.sqrt(variance) * noise
        else:
            return mu

    def get_target(self, data, noise, t):
        """
        Get the target for model prediction (what the network should learn to predict).
        For DDPM, the target is the noise.
        """
        return noise


class CustomGenerativeModel(BaseGenerativeModel):
    """
    Custom Generative Model Skeleton
    
    This is a simple DDPM-style generative model.
    """
    
    def __init__(self, network, scheduler, **kwargs):
        super().__init__(network, scheduler, **kwargs)
    
    def compute_loss(self, data, noise, **kwargs):
        """
        Compute the training loss.
        
        Args:
            data: Clean data batch
            noise: Noise batch
            **kwargs: Additional arguments
            
        Returns:
            Loss tensor
        """
        t = self.scheduler.sample_timesteps(data.shape[0], device=data.device)
        xt = self.scheduler.forward_process(data, noise, t)
        pred = self.predict(xt, t)
        target = self.scheduler.get_target(data, noise, t)
        loss = torch.nn.functional.mse_loss(pred, target)
        return loss
    
    def predict(self, xt, t, **kwargs):
        """
        Make prediction given noisy data and timestep.
        
        Args:
            xt: Noisy data
            t: Timestep
            **kwargs: Additional arguments
            
        Returns:
            Model prediction (predicted noise)
        """
        return self.network(xt, t)
    
    def sample(self, shape, num_inference_timesteps=20, return_traj=False, verbose=False, **kwargs):
        """
        Generate samples from noise using the reverse process.
        
        Args:
            shape: Shape of samples to generate (batch_size, channels, height, width)
            num_inference_timesteps: Number of denoising steps (NFE)
            return_traj: Whether to return the full trajectory
            verbose: Whether to show progress
            **kwargs: Additional arguments
            
        Returns:
            Generated samples (or trajectory if return_traj=True)
        """
        xt = torch.randn(shape, device=self.device)
        traj = [xt]
        
        timesteps = torch.linspace(self.scheduler.num_train_timesteps - 1, 0, num_inference_timesteps, dtype=torch.long, device=self.device)

        for i in range(num_inference_timesteps):
            t = timesteps[i].expand(shape[0])
            t_next = timesteps[i+1] if i < num_inference_timesteps - 1 else torch.tensor([-1], device=self.device)
            t_next = t_next.expand(shape[0])

            pred = self.predict(xt, t)
            xt = self.scheduler.reverse_process_step(xt, pred, t, t_next)
            if return_traj:
                traj.append(xt)

        return traj if return_traj else xt


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def create_custom_model(device="cpu", **kwargs):
    """
    Example function to create a custom generative model.
    
    Students should modify this function to create their specific model.
    
    Args:
        device: Device to place model on
        **kwargs: Additional arguments that can be passed to network or scheduler
                  (e.g., num_train_timesteps, use_additional_condition for scalar conditions
                   like step size in Shortcut Models or end timestep in Consistency Trajectory Models, etc.)
    """
    
    # Create U-Net backbone with FIXED hyperparameters
    # DO NOT MODIFY THESE HYPERPARAMETERS
    network = UNet(
        ch=128,
        ch_mult=[1, 2, 2, 2],
        attn=[1],
        num_res_blocks=4,
        dropout=0.1,
        use_additional_condition=kwargs.get('use_additional_condition', False)
    )
    
    # Extract scheduler parameters with defaults
    num_train_timesteps = kwargs.get('num_train_timesteps', 1000)
    
    # Create your scheduler
    scheduler = CustomScheduler(num_train_timesteps=num_train_timesteps, **{k: v for k, v in kwargs.items() if k != 'num_train_timesteps'})
    
    # Create your model
    model = CustomGenerativeModel(network, scheduler, **kwargs)
    
    return model.to(device)
