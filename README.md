# DLAV2024 PROJECT  : TEAM SPONGIFLEX

| Student Name  | Email                 | Sciper |
| ------------- | --------------------- | ------ |
| Martin Rollet | martin.rollet@epfl.ch | 300780 |
| Julien Ars    | julien.ars@epfl.ch    | 314545 |

> Github repo : https://github.com/merlebleue/DLAV2024-Spongiflex/tree/main

## Milestone 1 : 28th april 2024

### Introduction

This project is about prediting precise vehicle trajectory prediction, using the UniTraj framework from VITA lab @ EPFL. For this part, the objective is mainly familiarising ourselves with the framework. We use the provided `ptr` model, with the default configuration, evaluated using minADE6 error.

### Code

The code of the model can be consulted here : [ptr.py](motionnet/models/ptr/ptr.py)
The configuration is here : [ptr.yaml](motionnet/configs/method/ptr.yaml)

It consists of the provided code, with some parts we had to fill in. Here is our code :

- Function `temporal_attn_fn()` :
  
  ```python
  ######################## Your code here ########################
  for n in range(agents_emb.shape[2]): #per agent, assuming N is the number of agents
    agents_emb[:,:,n,:] = layer(agents_emb[:,:,n,:], src_key_padding_mask=agent_masks[:,:,n])
  ################################################################
  ```
- Function`social_attn_fn()` :
  
  ```python
  ######################## Your code here ########################
  for t in range(agents_emb.shape[0]): #per time step, assuming T is the mnumber of time steps
    agents_emb[t,:,:,:] = layer(agents_emb[t,:,:,:], src_key_padding_mask=agent_masks[:,t,:].permute(1,0))
  ################################################################
  ```
- In the function `_forward()`:
  
  ```python
  ######################## Your code here ########################
  # Apply temporal attention layers and then the social attention layers on agents_emb, each for L_enc times.
  for i in range(self.L_enc):
    agents_emb = self.temporal_attn_fn(agents_emb, opps_masks, self.temporal_attn_layers[i])
    agents_emb = self.social_attn_fn(agents_emb, opps_masks, self.social_attn_layers[i])
  ################################################################
  ```

### Results

Sadly, we encountered issues due to the presence of `nan` values in the dataset. This seemed to arrise from agents that do not exist in the sequence, but we did not manage to get rid of this error.