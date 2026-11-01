import math

import torch

from torch import nn

from torch.nn import functional as F

def timestep_embedding(t,dim=10):
    if dim<2:raise ValueError('Timestep embedding dimension must be at least two')
    half=dim//2;frequencies=torch.exp(-math.log(10000)*torch.arange(half,device=t.device,dtype=torch.float32)/half)
    phase=t[:,None].float()*frequencies[None,:]
    emb=torch.cat((phase.cos(),phase.sin()),dim=-1)
    return F.pad(emb,(0,dim%2))

class DenoisingMLP(nn.Module):
    def __init__(self,n_items,hidden=200,dropout=.5,normalize=False):
        super().__init__();self.normalize=normalize;self.time=nn.Linear(10,10);self.drop=nn.Dropout(dropout)
        self.encoder=nn.Linear(n_items+10,hidden);self.decoder=nn.Linear(hidden,n_items)
        for layer in (self.time,self.encoder,self.decoder):
            nn.init.xavier_normal_(layer.weight);nn.init.normal_(layer.bias,std=.001)
    def forward(self,x,t):
        if self.normalize:x=F.normalize(x,dim=-1)
        conditioned=torch.cat((self.drop(x),self.time(timestep_embedding(t))),dim=-1)
        return self.decoder(torch.tanh(self.encoder(conditioned)))

class SmallNoiseDiffusion(nn.Module):
    def __init__(self,steps=5,noise_scale=.001,noise_min=.0005,noise_max=.005):
        super().__init__();self.steps=steps
        if steps<2 or not 0<noise_min<noise_max or not 0<noise_scale*noise_max<1:raise ValueError('Invalid noise schedule')
        variance=torch.linspace(noise_scale*noise_min,noise_scale*noise_max,steps,dtype=torch.float64)
        intended=1-variance
        beta=1-intended/torch.cat((torch.ones(1,dtype=torch.float64),intended[:-1]));beta[0]=1e-5
        if not ((beta>0)&(beta<1)).all():raise ValueError('Invalid diffusion betas')
        cumulative=(1-beta).cumprod(0);previous=torch.cat((torch.ones(1,dtype=torch.float64),cumulative[:-1]))
        self.register_buffer('alpha_bar',cumulative)
        self.register_buffer('coef_clean',beta*previous.sqrt()/(1-cumulative))
        self.register_buffer('coef_noisy',(1-previous)*(1-beta).sqrt()/(1-cumulative))
        snr=cumulative/(1-cumulative);weights=torch.ones_like(snr);weights[1:]=snr[:-1]-snr[1:]
        self.register_buffer('loss_weights',weights/weights.mean())
    def corrupt(self,x,t,noise=None):
        if noise is None:noise=torch.randn_like(x)
        if noise.shape!=x.shape:raise ValueError('Noise shape mismatch')
        a=self.alpha_bar[t].to(x.dtype)[:,None]
        # Keep 1-alpha in float64 until after subtraction for very small noise.
        sigma=(1-self.alpha_bar[t]).sqrt().to(x.dtype)[:,None]
        return a.sqrt()*x+sigma*noise
    def posterior_mean(self,clean,noisy,t):
        return self.coef_clean[t].to(noisy.dtype)[:,None]*clean+self.coef_noisy[t].to(noisy.dtype)[:,None]*noisy
    def loss(self,model,clean,reweight=True):
        t=torch.randint(self.steps,(len(clean),),device=clean.device)
        estimate=model(self.corrupt(clean,t),t);mse=(estimate-clean).square().mean(1)
        if reweight:mse=mse*self.loss_weights[t].to(mse.dtype)
        return mse.mean()
    @torch.no_grad()
    def predict(self,model,observed):
        # Author-style sampling_steps=0: start at the observed vector and run
        # all reverse posterior means without adding sampling noise.
        value=observed
        for step in reversed(range(self.steps)):
            t=torch.full((len(value),),step,device=value.device,dtype=torch.long)
            value=self.posterior_mean(model(value,t),value,t)
        return value
