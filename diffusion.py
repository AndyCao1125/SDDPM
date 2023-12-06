import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import neuron, functional, surrogate, layer
import matplotlib.pyplot as plt


def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    out = torch.gather(v, index=t, dim=0).float()
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))

def extract2(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    out = torch.gather(v, index=t, dim=0)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


# class GaussianDiffusionTrainer(nn.Module):
#     def __init__(self, model, beta_1, beta_T, T):
#         super().__init__()

#         self.model = model
#         self.T = T

#         self.register_buffer(
#             'betas', torch.linspace(beta_1, beta_T, T).double())
#         alphas = 1. - self.betas
#         alphas_bar = torch.cumprod(alphas, dim=0)

#         # calculations for diffusion q(x_t | x_{t-1}) and others
#         self.register_buffer(
#             'sqrt_alphas_bar', torch.sqrt(alphas_bar))
#         self.register_buffer(
#             'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

#     def forward(self, x_0, y=None):
#         """
#         Algorithm 1.
#         """
#         t = torch.randint(self.T, size=(x_0.shape[0], ), device=x_0.device)
#         noise = torch.randn_like(x_0)
#         x_t = (
#             extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
#             extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)
#         #print(self.model(x_t, t).size())
#         #loss = F.mse_loss(self.model(x_t, t), noise, reduction='none')
#         print(y)
#         if y != None:
#             loss = F.mse_loss(self.model(x_t, t, torch.argmax(y)), noise, reduction='none')
#         else:
#             loss = F.mse_loss(self.model(x_t, t), noise, reduction='none')
#         return loss
        
class GaussianDiffusionTrainer(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()

        self.model = model
        self.T = T

        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

    def forward(self, x_0):
        """
        Algorithm 1.
        """
        t = torch.randint(self.T, size=(x_0.shape[0], ), device=x_0.device)
        noise = torch.randn_like(x_0)
        x_t = (
            extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
            extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)
        #print(self.model(x_t, t).size())
        #loss = F.mse_loss(self.model(x_t, t), noise, reduction='none')
        loss = F.mse_loss(self.model(x_t, t), noise, reduction='none')
        return loss


class LatentGaussianDiffusionTrainer(nn.Module):
    def __init__(self, model,vae, beta_1, beta_T, T):
        super().__init__()

        self.model = model
        self.vae = vae
        self.T = T

        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

    def forward(self,x_0):
        """
        Algorithm 1. Latent Diffusion
        """
        weight_dtype = torch.float32
        latents = self.vae.encode(
                x_0.to(dtype=weight_dtype)
            ).latent_dist.sample()
        latents = latents * 0.18215
        t = torch.randint(self.T, size=(latents.shape[0], ), device=latents.device)
        noise = torch.randn_like(latents)

        x_t = (
            extract(self.sqrt_alphas_bar, t, latents.shape) * latents +
            extract(self.sqrt_one_minus_alphas_bar, t, latents.shape) * noise)
        #print(self.model(x_t, t).size())
        #loss = F.mse_loss(self.model(x_t, t), noise, reduction='none')
        loss = F.mse_loss(self.model(x_t, t), noise, reduction='none')
        return loss
    

class GaussianDiffusionLogger(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()

        self.model = model
        self.T = T

        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

    
    def forward(self, x_0):
        '''
        Evaluate the loss through time
        '''

        print(f'{x_0.shape[0]} images start computing loss through time')
        t_list = torch.linspace(start=0, end=self.T-1, steps=self.T, dtype=torch.int64, device=x_0.device)
        # t_list = t_list.view(len(t_list))
        loss_list = []
        with torch.no_grad():
            for t in t_list:
                t = t.unsqueeze(0).repeat(x_0.shape[0])
                noise = torch.randn_like(x_0)
                x_t = (
                    extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
                    extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)
                loss = F.mse_loss(self.model(x_t, t), noise, reduction='mean')
                loss_list.append(loss.item())

                functional.reset_net(self.model)
        
        return t_list.cpu().numpy(), loss_list
        # fig = plt.figure()
        # plt.plot(t_list.cpu().numpy(),loss_list)
        # plt.title('Loss through Time')
        # plt.xlabel('t')
        # plt.ylabel('loss')

        # return fig



class GaussianDiffusionSampler(nn.Module):
    def __init__(self, model, beta_1, beta_T, T, img_size=32,
                 mean_type='xstart', var_type='fixedlarge',sample_type='ddpm',sample_steps=1000):
        print(mean_type)
        assert mean_type in ['xprev','xstart', 'epsilon']
        assert var_type in ['fixedlarge', 'fixedsmall']
        assert sample_type in ['ddpm', 'ddim','ddpm2']
        super().__init__()

        self.model = model
        self.T = T
        self.img_size = img_size
        self.mean_type = mean_type
        self.var_type = var_type
        self.sample_steps = sample_steps
        self.sample_type = sample_type

        self.ratio_raw = self.T/self.sample_steps
        self.t_list = [max(int(self.T-1-self.ratio_raw*x),0) for x in range(self.sample_steps)]
        logging.info(self.t_list)
        if self.t_list[-1] != 0:
            self.t_list.append(0)
        # print(self.t_list)

        # beta_t
        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        # alpha_t
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'one_minus_alphas_bar', (1.- alphas_bar))
        self.register_buffer(
            'sqrt_recip_one_minus_alphas_bar', 1./torch.sqrt(1.- alphas_bar))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_recip_alphas_bar', torch.sqrt(1. / alphas_bar))
        self.register_buffer(
            'sqrt_recipm1_alphas_bar', torch.sqrt(1. / alphas_bar - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.register_buffer(
            'posterior_var',
            self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))
        # below: log calculation clipped because the posterior variance is 0 at
        # the beginning of the diffusion chain
        self.register_buffer(
            'posterior_log_var_clipped',
            torch.log(
                torch.cat([self.posterior_var[1:2], self.posterior_var[1:]])))
        self.register_buffer(
            'posterior_mean_coef1',
            torch.sqrt(alphas_bar_prev) * self.betas / (1. - alphas_bar))
        self.register_buffer(
            'posterior_mean_coef2',
            torch.sqrt(alphas) * (1. - alphas_bar_prev) / (1. - alphas_bar))

    def q_mean_variance(self, x_0, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior
        q(x_{t-1} | x_t, x_0)
        """
        assert x_0.shape == x_t.shape
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_0 +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_log_var_clipped = extract(
            self.posterior_log_var_clipped, t, x_t.shape)
        return posterior_mean, posterior_log_var_clipped

    def predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        #print((extract(self.sqrt_recip_alphas_bar, t, x_t.shape)).dtype)
        return (
            extract(self.sqrt_recip_alphas_bar, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_bar, t, x_t.shape) * eps
        )

    def predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            extract(
                1. / self.posterior_mean_coef1, t, x_t.shape) * xprev -
            extract(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t,
                x_t.shape) * x_t
        )

    def p_mean_variance(self, x_t, t):
        # below: only log_variance is used in the KL computations
        # Mean parameterization
        if self.sample_type=='ddpm':
            model_log_var = {
                # for fixedlarge, we set the initial (log-)variance like so to
                # get a better decoder log likelihood
                'fixedlarge': torch.log(torch.cat([self.posterior_var[1:2],
                                                self.betas[1:]])),
                'fixedsmall': self.posterior_log_var_clipped,
            }[self.var_type]
            model_log_var = extract(model_log_var, t, x_t.shape)
            if self.mean_type == 'xprev':       # the model predicts x_{t-1}
                x_prev = self.model(x_t, t)
                x_0 = self.predict_xstart_from_xprev(x_t, t, xprev=x_prev)
                model_mean = x_prev
            elif self.mean_type == 'xstart':    # the model predicts x_0
                x_0 = self.model(x_t, t)
                model_mean, _ = self.q_mean_variance(x_0, x_t, t)
            elif self.mean_type == 'epsilon':   # the model predicts epsilon
                eps = self.model(x_t, t)
                x_0 = self.predict_xstart_from_eps(x_t, t, eps=eps)
                #print(x_0.dtype)
                x_0 = x_0.clamp(-1.,1.)
                model_mean, _ = self.q_mean_variance(x_0, x_t, t)
            else:
                raise NotImplementedError(self.mean_type)
            #(model_mean)
            x_0 = torch.clip(x_0, -1., 1.)
            
            functional.reset_net(self.model)

            return model_mean, model_log_var
        elif self.sample_type=='ddim':
            eps = self.model(x_t, t)
            a_t  = extract(self.sqrt_alphas_bar, t, x_t.shape)
            sigma_t = torch.sqrt(extract(self.one_minus_alphas_bar, t, x_t.shape))
            sigma_s = torch.sqrt(extract(self.one_minus_alphas_bar, t-self.ratio, x_t.shape))
            a_s  = extract(self.sqrt_alphas_bar, t-self.ratio, x_t.shape)

            a_ts = a_t/a_s
            beta_ts = sigma_t**2-a_ts**2*sigma_s**2

            x0_t = (x_t - eps*sigma_t)/(a_t)
            x0_t = x0_t.clamp(-1.,1.)
            eta = 0
            c_1 = eta * torch.sqrt((1-a_t.pow(2)/a_s.pow(2)) * (1-a_s.pow(2))/(1-a_t.pow(2)))
            c_2 = torch.sqrt((1-a_s.pow(2))-c_1.pow(2))
            mean = a_s * x0_t + c_2*eps + c_1 * torch.randn_like(x_t)
            functional.reset_net(self.model)
            return mean
        elif self.sample_type=='ddpm2':
            model_log_var = {
                # for fixedlarge, we set the initial (log-)variance like so to
                # get a better decoder log likelihood
                'fixedlarge': torch.log(torch.cat([self.posterior_var[1:2],
                                                self.betas[1:]])),
                'fixedsmall': self.posterior_log_var_clipped,
            }[self.var_type]
            model_log_var = extract(model_log_var, t, x_t.shape)

            eps = self.model(x_t, t)

            a_t  = extract2(self.sqrt_alphas_bar, t, x_t.shape)
            a_s  = extract2(self.sqrt_alphas_bar, t-self.ratio, x_t.shape)
            sigma_t = torch.sqrt(extract2(self.one_minus_alphas_bar, t, x_t.shape))
            sigma_s = torch.sqrt(extract2(self.one_minus_alphas_bar, t-self.ratio, x_t.shape))
            a_ts = a_t/a_s
            beta_ts = sigma_t**2-a_ts**2*sigma_s**2
            mean_x0 = ((1/a_t).float()*x_t - (sigma_t/a_t).float() * eps)
            mean_x0 = mean_x0.clamp(-1.,1.)
            mean_xs = (a_ts*sigma_s.pow(2)/(sigma_t.pow(2))).float() * x_t + (a_s*beta_ts/(sigma_t.pow(2))).float() * mean_x0

            functional.reset_net(self.model)
            return mean_xs, model_log_var
        else:
            pass

    def forward(self, x_T):
        x_t = x_T
        #for time_step in reversed(range(self.T)):
        for n_count1,time_step in enumerate(self.t_list):
            if n_count1 < len(self.t_list)-1:
                self.ratio = int(self.t_list[n_count1] - self.t_list[n_count1+1])

            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long)  * time_step
            if self.sample_type =='ddpm' or self.sample_type =='ddpm2':
                #print(x_t.dtype)
                # no noise when t == 0
                if time_step > 0:
                    mean, log_var = self.p_mean_variance(x_t=x_t, t=t)
                    noise = torch.randn_like(x_t)
                    x_t = mean + torch.exp(0.5 * log_var) * noise
                else:
                    eps = self.model(x_t, t)
                    a_ts = extract(self.sqrt_alphas_bar, t, x_t.shape)
                    sigma_t = torch.sqrt(extract(self.one_minus_alphas_bar, t, x_t.shape))
                    beta_ts = (1-a_ts**2)
                    x_0 = 1/a_ts*( x_t - eps * beta_ts/sigma_t)
                    return torch.clip(x_0, -1, 1)
            else:
                if time_step == 0: return x_t
                x_t = self.p_mean_variance(x_t=x_t, t=t)


# class GaussianDiffusionSampler(nn.Module):
#     def __init__(self, model, beta_1, beta_T, T, img_size=32,
#                  mean_type='xstart', var_type='fixedlarge',sample_type='ddpm',sample_steps=1000,cond=False):
#         print(mean_type)
#         assert mean_type in ['xprev','xstart', 'epsilon']
#         assert var_type in ['fixedlarge', 'fixedsmall']
#         assert sample_type in ['ddpm', 'ddim','ddpm2','analyticdpm']
#         super().__init__()
#         self.ms_pred = torch.load('./score/cifar10_ema_eps_400000.ms_eps.pth')
#         self.model = model
#         self.T = T
#         self.img_size = img_size
#         self.mean_type = mean_type
#         self.var_type = var_type
#         self.sample_steps = sample_steps
#         self.sample_type = sample_type
#         self.cond = cond

#         self.ratio_raw = self.T/self.sample_steps
#         self.t_list = [max(int(self.T-1-self.ratio_raw*x),0) for x in range(self.sample_steps)]
#         logging.info(self.t_list)
#         if self.t_list[-1] != 0:
#             self.t_list.append(0)
#         print(self.t_list)

#         # beta_t
#         self.register_buffer(
#             'betas', torch.linspace(beta_1, beta_T, T).double())
#         # alpha_t
#         alphas = 1. - self.betas
#         alphas_bar = torch.cumprod(alphas, dim=0)
#         alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]
#         self.register_buffer(
#             'sqrt_alphas_bar', torch.sqrt(alphas_bar))
#         self.register_buffer(
#             'one_minus_alphas_bar', (1.- alphas_bar))
#         self.register_buffer(
#             'sqrt_recip_one_minus_alphas_bar', 1./torch.sqrt(1.- alphas_bar))

#         # calculations for diffusion q(x_t | x_{t-1}) and others
#         self.register_buffer(
#             'sqrt_recip_alphas_bar', torch.sqrt(1. / alphas_bar))
#         self.register_buffer(
#             'sqrt_recipm1_alphas_bar', torch.sqrt(1. / alphas_bar - 1))

#         # calculations for posterior q(x_{t-1} | x_t, x_0)
#         self.register_buffer(
#             'posterior_var',
#             self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))
#         # below: log calculation clipped because the posterior variance is 0 at
#         # the beginning of the diffusion chain
#         self.register_buffer(
#             'posterior_log_var_clipped',
#             torch.log(
#                 torch.cat([self.posterior_var[1:2], self.posterior_var[1:]])))
#         self.register_buffer(
#             'posterior_mean_coef1',
#             torch.sqrt(alphas_bar_prev) * self.betas / (1. - alphas_bar))
#         self.register_buffer(
#             'posterior_mean_coef2',
#             torch.sqrt(alphas) * (1. - alphas_bar_prev) / (1. - alphas_bar))

#     def q_mean_variance(self, x_0, x_t, t):
#         """
#         Compute the mean and variance of the diffusion posterior
#         q(x_{t-1} | x_t, x_0)
#         """
#         assert x_0.shape == x_t.shape
#         posterior_mean = (
#             extract(self.posterior_mean_coef1, t, x_t.shape) * x_0 +
#             extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
#         )
#         posterior_log_var_clipped = extract(
#             self.posterior_log_var_clipped, t, x_t.shape)
#         return posterior_mean, posterior_log_var_clipped

#     def predict_xstart_from_eps(self, x_t, t, eps):
#         assert x_t.shape == eps.shape
#         #print((extract(self.sqrt_recip_alphas_bar, t, x_t.shape)).dtype)
#         return (
#             extract(self.sqrt_recip_alphas_bar, t, x_t.shape) * x_t -
#             extract(self.sqrt_recipm1_alphas_bar, t, x_t.shape) * eps
#         )

#     def predict_xstart_from_xprev(self, x_t, t, xprev):
#         assert x_t.shape == xprev.shape
#         return (  # (xprev - coef2*x_t) / coef1
#             extract(
#                 1. / self.posterior_mean_coef1, t, x_t.shape) * xprev -
#             extract(
#                 self.posterior_mean_coef2 / self.posterior_mean_coef1, t,
#                 x_t.shape) * x_t
#         )

#     def p_mean_variance(self, x_t, t):
#         # below: only log_variance is used in the KL computations
#         # Mean parameterization
#         if self.sample_type=='ddpm':
#             model_log_var = {
#                 # for fixedlarge, we set the initial (log-)variance like so to
#                 # get a better decoder log likelihood
#                 'fixedlarge': torch.log(torch.cat([self.posterior_var[1:2],
#                                                 self.betas[1:]])),
#                 'fixedsmall': self.posterior_log_var_clipped,
#             }[self.var_type]
#             model_log_var = extract(model_log_var, t, x_t.shape)
#             if self.mean_type == 'xprev':       # the model predicts x_{t-1}
#                 x_prev = self.model(x_t, t, self.label)
#                 x_0 = self.predict_xstart_from_xprev(x_t, t, xprev=x_prev)
#                 model_mean = x_prev
#             elif self.mean_type == 'xstart':    # the model predicts x_0
#                 x_0 = self.model(x_t, t, self.label)
#                 model_mean, _ = self.q_mean_variance(x_0, x_t, t)
#             elif self.mean_type == 'epsilon':   # the model predicts epsilon
#                 #eps = self.model(x_t, t, self.label)
#                 eps = self.model(x_t, t)
#                 x_0 = self.predict_xstart_from_eps(x_t, t, eps=eps)
#                 #print(x_0.dtype)
#                 x_0 = x_0.clamp(-1.,1.)
#                 model_mean, _ = self.q_mean_variance(x_0, x_t, t)
#             else:
#                 raise NotImplementedError(self.mean_type)
#             #(model_mean)
#             x_0 = torch.clip(x_0, -1., 1.)
            
#             functional.reset_net(self.model)

#             return model_mean, model_log_var
#         elif self.sample_type=='ddim':
#             eps = self.model(x_t, t, self.label)
#             a_t  = extract(self.sqrt_alphas_bar, t, x_t.shape)
#             sigma_t = torch.sqrt(extract(self.one_minus_alphas_bar, t, x_t.shape))
#             sigma_s = torch.sqrt(extract(self.one_minus_alphas_bar, t-self.ratio, x_t.shape))
#             a_s  = extract(self.sqrt_alphas_bar, t-self.ratio, x_t.shape)

#             a_ts = a_t/a_s
#             beta_ts = sigma_t**2-a_ts**2*sigma_s**2

#             x0_t = (x_t - eps*sigma_t)/(a_t)
#             x0_t = x0_t.clamp(-1.,1.)
#             eta = 0
#             c_1 = eta * torch.sqrt((1-a_t.pow(2)/a_s.pow(2)) * (1-a_s.pow(2))/(1-a_t.pow(2)))
#             c_2 = torch.sqrt((1-a_s.pow(2))-c_1.pow(2))
#             mean = a_s * x0_t + c_2*eps + c_1 * torch.randn_like(x_t)
#             functional.reset_net(self.model)
#             return mean
#         elif self.sample_type=='ddpm2':
#             model_log_var = {
#                 # for fixedlarge, we set the initial (log-)variance like so to
#                 # get a better decoder log likelihood
#                 'fixedlarge': torch.log(torch.cat([self.posterior_var[1:2],
#                                                 self.betas[1:]])),
#                 'fixedsmall': self.posterior_log_var_clipped,
#             }[self.var_type]
#             model_log_var = extract(model_log_var, t, x_t.shape)

#             eps = self.model(x_t, t, self.label)

#             a_t  = extract2(self.sqrt_alphas_bar, t, x_t.shape)
#             a_s  = extract2(self.sqrt_alphas_bar, t-self.ratio, x_t.shape)
#             sigma_t = torch.sqrt(extract2(self.one_minus_alphas_bar, t, x_t.shape))
#             sigma_s = torch.sqrt(extract2(self.one_minus_alphas_bar, t-self.ratio, x_t.shape))
#             a_ts = a_t/a_s
#             beta_ts = sigma_t**2-a_ts**2*sigma_s**2
#             mean_x0 = ((1/a_t).float()*x_t - (sigma_t/a_t).float() * eps)
#             mean_x0 = mean_x0.clamp(-1.,1.)
#             mean_xs = (a_ts*sigma_s.pow(2)/(sigma_t.pow(2))).float() * x_t + (a_s*beta_ts/(sigma_t.pow(2))).float() * mean_x0
#             return mean_xs, model_log_var
#         elif self.sample_type=='analyticdpm':
#             eps = self.model(x_t.float(), t, self.label)

#             a_t  = extract2(self.sqrt_alphas_bar, t, x_t.shape)
#             a_s  = extract2(self.sqrt_alphas_bar, t-self.ratio, x_t.shape)
#             sigma_t = torch.sqrt(extract2(self.one_minus_alphas_bar, t, x_t.shape))
#             sigma_s = torch.sqrt(extract2(self.one_minus_alphas_bar, t-self.ratio, x_t.shape))
#             a_ts = a_t/a_s
#             beta_ts = sigma_t**2-a_ts**2*sigma_s**2
#             mean_x0 = ((1/a_t).float()*x_t - (sigma_t/a_t).float() * eps)
#             mean_x0 = mean_x0.clamp(-1.,1.)
#             mean_xs = (a_ts*sigma_s.pow(2)/(sigma_t.pow(2))).float() * x_t + (a_s*beta_ts/(sigma_t.pow(2))).float() * mean_x0

#             sigma2_small = (sigma_s**2*beta_ts)/(sigma_t**2)
#             ms_pred_temp = ((torch.tensor(self.ms_pred[1+int(t[0].cpu())])).float()).to(x_t.device)

#             cov_x0_pred = sigma_t.pow(2)/a_t.pow(2) * (1-ms_pred_temp)
#             cov_x0_pred = cov_x0_pred.clamp(0., 1.)
#             offset = a_s.pow(2)*beta_ts.pow(2)/sigma_t.pow(4) * cov_x0_pred
#             model_var  = sigma2_small + offset
#             model_var  = model_var.clamp(0., 1.)
#             functional.reset_net(self.model)
#             return mean_xs,torch.log(model_var)
#         else:
#             pass

#     def forward(self, x_T, label=None):
#         self.label = label
#         x_t = x_T
#         #for time_step in reversed(range(self.T)):
#         for n_count1,time_step in enumerate(self.t_list):
#             if n_count1 < len(self.t_list)-1:
#                 self.ratio = int(self.t_list[n_count1] - self.t_list[n_count1+1])

#             t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long)  * time_step
#             if self.sample_type =='ddpm' or self.sample_type =='ddpm2' or self.sample_type =='analyticdpm':
#                 #print(x_t.dtype)
#                 # no noise when t == 0
#                 if time_step > 0:
#                     mean, log_var = self.p_mean_variance(x_t=x_t, t=t)
#                     noise = torch.randn_like(x_t)
#                     if time_step-self.ratio <= 0:
#                         var_threshold = (2 * 2. / 255. * (math.pi / 2.) ** 0.5) ** 2
#                         var = torch.exp(log_var)
#                         var = var.clamp(0., var_threshold)
#                         x_t = mean + var**0.5 * noise
#                         continue
#                     x_t = mean + torch.exp(0.5 * log_var) * noise
#                 else:
#                     #eps = self.model(x_t.float(), t, self.label)
#                     eps = self.model(x_t.float(), t)
#                     a_ts = extract(self.sqrt_alphas_bar, t, x_t.shape)
#                     sigma_t = torch.sqrt(extract(self.one_minus_alphas_bar, t, x_t.shape))
#                     beta_ts = (1-a_ts**2)
#                     x_0 = 1/a_ts*( x_t - eps * beta_ts/sigma_t)
#                     return torch.clip(x_0, -1, 1)
#             else:
#                 if time_step == 0: return x_t
#                 x_t = self.p_mean_variance(x_t=x_t, t=t)



class LatentGaussianDiffusionSampler(nn.Module):
    def __init__(self, model,vae, beta_1, beta_T, T, img_size=32,
                 mean_type='xstart', var_type='fixedlarge',sample_type='ddpm',sample_steps=1000):
        print(mean_type)
        assert mean_type in ['xprev','xstart', 'epsilon']
        assert var_type in ['fixedlarge', 'fixedsmall']
        assert sample_type in ['ddpm', 'ddim','ddpm2','analyticdpm']
        super().__init__()
        self.ms_pred = torch.load('./score/cifar10_ema_eps_400000.ms_eps.pth')
        self.model = model
        self.vae   = vae
        self.T = T
        self.img_size = img_size
        self.mean_type = mean_type
        self.var_type = var_type
        self.sample_steps = sample_steps
        self.sample_type = sample_type

        self.ratio_raw = self.T/self.sample_steps
        self.t_list = [max(int(self.T-1-self.ratio_raw*x),0) for x in range(self.sample_steps)]
        logging.info(self.t_list)
        if self.t_list[-1] != 0:
            self.t_list.append(0)
        print(self.t_list)

        # beta_t
        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        # alpha_t
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'one_minus_alphas_bar', (1.- alphas_bar))
        self.register_buffer(
            'sqrt_recip_one_minus_alphas_bar', 1./torch.sqrt(1.- alphas_bar))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_recip_alphas_bar', torch.sqrt(1. / alphas_bar))
        self.register_buffer(
            'sqrt_recipm1_alphas_bar', torch.sqrt(1. / alphas_bar - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.register_buffer(
            'posterior_var',
            self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))
        # below: log calculation clipped because the posterior variance is 0 at
        # the beginning of the diffusion chain
        self.register_buffer(
            'posterior_log_var_clipped',
            torch.log(
                torch.cat([self.posterior_var[1:2], self.posterior_var[1:]])))
        self.register_buffer(
            'posterior_mean_coef1',
            torch.sqrt(alphas_bar_prev) * self.betas / (1. - alphas_bar))
        self.register_buffer(
            'posterior_mean_coef2',
            torch.sqrt(alphas) * (1. - alphas_bar_prev) / (1. - alphas_bar))

    def q_mean_variance(self, x_0, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior
        q(x_{t-1} | x_t, x_0)
        """
        assert x_0.shape == x_t.shape
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_0 +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_log_var_clipped = extract(
            self.posterior_log_var_clipped, t, x_t.shape)
        return posterior_mean, posterior_log_var_clipped

    def predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        #print((extract(self.sqrt_recip_alphas_bar, t, x_t.shape)).dtype)
        return (
            extract(self.sqrt_recip_alphas_bar, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_bar, t, x_t.shape) * eps
        )

    def predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            extract(
                1. / self.posterior_mean_coef1, t, x_t.shape) * xprev -
            extract(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t,
                x_t.shape) * x_t
        )

    def p_mean_variance(self, x_t, t):
        if self.sample_type=='ddpm':
            model_log_var = {
                'fixedlarge': torch.log(torch.cat([self.posterior_var[1:2],
                                                self.betas[1:]])),
                'fixedsmall': self.posterior_log_var_clipped,
            }[self.var_type]
            model_log_var = extract(model_log_var, t, x_t.shape)
            if self.mean_type == 'xprev':       # the model predicts x_{t-1}
                with torch.no_grad():
                    x_prev = self.model(x_t, t, self.label)
                x_0 = self.predict_xstart_from_xprev(x_t, t, xprev=x_prev)
                model_mean = x_prev
            elif self.mean_type == 'xstart':    # the model predicts x_0
                with torch.no_grad():
                    x_0 = self.model(x_t, t, self.label)
                model_mean, _ = self.q_mean_variance(x_0, x_t, t)
            elif self.mean_type == 'epsilon':   # the model predicts epsilon
                with torch.no_grad():
                    eps = self.model(x_t, t, self.label)
                x_0 = self.predict_xstart_from_eps(x_t, t, eps=eps)
                #print(x_0.dtype)
                x_0 = x_0.clamp(-1.,1.)
                model_mean, _ = self.q_mean_variance(x_0, x_t, t)
            else:
                raise NotImplementedError(self.mean_type)
            x_0 = torch.clip(x_0, -1., 1.)
            
            functional.reset_net(self.model)
            return model_mean, model_log_var
        else:
            pass

    def forward(self, x_T):
        x_t = x_T
        #for time_step in reversed(range(self.T)):
        for n_count1,time_step in enumerate(self.t_list):
            if n_count1 < len(self.t_list)-1:
                self.ratio = int(self.t_list[n_count1] - self.t_list[n_count1+1])

            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long)  * time_step
            if self.sample_type =='ddpm' or self.sample_type =='ddpm2' or self.sample_type =='analyticdpm':
                #print(x_t.dtype)
                # no noise when t == 0
                if time_step > 0:
                    mean, log_var = self.p_mean_variance(x_t=x_t, t=t)
                    noise = torch.randn_like(x_t)
                    if time_step-self.ratio <= 0:
                        var_threshold = (2 * 2. / 255. * (math.pi / 2.) ** 0.5) ** 2
                        var = torch.exp(log_var)
                        var = var.clamp(0., var_threshold)
                        x_t = mean + var**0.5 * noise
                        continue
                    x_t = mean + torch.exp(0.5 * log_var) * noise
                else:
                    with torch.no_grad():
                        eps = self.model(x_t.float(), t, self.label)
                    a_ts = extract(self.sqrt_alphas_bar, t, x_t.shape)
                    sigma_t = torch.sqrt(extract(self.one_minus_alphas_bar, t, x_t.shape))
                    beta_ts = (1-a_ts**2)
                    x_0 = 1/a_ts*( x_t - eps * beta_ts/sigma_t)

                    weight_dtype = torch.float32
                    latents = 1 / 0.18215 * x_0.detach()
                    self.vae = self.vae.to(dtype=weight_dtype)
                    with torch.no_grad():
                        image = self.vae.decode(latents)['sample']
                    return torch.clip(image, -1, 1)
            else:
                if time_step == 0: return x_t
                x_t = self.p_mean_variance(x_t=x_t, t=t)