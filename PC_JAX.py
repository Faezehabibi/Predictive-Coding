import jax
import numpy as np
from jax import random, jit, value_and_grad
import jax.numpy as jnp
from jax import device_get
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


# Rc = Rp[1:] + [x]
# Mu = [relu(u @ rp) for u, rp in zip(U, Rp)]
# Ec = [mu - rc for mu, rc in zip(Mu, Rc)]
# Ep = [jnp.zeros((dims[0], 1), jnp.float32)] + Ec[:-1]
# dMu = [d_relu(u @ rp) for u,rp in zip(U, Rp)]
# dMu_had_Ec = [dmu * ec for dmu, ec in zip(dMu, Ec)]
# dF_drp = [ep - u.T @ dmu_had_ec for ep, u, dmu_had_ec in zip(Ep, U, dMu_had_Ec)]

@jax.jit
def relu(x):
    return jnp.maximum(0, x)

@jax.jit
def d_relu(x):
    # jnp.where(x > 0, 1, 0) ?
    return jnp.where(x < 0, 0, 1)


@jax.jit
def children(Rp, x):        #   Rc
    return Rp[1:] + [x]

@jax.jit
def prediction(U, Rp):      #   Mu
    return [relu(jnp.dot(u, rp)) for u, rp in zip(U, Rp)]

@jax.jit
def err_c(U, Rp, x):        #   Ec
    return [rc - mu for mu, rc in zip(prediction(U, Rp), children(Rp, x))]

@jax.jit
def err_p(U, Rp, x):        #   Ep
    return [jnp.zeros((Rp[0].shape[0], 1), jnp.float32)] + err_c(U, Rp, x)[:-1]


@jax.jit
def dprediction(U, Rp):
    return [d_relu(jnp.dot(u, rp)) for u, rp in zip(U, Rp)]


def train(U, Rp, x, n_epoch=20):
    Rp_init = [rp * 0 for rp in Rp.copy()]
    # Rp = Rp_init
    for i in range(n_epoch):
        # Rp = Rp_init
        Rp = inference(U, Rp, x)
        Loss = loss(U, Rp, x)
        U = learning(U, Rp, x)
        Free_eng = free_energy(U, Rp, x)

        # if i % 500 == 0:
        print(i, "Free_eng:", Free_eng)
        print(i, "Loss:", Loss)
    return U, Rp
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# @jax.jit
def free_energy(U, Rp, x):
    Ec = err_c(U, Rp, x)
    free_e = 0
    for ec in Ec:
        free_e = free_e + jnp.dot(ec.T, ec)
    return free_e

@jax.jit
def loss(U, Rp, x):
    Loss = err_c(U, Rp, x)[-1]  #jnp.dot(Loss.T, Loss)
    return jnp.mean(jnp.sum(Loss*Loss,axis=1))


@jax.jit
def dF_dr(U, Rp, x):
    return [- ep + jnp.dot(u.T, ec) * d_relu(rp)  for ep, u, rp, ec in zip(err_p(U, Rp, x), U, Rp, err_c(U, Rp, x))]

@jax.jit
def dF_du(U, Rp, x):
    return [jnp.matmul(ec, relu(rp).T) for ec, rp in zip(err_c(U, Rp, x), Rp)]
    # return [jnp.matmul(ec, d_relu(rp).T) for ec, rp in zip(err_c(U, Rp, x), Rp)]


@jax.jit
def inference(U, Rp, x, lr=5e-2, n_infer=30):   #   new Rp
    for i in range(n_infer):
        Rp = [rp + lr * df_dr for rp, df_dr in zip(Rp, dF_dr(U, Rp, x))]
    return Rp


@jax.jit
def learning(U, Rp, x, lr=1e-2):        #   new U
    return [u + lr * df_du for u, df_du in zip(U, dF_du(U, Rp, x))]




def recon(U, Rp, x):
    Rp = inference(U, Rp, x)
    x_recon = prediction(U, Rp)[-1]

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(x.reshape(28, 28), cmap='gray')
    axs[0].set_title('Original Image')
    axs[0].axis('off')
    axs[1].imshow(x_recon.reshape(28, 28), cmap='gray')
    axs[1].set_title('Reconstructed Image')
    axs[1].axis('off')

    plt.show()

    return x_recon

# -----------------------------------------------------------------


lr_R = 0.01
lr_U = 1e-2
Seed = 0
# e_steps = 150
# num_iterations = 40000

loaded_array_np = np.load('img_new.npy')
x = jnp.array(loaded_array_np, dtype=jnp.float32)
x = x.reshape(x.shape[0]*x.shape[1], 1)

dims = [128, 128, x.shape[0]]

# scale_r = 1e-2
scale_U = 1e-1

key = random.PRNGKey(seed=Seed)
# key_r, key_U = random.split(key, 2)

Rp = [jnp.zeros((Di, 1), jnp.float32) for Di in dims[:-1]]                   # rL > r0
U = [scale_U*random.normal(key, (Dc, Di), jnp.float32) for Dc, Di in zip(dims[1:], dims[:-1])]
# U = [u/(np.linalg.norm(u, axis=0)+eps) for u in U]

U, Rp = train(U, Rp, x, n_epoch=20)
x_recon = recon(U, Rp, x)











