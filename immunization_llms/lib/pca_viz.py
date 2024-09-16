import torch 
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def save_pca(
    hiddens_one,
    hiddens_two,
    noise=None,
    step=0,
    loss_fn_name=''
):
    plt.clf()
    pca = PCA(n_components=2)
    # select last layer
    hiddens_one = hiddens_one[:, :, :]
    hiddens_two = hiddens_two[:, :, :]
    hiddens_one = hiddens_one.view(hiddens_one.size(0), -1)
    hiddens_two = hiddens_two.view(hiddens_two.size(0), -1)
    if noise is not None:
        noise = noise[:, :, :]
        noise = noise.view(noise.size(0), -1)
    
    if noise is not None:
        pca.fit(
            torch.cat((hiddens_one, hiddens_two, noise)).detach().cpu().numpy()
        )
    else:
        pca.fit(
            torch.cat((hiddens_one, hiddens_two)).detach().cpu().numpy()
        )
    hiddens_one_pca = pca.transform(hiddens_one.detach().cpu().numpy())
    hiddens_two_pca = pca.transform(hiddens_two.detach().cpu().numpy())
    if noise is not None:
        noise_pca = pca.transform(noise.detach().cpu().numpy())
        plt.scatter(noise_pca[:, 0], noise_pca[:, 1], label='noise')

    plt.scatter(hiddens_one_pca[:, 0], hiddens_one_pca[:, 1], label='harmful')
    plt.scatter(hiddens_two_pca[:, 0], hiddens_two_pca[:, 1], label='harmless')
    plt.legend()
    # save plot as image
    plt.savefig(f'./results/viz/{loss_fn_name}_pca_hidden_states_{step}.png')