from utils.freq_imports import *
from utils.helper import create_dir
from sklearn.decomposition import PCA

def read_thetas(syn_data_dir):
    trial_ids = [1, 2, 3, 4, 5, 7, 8]
    thetas = []
    for trial_id in tqdm(trial_ids, dynamic_ncols=True):
        path_to_theta_dir = Path(f'{syn_data_dir}/trial_{trial_id:02d}/theta')
        theta_npys = sorted(path_to_theta_dir.glob('*.npy'))
        for theta_npy in theta_npys:
            theta = np.load(theta_npy)
            thetas.append(theta)
    thetas = np.array(thetas)   # (26000, 20)

    return thetas

def calculate_bounds(thetas):
    theta_min = np.amin(thetas, axis=0) # (20,)
    theta_max = np.amax(thetas, axis=0) # (20,)

    return theta_min, theta_max


def calculate_pca(thetas):
    # for PCA understanding refer https://www.youtube.com/watch?v=rng04VJxUt4&t=98s
    # 
    # sklearn PCA Algo:
    # mean_ = np.mean(X, axis=0)
    # X -= mean_
    # U, S, Vt = svd(X)
    # components_ = Vt  # (n_components, n_features); since we keep all components in this case, n_components = n_features
    # explained_variance_ = (S**2) / (n_samples-1)  # (n_components)
    #
    # pca_eigenvectors = components_.T  # this is U as per above video
    #
    # to project x_new (n,) onto PCA space: pca_eigenvectors.T @ (x_new - mu)
    # to project X_new (n_samples, n) onto PCA space: (X_new - mu[np.newaxis, :]) @ pca_eigenvectors
    #
    # as per convention in section 4.2 from https://lgg.epfl.ch/publications/2015/Htrack_ICP/paper.pdf
    # the PCA basis matrix, Pi = pca_eigenvectors
    # the diagonal matrix containing inverse of standard deviation of PCA basis, Sigma = np.diag(1/np.sqrt(explained_variance_))

    pca = PCA()
    pca.fit(thetas)
    mu = pca.mean_  # (20,)
    Pi = pca.components_.T   # (20, 20); each column is an eigenvector; each column is a component
    Sigma = np.diag(1/np.sqrt(pca.explained_variance_)) # (20, 20)

    return mu, Pi, Sigma

def main():
    syn_data_dir = Path('./output/syn_data')
    thetas = read_thetas(syn_data_dir)

    out_dir = f'./output/hand_model/prior'; create_dir(out_dir, False)

    theta_min, theta_max = calculate_bounds(thetas)
    out_bounds_dir = f'{out_dir}/bounds'; create_dir(out_bounds_dir, False)
    np.save(f'{out_bounds_dir}/theta_min.npy', theta_min)
    np.save(f'{out_bounds_dir}/theta_max.npy', theta_max)

    mu, Pi, Sigma = calculate_pca(thetas)
    out_pca_dir = f'{out_dir}/pca'; create_dir(out_pca_dir, False)
    np.save(f'{out_pca_dir}/mu.npy', mu)
    np.save(f'{out_pca_dir}/Pi.npy', Pi)
    np.save(f'{out_pca_dir}/Sigma.npy', Sigma)
    

if __name__ == "__main__":
    main()
