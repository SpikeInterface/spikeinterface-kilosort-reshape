import numpy as np

from spikeinterface.sortingcomponents.matching.base import BaseTemplateMatching, _base_matching_dtype

try:
    import torch
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False



import random, string
from spikeinterface.core import get_global_tmp_folder
from spikeinterface.sortingcomponents.peak_selection import select_peaks
from spikeinterface.sortingcomponents.waveforms.temporal_pca import TemporalPCAProjection
from spikeinterface.sortingcomponents.tools import extract_waveform_at_max_channel
from spikeinterface.core.recording_tools import get_channel_distances
import pickle, json
from spikeinterface.core.node_pipeline import (
    run_node_pipeline,
    ExtractSparseWaveforms,
    PeakRetriever,
)

import gc

import numpy as np
import torch
from torch import sparse_coo_tensor as coo
from scipy.sparse import csr_matrix
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks
from scipy.cluster.vq import kmeans
from pathlib import Path

try:
    import faiss
except ImportError:
    print('KiloSortClustering requires faiss installed')

from tqdm import tqdm 

spike_dtype = _base_matching_dtype


from scipy.sparse import csr_matrix
import numpy as np



class KiloSortClustering:
    """
    This code is an adaptation from the code hosted on
    https://github.com/MouseLand/Kilosort/blob/main/kilosort/clustering_qr.py
    and implementing the KiloSort4 spike sorting algorithm, published here
    https://www.nature.com/articles/s41592-024-02232-7
    by Marius Patchitariu and collaborators

    This code can only used in the context of spikeinterface.sortingcomponents in the context of the nodepipeline, and mostly
    for benchmark purposes. Note that parameters are taken/adapted from KS, results should be similar but not exactly alike
s    
    """

    _default_params = {
        "n_svd": 5,
        "tmp_folder": None,
        "ms_before": 2,
        "ms_after": 2,
        "verbose": True,
        "debug": False,
        "engine": "torch",
        "torch_device": "cpu",
        "cluster_downsampling": 20,
        "n_nearest_channels" : 10
    }


    @classmethod
    def main_function(cls, recording, peaks, params, job_kwargs=dict()):
        
        if params['engine'] != 'torch':
            raise Exception('Not yet implemented!')

        fs = recording.get_sampling_frequency()
        ms_before = params["ms_before"]
        ms_after = params["ms_after"]
        nbefore = int(ms_before * fs / 1000.0)
        nafter = int(ms_after * fs / 1000.0)
        if params["tmp_folder"] is None:
            name = "".join(random.choices(string.ascii_uppercase + string.digits, k=8))
            tmp_folder = get_global_tmp_folder() / name
        else:
            tmp_folder = Path(params["tmp_folder"]).absolute()

        tmp_folder.mkdir(parents=True, exist_ok=True)

        # SVD for time compression
        few_peaks = select_peaks(
            peaks, recording=recording, method="uniform", n_peaks=10000, margin=(nbefore, nafter)
        )
        few_wfs = extract_waveform_at_max_channel(
            recording, few_peaks, ms_before=ms_before, ms_after=ms_after, **job_kwargs
        )
        wfs = few_wfs[:, :, 0]
        
        # Remove outliers
        valid = np.argmax(np.abs(wfs), axis=1) == nbefore
        wfs = wfs[valid]

        prototype = np.nanmedian(wfs, 0) 

        # Ensure all waveforms have a positive max
        wfs *= np.sign(wfs[:, nbefore])[:, np.newaxis]

        from sklearn.decomposition import TruncatedSVD

        tsvd = TruncatedSVD(params["n_svd"])
        tsvd.fit(wfs)

        model_folder = tmp_folder / "tsvd_model"

        model_folder.mkdir(exist_ok=True)
        with open(model_folder / "pca_model.pkl", "wb") as f:
            pickle.dump(tsvd, f)

        model_params = {
            "ms_before": ms_before,
            "ms_after": ms_after,
            "sampling_frequency": float(fs),
        }

        with open(model_folder / "params.json", "w") as f:
            json.dump(model_params, f)

        # features
        node0 = PeakRetriever(recording, peaks)

        node1 = ExtractSparseWaveforms(
            recording,
            parents=[node0],
            return_output=False,
            ms_before=ms_before,
            ms_after=ms_after,
        )

        ### KS is considering the closest n_nearest_channels for every channels, so we need
        ### here a small hack for the waveformextractor
        closest_channels = np.argsort(node1.channel_distance, axis=1)
        node1.neighbours_mask[:] = False
        for count, valid in enumerate(closest_channels):
            node1.neighbours_mask[count, valid[:params["n_nearest_channels"]]] = True
        node1.max_num_chans = np.max(np.sum(node1.neighbours_mask, axis=1))

        node2 = TemporalPCAProjection(
            recording, parents=[node0, node1], return_output=True, model_folder_path=model_folder
        )

        pipeline_nodes = [node0, node1, node2]

        features_folder = tmp_folder / "tsvd_features"
        features_folder.mkdir(exist_ok=True)

        _ = run_node_pipeline(
            recording,
            pipeline_nodes,
            job_kwargs,
            job_name="extracting features",
            gather_mode="npy",
            gather_kwargs=dict(exist_ok=True),
            folder=features_folder,
            names=["sparse_tsvd"],
        )

        from spikeinterface.sortingcomponents.clustering.tools import FeaturesLoader

        tF = FeaturesLoader.from_dict_or_folder(features_folder)["sparse_tsvd"]
        tF = np.swapaxes(tF, 1, 2)
        tF = torch.as_tensor(tF, device=params["torch_device"])

        xcup, ycup = recording.get_channel_locations()[:, 0], recording.get_channel_locations()[:, 1]
        xy = xy_up(xcup, ycup)

        ## This is a key difference between KS 4 and this implementation. Currently, the peaks are
        ## not realigned wrt to the upsampled grid created by KS. This could be done using the grid convolution
        ## algorithm here, but this would need some adaptation. This template_index are used to initialize the 
        ## clustering algorithm.

        # from spikeinterface.sortingcomponents.peak_localization import localize_peaks
        # locations = localize_peaks(recording, peaks, method='grid_convolution', 
        #             ms_before=ms_before, 
        #             ms_after=ms_after, 
        #             prototype=prototype, **job_kwargs)

        iclust_template = peaks['channel_index']
        sparse_mask = node1.neighbours_mask

        iC = np.zeros((sparse_mask.sum(1).max(), len(sparse_mask)), dtype='int32')
        for channel_ind in range(len(iC)):
            chan_inds, = np.nonzero(sparse_mask[channel_ind])
            iC[:len(chan_inds), channel_ind] = chan_inds

        iC = torch.as_tensor(iC, device=params["torch_device"])

        dmin  = np.median(np.diff(np.unique(ycup)))
        dminx = 32
        nskip = params['cluster_downsampling']
        ycent = y_centers(ycup, dmin)
        print(ycent)
        xcent = x_centers(xcup)
        nsp = len(peaks)
        Nchan = recording.get_num_channels()
        n_pca = params['n_svd']
        nearest_center, _, _ = get_nearest_centers(xy, xcent, ycent)

        clu = np.zeros(nsp, 'int32')
        Wall = torch.zeros((0, Nchan, n_pca))
        Nfilt = None
        nearby_chans_empty = 0
        nmax = 0
        prog = np.arange(len(ycent))
        
        try:
            for kk in tqdm(prog):
                for jj in np.arange(len(xcent)):
                    # Get data for all templates that were closest to this x,y center.
                    ii = kk + jj*ycent.size
                    if ii not in nearest_center:
                        # No templates are nearest to this center, skip it.
                        continue
                    ix = (nearest_center == ii)
                    Xd, ch_min, ch_max, igood  = get_data_cpu(
                        xy, iC, iclust_template, tF, ycent[kk], xcent[jj],
                        dmin=dmin, dminx=dminx, ix=ix
                        )

                    if Xd is None:
                        nearby_chans_empty += 1
                        continue
                    elif Xd.shape[0] < 1000:
                        iclust = torch.zeros((Xd.shape[0],))
                    else:
                        st0 = None
                        # find new clusters
                        iclust, iclust0, M, _ = cluster(
                            Xd, nskip=nskip, lam=1, seed=5, device=params["torch_device"]
                            )
                        
                        gc.collect()
                        torch.cuda.empty_cache()

                        xtree, tstat, my_clus = maketree(M, iclust, iclust0)

                        xtree, tstat = split(
                            Xd.numpy(), xtree, tstat,iclust, my_clus, meta=st0
                            )

                        iclust = new_clusters(iclust, my_clus, xtree, tstat)

                    clu[igood] = iclust + nmax
                    Nfilt = int(iclust.max() + 1)
                    nmax += Nfilt

                    # we need the new templates here         
                    W = torch.zeros((Nfilt, Nchan, params['n_svd']))
                    for j in range(Nfilt):
                        w = Xd[iclust==j].mean(0)
                        W[j, ch_min:ch_max, :] = torch.reshape(w, (-1, params['n_svd'])).cpu()
                    
                    Wall = torch.cat((Wall, W), 0)

        except:
            raise

        if nearby_chans_empty == len(ycent):
            raise ValueError(
                f'`get_data_cpu` never found suitable channels in `clustering_qr.run`.'
                f'\ndmin, dminx, and xcenter are: {dmin, dminx, xcup.mean()}'
            )

        if Wall.sum() == 0:
            # Wall is empty, unspecified reason
            raise ValueError(
                'Wall is empty after `clustering_qr.run`, cannot continue clustering.'
            )

        return np.unique(clu), clu
        


def neigh_mat(Xd, nskip=10, n_neigh=30):
    # Xd is spikes by PCA features in a local neighborhood
    # finding n_neigh neighbors of each spike to a subset of every nskip spike

    # subsampling the feature matrix 
    Xsub = Xd[::nskip]

    # n_samples is the number of spikes, dim is number of features
    n_samples, dim = Xd.shape

    # n_nodes are the # subsampled spikes
    n_nodes = Xsub.shape[0]

    # search is much faster if array is contiguous
    Xd = np.ascontiguousarray(Xd)
    Xsub = np.ascontiguousarray(Xsub)

    # exact neighbor search ("brute force")
    # results is dn and kn, kn is n_samples by n_neigh, contains integer indices into Xsub
    index = faiss.IndexFlatL2(dim)   # build the index
    index.add(Xsub)    # add vectors to the index
    _, kn = index.search(Xd, n_neigh)     # actual search

    # create sparse matrix version of kn with ones where the neighbors are
    # M is n_samples by n_nodes
    dexp = np.ones(kn.shape, np.float32)    
    rows = np.tile(np.arange(n_samples)[:, np.newaxis], (1, n_neigh)).flatten()
    M   = csr_matrix((dexp.flatten(), (rows, kn.flatten())),
                   (kn.shape[0], n_nodes))

    # self connections are set to 0!
    M[np.arange(0,n_samples,nskip), np.arange(n_nodes)] = 0

    return kn, M

def assign_iclust(rows_neigh, isub, kn, tones2, nclust, lam, m, ki, kj, device='cuda'):
    NN = kn.shape[0]

    ij = torch.vstack((rows_neigh.flatten(), isub[kn].flatten()))
    xN = coo(ij, tones2.flatten(), (NN, nclust))
    xN = xN.to_dense()

    if lam > 0:
        tones = torch.ones(len(kj), device = device)
        tzeros = torch.zeros(len(kj), device = device)
        ij = torch.vstack((tzeros, isub))    
        kN = coo(ij, tones, (1, nclust))
    
        xN = xN - lam/m * (ki.unsqueeze(-1) * kN.to_dense()) 
    
    iclust = torch.argmax(xN, 1)

    return iclust

def assign_isub(iclust, kn, tones2, nclust, nsub, lam, m,ki,kj, device='cuda'):
    n_neigh = kn.shape[1]
    cols = iclust.unsqueeze(-1).tile((1, n_neigh))
    iis = torch.vstack((kn.flatten(), cols.flatten()))

    xS = coo(iis, tones2.flatten(), (nsub, nclust))
    xS = xS.to_dense()

    if lam > 0:
        tones = torch.ones(len(ki), device = device)
        tzeros = torch.zeros(len(ki), device = device)
        ij = torch.vstack((tzeros, iclust))    
        kN = coo(ij, tones, (1, nclust))
        xS = xS - lam / m * (kj.unsqueeze(-1) * kN.to_dense())

    isub = torch.argmax(xS, 1)
    return isub


def Mstats(M, device='cuda'):
    m = M.sum()
    ki = np.array(M.sum(1)).flatten()
    kj = np.array(M.sum(0)).flatten()
    ki = m * ki/ki.sum()
    kj = m * kj/kj.sum()

    ki = torch.from_numpy(ki).to(device)
    kj = torch.from_numpy(kj).to(device)
    
    return m, ki, kj


def cluster(Xd, iclust = None, kn = None, nskip = 20, n_neigh = 10, nclust = 200, 
            seed = 1, niter = 200, lam = 0, device='cuda'):  

    if kn is None:
        kn, M = neigh_mat(Xd, nskip=nskip, n_neigh=n_neigh)
    m, ki, kj = Mstats(M, device=device)

    Xg = Xd.to(device)
    kn = torch.from_numpy(kn).to(device)
    n_neigh = kn.shape[1]
    NN, nfeat = Xg.shape
    nsub = (NN-1)//nskip + 1
    rows_neigh = torch.arange(NN, device=device).unsqueeze(-1).tile((1,n_neigh))
    tones2 = torch.ones((NN, n_neigh), device=device)

    if iclust is None:
        iclust_init =  kmeans_plusplus(Xg, niter=nclust, seed=seed, 
                                       device=device)
        iclust = iclust_init.clone()
    else:
        iclust_init = iclust.clone()
        
    for t in range(niter):
        # given iclust, reassign isub
        isub = assign_isub(iclust, kn, tones2, nclust, nsub, lam, m,
                           ki, kj,device=device)
        # given mu and isub, reassign iclust
        iclust = assign_iclust(rows_neigh, isub, kn, tones2, nclust, lam, m,
                               ki, kj, device=device)

    _, iclust = torch.unique(iclust, return_inverse=True)    
    nclust = iclust.max() + 1
    isub = assign_isub(iclust, kn, tones2, nclust , nsub, lam, m,ki,kj, device=device)

    iclust = iclust.cpu().numpy()
    isub = isub.cpu().numpy()

    return iclust, isub, M, iclust_init


def subsample_idx(n1, n2):
    """Get boolean mask and reverse mapping for evenly distributed subsample.
    
    Parameters
    ----------
    n1 : int
        Size of index. Index is assumed to be sequential and not contain any
        missing values (i.e. 0, 1, 2, ... n1-1).
    n2 : int
        Number of indices to remove to create a subsample. Removed indices are
        evenly spaced across 
    
    Returns
    -------
    idx : np.ndarray
        Boolean mask, True for indices to be included in the subset.
    rev_idx : np.ndarray
        Map between subset indices and their position in the original index.

    Examples
    --------
    >>> subsample_idx(6, 3)
    array([False,  True, False,  True,  True, False], dtype=bool),
    array([1, 3, 4], dtype=int64)

    """
    remove = np.round(np.linspace(0, n1-1, n2)).astype(int)
    idx = np.ones(n1, dtype=bool)
    idx[remove] = False
    # Also need to map the indices from the subset back to indices for
    # the full tensor.
    rev_idx = idx.nonzero()[0]

    return idx, rev_idx

def kmeans_plusplus(Xg, niter = 200, seed = 1, device='cuda'):
    # Xg is number of spikes by number of features.
    # We are finding cluster centroids and assigning each spike to a centroid.
    vtot = torch.norm(Xg, 2, dim=1)**2

    n1 = vtot.shape[0]
    if n1 > 2**24:
        # This subsampling step is just for the candidate spikes to be considered
        # as new centroids. Sometimes need to subsample v2 since
        # torch.multinomial doesn't allow more than 2**24 elements. We're just
        # using this to sample some spikes, so it's fine to not use all of them.
        n2 = n1 - 2**24   # number of spikes to remove before sampling
        idx, rev_idx = subsample_idx(n1, n2)
        subsample = True
    else:
        subsample = False

    torch.manual_seed(seed)
    np.random.seed(seed)

    ntry = 100  # number of candidate cluster centroids to test on each iteration
    NN, nfeat = Xg.shape
    # Need to store the spike features used for each cluster centroid (mu),
    # best variance explained so far for each spike (vexp0),
    # and the cluster assignment for each spike (iclust).
    mu = torch.zeros((niter, nfeat), device = device)
    vexp0 = torch.zeros(NN, device = device)
    iclust = torch.zeros((NN,), dtype = torch.int, device = device)

    # On every iteration we choose one new centroid to keep.
    # We track how well n centroids so far explain each spike.
    # We ask, if we were to add another centroid, which spikes would that
    # increase the explained variance for and by how much?
    # We use ntry candidates on each iteration.
    for j in range(niter):
        # v2 is the un-explained variance so far for each spike
        v2 = torch.relu(vtot - vexp0)

        # We sample ntry new candidate centroids based on how much un-explained variance they have
        # more unexplained variance makes it more likely to be selected
        # Only one of these candidates will be added this iteration. 
        if subsample:
            isamp = rev_idx[torch.multinomial(v2[idx], ntry)]
        else:
            isamp = torch.multinomial(v2, ntry)

        try:
            # The new centroids to be tested, sampled from the spikes in Xg.
            Xc = Xg[isamp]
            # Variance explained for each spike for the new centroids.
            vexp = 2 * Xg @ Xc.T - (Xc**2).sum(1)
            # Difference between variance explained for new centroids
            # and best explained variance so far across all iterations.
            # This gets relu-ed, since only the positive increases will actually
            # re-assign a spike to this new cluster
            dexp = torch.relu(vexp - vexp0.unsqueeze(1))
            # Sum all positive increases to determine additional explained variance
            # for each candidate centroid.
            vsum = dexp.sum(0)
            # Pick the candidate which increases explained variance the most 
            imax = torch.argmax(vsum)

            # For that centroid (Xc[imax]), determine which spikes actually get
            # more variance from it
            ix = dexp[:, imax] > 0

            iclust[ix] = j    # assign new cluster identity
            mu[j] = Xc[imax]  # spike features used as centroid for cluster j
            # Update variance explained for the spikes assigned to cluster j
            vexp0[ix] = vexp[ix, imax]

            # Delete large variables between iterations
            # to prevent excessive memory reservation.
            del(vexp)
            del(dexp)

        except torch.cuda.OutOfMemoryError:
            raise

    # NOTE: For very large datasets, we may end up needing to subsample Xg.
    # If the clustering above is done on a subset of Xg,
    # then we need to assign all Xgs here to get an iclust 
    # for ii in range((len(Xg)-1)//nblock +1):
    #     vexp = 2 * Xg[ii*nblock:(ii+1)*nblock] @ mu.T - (mu**2).sum(1)
    #     iclust[ii*nblock:(ii+1)*nblock] = torch.argmax(vexp, dim=-1)

    return iclust


def xy_up(xcup, ycup):
    xy = np.vstack((xcup, ycup))
    xy = torch.from_numpy(xy)
    return xy


def x_centers(xc, x_centers=None, seed=5330, sigma=0.5, bin_width=50):
    if x_centers is not None:
        # Use this as the input for k-means, either a number of centers
        # or initial guesses.
        approx_centers = x_centers
    else:
        # NOTE: This automated method does not work well for 2D array probes.
        #       We recommend specifying `x_centers` manually for that case.

        # Originally bin_width was set equal to `dminx`, but decided it's better
        # to not couple this behavior with that setting. A bin size of 50 microns
        # seems to work well for NP1 and 2, tetrodes, and 2D arrays. We can make
        # this a parameter later on if it becomes a problem.
        min_x = xc.min()
        max_x = xc.max()

        # Make histogram of x-positions with bin size roughly equal to dminx,
        # with a bit of padding on either end of the probe so that peaks can be
        # detected at edges.
        num_bins = int((max_x-min_x)/(bin_width)) + 4
        bins = np.linspace(min_x - bin_width*2, max_x + bin_width*2, num_bins)
        hist, edges = np.histogram(xc, bins=bins)
        # Apply smoothing to make peak-finding simpler.
        smoothed = gaussian_filter(hist, sigma=sigma)
        peaks, _ = find_peaks(smoothed)
        # peaks are indices, translate back to position in microns
        approx_centers = [edges[p] for p in peaks]

        # Use these as initial guesses for centroids in k-means to get
        # a more accurate value for the actual centers. If there's one or none,
        # just look for one centroid.
        if len(approx_centers) <= 1: approx_centers = 1

    centers, distortion = kmeans(xc, approx_centers, seed=seed)

    # TODO: Maybe use distortion to raise warning if it seems too large?
    # "The mean (non-squared) Euclidean distance between the observations passed
    #  and the centroids generated. Note the difference to the standard definition
    #  of distortion in the context of the k-means algorithm, which is the sum of
    #  the squared distances."

    # For example, could raise a warning if this is greater than dminx*2?
    # Most probes should satisfy that criteria.

    return centers


def y_centers(ycup, dmin):
    # TODO: May want to add the -dmin/2 in the future to center these, but
    #       this changes the results for testing so we need to wait until we can
    #       check it with simulations.
    centers = np.arange(ycup.min()+dmin-1, ycup.max()+dmin+1, 2*dmin)# - dmin/2

    return centers


def get_nearest_centers(xy, xcent, ycent):
    # Get positions of all grouping centers
    ycent_pos, xcent_pos = np.meshgrid(ycent, xcent)
    ycent_pos = torch.from_numpy(ycent_pos.flatten())
    xcent_pos = torch.from_numpy(xcent_pos.flatten())
    # Compute distances from templates
    center_distance = (
        (xy[0,:] - xcent_pos.unsqueeze(-1))**2
        + (xy[1,:] - ycent_pos.unsqueeze(-1))**2
        )
    # Add some randomness in case of ties
    center_distance += 1e-20*torch.rand(center_distance.shape)
    # Get flattened index of x-y center that is closest to template
    minimum_distance = torch.min(center_distance, 0).indices

    return minimum_distance, xcent_pos, ycent_pos


def get_data_cpu(xy, iC, PID, tF, ycenter, xcenter, dmin=20, dminx=32,
                 ix=None, merge_dim=True):
    PID =  torch.from_numpy(PID).long()
    
    y0 = ycenter # xy[1].mean() - ycenter
    x0 = xcenter #xy[0].mean() - xcenter

    #print(dmin, dminx)
    if ix is None:
        ix = torch.logical_and(
            torch.abs(xy[1] - y0) < dmin,
            torch.abs(xy[0] - x0) < dminx
            )
    #print(ix.nonzero()[:,0])
    igood = ix[PID].nonzero()[:,0]

    if len(igood)==0:
        return None, None,  None, None

    pid = PID[igood]
    data = tF[igood]
    nspikes, _, nfeatures = data.shape

    ichan = torch.unique(iC[:, ix])
    ch_min = torch.min(ichan)
    ch_max = torch.max(ichan)+1
    nchan = ch_max - ch_min
    dd = torch.zeros((nspikes, nchan, nfeatures))

    for j in ix.nonzero()[:, 0]:
        ij = torch.nonzero(pid==j)[:, 0]
        dd[ij.unsqueeze(-1), iC[:,j]-ch_min] = data[ij]

    if merge_dim:
        Xd = torch.reshape(dd, (nspikes, -1))
    else:
        # Keep channels and features separate
        Xd = dd

    return Xd, ch_min, ch_max, igood



def assign_clust(rows_neigh, iclust, kn, tones2, nclust):    
    NN = len(iclust)

    ij = torch.vstack((rows_neigh.flatten(), iclust[kn].flatten()))
    xN = coo(ij, tones2.flatten(), (NN, nclust))
    
    xN = xN.to_dense() 
    iclust = torch.argmax(xN, 1)

    return iclust

def assign_iclust0(Xg, mu):
    vv = Xg @ mu.T
    nm = (mu**2).sum(1)
    iclust = torch.argmax(2*vv-nm, 1)
    return iclust


####################### Functions taken for hierarchical clustering ######################



def cluster_qr(M, iclust, iclust0):
    NN = M.shape[0]
    nr = M.shape[1]

    nc = iclust.max()+1
    q = csr_matrix((np.ones(NN,), (iclust, np.arange(NN))), (nc, NN))
    r  = csr_matrix((np.ones(nr,), (np.arange(nr), iclust0)), (nr, nc))
    return q,r

def Mstats_hierarchical(M):
    m = M.sum()
    ki = np.array(M.sum(1)).flatten()
    kj = np.array(M.sum(0)).flatten()
    ki = m * ki/ki.sum()
    kj = m * kj/kj.sum()
    return m, ki, kj

def prepare(M, iclust, iclust0, lam=1):
    m, ki, kj = Mstats_hierarchical(M)
    q,r = cluster_qr(M, iclust, iclust0)
    cc = (q @ M @ r).toarray()
    nc = cc.shape[0]
    cneg = .001 + np.outer(q @ ki , kj @ r)/m
    return cc, cneg

def merge_reduce(cc, cneg, iclust):

    cc = cc + cc.T
    cneg = cneg + cneg.T

    crat = cc/cneg
    crat = crat -np.diag(np.diag(crat)) - np.eye(crat.shape[0])

    xtree, tstat = find_merges(crat, cc, cneg)

    my_clus = get_my_clus(xtree, tstat)
    return xtree, tstat, my_clus

def find_merges(crat, cc, cneg):
    nc = cc.shape[0]
    xtree = np.zeros((nc-1,3), 'int32')
    tstat = np.zeros((nc-1,3), 'float32')
    xnow = np.arange(nc)
    ntot = np.ones(nc,)

    for nmerges in range(nc-1):
        y, x = np.unravel_index(np.argmax(crat), cc.shape)
        lam = crat[y,x]

        m      = cc[y,x] + cc[x,x] + cc[x,y] + cc[y,x]
        ki = cc[x,x] + cc[x,y]
        kj = cc[y,y] + cc[y,x]
        cneg_l = .5 * (ki * kj + (m-ki) * (m-kj)) / m
        cpos_l = cc[y,x] + cc[x,y]
        M      = cpos_l / cneg_l

        cc[y]   = cc[y] + cc[x]
        cc[:,y] = cc[:,y] + cc[:,x]
        cc[x]   = -1
        cc[:,x] = -1
        cneg[y]   = cneg[y]   + cneg[x]
        cneg[:,y] = cneg[:,y] + cneg[:,x]

        crat[y] = cc[y]/cneg[y]
        crat[:,y] = crat[y]
        crat[y,y] = -1
        crat[x] = -1
        crat[:,x]=-1

        xtree[nmerges,:] = [xnow[x], xnow[y], nmerges + nc]
        tstat[nmerges,:] = [lam, ntot[x]+ntot[y], M]

        ntot[y] +=ntot[x]
        xnow[y] = nc+nmerges

    return xtree, tstat

def get_my_clus(xtree, tstat):
    nc = xtree.shape[0]+1
    my_clus = [[j] for j in range(nc)]
    for t in range(nc-1):
        new_clus = my_clus[xtree[t,1]].copy()
        new_clus.extend(my_clus[xtree[t,0]])
        my_clus.append(new_clus)
    return my_clus

def maketree(M, iclust, iclust0):
    cc, cneg        = prepare(M, iclust, iclust0, lam = 1)
    xtree, tstat, my_clus  = merge_reduce(cc, cneg, iclust)
    return xtree, tstat, my_clus


####################### Functions taken from swarmsplitter ######################

import numpy as np
from numba import njit
import math

def count_elements(kk, iclust, my_clus, xtree):
    n1 = np.isin(iclust, my_clus[xtree[kk, 0]]).sum()
    n2 = np.isin(iclust, my_clus[xtree[kk, 1]]).sum()
    return n1, n2

def check_split(Xd, kk, xtree, iclust, my_clus):
    ixy = np.isin(iclust, my_clus[xtree[kk, 2]])
    iclu = iclust[ixy]
    labels = 2*np.isin(iclu, my_clus[xtree[kk, 0]]) - 1

    Xs = Xd[ixy]
    Xs[:,-1] = 1

    w = np.ones((Xs.shape[0],1))
    w[labels>0] = np.mean(labels<0)
    w[labels<0] = np.mean(labels>0)

    CC = Xs.T @ (Xs * w)
    CC = CC + .01 * np.eye(CC.shape[0])
    b = np.linalg.solve(CC, labels @ (Xs * w))
    xproj = Xs @ b

    score = bimod_score(xproj)
    return xproj, score

def clean_tree(valid_merge, xtree, inode):
    ix = (xtree[:,2]==inode).nonzero()[0]
    if len(ix)==0:
        return
    valid_merge[ix] = 0
    clean_tree(valid_merge, xtree, xtree[ix, 0])
    clean_tree(valid_merge, xtree, xtree[ix, 1])
    return

def bimod_score(xproj):
    from scipy.ndimage import gaussian_filter1d
    xbin, _ = np.histogram(xproj, np.linspace(-2,2,400))
    xbin = gaussian_filter1d(xbin.astype('float32'), 4)

    imin = np.argmin(xbin[175:225])
    xmin = np.min(xbin[175:225])
    xm1  = np.max(xbin[:imin+175])
    xm2  = np.max(xbin[imin+175:])

    score = 1 - np.maximum(xmin/xm1, xmin/xm2)
    return score


def refractoriness(st1, st2):
    # compute goodness of st1, st2, and both

    is_refractory = True #check_CCG(st1, st2)[1]
    if is_refractory:
        criterion = 1 # never split
    else:
        criterion = 0
    return criterion

def split(Xd, xtree, tstat, iclust, my_clus, verbose = True, meta = None):
    xtree = np.array(xtree)

    kk = xtree.shape[0]-1
    nc = xtree.shape[0] + 1
    valid_merge = np.ones((nc-1,), 'bool')


    for kk in range(nc-2,-1,-1):
        if not valid_merge[kk]:
            continue

        ix1 = np.isin(iclust, my_clus[xtree[kk, 0]])
        ix2 = np.isin(iclust, my_clus[xtree[kk, 1]])

        criterion = 0
        score = np.NaN
        if criterion==0:
            # first mutation is global modularity
            if tstat[kk,0] < 0.2:
                criterion = -1


        if meta is not None and criterion==0:
            # second mutation is based on meta_data
            criterion = refractoriness(meta[ix1],meta[ix2])
            #criterion = 0
        
        if criterion==0:
            xproj, score = check_split(Xd, kk, xtree, iclust, my_clus)
            # third mutation is bimodality
            #xproj, score = check_split(Xd, kk, xtree, iclust, my_clus)
            criterion = 2 * (score <  .6) - 1

        if criterion==0:
            # fourth mutation is local modularity (not reachable)
            score = tstat[kk,-1]
            criterion = score > .15

        if verbose:
            n1,n2 = ix1.sum(), ix2.sum()
            #print('%3.0d, %6.0d, %6.0d, %6.0d, %2.2f,%4.2f, %2.2f'%(kk, n1, n2,n1+n2,
            #tstat[kk,0], tstat[kk,-1], score))

        if criterion==1:
            valid_merge[kk] = 0
            clean_tree(valid_merge, xtree, xtree[kk,0])
            clean_tree(valid_merge, xtree, xtree[kk,1])

    tstat = tstat[valid_merge]
    xtree = xtree[valid_merge]

    return xtree, tstat


def new_clusters(iclust, my_clus, xtree, tstat):

    if len(xtree)==0:
        return np.zeros_like(iclust)
         

    nc = xtree.max() + 1

    isleaf = np.zeros(2*nc-1,)
    isleaf[xtree[:,0]] = 1
    isleaf[xtree[:,1]] = 1
    isleaf[xtree[:,2]] = 0

    ind = np.nonzero(isleaf)[0]
    iclust1 = iclust.copy()
    for j in range(len(ind)):
        ix = np.isin(iclust, my_clus[ind[j]])
        iclust1[ix] = j
        xtree[xtree[:,0] == ind[j], 0] = j
        xtree[xtree[:,1] == ind[j], 1] = j


    return iclust1
