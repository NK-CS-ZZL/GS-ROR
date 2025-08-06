import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

def linear_to_srgb(linear):
    if isinstance(linear, torch.Tensor):
        """Assumes `linear` is in [0, 1], see https://en.wikipedia.org/wiki/SRGB."""
        eps = torch.finfo(torch.float32).eps
        srgb0 = 323 / 25 * linear
        srgb1 = (211 * torch.clamp(linear, min=eps)**(5 / 12) - 11) / 200
        return torch.where(linear <= 0.0031308, srgb0, srgb1)
    elif isinstance(linear, np.ndarray):
        eps = np.finfo(np.float32).eps
        srgb0 = 323 / 25 * linear
        srgb1 = (211 * np.maximum(eps, linear) ** (5 / 12) - 11) / 200
        return np.where(linear <= 0.0031308, srgb0, srgb1)
    else:
        raise NotImplementedError

# Positional encoding embedding. Code was taken from https://github.com/bmild/nerf.
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

def get_embedder(multires, input_dims=3):
    embed_kwargs = {
        'include_input': True,
        'input_dims': input_dims,
        'max_freq_log2': multires-1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    def embed(x, eo=embedder_obj): return eo.embed(x)
    return embed, embedder_obj.out_dim

def expected_sin(mean, var):
  """Compute the mean of sin(x), x ~ N(mean, var)."""
  return torch.exp(-0.5 * var) * torch.sin(mean)  # large var -> small value.

def IPE(mean,var,min_deg,max_deg):
    scales = 2**torch.arange(min_deg, max_deg)
    shape = mean.shape[:-1] + (-1,)
    scaled_mean = torch.reshape(mean[..., None, :] * scales[:, None], shape)
    scaled_var = torch.reshape(var[..., None, :] * scales[:, None]**2, shape)
    return expected_sin(torch.concat([scaled_mean, scaled_mean + 0.5 * np.pi], dim=-1), torch.concat([scaled_var] * 2, dim=-1))

def saturate_dot(v0,v1):
    return torch.clamp(torch.sum(v0*v1,dim=-1,keepdim=True),min=0.0,max=1.0)

def safe_l2_normalize(x, dim=None, eps=1e-6):
    return torch.nn.functional.normalize(x, p=2, dim=dim, eps=eps)

def get_camera_plane_intersection(pts, dirs, poses):
    """
    compute the intersection between the rays and the camera XoY plane
    :param pts:      pn,3
    :param dirs:     pn,3
    :param poses:    pn,3,4
    :return:
    """
    R, t = poses[:,:,:3], poses[:,:,3:]

    # transfer into human coordinate
    pts_ = (R @ pts[:,:,None] + t)[..., 0] # pn,3
    dirs_ = (R @ dirs[:,:,None])[..., 0]   # pn,3

    hits = torch.abs(dirs_[..., 2]) > 1e-4
    dirs_z = dirs_[:, 2]
    dirs_z[~hits] = 1e-4
    dist = -pts_[:, 2] / dirs_z
    inter = pts_ + dist.unsqueeze(-1) * dirs_
    return inter, dist, hits

def contraction(xyz_sampled, aabb):
    return (xyz_sampled - aabb[0]) / (aabb[1] - aabb[0])

def normalize_coord(xyz_sampled, aabb):
    return 2 * (xyz_sampled - aabb[0]) / (aabb[1] - aabb[0]) - 1

def to_sphere_dir(xyz_dir):
    theta = torch.arccos(xyz_dir[..., 2])
    phi = torch.arctan2(xyz_dir[..., 1], xyz_dir[..., 0])
    return phi[..., None], theta[..., None]

def to_cartesian_dir(sphere_dir):
    phi, theta = sphere_dir[..., 0], sphere_dir[..., 1]
    x = torch.sin(theta) * torch.cos(phi)
    y = torch.sin(theta) * torch.sin(phi)
    z = torch.cos(theta)
    return torch.cat([x, y, z], dim=-1)

def get_sphere_intersection(pts, dirs):
    dtx = torch.sum(pts*dirs,dim=-1,keepdim=True) # rn,1
    xtx = torch.sum(pts**2,dim=-1,keepdim=True) # rn,1
    dist = dtx ** 2 - xtx + 1
    assert torch.sum(dist<0)==0
    dist = -dtx + torch.sqrt(dist+1e-6) # rn,1
    return dist

# this function is borrowed from NeuS
def sample_pdf(bins, weights, n_samples, det=False):
    # This implementation is from NeRF
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    # Take uniform samples
    if det:
        u = torch.linspace(0. + 0.5 / n_samples, 1. - 0.5 / n_samples, steps=n_samples)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples])

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples

def get_weights(sdf_fun, inv_fun, z_vals, origins, dirs):
    points = z_vals.unsqueeze(-1) * dirs.unsqueeze(-2) + origins.unsqueeze(-2) # pn,sn,3
    inv_s = inv_fun(points[:, :-1, :])[..., 0]  # pn,sn-1
    pn, sn = points.shape[:2]
    sdf = sdf_fun(points.reshape(-1, 3)).reshape(pn, sn, -1)[..., 0]  # pn,sn

    prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]  # pn,sn-1
    prev_z_vals, next_z_vals = z_vals[:, :-1], z_vals[:, 1:]
    mid_sdf = (prev_sdf + next_sdf) * 0.5
    cos_val = (next_sdf - prev_sdf) / (next_z_vals - prev_z_vals + 1e-5)  # pn,sn-1
    surface_mask = (cos_val < 0)  # pn,sn-1
    cos_val = torch.clamp(cos_val, max=0)

    dist = next_z_vals - prev_z_vals  # pn,sn-1
    prev_esti_sdf = mid_sdf - cos_val * dist * 0.5  # pn, sn-1
    next_esti_sdf = mid_sdf + cos_val * dist * 0.5
    prev_cdf = torch.sigmoid(prev_esti_sdf * inv_s)
    next_cdf = torch.sigmoid(next_esti_sdf * inv_s)
    alpha = (prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5) * surface_mask.float()
    weights = alpha * torch.cumprod(torch.cat([torch.ones([alpha.shape[0], 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]
    mid_sdf[~surface_mask]=-1.0
    return weights, mid_sdf

def get_intersection(sdf_fun, inv_fun, pts, dirs, sn0=128, sn1=9):
    """
    :param sdf_fun:
    :param inv_fun:
    :param pts:    pn,3
    :param dirs:   pn,3
    :param sn0:
    :param sn1:
    :return:
    """
    inside_mask = torch.norm(pts, dim=-1) < 0.999 # left some margin
    pn, _ = pts.shape
    hit_z_vals = torch.zeros([pn, sn1-1])
    hit_weights = torch.zeros([pn, sn1-1])
    hit_sdf = -torch.ones([pn, sn1-1])
    if torch.sum(inside_mask)>0:
        pts = pts[inside_mask]
        dirs = dirs[inside_mask]
        max_dist = get_sphere_intersection(pts, dirs) # pn,1
        with torch.no_grad():
            z_vals = torch.linspace(0, 1, sn0) # sn0
            z_vals = max_dist * z_vals.unsqueeze(0) # pn,sn0
            weights, mid_sdf = get_weights(sdf_fun, inv_fun, z_vals, pts, dirs) # pn,sn0-1
            z_vals_new = sample_pdf(z_vals, weights, sn1, True) # pn,sn1
            weights, mid_sdf = get_weights(sdf_fun, inv_fun, z_vals_new, pts, dirs) # pn,sn1-1
            z_vals_mid = (z_vals_new[:,1:] + z_vals_new[:,:-1]) * 0.5

        hit_z_vals[inside_mask] = z_vals_mid
        hit_weights[inside_mask] = weights
        hit_sdf[inside_mask] = mid_sdf
    return hit_z_vals, hit_weights, hit_sdf

def extract_fields(bound_min, bound_max, resolution, query_func, batch_size=64, outside_val=1.0):
    N = batch_size
    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = torch.meshgrid(xs, ys, zs)
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)
                    val = query_func(pts).detach()
                    outside_mask = torch.norm(pts,dim=-1)>=1.0
                    val[outside_mask]=outside_val
                    val = val.reshape(len(xs), len(ys), len(zs)).cpu().numpy()
                    u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val
    return u

# def extract_geometry(bound_min, bound_max, resolution, threshold, query_func, outside_val=1.0):
#     u = extract_fields(bound_min, bound_max, resolution, query_func, outside_val=outside_val)
#     vertices, triangles = mcubes.marching_cubes(u, threshold)
#     b_max_np = bound_max.detach().cpu().numpy()
#     b_min_np = bound_min.detach().cpu().numpy()

#     vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
#     return vertices, triangles

import plyfile
import skimage.measure
def convert_sdf_samples_to_ply(
    pytorch_3d_sdf_tensor,
    ply_filename_out,
    bbox,
    level=0.5,
    offset=None,
    scale=None,
):
    """
    Convert sdf samples to .ply

    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to

    This function adapted from: https://github.com/RobotLocomotion/spartan
    """

    numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.numpy()
    voxel_size = list((bbox[1]-bbox[0]) / np.array(pytorch_3d_sdf_tensor.shape))

    verts, faces, normals, values = skimage.measure.marching_cubes(
        numpy_3d_sdf_tensor, level=level, spacing=voxel_size
    )
    faces = faces[...,::-1] # inverse face orientation

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = bbox[0,0] + verts[:, 0]
    mesh_points[:, 1] = bbox[0,1] + verts[:, 1]
    mesh_points[:, 2] = bbox[0,2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    # try writing to the ply file

    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(mesh_points[i, :])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    print("saving mesh to %s" % (ply_filename_out))
    ply_data.write(ply_filename_out)


def generalized_binomial_coeff(a, k):
    """Compute generalized binomial coefficients."""
    return np.prod(a - np.arange(k)) / np.math.factorial(k)


def assoc_legendre_coeff(l, m, k):
    """Compute associated Legendre polynomial coefficients.

      Returns the coefficient of the cos^k(theta)*sin^m(theta) term in the
      (l, m)th associated Legendre polynomial, P_l^m(cos(theta)).

      Args:
        l: associated Legendre polynomial degree.
        m: associated Legendre polynomial order.
        k: power of cos(theta).

      Returns:
        A float, the coefficient of the term corresponding to the inputs.
    """
    return ((-1)**m * 2**l * np.math.factorial(l) / np.math.factorial(k) /
          np.math.factorial(l - k - m) *
          generalized_binomial_coeff(0.5 * (l + k + m - 1.0), l))


def sph_harm_coeff(l, m, k):
  """Compute spherical harmonic coefficients."""
  return (np.sqrt(
      (2.0 * l + 1.0) * np.math.factorial(l - m) /
      (4.0 * np.pi * np.math.factorial(l + m))) * assoc_legendre_coeff(l, m, k))



def get_ml_array(deg_view):
    """Create a list with all pairs of (l, m) values to use in the encoding."""
    ml_list = []
    for i in range(deg_view):
        l = 2**i
        # Only use nonnegative m values, later splitting real and imaginary parts.
        for m in range(l + 1):
            ml_list.append((m, l))

    # Convert list into a numpy array.
    ml_array = np.array(ml_list).T
    return ml_array

def generate_ide_fn(deg_view):
    """Generate integrated directional encoding (IDE) function.

      This function returns a function that computes the integrated directional
      encoding from Equations 6-8 of arxiv.org/abs/2112.03907.

      Args:
        deg_view: number of spherical harmonics degrees to use.

      Returns:
        A function for evaluating integrated directional encoding.

      Raises:
        ValueError: if deg_view is larger than 5.
    """
    if deg_view > 5:
        raise ValueError('Only deg_view of at most 5 is numerically stable.')

    ml_array = get_ml_array(deg_view)
    l_max = 2**(deg_view - 1)

    # Create a matrix corresponding to ml_array holding all coefficients, which,
    # when multiplied (from the right) by the z coordinate Vandermonde matrix,
    # results in the z component of the encoding.
    mat = np.zeros((l_max + 1, ml_array.shape[1]))
    for i, (m, l) in enumerate(ml_array.T):
        for k in range(l - m + 1):
            mat[k, i] = sph_harm_coeff(l, m, k)

    mat = torch.from_numpy(mat.astype(np.float32)).cuda()
    ml_array = torch.from_numpy(ml_array.astype(np.float32)).cuda()

    def integrated_dir_enc_fn(xyz, kappa_inv):
        """Function returning integrated directional encoding (IDE).

        Args:
          xyz: [..., 3] array of Cartesian coordinates of directions to evaluate at.
          kappa_inv: [..., 1] reciprocal of the concentration parameter of the von
            Mises-Fisher distribution.

        Returns:
          An array with the resulting IDE.
        """
        x = xyz[..., 0:1]
        y = xyz[..., 1:2]
        z = xyz[..., 2:3]

        # Compute z Vandermonde matrix.
        vmz = torch.concat([z**i for i in range(mat.shape[0])], dim=-1)

        # Compute x+iy Vandermonde matrix.
        vmxy = torch.concat([(x + 1j * y)**m for m in ml_array[0, :]], dim=-1)

        # Get spherical harmonics.
        sph_harms = vmxy * torch.matmul(vmz, mat)

        # Apply attenuation function using the von Mises-Fisher distribution
        # concentration parameter, kappa.
        sigma = 0.5 * ml_array[1, :] * (ml_array[1, :] + 1)
        ide = sph_harms * torch.exp(-sigma * kappa_inv)

        # Split into real and imaginary parts and return
        return torch.concat([torch.real(ide), torch.imag(ide)], dim=-1)

    return integrated_dir_enc_fn

def get_lat_long():
    res = (1080, 1080*3)
    gy, gx = torch.meshgrid(torch.linspace(0.0 + 1.0 / res[0], 1.0 - 1.0 / res[0], res[0], device='cuda'),
                            torch.linspace(-1.0 + 1.0 / res[1], 1.0 - 1.0 / res[1], res[1], device='cuda'),
                            indexing='ij') # [h,w]

    sintheta, costheta = torch.sin(gy * np.pi), torch.cos(gy * np.pi)
    sinphi, cosphi = torch.sin(gx * np.pi), torch.cos(gx * np.pi)
    reflvec = torch.stack((sintheta * sinphi, costheta, -sintheta * cosphi), dim=-1)
    return reflvec
  
def components_from_spherical_harmonics(levels, directions):
    """
    levels: int
    directions : Tensor(batch, 3)
    Returns value for each component of spherical harmonics.

    Args:
        levels: Number of spherical harmonic levels to compute.
        directions: Spherical harmonic coefficients
    """
    num_components = levels**2
    components = torch.zeros((*directions.shape[:-1], num_components), device=directions.device)

    assert 1 <= levels <= 5, f"SH levels must be in [1,4], got {levels}"
    assert directions.shape[-1] == 3, f"Direction input should have three dimensions. Got {directions.shape[-1]}"

    x = directions[..., 0]
    y = directions[..., 1]
    z = directions[..., 2]

    xx = x**2
    yy = y**2
    zz = z**2

    # l0
    components[..., 0] = 0.28209479177387814

    # l1
    if levels > 1:
        components[..., 1] = 0.4886025119029199 * y
        components[..., 2] = 0.4886025119029199 * z
        components[..., 3] = 0.4886025119029199 * x

    # l2
    if levels > 2:
        components[..., 4] = 1.0925484305920792 * x * y
        components[..., 5] = 1.0925484305920792 * y * z
        components[..., 6] = 0.9461746957575601 * zz - 0.31539156525251999
        components[..., 7] = 1.0925484305920792 * x * z
        components[..., 8] = 0.5462742152960396 * (xx - yy)

    # l3
    if levels > 3:
        components[..., 9] = 0.5900435899266435 * y * (3 * xx - yy)
        components[..., 10] = 2.890611442640554 * x * y * z
        components[..., 11] = 0.4570457994644658 * y * (5 * zz - 1)
        components[..., 12] = 0.3731763325901154 * z * (5 * zz - 3)
        components[..., 13] = 0.4570457994644658 * x * (5 * zz - 1)
        components[..., 14] = 1.445305721320277 * z * (xx - yy)
        components[..., 15] = 0.5900435899266435 * x * (xx - 3 * yy)

    # l4
    if levels > 4:
        components[..., 16] = 2.5033429417967046 * x * y * (xx - yy)
        components[..., 17] = 1.7701307697799304 * y * z * (3 * xx - yy)
        components[..., 18] = 0.9461746957575601 * x * y * (7 * zz - 1)
        components[..., 19] = 0.6690465435572892 * y * z * (7 * zz - 3)
        components[..., 20] = 0.10578554691520431 * (35 * zz * zz - 30 * zz + 3)
        components[..., 21] = 0.6690465435572892 * x * z * (7 * zz - 3)
        components[..., 22] = 0.47308734787878004 * (xx - yy) * (7 * zz - 1)
        components[..., 23] = 1.7701307697799304 * x * z * (xx - 3 * yy)
        components[..., 24] = 0.6258357354491761 * (xx * (xx - 3 * yy) - yy * (3 * xx - yy))

    return components

class IdentityActivation(nn.Module):
    def forward(self, x): return x

class ExpActivation(nn.Module):
    def __init__(self, max_light=5.0):
        super().__init__()
        self.max_light=max_light

    def forward(self, x):
        return torch.exp(torch.clamp(x, max=self.max_light))

def make_predictor_2layer(feats_dim: object, output_dim: object, weight_norm: object = True, activation='sigmoid', exp_max=0.0, run_dim=128) -> object:
    if activation == 'sigmoid':
        activation = nn.Sigmoid()
    elif activation=='exp':
        activation = ExpActivation(max_light=exp_max)
    elif activation=='none':
        activation = IdentityActivation()
    elif activation=='relu':
        activation = nn.ReLU()
    elif activation=='softplus':
        activation = nn.Softplus()
    else:
        raise NotImplementedError

    if weight_norm:
        module=nn.Sequential(
            nn.utils.weight_norm(nn.Linear(feats_dim, run_dim)),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Linear(run_dim, output_dim)),
            activation,
        )
    else:
        module=nn.Sequential(
            nn.Linear(feats_dim, run_dim),
            nn.ReLU(),
            nn.Linear(run_dim, output_dim),
            activation,
        )
    return module
        
def make_predictor_3layer(feats_dim: object, output_dim: object, weight_norm: object = True, activation='sigmoid', exp_max=0.0, run_dim=128) -> object:
    if activation == 'sigmoid':
        activation = nn.Sigmoid()
    elif activation=='exp':
        activation = ExpActivation(max_light=exp_max)
    elif activation=='none':
        activation = IdentityActivation()
    elif activation=='relu':
        activation = nn.ReLU()
    elif activation=='softplus':
        activation = nn.Softplus()
    elif activation=='tanh':
        activation = nn.Tanh()
    else:
        raise NotImplementedError

    if weight_norm:
        module=nn.Sequential(
            nn.utils.weight_norm(nn.Linear(feats_dim, run_dim)),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Linear(run_dim, run_dim)),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Linear(run_dim, output_dim)),
            activation,
        )
    else:
        module=nn.Sequential(
            nn.Linear(feats_dim, run_dim),
            nn.ReLU(),
            nn.Linear(run_dim, run_dim),
            nn.ReLU(),
            nn.Linear(run_dim, output_dim),
            activation,
        )
    return module

def make_predictor_4layer(feats_dim: object, output_dim: object, weight_norm: object = True, activation='sigmoid', exp_max=0.0, run_dim = 256) -> object:
    if activation == 'sigmoid':
        activation = nn.Sigmoid()
    elif activation=='exp':
        activation = ExpActivation(max_light=exp_max)
    elif activation=='none':
        activation = IdentityActivation()
    elif activation=='relu':
        activation = nn.ReLU()
    else:
        raise NotImplementedError

    if weight_norm:
        module=nn.Sequential(
            nn.utils.weight_norm(nn.Linear(feats_dim, run_dim)),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Linear(run_dim, run_dim)),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Linear(run_dim, run_dim)),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Linear(run_dim, output_dim)),
            activation,
        )
    else:
        module=nn.Sequential(
            nn.Linear(feats_dim, run_dim),
            nn.ReLU(),
            nn.Linear(run_dim, run_dim),
            nn.ReLU(),
            nn.Linear(run_dim, run_dim),
            nn.ReLU(),
            nn.Linear(run_dim, output_dim),
            activation,
        )

    return module

def gaussian_kernel2D(kernel_size=5, sigma=1.0):
    x = torch.arange(-kernel_size // 2 + 1.0, kernel_size // 2 + 1.0)
    y = torch.arange(-kernel_size // 2 + 1.0, kernel_size // 2 + 1.0)
    xx, yy = torch.meshgrid(x, y)
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = (kernel[None, None, ...] / kernel.sum())
    return kernel

def gaussian_kernel1D(kernel_size=5, sigma=1.0):
    x = torch.arange(-kernel_size // 2 + 1.0, kernel_size // 2 + 1.0)
    kernel = torch.exp(-x**2 / (2 * sigma**2))
    kernel = (kernel[None, None, ...] / kernel.sum())
    return kernel     
    
class SHEncoding(nn.Module):
    def __init__(self, levels):
        super().__init__()
        if levels <= 0 or levels > 4:
            raise ValueError(f"Spherical harmonic encoding only supports 1 to 4 levels, requested {levels}")
        self.levels = levels        
        self.out_dims = self.levels**2
        
    def forward(self, inputs):
        return components_from_spherical_harmonics(self.levels, inputs)
        
class GaussianBlur2D(nn.Module):
    def __init__(self, kernel_size=5, sigma=1.0, stride=2, device='cuda'):
        super(GaussianBlur2D, self).__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.stride = stride
        self.register_buffer('kernel', gaussian_kernel2D(kernel_size, sigma).to(device))
        
    def forward(self, x):
        return F.conv2d(x, self.kernel, stride=self.stride, padding=self.kernel_size // 2)

class GaussianBlur1D(nn.Module):
    def __init__(self, kernel_size=5, sigma=1.0, stride=2, device='cuda'):
        super(GaussianBlur1D, self).__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.stride = stride
        self.register_buffer('kernel', gaussian_kernel1D(kernel_size, sigma).to(device))
        
    def forward(self, x):
        return F.conv1d(x, self.kernel, stride=self.stride, padding=self.kernel_size // 2)

class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        total_loss = 0.
        if count_h != 0:
            h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
            total_loss += h_tv/count_h
        if count_w != 0:
            w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
            total_loss += w_tv/count_w
        return self.TVLoss_weight*2*(total_loss)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]

class SingleVarianceNetwork(nn.Module):
    def __init__(self, init_val, activation='exp'):
        super(SingleVarianceNetwork, self).__init__()
        self.act = activation
        self.register_parameter('variance', nn.Parameter(torch.tensor(init_val)))

    def forward(self, x):
        if self.act=='exp':
            return torch.ones([*x.shape[:-1], 1]) * torch.exp(self.variance * 10.0)
        elif self.act=='linear':
            return torch.ones([*x.shape[:-1], 1]) * self.variance * 10.0
        elif self.act=='square':
            return torch.ones([*x.shape[:-1], 1]) * (self.variance * 10.0) ** 2
        else:
            raise NotImplementedError

    def warp(self, x, inv_s):
        return torch.ones([*x.shape[:-1], 1]) * inv_s

# This implementation is borrowed from nerf-pytorch: https://github.com/yenchenlin/nerf-pytorch
class NeRFNetwork(nn.Module):
    def __init__(self,
                 D=8,
                 W=256,
                 d_in=3,
                 d_in_view=3,
                 multires=0,
                 multires_view=0,
                 output_ch=4,
                 skips=[4],
                 use_viewdirs=False):
        super(NeRFNetwork, self).__init__()
        self.D = D
        self.W = W
        self.d_in = d_in
        self.d_in_view = d_in_view
        self.input_ch = 3
        self.input_ch_view = 3
        self.embed_fn = None
        self.embed_fn_view = None

        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn = embed_fn
            self.input_ch = input_ch

        if multires_view > 0:
            embed_fn_view, input_ch_view = get_embedder(multires_view, input_dims=d_in_view)
            self.embed_fn_view = embed_fn_view
            self.input_ch_view = input_ch_view

        self.skips = skips
        self.use_viewdirs = use_viewdirs

        self.pts_linears = nn.ModuleList(
            [nn.Linear(self.input_ch, W)] +
            [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_ch, W) for i in range(D - 1)])

        ### Implementation according to the official code release
        ### (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(self.input_ch_view + W, W // 2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W // 2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, input_pts, input_views):
        if self.embed_fn is not None:
            input_pts = self.embed_fn(input_pts)
        if self.embed_fn_view is not None:
            input_views = self.embed_fn_view(input_views)

        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            return alpha, rgb
        else:
            assert False

    def density(self, input_pts):
        if self.embed_fn is not None:
            input_pts = self.embed_fn(input_pts)

        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        alpha = self.alpha_linear(h)
        return alpha
    
class AlphaGridMask(torch.nn.Module):
    def __init__(self, device, aabb, alpha_volume):
        super(AlphaGridMask, self).__init__()
        self.device = device

        self.aabb=aabb.to(self.device)
        self.aabbSize = self.aabb[1] - self.aabb[0]
        self.invgridSize = 1.0/self.aabbSize * 2
        self.alpha_volume = alpha_volume.view(1,1,*alpha_volume.shape[-3:])
        self.gridSize = torch.LongTensor([alpha_volume.shape[-1],alpha_volume.shape[-2],alpha_volume.shape[-3]]).to(self.device)

    def sample_alpha(self, xyz_sampled):
        xyz_sampled = self.normalize_coord(xyz_sampled)
        alpha_vals = F.grid_sample(self.alpha_volume, xyz_sampled.view(1,-1,1,1,3), align_corners=True).view(-1)

        return alpha_vals

    def normalize_coord(self, xyz_sampled):
        return (xyz_sampled-self.aabb[0]) * self.invgridSize - 1



class InitSDFRegLoss:
    def __init__(self):
        pass

    def __call__(self, data_pr, step):
        reg_step = 1000
        # small_threshold = 0.1
        small_threshold = 0.4
        large_threshold = 1.05
        if 'sdf_vals' in data_pr and 'sdf_pts' in data_pr and step<reg_step:
            norm = torch.norm(data_pr['sdf_pts'], dim=-1)
            sdf = data_pr['sdf_vals']
            small_mask = norm<small_threshold
            if torch.sum(small_mask)>0:
                bounds = norm[small_mask] - small_threshold # 0-small_threshold -> 0
                # we want sdf - bounds < 0
                small_loss = torch.mean(torch.clamp(sdf[small_mask] - bounds, min=0.0))
                small_loss = torch.sum(small_loss) / (torch.sum(small_loss > 1e-5) + 1e-3)
            else:
                small_loss = torch.zeros(1)

            large_mask = norm > large_threshold
            if torch.sum(large_mask)>0:
                bounds = norm[large_mask] - large_threshold # 0 -> 1 - large_threshold
                # we want sdf - bounds > 0 => bounds - sdf < 0
                large_loss = torch.clamp(bounds - sdf[large_mask], min=0.0)
                large_loss = torch.sum(large_loss) / (torch.sum(large_loss > 1e-5) + 1e-3)
            else:
                large_loss = torch.zeros(1)

            anneal_weights = (np.cos((step / reg_step) * np.pi) + 1) / 2
            return {'loss_sdf_large': large_loss*anneal_weights, 'loss_sdf_small': small_loss*anneal_weights}
        else:
            return {}
