# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.

import os
import numpy as np
import torch
import torch.nn as nn
import nvdiffrast.torch as dr
import imageio
import torch.nn.functional as F
#----------------------------------------------------------------------------
# Vector operations
#----------------------------------------------------------------------------

def dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sum(x*y, -1, keepdim=True)

def reflect(x: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    return 2*dot(x, n)*n - x

def length(x: torch.Tensor, eps: float =1e-20) -> torch.Tensor:
    return torch.sqrt(torch.clamp(dot(x,x), min=eps)) # Clamp to avoid nan gradients because grad(sqrt(0)) = NaN

def safe_normalize(x: torch.Tensor, eps: float =1e-20) -> torch.Tensor:
    return x / length(x, eps)

def to_hvec(x: torch.Tensor, w: float) -> torch.Tensor:
    return torch.nn.functional.pad(x, pad=(0,1), mode='constant', value=w)


#-----------------------------------------------------------------------------------
# Metrics (taken from jaxNerf source code, in order to replicate their measurements)
#
# https://github.com/google-research/google-research/blob/301451a62102b046bbeebff49a760ebeec9707b8/jaxnerf/nerf/utils.py#L266
#
#-----------------------------------------------------------------------------------

def mse_to_psnr(mse):
  """Compute PSNR given an MSE (we assume the maximum pixel value is 1)."""
  return -10. / np.log(10.) * np.log(mse)

def psnr_to_mse(psnr):
  """Compute MSE given a PSNR (we assume the maximum pixel value is 1)."""
  return np.exp(-0.1 * np.log(10.) * psnr)

#----------------------------------------------------------------------------
# Displacement texture lookup
#----------------------------------------------------------------------------

def get_miplevels(texture: np.ndarray) -> float:
    minDim = min(texture.shape[0], texture.shape[1])
    return np.floor(np.log2(minDim))

def tex_2d(tex_map : torch.Tensor, coords : torch.Tensor, filter='nearest') -> torch.Tensor:
    tex_map = tex_map[None, ...]    # Add batch dimension
    tex_map = tex_map.permute(0, 3, 1, 2) # NHWC -> NCHW
    tex = torch.nn.functional.grid_sample(tex_map, coords[None, None, ...] * 2 - 1, mode=filter, align_corners=False)
    tex = tex.permute(0, 2, 3, 1) # NCHW -> NHWC
    return tex[0, 0, ...]

#----------------------------------------------------------------------------
# Cubemap utility functions
#----------------------------------------------------------------------------

def cube_to_dir(s, x, y):
    if s == 0:   rx, ry, rz = torch.ones_like(x), -y, -x
    elif s == 1: rx, ry, rz = -torch.ones_like(x), -y, x
    elif s == 2: rx, ry, rz = x, torch.ones_like(x), y
    elif s == 3: rx, ry, rz = x, -torch.ones_like(x), -y
    elif s == 4: rx, ry, rz = x, -y, torch.ones_like(x)
    elif s == 5: rx, ry, rz = -x, -y, -torch.ones_like(x)
    return torch.stack((rx, ry, rz), dim=-1)

def latlong_to_cubemap(latlong_map, res):
    cubemap = torch.zeros(6, res[0], res[1], latlong_map.shape[-1], dtype=torch.float32, device='cuda')
    for s in range(6):
        gy, gx = torch.meshgrid(torch.linspace(-1.0 + 1.0 / res[0], 1.0 - 1.0 / res[0], res[0], device='cuda'), 
                                torch.linspace(-1.0 + 1.0 / res[1], 1.0 - 1.0 / res[1], res[1], device='cuda'),
                                # indexing='ij')
                                )
        v = safe_normalize(cube_to_dir(s, gx, gy))

        tu = torch.atan2(v[..., 0:1], -v[..., 2:3]) / (2 * np.pi) + 0.5
        tv = torch.acos(torch.clamp(v[..., 1:2], min=-1, max=1)) / np.pi
        texcoord = torch.cat((tu, tv), dim=-1)

        cubemap[s, ...] = dr.texture(latlong_map[None, ...], texcoord[None, ...], filter_mode='linear')[0]
    return cubemap

def latlong_to_cubemap2(latlong_map, res, trans_mat=None):
    cubemap = torch.zeros(6, res[0], res[1], latlong_map.shape[-1], dtype=torch.float32, device='cuda')
    for s in range(6):
        gy, gx = torch.meshgrid(torch.linspace(-1.0 + 1.0 / res[0], 1.0 - 1.0 / res[0], res[0], device='cuda'), 
                                torch.linspace(-1.0 + 1.0 / res[1], 1.0 - 1.0 / res[1], res[1], device='cuda'),
                                # indexing='ij')
                                )
        if trans_mat is None:
            # for glossy
            transform_mat = torch.tensor([[0, 0, -1], [0, 1, 0], [1, 0, 0]]).cuda().float()
        else:
            transform_mat = trans_mat.cuda().float()
        v = safe_normalize(cube_to_dir(s, gx, gy)) @ transform_mat

        tu = torch.atan2(v[..., 0:1], -v[..., 2:3]) / (2 * np.pi) + 0.5
        tv = torch.acos(torch.clamp(v[..., 1:2], min=-1, max=1)) / np.pi
        texcoord = torch.cat((tu, tv), dim=-1)

        cubemap[s, ...] = dr.texture(latlong_map[None, ...], texcoord[None, ...], filter_mode='linear')[0]
    return cubemap

def cubemap_to_latlong(cubemap, res):
    gy, gx = torch.meshgrid(torch.linspace( 0.0 + 1.0 / res[0], 1.0 - 1.0 / res[0], res[0], device='cuda'), 
                            torch.linspace(-1.0 + 1.0 / res[1], 1.0 - 1.0 / res[1], res[1], device='cuda'),
                            # indexing='ij')
                            )
    
    sintheta, costheta = torch.sin(gy*np.pi), torch.cos(gy*np.pi)
    sinphi, cosphi     = torch.sin(gx*np.pi), torch.cos(gx*np.pi)
    
    reflvec = torch.stack((
        sintheta*sinphi, 
        costheta, 
        -sintheta*cosphi
        ), dim=-1)
    return dr.texture(cubemap[None, ...], reflvec[None, ...].contiguous(), filter_mode='linear', boundary_mode='cube')[0]

def cubemap_to_latlong2(cubemap, res):
    gy, gx = torch.meshgrid(torch.linspace( 0.0 + 1.0 / res[0], 1.0 - 1.0 / res[0], res[0], device='cuda'), 
                            torch.linspace(-1.0 + 1.0 / res[1], 1.0 - 1.0 / res[1], res[1], device='cuda'),
                            # indexing='ij')
                            )
    
    sintheta, costheta = torch.sin(gx*np.pi), torch.cos(gx*np.pi)
    sinphi, cosphi     = torch.sin(gy*np.pi), torch.cos(gy*np.pi)
    
    reflvec = torch.stack((
        sintheta*sinphi, 
        costheta, 
        -sintheta*cosphi
        ), dim=-1)
    return dr.texture(cubemap[None, ...], reflvec[None, ...].contiguous(), filter_mode='linear', boundary_mode='cube')[0]

#----------------------------------------------------------------------------
# Image scaling
#----------------------------------------------------------------------------

def scale_img_hwc(x : torch.Tensor, size, mag='bilinear', min='area') -> torch.Tensor:
    return scale_img_nhwc(x[None, ...], size, mag, min)[0]

def scale_img_nhwc(x  : torch.Tensor, size, mag='bilinear', min='area') -> torch.Tensor:
    assert (x.shape[1] >= size[0] and x.shape[2] >= size[1]) or (x.shape[1] < size[0] and x.shape[2] < size[1]), "Trying to magnify image in one dimension and minify in the other"
    y = x.permute(0, 3, 1, 2) # NHWC -> NCHW
    if x.shape[1] > size[0] and x.shape[2] > size[1]: # Minification, previous size was bigger
        y = torch.nn.functional.interpolate(y, size, mode=min)
    else: # Magnification
        if mag == 'bilinear' or mag == 'bicubic':
            y = torch.nn.functional.interpolate(y, size, mode=mag, align_corners=True)
        else:
            y = torch.nn.functional.interpolate(y, size, mode=mag)
    return y.permute(0, 2, 3, 1).contiguous() # NCHW -> NHWC

def avg_pool_nhwc(x  : torch.Tensor, size) -> torch.Tensor:
    y = x.permute(0, 3, 1, 2) # NHWC -> NCHW
    y = torch.nn.functional.avg_pool2d(y, size)
    return y.permute(0, 2, 3, 1).contiguous() # NCHW -> NHWC

#----------------------------------------------------------------------------
# Behaves similar to tf.segment_sum
#----------------------------------------------------------------------------

def segment_sum(data: torch.Tensor, segment_ids: torch.Tensor) -> torch.Tensor:
    num_segments = torch.unique_consecutive(segment_ids).shape[0]

    # Repeats ids until same dimension as data
    if len(segment_ids.shape) == 1:
        s = torch.prod(torch.tensor(data.shape[1:], dtype=torch.int64, device='cuda')).long()
        segment_ids = segment_ids.repeat_interleave(s).view(segment_ids.shape[0], *data.shape[1:])

    assert data.shape == segment_ids.shape, "data.shape and segment_ids.shape should be equal"

    shape = [num_segments] + list(data.shape[1:])
    result = torch.zeros(*shape, dtype=torch.float32, device='cuda')
    result = result.scatter_add(0, segment_ids, data)
    return result

#----------------------------------------------------------------------------
# Matrix helpers.
#----------------------------------------------------------------------------

def fovx_to_fovy(fovx, aspect):
    return np.arctan(np.tan(fovx / 2) / aspect) * 2.0

def focal_length_to_fovy(focal_length, sensor_height):
    return 2 * np.arctan(0.5 * sensor_height / focal_length)

# Reworked so this matches gluPerspective / glm::perspective, using fovy
def perspective(fovy=0.7854, aspect=1.0, n=0.1, f=1000.0, device=None):
    y = np.tan(fovy / 2)
    return torch.tensor([[1/(y*aspect),    0,            0,              0], 
                         [           0, 1/-y,            0,              0], 
                         [           0,    0, -(f+n)/(f-n), -(2*f*n)/(f-n)], 
                         [           0,    0,           -1,              0]], dtype=torch.float32, device=device)

# Reworked so this matches gluPerspective / glm::perspective, using fovy
def perspective_offcenter(fovy, fraction, rx, ry, aspect=1.0, n=0.1, f=1000.0, device=None):
    y = np.tan(fovy / 2)

    # Full frustum
    R, L = aspect*y, -aspect*y
    T, B = y, -y

    # Create a randomized sub-frustum
    width  = (R-L)*fraction
    height = (T-B)*fraction
    xstart = (R-L)*rx
    ystart = (T-B)*ry

    l = L + xstart
    r = l + width
    b = B + ystart
    t = b + height
    
    # https://www.scratchapixel.com/lessons/3d-basic-rendering/perspective-and-orthographic-projection-matrix/opengl-perspective-projection-matrix
    return torch.tensor([[2/(r-l),        0,  (r+l)/(r-l),              0], 
                         [      0, -2/(t-b),  (t+b)/(t-b),              0], 
                         [      0,        0, -(f+n)/(f-n), -(2*f*n)/(f-n)], 
                         [      0,        0,           -1,              0]], dtype=torch.float32, device=device)

def translate(x, y, z, device=None):
    return torch.tensor([[1, 0, 0, x], 
                         [0, 1, 0, y], 
                         [0, 0, 1, z], 
                         [0, 0, 0, 1]], dtype=torch.float32, device=device)

def rotate_x(a, device=None):
    s, c = np.sin(a), np.cos(a)
    return torch.tensor([[1,  0, 0, 0], 
                         [0,  c, s, 0], 
                         [0, -s, c, 0], 
                         [0,  0, 0, 1]], dtype=torch.float32, device=device)

def rotate_y(a, device=None):
    s, c = np.sin(a), np.cos(a)
    return torch.tensor([[ c, 0, s, 0], 
                         [ 0, 1, 0, 0], 
                         [-s, 0, c, 0], 
                         [ 0, 0, 0, 1]], dtype=torch.float32, device=device)

def scale(s, device=None):
    return torch.tensor([[ s, 0, 0, 0], 
                         [ 0, s, 0, 0], 
                         [ 0, 0, s, 0], 
                         [ 0, 0, 0, 1]], dtype=torch.float32, device=device)

def lookAt(eye, at, up):
    a = eye - at
    w = a / torch.linalg.norm(a)
    u = torch.cross(up, w)
    u = u / torch.linalg.norm(u)
    v = torch.cross(w, u)
    translate = torch.tensor([[1, 0, 0, -eye[0]], 
                              [0, 1, 0, -eye[1]], 
                              [0, 0, 1, -eye[2]], 
                              [0, 0, 0, 1]], dtype=eye.dtype, device=eye.device)
    rotate = torch.tensor([[u[0], u[1], u[2], 0], 
                           [v[0], v[1], v[2], 0], 
                           [w[0], w[1], w[2], 0], 
                           [0, 0, 0, 1]], dtype=eye.dtype, device=eye.device)
    return rotate @ translate

@torch.no_grad()
def random_rotation_translation(t, device=None):
    m = np.random.normal(size=[3, 3])
    m[1] = np.cross(m[0], m[2])
    m[2] = np.cross(m[0], m[1])
    m = m / np.linalg.norm(m, axis=1, keepdims=True)
    m = np.pad(m, [[0, 1], [0, 1]], mode='constant')
    m[3, 3] = 1.0
    m[:3, 3] = np.random.uniform(-t, t, size=[3])
    return torch.tensor(m, dtype=torch.float32, device=device)

@torch.no_grad()
def random_rotation(device=None):
    m = np.random.normal(size=[3, 3])
    m[1] = np.cross(m[0], m[2])
    m[2] = np.cross(m[0], m[1])
    m = m / np.linalg.norm(m, axis=1, keepdims=True)
    m = np.pad(m, [[0, 1], [0, 1]], mode='constant')
    m[3, 3] = 1.0
    m[:3, 3] = np.array([0,0,0]).astype(np.float32)
    return torch.tensor(m, dtype=torch.float32, device=device)

#----------------------------------------------------------------------------
# Compute focal points of a set of lines using least squares. 
# handy for poorly centered datasets
#----------------------------------------------------------------------------

def lines_focal(o, d):
    d = safe_normalize(d)
    I = torch.eye(3, dtype=o.dtype, device=o.device)
    S = torch.sum(d[..., None] @ torch.transpose(d[..., None], 1, 2) - I[None, ...], dim=0)
    C = torch.sum((d[..., None] @ torch.transpose(d[..., None], 1, 2) - I[None, ...]) @ o[..., None], dim=0).squeeze(1)
    return torch.linalg.pinv(S) @ C

#----------------------------------------------------------------------------
# Cosine sample around a vector N
#----------------------------------------------------------------------------
@torch.no_grad()
def cosine_sample(N, size=None):
    # construct local frame
    N = N/torch.linalg.norm(N)

    dx0 = torch.tensor([0, N[2], -N[1]], dtype=N.dtype, device=N.device)
    dx1 = torch.tensor([-N[2], 0, N[0]], dtype=N.dtype, device=N.device)

    dx = torch.where(dot(dx0, dx0) > dot(dx1, dx1), dx0, dx1)
    #dx = dx0 if np.dot(dx0,dx0) > np.dot(dx1,dx1) else dx1
    dx = dx / torch.linalg.norm(dx)
    dy = torch.cross(N,dx)
    dy = dy / torch.linalg.norm(dy)

    # cosine sampling in local frame
    if size is None:
        phi = 2.0 * np.pi * np.random.uniform()
        s = np.random.uniform()
    else:
        phi = 2.0 * np.pi * torch.rand(*size, 1, dtype=N.dtype, device=N.device)
        s = torch.rand(*size, 1, dtype=N.dtype, device=N.device)
    costheta = np.sqrt(s)
    sintheta = np.sqrt(1.0 - s)

    # cartesian vector in local space
    x = np.cos(phi)*sintheta
    y = np.sin(phi)*sintheta
    z = costheta

    # local to world
    return dx*x + dy*y + N*z

#----------------------------------------------------------------------------
# Bilinear downsample by 2x.
#----------------------------------------------------------------------------

def bilinear_downsample(x : torch.tensor) -> torch.Tensor:
    w = torch.tensor([[1, 3, 3, 1], [3, 9, 9, 3], [3, 9, 9, 3], [1, 3, 3, 1]], dtype=torch.float32, device=x.device) / 64.0
    w = w.expand(x.shape[-1], 1, 4, 4) 
    x = torch.nn.functional.conv2d(x.permute(0, 3, 1, 2), w, padding=1, stride=2, groups=x.shape[-1])
    return x.permute(0, 2, 3, 1)

#----------------------------------------------------------------------------
# Bilinear downsample log(spp) steps
#----------------------------------------------------------------------------

def bilinear_downsample(x : torch.tensor, spp) -> torch.Tensor:
    w = torch.tensor([[1, 3, 3, 1], [3, 9, 9, 3], [3, 9, 9, 3], [1, 3, 3, 1]], dtype=torch.float32, device=x.device) / 64.0
    g = x.shape[-1]
    w = w.expand(g, 1, 4, 4) 
    x = x.permute(0, 3, 1, 2) # NHWC -> NCHW
    steps = int(np.log2(spp))
    for _ in range(steps):
        xp = torch.nn.functional.pad(x, (1,1,1,1), mode='replicate')
        x = torch.nn.functional.conv2d(xp, w, padding=0, stride=2, groups=g)
    return x.permute(0, 2, 3, 1).contiguous() # NCHW -> NHWC

#----------------------------------------------------------------------------
# Singleton initialize GLFW
#----------------------------------------------------------------------------

_glfw_initialized = False
def init_glfw():
    global _glfw_initialized
    try:
        import glfw
        glfw.ERROR_REPORTING = 'raise'
        glfw.default_window_hints()
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        test = glfw.create_window(8, 8, "Test", None, None) # Create a window and see if not initialized yet
    except glfw.GLFWError as e:
        if e.error_code == glfw.NOT_INITIALIZED:
            glfw.init()
            _glfw_initialized = True

#----------------------------------------------------------------------------
# Image display function using OpenGL.
#----------------------------------------------------------------------------

_glfw_window = None
def display_image(image, title=None):
    # Import OpenGL
    import OpenGL.GL as gl
    import glfw

    # Zoom image if requested.
    image = np.asarray(image[..., 0:3]) if image.shape[-1] == 4 else np.asarray(image)
    height, width, channels = image.shape

    # Initialize window.
    init_glfw()
    if title is None:
        title = 'Debug window'
    global _glfw_window
    if _glfw_window is None:
        glfw.default_window_hints()
        _glfw_window = glfw.create_window(width, height, title, None, None)
        glfw.make_context_current(_glfw_window)
        glfw.show_window(_glfw_window)
        glfw.swap_interval(0)
    else:
        glfw.make_context_current(_glfw_window)
        glfw.set_window_title(_glfw_window, title)
        glfw.set_window_size(_glfw_window, width, height)

    # Update window.
    glfw.poll_events()
    gl.glClearColor(0, 0, 0, 1)
    gl.glClear(gl.GL_COLOR_BUFFER_BIT)
    gl.glWindowPos2f(0, 0)
    gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
    gl_format = {3: gl.GL_RGB, 2: gl.GL_RG, 1: gl.GL_LUMINANCE}[channels]
    gl_dtype = {'uint8': gl.GL_UNSIGNED_BYTE, 'float32': gl.GL_FLOAT}[image.dtype.name]
    gl.glDrawPixels(width, height, gl_format, gl_dtype, image[::-1])
    glfw.swap_buffers(_glfw_window)
    if glfw.window_should_close(_glfw_window):
        return False
    return True

#----------------------------------------------------------------------------
# Image save/load helper.
#----------------------------------------------------------------------------

def save_image(fn, x : np.ndarray):
    try:
        if os.path.splitext(fn)[1] == ".png":
            imageio.imwrite(fn, np.clip(np.rint(x * 255.0), 0, 255).astype(np.uint8), compress_level=3) # Low compression for faster saving
        else:
            imageio.imwrite(fn, np.clip(np.rint(x * 255.0), 0, 255).astype(np.uint8))
    except:
        print("WARNING: FAILED to save image %s" % fn)

def save_image_raw(fn, x : np.ndarray):
    try:
        imageio.imwrite(fn, x)
    except:
        print("WARNING: FAILED to save image %s" % fn)


def load_image_raw(fn) -> np.ndarray:
    return imageio.imread(fn)

def load_image(fn) -> np.ndarray:
    img = load_image_raw(fn)
    if img.dtype == np.float32: # HDR image
        return img
    else: # LDR image
        return img.astype(np.float32) / 255

#----------------------------------------------------------------------------

def time_to_text(x):
    if x > 3600:
        return "%.2f h" % (x / 3600)
    elif x > 60:
        return "%.2f m" % (x / 60)
    else:
        return "%.2f s" % x

#----------------------------------------------------------------------------

def checkerboard(res, checker_size) -> np.ndarray:
    tiles_y = (res[0] + (checker_size*2) - 1) // (checker_size*2)
    tiles_x = (res[1] + (checker_size*2) - 1) // (checker_size*2)
    check = np.kron([[1, 0] * tiles_x, [0, 1] * tiles_x] * tiles_y, np.ones((checker_size, checker_size)))*0.33 + 0.33
    check = check[:res[0], :res[1]]
    return np.stack((check, check, check), axis=-1)

#----------------------------------------------------------------------------
# NeRO utils
class IdentityActivation(nn.Module):
    def forward(self, x): return x

class ExpActivation(nn.Module):
    def __init__(self, max_light=5.0):
        super().__init__()
        self.max_light=max_light

    def forward(self, x):
        return torch.exp(torch.clamp(x, max=self.max_light))
    
import torch.nn as nn
def make_predictor(feats_dim: object, output_dim: object, weight_norm: object = True, activation='sigmoid', exp_max=0.0) -> object:
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

    run_dim = 256
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

def offset_points_to_sphere(points):
    points_norm = torch.norm(points, dim=-1)
    mask = points_norm > 0.999
    if torch.sum(mask)>0:
        points = torch.clone(points)
        points[mask] /= points_norm[mask].unsqueeze(-1)
        points[mask] *= 0.999
        # points[points_norm>0.999] = 0
    return points

def get_sphere_intersection(pts, dirs):
    dtx = torch.sum(pts*dirs,dim=-1,keepdim=True) # rn,1
    xtx = torch.sum(pts**2,dim=-1,keepdim=True) # rn,1
    dist = dtx ** 2 - xtx + 1
    assert torch.sum(dist<0)==0
    dist = -dtx + torch.sqrt(dist+1e-6) # rn,1
    return dist


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
        vmxy = torch.concat([(x + 1j * y + 1e-10)**m for m in ml_array[0, :]], dim=-1)
       
        # Get spherical harmonics.
        sph_harms = vmxy * torch.matmul(vmz, mat)

        # Apply attenuation function using the von Mises-Fisher distribution
        # concentration parameter, kappa.
        sigma = 0.5 * ml_array[1, :] * (ml_array[1, :] + 1)
        sigma = sigma.cuda()
        ide = sph_harms.cuda() * torch.exp(-sigma * kappa_inv).cuda()

        # Split into real and imaginary parts and return
        return torch.concat([torch.real(ide), torch.imag(ide)], dim=-1).cuda()

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


def linear2srgb_torch(tensor_0to1):
    if isinstance(tensor_0to1, torch.Tensor):
        pow_func = torch.pow
        where_func = torch.where
    elif isinstance(tensor_0to1, np.ndarray):
        pow_func = np.power
        where_func = np.where
    else:
        raise NotImplementedError(f'Do not support dtype {type(tensor_0to1)}')

    srgb_linear_thres = 0.0031308
    srgb_linear_coeff = 12.92
    srgb_exponential_coeff = 1.055
    srgb_exponent = 2.4

    tensor_linear = tensor_0to1 * srgb_linear_coeff
    
    tensor_nonlinear = srgb_exponential_coeff * (
        pow_func(tensor_0to1 + 1e-6, 1 / srgb_exponent)
    ) - (srgb_exponential_coeff - 1)

    is_linear = tensor_0to1 <= srgb_linear_thres
    tensor_srgb = where_func(is_linear, tensor_linear, tensor_nonlinear)

    return tensor_srgb

#-----------------------------------------------------------------------------------
# Relightable Gaussian

C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
]
C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
]
C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]

def rotation_between_z(vec):
    """
    https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d/476311#476311
    Args:
        vec: [..., 3]

    Returns:
        R: [..., 3, 3]

    """
    v1 = -vec[..., 1]
    v2 = vec[..., 0]
    v3 = torch.zeros_like(v1)
    v11 = v1 * v1
    v22 = v2 * v2
    v33 = v3 * v3
    v12 = v1 * v2
    v13 = v1 * v3
    v23 = v2 * v3
    cos_p_1 = (vec[..., 2] + 1).clamp_min(1e-7)
    R = torch.zeros(vec.shape[:-1] + (3, 3,), dtype=torch.float32, device="cuda")
    R[..., 0, 0] = 1 + (-v33 - v22) / cos_p_1
    R[..., 0, 1] = -v3 + v12 / cos_p_1
    R[..., 0, 2] = v2 + v13 / cos_p_1
    R[..., 1, 0] = v3 + v12 / cos_p_1
    R[..., 1, 1] = 1 + (-v33 - v11) / cos_p_1
    R[..., 1, 2] = -v1 + v23 / cos_p_1
    R[..., 2, 0] = -v2 + v13 / cos_p_1
    R[..., 2, 1] = v1 + v23 / cos_p_1
    R[..., 2, 2] = 1 + (-v22 - v11) / cos_p_1
    R = torch.where((vec[..., 2] + 1 > 0)[..., None, None], R,
                    -torch.eye(3, dtype=torch.float32, device="cuda").expand_as(R))
    return R

def fibonacci_sphere_sampling(normals, sample_num, random_rotate=True):
    pre_shape = normals.shape[:-1]
    if len(pre_shape) > 1:
        normals = normals.reshape(-1, 3)
    delta = np.pi * (3.0 - np.sqrt(5.0))

    # fibonacci sphere sample around z axis
    idx = torch.arange(sample_num, dtype=torch.float, device='cuda')[None]
    z = 1 - 2 * idx / (2 * sample_num - 1)
    rad = torch.sqrt(1 - z ** 2)
    theta = delta * idx
    if random_rotate:
        theta = torch.rand(*pre_shape, 1, device='cuda') * 2 * np.pi + theta
    y = torch.cos(theta) * rad
    x = torch.sin(theta) * rad
    z_samples = torch.stack([x, y, z.expand_as(y)], dim=-2)

    # rotate to normal
    # z_vector = torch.zeros_like(normals)
    # z_vector[..., 2] = 1  # [H, W, 3]
    # rotation_matrix = rotation_between_vectors(z_vector, normals)
    rotation_matrix = rotation_between_z(normals)
    incident_dirs = rotation_matrix @ z_samples
    incident_dirs = F.normalize(incident_dirs, dim=-2).transpose(-1, -2)
    incident_areas = torch.ones_like(incident_dirs)[..., 0:1] * 2 * np.pi
    if len(pre_shape) > 1:
        incident_dirs = incident_dirs.reshape(*pre_shape, sample_num, 3)
        incident_areas = incident_areas.reshape(*pre_shape, sample_num, 1)
    return incident_dirs, incident_areas

def simple_fibonacci_sphere_sampling(sample_num):
    phi = (np.sqrt(5)-1) * np.pi

    # fibonacci sphere sample around z axis
    idx = np.arange(sample_num) + 1
    print(idx.shape)
    z = (2*idx - 1) / sample_num - 1
    rad = np.sqrt(1 - z ** 2)
    theta = phi * idx 
    y = np.cos(theta) * rad
    x = np.sin(theta) * rad
    z_samples = np.stack([x, y, z], axis=-1)
    return z_samples


def sample_incident_rays(normals, is_training=False, sample_num=24):
    if is_training:
        incident_dirs, incident_areas = fibonacci_sphere_sampling(
            normals, sample_num, random_rotate=True)
    else:
        incident_dirs, incident_areas = fibonacci_sphere_sampling(
            normals, sample_num, random_rotate=False)

    return incident_dirs, incident_areas  # [N, S, 3], [N, S, 1]

def eval_sh_coef(deg, dirs):
    """
    Evaluate spherical harmonics at unit directions
    using hardcoded SH polynomials.
    Works with torch/np/jnp.
    ... Can be 0 or more batch dimensions.
    Args:
        deg: int SH deg. Currently, 0-3 supported
        dirs: jnp.ndarray unit directions [..., 3]
    Returns:
        [..., C]
    """
    assert 4 >= deg >= 0
    coeff = (deg + 1) ** 2
    results = torch.zeros(dirs.shape[:-1] + (coeff,), device=dirs.device)
    results[..., 0] = C0
    if deg > 0:
        x, y, z = dirs[..., 0], dirs[..., 1], dirs[..., 2]
        results[..., 1] = -C1 * y
        results[..., 2] = C1 * z
        results[..., 3] = -C1 * x

        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            results[..., 4] = C2[0] * xy
            results[..., 5] = C2[1] * yz
            results[..., 6] = C2[2] * (2.0 * zz - xx - yy)
            results[..., 7] = C2[3] * xz
            results[..., 8] = C2[4] * (xx - yy)

            if deg > 2:
                results[..., 9] = C3[0] * y * (3 * xx - yy)
                results[..., 10] = C3[1] * xy * z
                results[..., 11] = C3[2] * y * (4 * zz - xx - yy)
                results[..., 12] = C3[3] * z * (2 * zz - 3 * xx - 3 * yy)
                results[..., 13] = C3[4] * x * (4 * zz - xx - yy)
                results[..., 14] = C3[5] * z * (xx - yy)
                results[..., 15] = C3[6] * x * (xx - 3 * yy)

                if deg > 3:
                    results[..., 16] = C4[0] * xy * (xx - yy)
                    results[..., 17] = C4[1] * yz * (3 * xx - yy)
                    results[..., 18] = C4[2] * xy * (7 * zz - 1)
                    results[..., 19] = C4[3] * yz * (7 * zz - 3)
                    results[..., 20] = C4[4] * (zz * (35 * zz - 30) + 3)
                    results[..., 21] = C4[5] * xz * (7 * zz - 3)
                    results[..., 22] = C4[6] * (xx - yy) * (7 * zz - 1)
                    results[..., 23] = C4[7] * xz * (xx - 3 * yy)
                    results[..., 24] = C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy))

    return results 

def _dot(a, b):
        return (a * b).sum(dim=-1, keepdim=True)  # [H, W, 1, 1]

def _f_diffuse(base_color, metallic):
    return (1 - metallic) * base_color / np.pi  # [H, W, 1, 3]

def _f_specular(h_d_n, h_d_o, n_d_i, n_d_o, base_color, roughness, metallic):
    # used in SG, wrongly normalized
    def _d_sg(r, cos):
        r2 = (r * r).clamp(min=1e-7)
        amp = 1 / (r2 * np.pi)
        sharp = 2 / r2
        return amp * torch.exp(sharp * (cos - 1))

    D = _d_sg(roughness, h_d_n)

    # Fresnel term F
    F_0 = 0.04 * (1 - metallic) + base_color * metallic  # [H, W, 1, 3]
    F = F_0 + (1.0 - F_0) * ((1.0 - h_d_o) ** 5)  # [H, W, S, 3]

    # geometry term V, we use V = G / (4 * cos * cos) here
    def _v_schlick_ggx(r, cos):
        r2 = ((1 + r) ** 2) / 8
        return 0.5 / (cos * (1 - r2) + r2).clamp(min=1e-7)

    V = _v_schlick_ggx(roughness, n_d_i) * _v_schlick_ggx(roughness, n_d_o)  # [H, W, S, 1]

    return D * F * V

def rendering_equation_python(base_color, roughness, metallic, normals, viewdirs,
                              incidents, is_training=False, direct_light_env_light=None,
                              visibility=None, sample_num=24):
    incident_dirs, incident_areas = sample_incident_rays(normals, is_training, sample_num)

    base_color = base_color.unsqueeze(-2).contiguous()
    roughness = roughness.unsqueeze(-2).contiguous()
    metallic = metallic.unsqueeze(-2).contiguous()
    normals = normals.unsqueeze(-2).contiguous()
    viewdirs = viewdirs.unsqueeze(-2).contiguous()

    deg = int(np.sqrt(visibility.shape[1]) - 1)
    incident_dirs_coef = eval_sh_coef(deg, incident_dirs).unsqueeze(2)
    shs_view = incidents.transpose(1, 2).view(base_color.shape[0], 1, 3, -1)
    shs_visibility = visibility.transpose(1, 2).view(base_color.shape[0], 1, 1, -1)
    local_incident_lights = torch.clamp_min((incident_dirs_coef[..., :shs_view.shape[-1]] * shs_view).sum(-1), 0)
    if direct_light_env_light is not None:
        shs_view_direct = direct_light_env_light.get_env_shs.transpose(1, 2).unsqueeze(1)
        global_incident_lights = torch.clamp_min(
            (incident_dirs_coef[..., :shs_view_direct.shape[-1]] * shs_view_direct).sum(-1) + 0.5, 0)
    else:
        global_incident_lights = torch.zeros_like(local_incident_lights, requires_grad=False)

    incident_visibility = torch.clamp(
        (incident_dirs_coef[..., :shs_visibility.shape[-1]] * shs_visibility).sum(-1) + 0.5, 0, 1)
    global_incident_lights = global_incident_lights * incident_visibility
    incident_lights = local_incident_lights + global_incident_lights

    

    half_dirs = incident_dirs + viewdirs
    half_dirs = F.normalize(half_dirs, dim=-1)

    h_d_n = _dot(half_dirs, normals).clamp(min=0)
    h_d_o = _dot(half_dirs, viewdirs).clamp(min=0)
    n_d_i = _dot(normals, incident_dirs).clamp(min=0)
    n_d_o = _dot(normals, viewdirs).clamp(min=0)
    f_d = _f_diffuse(base_color, metallic)
    f_s = _f_specular(h_d_n, h_d_o, n_d_i, n_d_o, base_color, roughness, metallic)

    transport = incident_lights * incident_areas * n_d_i  # （num_pts, num_sample, 3)
    rgb_d = (f_d * transport).mean(dim=-2)
    rgb_s = (f_s * transport).mean(dim=-2)
    pbr = rgb_d + rgb_s
    diffuse_light = transport.mean(dim=-2)

    extra_results = {
        "incident_dirs": incident_dirs,
        "incident_lights": incident_lights,
        "local_incident_lights": local_incident_lights,
        "global_incident_lights": global_incident_lights,
        "incident_visibility": incident_visibility,
        "diffuse_light": diffuse_light,
    }

    return pbr, extra_results

#------------------------------------------------------------------------------------
# InvRender enc
class MaterialNetwork(nn.Module):
    def __init__(self, multires=8, out_dim=7, 
                 brdf_encoder_dims=[256, 256, 256, 256],
                 brdf_decoder_dims=[128, 128],
                 latent_dim=32,
                 ):
        super().__init__()

        self.embed_fn = None
        if multires > 0:
            self.brdf_embed_fn, brdf_input_dim = get_embedder(multires)


        self.latent_dim = latent_dim
        self.actv_fn = nn.LeakyReLU(0.2)
        ############## spatially-varying BRDF ############
        
        print('BRDF encoder network size: ', brdf_encoder_dims)
        print('BRDF decoder network size: ', brdf_decoder_dims)

        brdf_encoder_layer = []
        dim = brdf_input_dim
        for i in range(len(brdf_encoder_dims)):
            brdf_encoder_layer.append(nn.Linear(dim, brdf_encoder_dims[i]))
            brdf_encoder_layer.append(self.actv_fn)
            dim = brdf_encoder_dims[i]
        brdf_encoder_layer.append(nn.Linear(dim, self.latent_dim))
        self.brdf_encoder_layer = nn.Sequential(*brdf_encoder_layer)
        
        brdf_decoder_layer = []
        dim = self.latent_dim
        for i in range(len(brdf_decoder_dims)):
            brdf_decoder_layer.append(nn.Linear(dim, brdf_decoder_dims[i]))
            brdf_decoder_layer.append(self.actv_fn)
            dim = brdf_decoder_dims[i]
        brdf_decoder_layer.append(nn.Linear(dim, out_dim))
        self.brdf_decoder_layer = nn.Sequential(*brdf_decoder_layer)


    def forward(self, points):
        if self.brdf_embed_fn is not None:
            points = self.brdf_embed_fn(points)

        brdf_lc = torch.sigmoid(self.brdf_encoder_layer(points))
        brdf = torch.sigmoid(self.brdf_decoder_layer(brdf_lc))

        rand_lc = brdf_lc + torch.randn(brdf_lc.shape).cuda() * 0.01
        random_xi_brdf = torch.sigmoid(self.brdf_decoder_layer(rand_lc))

        ret = dict([
            ('brdf', brdf),
            ('random_brdf', random_xi_brdf),
        ])
        return ret

##### Visibility MLP
def positional_encoding(positions, freqs):
        freq_bands = (2**torch.arange(freqs).float()).to(positions.device)  # (F,)
        pts = (positions[..., None] * freq_bands).reshape(
            positions.shape[:-1] + (freqs * positions.shape[-1], ))  # (..., DF)
        pts = torch.cat([torch.sin(pts), torch.cos(pts)], dim=-1)
        return pts

class VisibilityMLP(nn.Module):
    def __init__(self, pospe=3, viewpe=3, hidden_dim=64) -> None:
        super().__init__()
        self.pospe = pospe
        self.viewpe = viewpe
        self.pos_mlp = nn.Sequential(
            nn.Linear(2 * pospe * 3 + 3, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.view_mlp = nn.Sequential(
            nn.Linear(hidden_dim + 2 * viewpe * 3 + 3, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, pos, viewdir):
        feat = self.pos_mlp(torch.cat([pos, positional_encoding(pos, self.pospe)], -1))
        vis = self.view_mlp(torch.cat([pos, positional_encoding(viewdir, self.viewpe), feat], -1))
        return vis

class VisibilitySHMLP(nn.Module):
    def __init__(self, pospe=3, hidden_dim=64) -> None:
        super().__init__()
        self.pospe = pospe
        self.vis_mlp = nn.Sequential(
            nn.Linear(2 * pospe * 3 + 3, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, 16)
        )
        
    def forward(self, pos):
        feat = self.vis_mlp(torch.cat([pos, positional_encoding(pos, self.pospe)], -1))
        return feat