import os
import torch
import trimesh
import numpy as np
from utils import decode_latent_mesh
from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.rendering.pytorch3d_util import convert_meshes

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('qtagg')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
xm = load_model('transmitter', device=device)
model = load_model('text300M', device=device)
diffusion = diffusion_from_config(load_config('diffusion'))
batch_size = 1
guidance_scale = 15.0
voxel_grid_max_size = 20


def get_voxel_length_for_grid_size(vertices: torch.Tensor, grid_size: int) -> float:
    """
    Args:
        vertices: [N, 3] tensor, each row is a vertex
    Returns:
        Voxel length for voxelization process
    """
    max_bound = float(torch.max(torch.max(vertices, dim=0).values - torch.min(vertices, dim=0).values))
    return max_bound / (grid_size - 1)


def pad_voxels_to_grid_size(voxels: np.ndarray, grid_size: int) -> np.ndarray:
    """
    Args:
        voxels: A minimum [M, N, P] numpy boolean array that contains the target shape

    Returns:
        Padded voxels of shape [grid_size, grid_size, grid_size]
    """
    new_voxels = np.zeros([grid_size, grid_size, grid_size], dtype=voxels.dtype)
    padding = (np.array(new_voxels.shape) - np.array(voxels.shape)) / 2
    padding = padding.astype(int)
    new_voxels[padding[0]:padding[0] + voxels.shape[0],
    padding[1]:padding[1] + voxels.shape[1], padding[2]:padding[2] + voxels.shape[2]] = voxels
    return new_voxels


if __name__ == '__main__':
    with open("prompts.txt", "r") as file:
        prompts = [line.strip() for line in file]

    if not os.path.exists("shapes"):
        os.mkdir("shapes")

    for prompt in prompts:
        print(f"Generating {prompt}")
        latents = sample_latents(
            batch_size=batch_size,
            model=model,
            diffusion=diffusion,
            guidance_scale=guidance_scale,
            model_kwargs=dict(texts=["a {}".format(prompt)] * batch_size),
            progress=True,
            clip_denoised=True,
            use_fp16=True,
            use_karras=True,
            karras_steps=64,
            sigma_min=1e-3,
            sigma_max=160,
            s_churn=0,
        )

        raw_mesh = decode_latent_mesh(xm, latents[0])
        mesh = convert_meshes([raw_mesh])
        verts = torch.Tensor.cpu(raw_mesh.verts)
        faces = torch.Tensor.cpu(raw_mesh.faces)
        vert_norm = torch.Tensor.cpu(mesh.verts_normals_list()[0])
        mesh = trimesh.Trimesh(np.asarray(verts), np.asarray(faces),
                               vertex_normals=np.asarray(vert_norm))

        vox_ = mesh.voxelized(get_voxel_length_for_grid_size(verts, voxel_grid_max_size))
        vox_ = vox_.fill()

        voxels = pad_voxels_to_grid_size(np.array(vox_.matrix), voxel_grid_max_size)

        np.save(str(os.path.join("shapes", f"{prompt}.npy")), voxels)

        fig, ax = plt.subplots(1, 1, subplot_kw={"projection": "3d"})
        colors = np.empty(voxels.shape, dtype=object)
        colors[voxels] = "grey"
        ax.voxels(voxels, facecolors=colors)
        ax.axis("equal")
        fig.savefig(str(os.path.join("shapes", f"{prompt}.png")), bbox_inches="tight", pad_inches=0)
