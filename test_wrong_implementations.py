"""
Test script for visualizing wrong implementations.
각 잘못된 구현의 시각적 효과를 확인하기 위한 스크립트
"""

import json
from pathlib import Path
import imageio.v2 as imageio
import numpy as np
import torch
import torchvision.utils as tvu
from tqdm import tqdm

from src.camera import Camera
from src.constants import USE_HALF
from src.renderer import GSRasterizer, homogenize
from src.scene import Scene
from render import load_scene, load_camera_params


class WrongRenderer1_NoPerspectiveDivision(GSRasterizer):
    """실험 1: Perspective Division 생략"""

    def project_ndc(self, points, w2c, proj_mat, z_near):
        p_homogeneous = homogenize(points)
        p_view = p_homogeneous @ w2c
        p_proj = p_view @ proj_mat

        # ❌ 잘못: perspective division 생략
        p_ndc = p_proj  # w로 나누지 않음!

        in_mask = p_view[..., 2] > z_near
        return p_ndc, p_view, in_mask


class WrongRenderer2_WrongJacobianSign(GSRasterizer):
    """실험 2: Jacobian 부호 오류"""

    def compute_cov_2d(self, mean_3d, cov_3d, w2c, f_x, f_y):
        mean_camera_homogeneous = homogenize(mean_3d) @ w2c
        mean_camera = mean_camera_homogeneous[:, :3]

        J = torch.zeros(mean_3d.shape[0], 3, 3).to(mean_3d)
        W = w2c[:3, :3].T

        t_x = mean_camera[:, 0]
        t_y = mean_camera[:, 1]
        t_z = mean_camera[:, 2]

        J[:, 0, 0] = f_x / t_z
        J[:, 1, 1] = f_y / t_z

        # ❌ 잘못: 부호 반대 (양수!)
        J[:, 0, 2] = f_x * t_x / (t_z * t_z)
        J[:, 1, 2] = f_y * t_y / (t_z * t_z)

        cov_2d = J @ W @ cov_3d @ W.T @ J.transpose(1, 2)
        filter = torch.eye(2, 2).to(cov_2d) * 0.3
        return cov_2d[:, :2, :2] + filter[None]


class WrongRenderer3_ReverseDepthSort(GSRasterizer):
    """실험 3: Depth Sorting 역순"""

    def render(self, camera, mean_2d, cov_2d, color, opacities, depths):
        from src.renderer import get_radius, get_rect

        radii = get_radius(cov_2d)
        rect = get_rect(mean_2d, radii, width=camera.image_width, height=camera.image_height)

        pix_coord = torch.stack(
            torch.meshgrid(torch.arange(camera.image_height), torch.arange(camera.image_width), indexing='xy'),
            dim=-1,
        ).to(mean_2d.device)

        render_color = torch.ones(*pix_coord.shape[:2], 3).to(mean_2d.device)

        for h in range(0, camera.image_height, self.tile_size):
            for w in range(0, camera.image_width, self.tile_size):
                over_tl = rect[0][..., 0].clip(min=w), rect[0][..., 1].clip(min=h)
                over_br = rect[1][..., 0].clip(max=w+self.tile_size-1), rect[1][..., 1].clip(max=h+self.tile_size-1)

                in_mask = (over_br[0] > over_tl[0]) & (over_br[1] > over_tl[1])
                if not in_mask.sum() > 0:
                    continue

                # ❌ 잘못: 내림차순 정렬 (먼 것부터)
                sorted_indices = torch.argsort(depths[in_mask], descending=True)

                sorted_mean_2d = mean_2d[in_mask][sorted_indices]
                sorted_cov_2d = cov_2d[in_mask][sorted_indices]
                sorted_color = color[in_mask][sorted_indices]
                sorted_opacity = opacities[in_mask][sorted_indices]

                tile_pix_coord = pix_coord[h:h+self.tile_size, w:w+self.tile_size]
                displacement = tile_pix_coord[:, :, None, :] - sorted_mean_2d[None, None, :, :]

                cov_2d_inv = torch.inverse(sorted_cov_2d)
                mahalanobis = torch.einsum('hwmi,mij,hwmj->hwm', displacement, cov_2d_inv, displacement)
                gaussian_weight = torch.exp(-0.5 * mahalanobis)

                alpha_tilde = gaussian_weight * sorted_opacity.squeeze(-1)[None, None, :]
                alpha_tilde = torch.clamp(alpha_tilde, 0.0, 0.99)

                one_minus_alpha = 1.0 - alpha_tilde
                transmittance = torch.cumprod(one_minus_alpha, dim=-1)
                transmittance = torch.cat([
                    torch.ones_like(transmittance[:, :, :1]),
                    transmittance[:, :, :-1]
                ], dim=-1)

                weights = alpha_tilde * transmittance
                tile_color = torch.einsum('hwm,mc->hwc', weights, sorted_color)

                if self.white_bkgd:
                    accumulated_alpha = 1.0 - torch.prod(one_minus_alpha, dim=-1, keepdim=True)
                    tile_color = tile_color + (1.0 - accumulated_alpha)

                render_color[h:h+self.tile_size, w:w+self.tile_size] = tile_color.reshape(self.tile_size, self.tile_size, -1)

        return render_color


class WrongRenderer4_NoTransmittanceShift(GSRasterizer):
    """실험 4: Transmittance Shift 생략"""

    def render(self, camera, mean_2d, cov_2d, color, opacities, depths):
        from src.renderer import get_radius, get_rect

        radii = get_radius(cov_2d)
        rect = get_rect(mean_2d, radii, width=camera.image_width, height=camera.image_height)

        pix_coord = torch.stack(
            torch.meshgrid(torch.arange(camera.image_height), torch.arange(camera.image_width), indexing='xy'),
            dim=-1,
        ).to(mean_2d.device)

        render_color = torch.ones(*pix_coord.shape[:2], 3).to(mean_2d.device)

        for h in range(0, camera.image_height, self.tile_size):
            for w in range(0, camera.image_width, self.tile_size):
                over_tl = rect[0][..., 0].clip(min=w), rect[0][..., 1].clip(min=h)
                over_br = rect[1][..., 0].clip(max=w+self.tile_size-1), rect[1][..., 1].clip(max=h+self.tile_size-1)

                in_mask = (over_br[0] > over_tl[0]) & (over_br[1] > over_tl[1])
                if not in_mask.sum() > 0:
                    continue

                sorted_indices = torch.argsort(depths[in_mask])

                sorted_mean_2d = mean_2d[in_mask][sorted_indices]
                sorted_cov_2d = cov_2d[in_mask][sorted_indices]
                sorted_color = color[in_mask][sorted_indices]
                sorted_opacity = opacities[in_mask][sorted_indices]

                tile_pix_coord = pix_coord[h:h+self.tile_size, w:w+self.tile_size]
                displacement = tile_pix_coord[:, :, None, :] - sorted_mean_2d[None, None, :, :]

                cov_2d_inv = torch.inverse(sorted_cov_2d)
                mahalanobis = torch.einsum('hwmi,mij,hwmj->hwm', displacement, cov_2d_inv, displacement)
                gaussian_weight = torch.exp(-0.5 * mahalanobis)

                alpha_tilde = gaussian_weight * sorted_opacity.squeeze(-1)[None, None, :]
                alpha_tilde = torch.clamp(alpha_tilde, 0.0, 0.99)

                one_minus_alpha = 1.0 - alpha_tilde
                transmittance = torch.cumprod(one_minus_alpha, dim=-1)

                # ❌ 잘못: shift 생략 (transmittance를 그대로 사용)

                weights = alpha_tilde * transmittance
                tile_color = torch.einsum('hwm,mc->hwc', weights, sorted_color)

                if self.white_bkgd:
                    accumulated_alpha = 1.0 - torch.prod(one_minus_alpha, dim=-1, keepdim=True)
                    tile_color = tile_color + (1.0 - accumulated_alpha)

                render_color[h:h+self.tile_size, w:w+self.tile_size] = tile_color.reshape(self.tile_size, self.tile_size, -1)

        return render_color


def render_comparison(scene_type="lego", device_type="cpu", num_views=5):
    """
    여러 잘못된 구현과 올바른 구현을 비교하여 렌더링
    """
    device = torch.device(device_type)
    print(f"Using device: {device}")

    # Load scene and camera
    print(f"Loading Scene: {scene_type}")
    scene = load_scene(scene_type, device)

    c2ws, proj_mat, fov, focal, near, far, img_width, img_height = load_camera_params(
        scene_type, device, use_half=USE_HALF
    )

    # Select views to render (균등하게 분산)
    total_views = len(c2ws)
    view_indices = [int(i * total_views / num_views) for i in range(num_views)]

    print(f"Rendering {num_views} views: {view_indices}")

    # Initialize renderers
    renderers = {
        "correct": GSRasterizer(),
        "exp1_no_perspective": WrongRenderer1_NoPerspectiveDivision(),
        "exp2_wrong_jacobian": WrongRenderer2_WrongJacobianSign(),
        "exp3_reverse_depth": WrongRenderer3_ReverseDepthSort(),
        "exp4_no_shift": WrongRenderer4_NoTransmittanceShift(),
    }

    # Create output directory
    out_dir = Path(f"./experiments/{scene_type}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Render each view with each renderer
    for view_idx in tqdm(view_indices, desc="Rendering views"):
        c2w = c2ws[view_idx]
        c2w_ = torch.from_numpy(c2w).float().to(device)
        proj_mat_ = proj_mat.float().to(device)

        cam = Camera(
            camera_to_world=c2w_, proj_mat=proj_mat_, cam_center=c2w_[:3, 3],
            fov_x=fov, fov_y=fov, near=near, far=far,
            image_width=img_width, image_height=img_height,
            f_x=focal, f_y=focal,
            c_x=img_width / 2, c_y=img_height / 2,
        )

        for name, renderer in renderers.items():
            try:
                img = renderer.render_scene(scene, cam)
                img = img.reshape(img_height, img_width, 3)
                img = torch.clamp(img, 0.0, 1.0)

                # Save image
                out_path = out_dir / f"view{view_idx:03d}_{name}.png"
                img_save = img.permute(2, 0, 1)
                tvu.save_image(img_save, out_path)

            except Exception as e:
                print(f"Error rendering {name} for view {view_idx}: {e}")

    print(f"\n✅ Rendering completed! Check results in: {out_dir}")
    print(f"\nCompare images:")
    for view_idx in view_indices:
        print(f"  View {view_idx}:")
        print(f"    - Correct:          view{view_idx:03d}_correct.png")
        print(f"    - Exp1 (No Persp):  view{view_idx:03d}_exp1_no_perspective.png")
        print(f"    - Exp2 (Jacobian):  view{view_idx:03d}_exp2_wrong_jacobian.png")
        print(f"    - Exp3 (Depth):     view{view_idx:03d}_exp3_reverse_depth.png")
        print(f"    - Exp4 (Transmit):  view{view_idx:03d}_exp4_no_shift.png")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_type", type=str, default="lego")
    parser.add_argument("--device_type", type=str, default="cpu")
    parser.add_argument("--num_views", type=int, default=5)

    args = parser.parse_args()

    render_comparison(
        scene_type=args.scene_type,
        device_type=args.device_type,
        num_views=args.num_views
    )
