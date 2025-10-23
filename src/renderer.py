"""
PyTorch implementation of Gaussian Splat Rasterizer.

The implementation is based on torch-splatting: https://github.com/hbb1/torch-splatting
"""

from jaxtyping import Bool, Float, jaxtyped
import torch
from typeguard import typechecked


from .camera import Camera
from .scene import Scene
from .sh import eval_sh


class GSRasterizer(object):
    """
    Gaussian Splat Rasterizer.
    """

    def __init__(self):

        self.sh_degree = 3
        self.white_bkgd = True
        self.tile_size = 25

    def render_scene(self, scene: Scene, camera: Camera):

        # Retrieve Gaussian parameters
        mean_3d = scene.mean_3d
        scales = scene.scales
        rotations = scene.rotations
        shs = scene.shs
        opacities = scene.opacities
        
        # ============================================================================
        # Process camera parameters
        # NOTE: We transpose both camera extrinsic and projection matrices
        # assuming that these transforms are applied to points in row vector format.
        # NOTE: Do NOT modify this block.
        # Retrieve camera pose (extrinsic)
        R = camera.camera_to_world[:3, :3]  # 3 x 3
        T = camera.camera_to_world[:3, 3:4]  # 3 x 1
        R_edit = torch.diag(torch.tensor([1, -1, -1], device=R.device, dtype=R.dtype))
        R = R @ R_edit
        R_inv = R.T
        T_inv = -R_inv @ T
        world_to_camera = torch.eye(4, device=R.device, dtype=R.dtype)
        world_to_camera[:3, :3] = R_inv
        world_to_camera[:3, 3:4] = T_inv
        world_to_camera = world_to_camera.permute(1, 0)

        # Retrieve camera intrinsic
        proj_mat = camera.proj_mat.permute(1, 0)
        world_to_camera = world_to_camera.to(mean_3d.device)
        proj_mat = proj_mat.to(mean_3d.device)
        # ============================================================================

        # Project Gaussian center positions to NDC
        mean_ndc, mean_view, in_mask = self.project_ndc(
            mean_3d, world_to_camera, proj_mat, camera.near,
        )
        mean_ndc = mean_ndc[in_mask]
        mean_view = mean_view[in_mask]
        assert mean_ndc.shape[0] > 0, "No points in the frustum"
        assert mean_view.shape[0] > 0, "No points in the frustum"
        depths = mean_view[:, 2]

        # Compute RGB from spherical harmonics
        color = self.get_rgb_from_sh(mean_3d, shs, camera)

        # Compute 3D covariance matrix
        cov_3d = self.compute_cov_3d(scales, rotations)

        # Project covariance matrices to 2D
        cov_2d = self.compute_cov_2d(
            mean_3d=mean_3d, 
            cov_3d=cov_3d, 
            w2c=world_to_camera,
            f_x=camera.f_x, 
            f_y=camera.f_y,
        )
        
        # Compute pixel space coordinates of the projected Gaussian centers
        mean_coord_x = ((mean_ndc[..., 0] + 1) * camera.image_width - 1.0) * 0.5
        mean_coord_y = ((mean_ndc[..., 1] + 1) * camera.image_height - 1.0) * 0.5
        mean_2d = torch.stack([mean_coord_x, mean_coord_y], dim=-1)

        color = self.render(
            camera=camera, 
            mean_2d=mean_2d,
            cov_2d=cov_2d,
            color=color,
            opacities=opacities, 
            depths=depths,
        )
        color = color.reshape(-1, 3)

        return color

    @torch.no_grad()
    def get_rgb_from_sh(self, mean_3d, shs, camera):
        rays_o = camera.cam_center        
        rays_d = mean_3d - rays_o
        rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
        color = eval_sh(self.sh_degree, shs.permute(0, 2, 1), rays_d)
        color = torch.clamp_min(color + 0.5, 0.0)
        return color
    
    @jaxtyped(typechecker=typechecked)
    @torch.no_grad()
    def project_ndc(
        self,
        points: Float[torch.Tensor, "N 3"],
        w2c: Float[torch.Tensor, "4 4"],
        proj_mat: Float[torch.Tensor, "4 4"],
        z_near: float,
    ) -> tuple[
        Float[torch.Tensor, "N 4"],
        Float[torch.Tensor, "N 4"],
        Bool[torch.Tensor, "N"],
    ]:
        """
        Projects points to NDC space.
        
        Args:
        - points: 3D points in object space.
        - w2c: World-to-camera matrix.
        - proj_mat: Projection matrix.
        - z_near: Near plane distance.

        Returns:
        - p_ndc: NDC coordinates.
        - p_view: View space coordinates.
        - in_mask: Mask of points that are in the frustum.
        """
        # ========================================================
        # TODO: NDC 공간으로의 투영 구현
        # Step 1: 3D 점을 homogeneous 좌표로 변환 (N x 4)
        p_homogeneous = homogenize(points)

        # Step 2: View space으로 변환: p_view = p @ w2c
        p_view = p_homogeneous @ w2c

        # Step 3: 클립 공간으로 변환: p_proj = p_view @ proj_mat
        p_proj = p_view @ proj_mat

        # Step 4: w 성분으로 정규화하여 NDC 좌표 얻기
        p_ndc = p_proj / p_proj[..., 3:4]

        # View space의 z 좌표가 z_near보다 큰지 확인
        in_mask = p_view[..., 2] > z_near
        # ========================================================

        return p_ndc, p_view, in_mask

    @torch.no_grad()
    def compute_cov_3d(self, s, r):
        L = build_scaling_rotation(s, r)
        cov3d = L @ L.transpose(1, 2)
        return cov3d

    @jaxtyped(typechecker=typechecked)
    @torch.no_grad()
    def compute_cov_2d(
        self,
        mean_3d: Float[torch.Tensor, "N 3"],
        cov_3d: Float[torch.Tensor, "N 3 3"],
        w2c: Float[torch.Tensor, "4 4"],
        f_x: Float[torch.Tensor, ""],
        f_y: Float[torch.Tensor, ""],
    ) -> Float[torch.Tensor, "N 2 2"]:
        """
        Projects 3D covariances to 2D image plane.

        Args:
        - mean_3d: Coordinates of center of 3D Gaussians.
        - cov_3d: 3D covariance matrix.
        - w2c: World-to-camera matrix.
        - f_x: Focal length along x-axis.
        - f_y: Focal length along y-axis.

        Returns:
        - cov_2d: 2D covariance matrix.
        """ 
        # ========================================================
        # TODO: 3D mean coordinate를 Camera space로 변환
        # Homogeneous로 변환하고 Camera space으로 변환
        mean_camera_homogeneous = homogenize(mean_3d) @ w2c  # N x 4
        mean_camera = mean_camera_homogeneous[:, :3]  # N x 3
        # ========================================================

        # Transpose the rigid transformation part of the world-to-camera matrix
        J = torch.zeros(mean_3d.shape[0], 3, 3).to(mean_3d)
        W = w2c[:3, :3].T
        # ========================================================
        # TODO: View transform과 projection의 Jacobian 계산
        # Camera space coord 추출
        t_x = mean_camera[:, 0]  # N
        t_y = mean_camera[:, 1]  # N
        t_z = mean_camera[:, 2]  # N

        # 다음 공식에 따라 Jacobian 행렬 채우기:
        # J = [[f_x/t_z,    0,       -f_x*t_x/t_z^2],
        #      [0,          f_y/t_z, -f_y*t_y/t_z^2],
        #      [0,          0,        0             ]]
        J[:, 0, 0] = f_x / t_z
        J[:, 1, 1] = f_y / t_z
        J[:, 0, 2] = -f_x * t_x / (t_z * t_z)
        J[:, 1, 2] = -f_y * t_y / (t_z * t_z)

        # 2D Covariance 계산: Σ_2D = J @ W @ Σ_3D @ W^T @ J^T
        cov_2d = J @ W @ cov_3d @ W.T @ J.transpose(1, 2)
        # ========================================================

        # add low pass filter here according to E.q. 32
        filter = torch.eye(2, 2).to(cov_2d) * 0.3
        return cov_2d[:, :2, :2] + filter[None]

    @jaxtyped(typechecker=typechecked)
    @torch.no_grad()
    def render(
        self,
        camera: Camera,
        mean_2d: Float[torch.Tensor, "N 2"],
        cov_2d: Float[torch.Tensor, "N 2 2"],
        color: Float[torch.Tensor, "N 3"],
        opacities: Float[torch.Tensor, "N 1"],
        depths: Float[torch.Tensor, "N"],
    ) -> Float[torch.Tensor, "H W 3"]:
        radii = get_radius(cov_2d)
        rect = get_rect(mean_2d, radii, width=camera.image_width, height=camera.image_height)

        pix_coord = torch.stack(
            torch.meshgrid(torch.arange(camera.image_height), torch.arange(camera.image_width), indexing='xy'),
            dim=-1,
        ).to(mean_2d.device)
        
        render_color = torch.ones(*pix_coord.shape[:2], 3).to(mean_2d.device)

        assert camera.image_height % self.tile_size == 0, "Image height must be divisible by the tile_size."
        assert camera.image_width % self.tile_size == 0, "Image width must be divisible by the tile_size."
        for h in range(0, camera.image_height, self.tile_size):
            for w in range(0, camera.image_width, self.tile_size):
                # check if the rectangle penetrate the tile
                over_tl = rect[0][..., 0].clip(min=w), rect[0][..., 1].clip(min=h)
                over_br = rect[1][..., 0].clip(max=w+self.tile_size-1), rect[1][..., 1].clip(max=h+self.tile_size-1)
                
                # a binary mask indicating projected Gaussians that lie in the current tile
                in_mask = (over_br[0] > over_tl[0]) & (over_br[1] > over_tl[1])
                if not in_mask.sum() > 0:
                    continue

                # ========== TODO 1: Depth Sorting ==========
                sorted_indices = torch.argsort(depths[in_mask])

                # Sorted attributes 가져오기
                sorted_mean_2d = mean_2d[in_mask][sorted_indices]  # (M, 2)
                sorted_cov_2d = cov_2d[in_mask][sorted_indices]    # (M, 2, 2)
                sorted_color = color[in_mask][sorted_indices]       # (M, 3)
                sorted_opacity = opacities[in_mask][sorted_indices] # (M, 1)
                # ========================================================

                # ========== TODO 2: Displacement Vector ==========
                # 현재 타일의 픽셀 좌표 가져오기
                tile_pix_coord = pix_coord[h:h+self.tile_size, w:w+self.tile_size]  # (tile_size, tile_size, 2)

                # Displacement 계산: (tile_size, tile_size, 1, 2) - (1, 1, M, 2) = (tile_size, tile_size, M, 2)
                displacement = tile_pix_coord[:, :, None, :] - sorted_mean_2d[None, None, :, :]
                # ========================================================

                # ========== TODO 3: Gaussian Weight ==========
                # Inverse of Covariance
                cov_2d_inv = torch.inverse(sorted_cov_2d)  # (M, 2, 2)

                # Mahalanobis 거리 계산: d^T @ Σ^-1 @ d
                # displacement: (tile_size, tile_size, M, 2)
                # cov_2d_inv: (M, 2, 2)
                mahalanobis = torch.einsum('hwmi,mij,hwmj->hwm', displacement, cov_2d_inv, displacement)

                # Gaussian weight: exp(-0.5 * mahalanobis)
                gaussian_weight = torch.exp(-0.5 * mahalanobis)  # (tile_size, tile_size, M)
                # ========================================================

                # ========== TODO 4: Alpha Blending ==========
                # alpha_tilde = gaussian_weight * opacity 계산
                alpha_tilde = gaussian_weight * sorted_opacity.squeeze(-1)[None, None, :]  # (tile_size, tile_size, M)

                # 수치 안정성을 위해 클램핑
                alpha_tilde = torch.clamp(alpha_tilde, 0.0, 0.99)

                # Transmittance 계산: T_j = ∏_{k<j}(1 - α_k)
                one_minus_alpha = 1.0 - alpha_tilde  # (tile_size, tile_size, M)
                transmittance = torch.cumprod(one_minus_alpha, dim=-1)  # (tile_size, tile_size, M)

                # Shift transmittance: 맨 앞에 1을 추가하고 마지막 요소 제거
                # T_j는 j를 포함하지 않는 이전 요소들의 곱이어야 함
                transmittance = torch.cat([
                    torch.ones_like(transmittance[:, :, :1]),
                    transmittance[:, :, :-1]
                ], dim=-1)  # (tile_size, tile_size, M)

                # Blending 가중치 계산
                weights = alpha_tilde * transmittance  # (tile_size, tile_size, M)

                # Accumulate colors: C = Σ_j weights_j * color_j
                tile_color = torch.einsum('hwm,mc->hwc', weights, sorted_color)  # (tile_size, tile_size, 3)

                # 흰색 배경
                if self.white_bkgd:
                    accumulated_alpha = 1.0 - torch.prod(one_minus_alpha, dim=-1, keepdim=True)  # (tile_size, tile_size, 1)
                    tile_color = tile_color + (1.0 - accumulated_alpha)
                # ========================================================

                render_color[h:h+self.tile_size, w:w+self.tile_size] = tile_color.reshape(self.tile_size, self.tile_size, -1)

        return render_color

@torch.no_grad()
def homogenize(points):
    """
    homogeneous points
    :param points: [..., 3]
    """
    return torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)

@torch.no_grad()
def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device=r.device)

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

@torch.no_grad()
def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device=s.device)
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

@torch.no_grad()
def get_radius(cov2d):
    det = cov2d[:, 0, 0] * cov2d[:, 1, 1] - cov2d[:, 0, 1] * cov2d[:, 1, 0]
    mid = 0.5 * (cov2d[:, 0, 0] + cov2d[:, 1, 1])
    lambda1 = mid + torch.sqrt((mid**2-det).clip(min=0.1))
    lambda2 = mid - torch.sqrt((mid**2-det).clip(min=0.1))
    return 3.0 * torch.sqrt(torch.max(lambda1, lambda2)).ceil()

@torch.no_grad()
def get_rect(pix_coord, radii, width, height):
    rect_min = (pix_coord - radii[:,None])
    rect_max = (pix_coord + radii[:,None])
    rect_min[..., 0] = rect_min[..., 0].clip(0, width - 1.0)
    rect_min[..., 1] = rect_min[..., 1].clip(0, height - 1.0)
    rect_max[..., 0] = rect_max[..., 0].clip(0, width - 1.0)
    rect_max[..., 1] = rect_max[..., 1].clip(0, height - 1.0)
    return rect_min, rect_max
