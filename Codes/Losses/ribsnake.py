import torch as th
import torch
import numpy as np
from Codes.Losses.snake import Snake
from Codes.Losses.gradImSnake import cmptExtGrad
import torch.nn.functional as F
from matplotlib.path import Path

#######################
# DIST MAP 2D

def dist_perfectV(samples_with_widths, size, normals, iscuda=False):
    w, h = size

    centers = np.array([c for c, _ in samples_with_widths], dtype=np.float32)
    widths = np.array([w for _, w in samples_with_widths], dtype=np.float32)
    normals = np.array(normals, dtype=np.float32)

    if len(centers) == 0:
        dist_map_t = torch.zeros((w, h), dtype=torch.float32)
        return dist_map_t.cuda() if iscuda else dist_map_t

    left_pts = centers - (normals * widths[:, None] / 2)
    right_pts = centers + (normals * widths[:, None] / 2)

    end_cap_points = np.zeros((0, 2), dtype=np.float32)
    if len(centers) > 0:
        C_end = centers[-1]
        r_end = widths[-1] / 2
        P_end = left_pts[-1]

        dx = P_end[0] - C_end[0]
        dy = P_end[1] - C_end[1]
        start_angle = np.arctan2(dy, dx)
        end_angle = start_angle + np.pi

        num_segments = 20
        theta = np.linspace(start_angle, end_angle, num_segments + 1)
        x = C_end[0] + r_end * np.cos(theta)
        y = C_end[1] + r_end * np.sin(theta)
        end_cap_points = np.column_stack((x, y))[1:-1]

    start_cap_points = np.zeros((0, 2), dtype=np.float32)
    if len(centers) > 0:
        C_start = centers[0]
        r_start = widths[0] / 2
        P_start = right_pts[0]

        dx = P_start[0] - C_start[0]
        dy = P_start[1] - C_start[1]
        start_angle = np.arctan2(dy, dx)
        end_angle = start_angle + np.pi

        num_segments = 20
        theta = np.linspace(start_angle, end_angle, num_segments + 1)
        x = C_start[0] + r_start * np.cos(theta)
        y = C_start[1] + r_start * np.sin(theta)
        start_cap_points = np.column_stack((x, y))[1:]

    reversed_right_pts = right_pts[::-1]
    if len(reversed_right_pts) > 0:
        reversed_right_pts = reversed_right_pts[1:]

    polygon = np.vstack([
        left_pts,
        end_cap_points,
        reversed_right_pts,
        start_cap_points
    ])
    path = Path(polygon)

    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    grid_points = np.column_stack([xx.ravel(), yy.ravel()])
    is_inside = path.contains_points(grid_points).reshape(h, w).T

    dist_map = np.full((w, h), np.inf, dtype=np.float32)
    num_vertices = polygon.shape[0]
    for i in range(num_vertices):
        p1 = polygon[i]
        p2 = polygon[(i + 1) % num_vertices]
        seg_dist = qq(grid_points, p1, p2).reshape(h, w).T
        dist_map = np.minimum(dist_map, seg_dist)

    dist_map = np.where(is_inside, -dist_map, dist_map)

    dist_map_t = torch.from_numpy(dist_map).float()
    if iscuda:
        dist_map_t = dist_map_t.cuda()
    return dist_map_t

def qq(points, seg_p1, seg_p2):
    seg_vec = seg_p2 - seg_p1
    seg_len = np.linalg.norm(seg_vec)
    if seg_len < 1e-8:
        return np.linalg.norm(points - seg_p1, axis=1)
    
    # treat seg_p1 as the origin, then calculate 
    t = np.dot(points - seg_p1, seg_vec) / (seg_len ** 2)
    t = np.clip(t, 0, 1)
    proj = seg_p1 + t[:, None] * seg_vec
    return np.linalg.norm(points - proj, axis=1)


#######################
# DIST MAP 3D

def compute_3d_distance_map(centers, widths, tangents, grid_shape, voxel_size, origin, device='cuda'):
    centers_tensor = torch.tensor(centers, dtype=torch.float32, device=device)
    widths_tensor = torch.tensor(widths, dtype=torch.float32, device=device)
    tangents_tensor = torch.tensor(tangents, dtype=torch.float32, device=device)

    nx, ny, nz = grid_shape
    x = torch.linspace(origin[0], origin[0] + (nx-1)*voxel_size[0], nx, device=device)
    y = torch.linspace(origin[1], origin[1] + (ny-1)*voxel_size[1], ny, device=device)
    z = torch.linspace(origin[2], origin[2] + (nz-1)*voxel_size[2], nz, device=device)
    grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing='ij')
    points = torch.stack([grid_x.flatten(), grid_y.flatten(), grid_z.flatten()], dim=1)

    seg_start = centers_tensor[:-1]
    seg_end = centers_tensor[1:]
    seg_radii_start = widths_tensor[:-1] / 2
    seg_radii_end = widths_tensor[1:] / 2
    
    vec_seg = seg_end - seg_start
    seg_length = torch.norm(vec_seg, dim=1, keepdim=True)
    seg_dir = vec_seg / (seg_length + 1e-8)
    
    points_exp = points.unsqueeze(1)
    seg_start_exp = seg_start.unsqueeze(0)
    seg_dir_exp = seg_dir.unsqueeze(0)
    seg_length_exp = seg_length.unsqueeze(0)
    
    vec_to_segment = points_exp - seg_start_exp
    
    proj = torch.sum(vec_to_segment * seg_dir_exp, dim=2)
    t = torch.clamp(proj, torch.tensor(0, device=device), seg_length_exp.squeeze(-1))
    
    closest_points = seg_start_exp + t.unsqueeze(-1) * seg_dir_exp
    
    dist_to_axis = torch.norm(points_exp - closest_points, dim=2)
    
    t_ratio = t / (seg_length_exp.squeeze(-1) + 1e-8)
    radii_interp = seg_radii_start * (1 - t_ratio) + seg_radii_end * t_ratio
    
    signed_dist_segments = dist_to_axis - radii_interp
    min_segment_dist, _ = torch.min(signed_dist_segments, dim=1)
    
    def hemisphere_dist(center, radius, direction, points):
        vec = points - center
        sphere_dist = torch.norm(vec, dim=1) - radius
        direction = direction / torch.norm(direction)
        dot = torch.sum(vec * direction, dim=1)
        valid = dot >= 0
        return torch.where(valid, sphere_dist, torch.inf)
    
    start_center = centers_tensor[0]
    start_radius = widths_tensor[0] / 2
    start_dir = -tangents_tensor[0]
    dist_start = hemisphere_dist(start_center, start_radius, start_dir, points)
    
    end_center = centers_tensor[-1]
    end_radius = widths_tensor[-1] / 2
    end_dir = tangents_tensor[-1]
    dist_end = hemisphere_dist(end_center, end_radius, end_dir, points)
    
    final_dist = torch.minimum(min_segment_dist, dist_start)
    final_dist = torch.minimum(final_dist, dist_end)
    
    return final_dist.reshape(grid_shape).cpu()

# need to be adjusted to calculate the dist map correctly
# TODO: doesn'T calculate the grid_shape
def calculate_grid_parameters(centers, widths, padding=10, voxel_resolution=1.0):
    centers = np.array(centers)
    
    #min_coords = np.min(centers - widths[:, None]/2, axis=0) - padding
    min_coords = np.array([0,0,0])
    max_coords = np.max(centers + widths[:, None]/2, axis=0) + padding
    
    grid_size = max_coords - min_coords
    #grid_shape = tuple(np.ceil(grid_size / voxel_resolution).astype(int))
    grid_shape = (100,100,100)
    
    return {
        'origin': min_coords,
        'voxel_size': (voxel_resolution, voxel_resolution, voxel_resolution),
        'grid_shape': grid_shape
    }


class RibbonSnake(Snake):
    def __init__(self, graph, crop, stepsz, alpha, beta, gimgN, gimgW, step_type="original", ndims=2):
        super().__init__(graph, crop, stepsz, alpha, beta, ndims)
        # TODO sample widths between 0,1
        self.w = th.ones(len(self.s), 1, dtype=th.float32) * 1.0
        self.step_type = step_type
        if self.s.is_cuda:
            self.s = self.s.cuda()
        
        self.gimgN = gimgN
        self.gimgW = gimgW

    def cuda(self):
        super().cuda()
        if self.gimgN is not None:
            self.gimgN = self.gimgN.cuda()
        if self.gimgW is not None:
            self.gimgW = self.gimgW.cuda()
    
    def _compute_normals_complex(self, pos_xy):
        if self.ndims == 2:
            k = pos_xy.size(0)
            normals = th.zeros_like(pos_xy)
            eps = 1e-8

            def perp(vec):
                return th.tensor([-vec[1], vec[0]], dtype=vec.dtype, device=vec.device)

            if k >= 2:
                tang = pos_xy[1] - pos_xy[0]
                tang_norm = th.norm(tang) + eps
                tang = tang / tang_norm
                normals[0] = perp(tang)
            
            for i in range(1, k-1):
                t1 = pos_xy[i] - pos_xy[i-1]
                t2 = pos_xy[i+1] - pos_xy[i]
                n1 = th.norm(t1) + eps
                n2 = th.norm(t2) + eps
                t1 = t1 / n1
                t2 = t2 / n2
                tangent = t1 + t2
                
                if th.norm(tangent) < 1e-5:
                    tangent = t1

                tangent = tangent / (th.norm(tangent) + eps)
                normals[i] = perp(tangent)
            
            if k >= 2:
                tang = pos_xy[-1] - pos_xy[-2]
                tang = tang / (th.norm(tang) + eps)
                normals[-1] = perp(tang)
            return normals
        
        elif self.ndims == 3:
            pos_xyz = pos_xy
            k = pos_xyz.size(0)
            normals1 = th.zeros_like(pos_xyz)
            normals2 = th.zeros_like(pos_xyz)
            eps = 1e-8
            tangents = []
            if k >= 2:
                tang = pos_xyz[1] - pos_xyz[0]
                tang_norm = th.norm(tang) + eps
                tang = tang / tang_norm
                tangents.append(tang)
                arbitrary_vector = th.tensor([1.0, 0.0, 0.0], dtype=tang.dtype, device=tang.device)
                if th.allclose(tang, arbitrary_vector):
                    arbitrary_vector = th.tensor([0.0, 1.0, 0.0], dtype=tang.dtype, device=tang.device)
                normals1[0] = th.cross(tang, arbitrary_vector)
                normals1[0] = normals1[0] / (th.norm(normals1[0]) + eps)
                normals2[0] = th.cross(tang, normals1[0])
                normals2[0] = normals2[0] / (th.norm(normals2[0]) + eps)
            
            for i in range(1, k-1):
                t1 = pos_xyz[i] - pos_xyz[i-1]
                t2 = pos_xyz[i+1] - pos_xyz[i]
                n1 = th.norm(t1) + eps
                n2 = th.norm(t2) + eps
                t1 = t1 / n1
                t2 = t2 / n2

                tangent = t1 + t2
                if th.norm(tangent) < 1e-5:
                    tangent = t1

                tangent = tangent / (th.norm(tangent) + eps)
                tangents.append(tangent)
                arbitrary_vector = th.tensor([1.0, 0.0, 0.0], dtype=tangent.dtype, device=tangent.device)
                if th.allclose(tangent, arbitrary_vector):
                    arbitrary_vector = th.tensor([0.0, 1.0, 0.0], dtype=tangent.dtype, device=tangent.device)
                normals1[i] = th.cross(tangent, arbitrary_vector)
                normals1[i] = normals1[i] / (th.norm(normals1[i]) + eps)
                normals2[i] = th.cross(tang, normals1[i])
                normals2[i] = normals2[i] / (th.norm(normals2[i]) + eps)
            
            if k >= 2:
                tang = pos_xyz[-1] - pos_xyz[-2]
                tang = tang / (th.norm(tang) + eps)
                tangents.append(tang)
                arbitrary_vector = th.tensor([1.0, 0.0, 0.0], dtype=tangent.dtype, device=tangent.device)
                if th.allclose(tangent, arbitrary_vector):
                    arbitrary_vector = th.tensor([0.0, 1.0, 0.0], dtype=tangent.dtype, device=tangent.device)
                normals1[-1] = th.cross(tang, arbitrary_vector)
                normals1[-1] = normals1[-1] / (th.norm(normals1[-1]) + eps)
                normals2[-1] = th.cross(tang, normals1[-1])
                normals2[-1] = normals2[-1] / (th.norm(normals2[-1]) + eps)

            return normals1, normals2, tangents

    def step(self):
        if len(self.s) == 0:
            return
        super().step(cmptExtGrad(self.s, self.gimgN))
        return self.s

    def comp_second_deriv(self):
        w = self.w
        n = w.shape[0]
        if n < 5:
            grad_norm = th.zeros_like(w)
            if n >= 3:
                grad_norm[1:-1] = (w[:-2] - 2 * w[1:-1] + w[2:])
            return grad_norm

        w_reshaped = w.view(1, 1, -1)
        kernel = th.tensor([1.0, -4.0, 6.0, -4.0, 1.0], dtype=w.dtype, device=w.device).view(1, 1, 5)

        grad_norm = F.conv1d(w_reshaped, kernel, padding=2)
        grad_norm = grad_norm
        return grad_norm.view(-1, 1)
    
    def step_widths(self):
        if len(self.s) == 0:
            return

        if self.ndims == 2:
            pos_xy = self.s
            normals = self._compute_normals_complex(pos_xy)
            v_L = pos_xy - self.w * normals / 2
            v_R = pos_xy + self.w * normals / 2

            grad_L = cmptExtGrad(v_L, self.gimgW)
            grad_R = cmptExtGrad(v_R, self.gimgW)
            grad_w = th.sum((-grad_L + grad_R) * normals, dim=1, keepdim=True)

            smooth_grad = th.zeros_like(grad_w)
            if len(self.s) > 1:
                smooth_grad[0] = (self.w[0] - self.w[1])
                smooth_grad[1:-1] = (2 * self.w[1:-1] - self.w[:-2] - self.w[2:])
                smooth_grad[-1] = (self.w[-1] - self.w[-2])

            norm_grad = self.comp_second_deriv()
            internal = norm_grad + smooth_grad
            alpha = abs(grad_w) / abs(internal + 1e-8)

            total_grad = grad_w + alpha * internal
            self.w = self.w - self.stepsz * total_grad
        
        # TODO: 3D case
        elif self.ndims == 3:
            pos_xyz = self.s
            k = pos_xyz.size(0)
            N = 8
            normals1, normals2, _ = self._compute_normals_complex(pos_xyz)

            grad_w = th.zeros(k, 1, dtype=th.float32, device=pos_xyz.device)

            theta = th.linspace(0, th.pi, N, device=pos_xyz.device, dtype=th.float32)[:-1]
            cos_theta = th.cos(theta)
            sin_theta = th.sin(theta)

            for i in range(k):
                n1 = normals1[i]
                n2 = normals2[i]

                r_i = (cos_theta[:, None] * n1 + sin_theta[:, None] * n2)
                v_i = pos_xyz[i] + (self.w[i] / 2) * r_i
                v_opp = pos_xyz[i] - (self.w[i] / 2) * r_i

                # TODO: might need to take samples for each dimension in different orders
                # (positive negative issue created by distance map gradients)

                grad_i = cmptExtGrad(v_i, self.gimgW)
                grad_opp = cmptExtGrad(v_opp, self.gimgW)

                contrib = th.sum((grad_i - grad_opp) * r_i, dim=1)
                grad_w[i] = contrib.mean()

                smooth_grad = th.zeros_like(grad_w)
                if k > 1:
                    smooth_grad[0] = (self.w[0] - self.w[1])
                    smooth_grad[1:-1] = (2 * self.w[1:-1] - self.w[:-2] - self.w[2:])
                    smooth_grad[-1] = (self.w[-1] - self.w[-2])
                norm_grad = self.comp_second_deriv()
                internal = norm_grad + smooth_grad
                alpha = abs(grad_w) / abs(internal + 1e-8)
                total_grad = grad_w + alpha * internal
            self.w = self.w - self.stepsz * grad_w

        return self.w

    def optim(self, niter):
        if self.step_type == "original":
            for i in range(niter):
                if i < 50:
                    self.step()
                else:
                    self.step_widths()
        elif self.step_type == "combined":
            for i in range(niter):
                self.step()
                self.step_widths()
        return self.s

    def render_distance_map(self, size):
        widths = self.get_w().flatten()
        gra = self.getGraph()
        nodes = [gra.nodes[n]['pos'] for n in gra.nodes()]
        samples_to_widths = list(zip(nodes, widths))

        if self.ndims == 2:
            distmap = dist_perfectV(samples_to_widths, size, self._compute_normals_complex(self.s))
            return torch.clamp(distmap, max=16)
        elif self.ndims == 3:
            self._compute_normals_complex(self.s)
            centers = np.array([c for c, _ in samples_to_widths], dtype=np.float32)
            widths = np.array([w for _, w in samples_to_widths], dtype=np.float32)
            _, _, tangents = self._compute_normals_complex(self.s)
            tangents = np.array(tangents)
            params = calculate_grid_parameters(centers, widths, padding=10, voxel_resolution=1.0)
            distmap = compute_3d_distance_map(
                            centers, widths, tangents,
                            grid_shape=params['grid_shape'],
                            voxel_size=params['voxel_size'],
                            origin=params['origin']
                        )
            return torch.clamp(distmap, max=16)
    
    def get_s(self):
        return self.s
    
    def get_w(self):
        return self.w
    
    def set_w(self, w):
        self.w = w