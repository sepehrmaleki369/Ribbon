from snake import Snake
from gradRib import cmptExtGrad
import torch
import torch.nn.functional as F
import math

class RibbonSnake(Snake):
    def __init__(self, graph, crop, stepsz, alpha, beta, dim):
        # In the new version grad will be separate, so image gradients will not be here
        # Normal initialization of Snake super class
        super().__init__(graph, crop, stepsz, alpha, beta, dim)
        # Additionally we sample from a normal distrubution for widths of nodes
        #self.w = torch.randn(self.s.shape[0]).abs()
        self.w = torch.ones(self.s.shape[0])

    def cuda(self):
        super().cuda()
        # move the widths to gpu
        self.w = self.w.cuda()

    def set_w(self, widths):
        self.w = widths

    def get_w(self):
        return self.w

    def _compute_normals(self, pos):
        """
        Compute normals (and tangents for 3D) for each center point.
        Returns:
         - 2D: (normals,) where normals is (N,2)
         - 3D: (n1, n2, tangents) each (N,3)
        """
        N, d = pos.shape
        eps = 1e-8
        t = torch.zeros_like(pos)
        if N > 1:
            t[1:-1] = (pos[2:] - pos[:-2]) * 0.5
            t[0] = pos[1] - pos[0]
            t[-1] = pos[-1] - pos[-2]
        t = t / (t.norm(dim=1, keepdim=True) + eps)

        if self.ndims == 2:
            normals = torch.stack([-t[:,1], t[:,0]], dim=1)
            normals = normals / (normals.norm(dim=1, keepdim=True) + eps)
            return (normals,)
        else:
            a = torch.zeros_like(pos)
            a[:] = torch.tensor([1.0, 0.0, 0.0], device=pos.device)
            mask = (t * a).abs().sum(dim=1) > 0.9
            a[mask] = torch.tensor([0.0, 1.0, 0.0], device=pos.device)
            n1 = torch.cross(t, a, dim=1)
            n1 = n1 / (n1.norm(dim=1, keepdim=True) + eps)
            n2 = torch.cross(t, n1, dim=1)
            n2 = n2 / (n2.norm(dim=1, keepdim=True) + eps)
            return (n1, n2, t)
        
    def comp_second_deriv(self):
        """
        1D second-derivative smoothing for widths via convolution.
        """
        w = self.w.view(1, 1, -1)
        kernel = torch.tensor([1.0, -4.0, 6.0, -4.0, 1.0], device=w.device).view(1,1,5)
        grad_norm = F.conv1d(w, kernel, padding=2)
        return grad_norm.view(-1,1)
    
    def step_widths(self, gimgW):
        """
        Update widths by sampling gradient of image W at ribbon edges and
        adding internal smoothness via second derivative.
        """
        if self.s.numel() == 0:
            return self.w

        pos = self.s                  # (N, d)
        K, d = pos.shape
        device = pos.device
        half_r = self.w * 0.5 # N

        if d == 2:
            (normals,) = self._compute_normals(pos)
            left_pts  = pos - normals * half_r.unsqueeze(1)
            right_pts = pos + normals * half_r.unsqueeze(1)

            grad_L = cmptExtGrad(left_pts,  gimgW)
            grad_R = cmptExtGrad(right_pts, gimgW)
            # radial derivative
            grad_w = ((grad_R - grad_L) * normals).sum(dim=1, keepdim=True)

        else:  # d == 3
            n1, n2, _ = self._compute_normals(pos)   # each (K,3)

            N = 8
            theta = torch.linspace(0, 2*math.pi, N, device=device, dtype=pos.dtype)[:-1]  # (N-1,)
            dirs = (
                theta.cos().unsqueeze(1).unsqueeze(2) * n1.unsqueeze(0) +
                theta.sin().unsqueeze(1).unsqueeze(2) * n2.unsqueeze(0)
            )  # (N-1, K, 3)

            pts_out = pos.unsqueeze(0) + half_r.unsqueeze(0).unsqueeze(2) * dirs   # (N-1, K, 3)
            pts_in = pos.unsqueeze(0) - half_r.unsqueeze(0).unsqueeze(2) * dirs   # (N-1, K, 3)
            all_pts = torch.cat([pts_out, pts_in], dim=0) # (2(N-1), K, 3)

            grads = cmptExtGrad(all_pts.view(-1,3), gimgW)# (2(N-1)*K, 3)
            grads = grads.view(2*(N-1), -1,3)

            grad_diff = grads[:(N-1)] - grads[(N-1):] # (N-1, K, 3)
            radial = (grad_diff * dirs).sum(dim=2) # (N-1, K)
            grad_w = radial.mean(dim=0, keepdim=True).t() # (K,1)
            norm_grad = self.comp_second_deriv() # (K,1)
            w_flat    = self.w.view(-1) # (K,)
            smooth    = torch.zeros_like(w_flat) # (K,)

            if K > 1:
                smooth[0]   = w_flat[0]   - w_flat[1]
                smooth[-1]  = w_flat[-1]  - w_flat[-2]
                if K > 2:
                    smooth[1:-1] = 2*w_flat[1:-1] - w_flat[:-2] - w_flat[2:]

            smooth = smooth.view(K,1) # (K,1)
            internal = norm_grad + smooth # (K,1)
            alpha = grad_w.abs() / (internal.abs() + 1e-8)
            total = grad_w + alpha * internal # (K,1)
            self.w = self.w - self.stepsz * total.squeeze(1)
            return self.w

        # internal smoothness
        internal = self.comp_second_deriv()
        alpha = grad_w.abs() / (internal.abs() + 1e-8)
        total = (grad_w + alpha * internal).squeeze(1)
        # gradient step
        self.w = self.w - self.stepsz * total
        return self.w
    
    def render_distance_map_with_widths(self, size):
        """
        Unified 2D/3D signed distance map for the ribbon snake.

        Args:
            size (tuple): (W, H) for 2D or (X, Y, Z) for 3D grid dimensions.

        Returns:
            torch.Tensor: Signed distance map of shape `size`, where negative values indicate points inside the ribbon and positive values the distance to the nearest edge.
        """
        device = self.s.device
        centers = self.s
        radii   = (self.w.flatten() / 2)
        axes    = [torch.arange(sz, device=device, dtype=torch.float32) for sz in size]
        mesh    = torch.meshgrid(*axes, indexing='ij')
        points  = torch.stack([m.flatten() for m in mesh], dim=1)

        # capsule sides
        if centers.shape[0] > 1:
            starts, ends = centers[:-1], centers[1:]
            r0, r1 = radii[:-1], radii[1:]
            vec = ends - starts
            L   = vec.norm(dim=1, keepdim=True)
            D   = vec / (L + 1e-8)

            P = points.unsqueeze(1)
            S = starts.unsqueeze(0)
            D = D.unsqueeze(0)
            L = L.unsqueeze(0).squeeze(-1)

            v    = P - S
            proj = (v * D).sum(dim=2)
            # clamp proj to [0, L]
            low  = proj.clamp(min=0.0)
            t    = torch.min(low, L)

            closest       = S + D * t.unsqueeze(-1)
            dist_axis     = (P - closest).norm(dim=2)
            frac          = t / (L + 1e-8)
            interp_radius = r0 * (1-frac) + r1 * frac
            dist_capsule, _ = (dist_axis - interp_radius).min(dim=1)
        else:
            dist_capsule = torch.full((points.shape[0],), float('inf'), device=device)

        # end caps
        start_c = centers[0]; end_c = centers[-1]
        dist_start = (points - start_c).norm(dim=1) - radii[0]
        dist_end   = (points - end_c).norm(dim=1)   - radii[-1]

        dist = torch.min(dist_capsule, dist_start)
        dist = torch.min(dist, dist_end)
        #dist.requires_grad_(True)
        return torch.clamp(dist.reshape(*size), max=16)