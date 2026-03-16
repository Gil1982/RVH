"""
fit smplh to scans

crated by Xianghui, 12, January 2022

the code is tested
"""

import sys, os
sys.path.append(os.getcwd())
import json
import math
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from os.path import exists
from pytorch3d.loss import point_mesh_face_distance, chamfer_distance
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.ops import sample_points_from_meshes
from smpl_registration.base_fitter import BaseFitter
from lib.body_objectives import batch_get_pose_obj, batch_3djoints_loss
from lib.smpl.priors.th_smpl_prior import get_prior
from lib.smpl.priors.th_hand_prior import HandPrior
from lib.smpl.wrapper_pytorch import SMPLPyTorchWrapperBatchSplitParams


# ---------------------------------------------------------------------
# Landmark / joint definitions inspired by DavidBoja/SMPL-Anthropometry
# ---------------------------------------------------------------------
SMPL_LANDMARK_INDICES = {
    "HEAD_TOP": 412,
    "LEFT_HEEL": 3458,
    "RIGHT_HEEL": 6858,
    "BELLY_BUTTON": 3501,
    "BACK_BELLY_BUTTON": 3022,
    "PUBIC_BONE": 3145,
    "RIGHT_BICEP": 4855,
    "LEFT_THIGH": 947,
}

SMPL_JOINT2IND = {
    'pelvis': 0,
    'left_hip': 1,
    'right_hip': 2,
    'spine1': 3,
    'left_knee': 4,
    'right_knee': 5,
    'spine2': 6,
    'left_ankle': 7,
    'right_ankle': 8,
    'spine3': 9,
    'left_foot': 10,
    'right_foot': 11,
    'neck': 12,
    'left_collar': 13,
    'right_collar': 14,
    'head': 15,
    'left_shoulder': 16,
    'right_shoulder': 17,
    'left_elbow': 18,
    'right_elbow': 19,
    'left_wrist': 20,
    'right_wrist': 21,
    'left_hand': 22,
    'right_hand': 23,
}



class SMPLHFitter(BaseFitter):
    def fit(self, scans, pose_files, gender='male', save_path=None):
        # Batch size
        batch_sz = len(scans)


        # Load scans and center them. Once smpl is registered, move it accordingly.
        th_scan_meshes, centers = self.load_scans(scans, ret_cent=True)

        # init smpl
        smpl = self.init_smpl(batch_sz, gender, trans=centers) # add centers as initial SMPL translation

        # Set optimization hyper parameters
        iterations = getattr(self, 'iterations', 8)
        pose_iterations = getattr(self, 'pose_iterations', 0)
        steps_per_iter = getattr(self, 'steps_per_iter', 40)
        pose_steps_per_iter = getattr(self, 'pose_steps_per_iter', 0)

        th_pose_3d = None
        #if pose_files is not None:
        if th_pose_3d is not None:
            #th_pose_3d = self.load_j3d(pose_files)
            #th_pose_3d = None

            # Optimize pose first
            self.optimize_pose_only(th_scan_meshes, smpl, pose_iterations, pose_steps_per_iter, th_pose_3d)

        # Optimize pose and shape
        self.optimize_pose_shape(th_scan_meshes, smpl, iterations, steps_per_iter, th_pose_3d)

        fit_score = self.compute_fit_score(th_scan_meshes, smpl)
        print('** Fit score **', json.dumps(fit_score, indent=2))

        if save_path is not None:
            if not exists(save_path):
                os.makedirs(save_path)
            with open(os.path.join(save_path, 'fit_score.json'), 'w') as f:
                json.dump(fit_score, f, indent=2)
            return self.save_outputs(save_path, scans, smpl, th_scan_meshes, save_name='smplh' if self.hands else 'smpl')

        return fit_score

    def optimize_pose_shape(self, th_scan_meshes, smpl, iterations, steps_per_iter, th_pose_3d=None):
        # Optimizer
        lr = getattr(self, 'pose_shape_lr', 0.02)
        optimizer = torch.optim.Adam([smpl.trans, smpl.betas, smpl.pose], lr, betas=(0.9, 0.999))
        # Get loss_weights
        weight_dict = self.get_loss_weights()

        for it in range(iterations):
            loop = tqdm(range(steps_per_iter))
            loop.set_description('Optimizing SMPL')
            for i in loop:
                optimizer.zero_grad()
                # Get losses for a forward pass
                loss_dict = self.forward_pose_shape(th_scan_meshes, smpl, th_pose_3d)
                # Get total loss for backward pass
                tot_loss = self.backward_step(loss_dict, weight_dict, it)
                tot_loss.backward()
                optimizer.step()

                l_str = 'Iter: {}'.format(i)
                for k in loss_dict:
                    l_str += ', {}: {:0.4f}'.format(k, weight_dict[k](loss_dict[k], it).mean().item())
                    loop.set_description(l_str)

                if self.debug:
                    self.viz_fitting(smpl, th_scan_meshes)

        print('** Optimised smpl pose and shape **')

    # def forward_pose_shape(self, th_scan_meshes, smpl, th_pose_3d=None):
        # # Get pose prior
        # prior = get_prior(self.model_root, smpl.gender)

        # # forward
        # verts, _, _, _ = smpl()
        # th_smpl_meshes = Meshes(verts=verts, faces=torch.stack([smpl.faces] * len(verts), dim=0))

        # # losses
        # loss = dict()
        # loss['s2m'] = point_mesh_face_distance(th_smpl_meshes, Pointclouds(points=th_scan_meshes.verts_list()))
        # loss['m2s'] = point_mesh_face_distance(th_scan_meshes, Pointclouds(points=th_smpl_meshes.verts_list()))
        # loss['betas'] = torch.mean(smpl.betas ** 2)
        # loss['pose_pr'] = torch.mean(prior(smpl.pose))
        # if self.hands:
            # hand_prior = HandPrior(self.model_root, type='grab')
            # loss['hand'] = torch.mean(hand_prior(smpl.pose)) # add hand prior if smplh is used
        # if th_pose_3d is not None:
            # # 3D joints loss
            # J, face, hands = smpl.get_landmarks()
            # joints = self.compose_smpl_joints(J, face, hands, th_pose_3d)
            # j3d_loss = batch_3djoints_loss(th_pose_3d, joints)
            # loss['pose_obj'] = j3d_loss
            # # loss['pose_obj'] = batch_get_pose_obj(th_pose_3d, smpl).mean()
        # return loss
    
    def forward_pose_shape(self, th_scan_meshes, smpl, th_pose_3d=None):
        # Get pose prior
        prior = get_prior(self.model_root, smpl.gender)

        # forward
        verts, _, _, _ = smpl()
        th_smpl_meshes = Meshes(
            verts=verts,
            faces=torch.stack([smpl.faces] * len(verts), dim=0)
        )

        # losses
        loss = dict()

        # point-to-mesh losses originais
        loss['s2m'] = point_mesh_face_distance(
            th_smpl_meshes,
            Pointclouds(points=th_scan_meshes.verts_list())
        )

        loss['m2s'] = point_mesh_face_distance(
            th_scan_meshes,
            Pointclouds(points=th_smpl_meshes.verts_list())
        )

        # ------------------------------------------------------------------
        # NOVA LOSS: Chamfer distance entre pontos amostrados do SMPL e scan
        # ------------------------------------------------------------------
        
        num_samples = 10000

        smpl_samples = sample_points_from_meshes(
            th_smpl_meshes,
            num_samples=num_samples
        )

        # Usa apenas a região mais central do scan para a Chamfer.
        # Isso reduz a tendência de distorcer extremidades (ombros/braços/mãos)
        # para explicar ruído ou oclusões locais.
        filtered_scan_cloud = self.mask_scan_vertices(th_scan_meshes, x_quantile_hands=0.88)
        scan_points_list = []
        for scan_verts in filtered_scan_cloud.points_list():
            n_scan = scan_verts.shape[0]
            if n_scan == 0:
                scan_verts = th_scan_meshes.verts_list()[0]
                n_scan = scan_verts.shape[0]

            if n_scan >= num_samples:
                idx = torch.randperm(n_scan, device=scan_verts.device)[:num_samples]
            else:
                idx = torch.randint(0, n_scan, (num_samples,), device=scan_verts.device)
            scan_points_list.append(scan_verts[idx])

        scan_points = torch.stack(scan_points_list, dim=0)

        chamfer_loss, _ = chamfer_distance(
            smpl_samples,
            scan_points)
            
        
        
        loss['chamfer'] = chamfer_loss


        # --------------------------------------------------------------
        # Anthropometric losses: scan target vs SMPL (DavidBoja-inspired)
        # --------------------------------------------------------------
        J, face, hands = smpl.get_landmarks()
        scan_targets = self.scan_measurement_targets(th_scan_meshes)
        smpl_meas = self.measure_smpl_anthropometrics(verts, J)

        loss['meas_height'] = torch.mean(torch.abs(smpl_meas['height'] - scan_targets['height']))
        loss['meas_waist'] = torch.mean(torch.abs(smpl_meas['waist_circ'] - scan_targets['waist_circ']))
        loss['meas_hip'] = torch.mean(torch.abs(smpl_meas['hip_circ'] - scan_targets['hip_circ']))
        loss['meas_biceps'] = torch.mean(torch.abs(smpl_meas['biceps_circ'] - scan_targets['biceps_circ']))
        loss['meas_thigh'] = torch.mean(torch.abs(smpl_meas['thigh_circ'] - scan_targets['thigh_circ']))

        # regularização de shape
        loss['betas'] = torch.mean(smpl.betas ** 2)

        # prior de pose
        loss['pose_pr'] = torch.mean(prior(smpl.pose))



        if self.hands:
            hand_prior = HandPrior(self.model_root, type='grab')
            loss['hand'] = torch.mean(hand_prior(smpl.pose))

        if th_pose_3d is not None:
            # 3D joints loss
            J, face, hands = smpl.get_landmarks()
            joints = self.compose_smpl_joints(J, face, hands, th_pose_3d)
            j3d_loss = batch_3djoints_loss(th_pose_3d, joints)
            loss['pose_obj'] = j3d_loss

        return loss

    
    @staticmethod
    def _convex_hull_perimeter(points_2d):
        """Perimeter of the convex hull for a Nx2 tensor/array."""
        pts = points_2d.detach().cpu().numpy() if torch.is_tensor(points_2d) else points_2d
        if pts.shape[0] < 3:
            return 0.0

        pts = sorted(set((float(x), float(y)) for x, y in pts))
        if len(pts) < 3:
            return 0.0

        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        lower = []
        for p in pts:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)

        upper = []
        for p in reversed(pts):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)

        hull = lower[:-1] + upper[:-1]
        if len(hull) < 3:
            return 0.0

        perim = 0.0
        for i in range(len(hull)):
            x1, y1 = hull[i]
            x2, y2 = hull[(i + 1) % len(hull)]
            perim += math.hypot(x2 - x1, y2 - y1)
        return perim

    def _infer_body_axes(self, verts):
        """Infer (up, left-right, depth) axes from bbox extents."""
        ext = torch.max(verts, dim=0).values - torch.min(verts, dim=0).values
        up_axis = int(torch.argmax(ext).item())
        rem = [a for a in [0, 1, 2] if a != up_axis]
        if ext[rem[0]] >= ext[rem[1]]:
            lr_axis, depth_axis = rem[0], rem[1]
        else:
            lr_axis, depth_axis = rem[1], rem[0]
        return up_axis, lr_axis, depth_axis

    def _circumference_at_rel_height(self, verts, rel_height, band_rel=0.01, side=None):
        up_axis, lr_axis, depth_axis = self._infer_body_axes(verts)
        coord_up = verts[:, up_axis]
        up_min = torch.min(coord_up)
        up_max = torch.max(coord_up)
        h = torch.clamp(up_max - up_min, min=1e-8)

        up_target = up_min + rel_height * h
        band = torch.clamp(band_rel * h, min=1e-5)

        keep = torch.abs(coord_up - up_target) <= band
        if side == 'left':
            keep = keep & (verts[:, lr_axis] > 0)
        elif side == 'right':
            keep = keep & (verts[:, lr_axis] < 0)

        pts = verts[keep][:, [lr_axis, depth_axis]]
        if pts.shape[0] < 16:
            return float('nan')

        return self._convex_hull_perimeter(pts)

    def _best_circumference_in_window(self, verts, rel_range, mode='max', side=None, band_rel=0.012, steps=15):
        rel_values = np.linspace(rel_range[0], rel_range[1], steps)
        best = float('nan')
        best_rel = float('nan')

        for rel_h in rel_values:
            c = self._circumference_at_rel_height(
                verts,
                rel_height=float(rel_h),
                band_rel=band_rel,
                side=side
            )
            if math.isnan(c):
                continue

            if math.isnan(best):
                best, best_rel = c, float(rel_h)
                continue

            if (mode == 'max' and c > best) or (mode == 'min' and c < best):
                best, best_rel = c, float(rel_h)

        return best, best_rel

    def anthropometric_measurements(self, verts):
        """
        Heuristic scan measurements (meters).
        Kept as a stable target for scan-vs-SMPL comparison.
        """
        up_axis, _, _ = self._infer_body_axes(verts)
        up = verts[:, up_axis]
        height = torch.max(up) - torch.min(up)

        waist, waist_rel = self._best_circumference_in_window(
            verts, rel_range=(0.56, 0.70), mode='min', band_rel=0.012
        )
        hip, hip_rel = self._best_circumference_in_window(
            verts, rel_range=(0.45, 0.60), mode='max', band_rel=0.012
        )
        biceps_l, biceps_l_rel = self._best_circumference_in_window(
            verts, rel_range=(0.67, 0.80), mode='max', side='left', band_rel=0.010
        )
        biceps_r, biceps_r_rel = self._best_circumference_in_window(
            verts, rel_range=(0.67, 0.80), mode='max', side='right', band_rel=0.010
        )
        thigh_l, thigh_l_rel = self._best_circumference_in_window(
            verts, rel_range=(0.30, 0.48), mode='max', side='left', band_rel=0.012
        )
        thigh_r, thigh_r_rel = self._best_circumference_in_window(
            verts, rel_range=(0.30, 0.48), mode='max', side='right', band_rel=0.012
        )

        def safe_mean(a, b):
            vals = [v for v in [a, b] if not math.isnan(v)]
            return float(sum(vals) / len(vals)) if vals else float('nan')

        return {
            'height': float(height.detach().cpu().item()),
            'waist_circ': float(waist),
            'hip_circ': float(hip),
            'biceps_circ': safe_mean(biceps_l, biceps_r),
            'thigh_circ': safe_mean(thigh_l, thigh_r),
            'levels_rel': {
                'waist': waist_rel,
                'hip': hip_rel,
                'biceps_left': biceps_l_rel,
                'biceps_right': biceps_r_rel,
                'thigh_left': thigh_l_rel,
                'thigh_right': thigh_r_rel,
            }
        }

    # ------------------------------------------------------------------
    # Differentiable anthropometric measurements for SMPL
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_vertex_mean(verts_b, index_or_tuple):
        if isinstance(index_or_tuple, (tuple, list)):
            idx = torch.as_tensor(index_or_tuple, device=verts_b.device, dtype=torch.long)
            return verts_b[idx].mean(dim=0)
        return verts_b[index_or_tuple]

    @staticmethod
    def _plane_basis_from_normal(normal):
        normal = normal / torch.clamp(torch.linalg.norm(normal), min=1e-8)

        ref = torch.tensor([1.0, 0.0, 0.0], device=normal.device, dtype=normal.dtype)
        if torch.abs(torch.dot(normal, ref)) > 0.9:
            ref = torch.tensor([0.0, 1.0, 0.0], device=normal.device, dtype=normal.dtype)

        u = torch.cross(normal, ref, dim=0)
        u = u / torch.clamp(torch.linalg.norm(u), min=1e-8)
        v = torch.cross(normal, u, dim=0)
        v = v / torch.clamp(torch.linalg.norm(v), min=1e-8)
        return u, v

    def _soft_slice_circumference(
        self,
        verts_b,
        plane_origin,
        plane_normal,
        sigma=0.008,
        topk=1024
        ):
        """
        Smooth approximation of a circumference by softly selecting
        vertices near a cutting plane and computing an ordered polygonal loop.
        """
        normal = plane_normal / torch.clamp(torch.linalg.norm(plane_normal), min=1e-8)
        rel = verts_b - plane_origin[None, :]                       # (V, 3)
        signed_dist = (rel * normal[None, :]).sum(dim=-1)          # (V,)
        weights = torch.exp(-0.5 * (signed_dist / sigma) ** 2)     # (V,)

        k = min(topk, verts_b.shape[0])
        topw, idx = torch.topk(weights, k=k, dim=0)
        pts = rel[idx]                                             # (k, 3)

        u, v = self._plane_basis_from_normal(normal)
        x = (pts * u[None, :]).sum(dim=-1)
        y = (pts * v[None, :]).sum(dim=-1)

        denom = torch.clamp(topw.sum(), min=1e-8)
        cx = (topw * x).sum() / denom
        cy = (topw * y).sum() / denom

        ang = torch.atan2(y - cy, x - cx)
        order = torch.argsort(ang)

        xs = x[order]
        ys = y[order]
        ws = topw[order]

        xs_next = torch.roll(xs, shifts=-1, dims=0)
        ys_next = torch.roll(ys, shifts=-1, dims=0)
        ws_next = torch.roll(ws, shifts=-1, dims=0)

        seg = torch.sqrt((xs_next - xs) ** 2 + (ys_next - ys) ** 2 + 1e-8)
        seg_w = 0.5 * (ws + ws_next)

        perim = (seg * seg_w).sum() / torch.clamp(seg_w.sum(), min=1e-8)
        return perim

    def _measure_height_smpl(self, verts_b):
        up_axis, _, _ = self._infer_body_axes(verts_b)
        head_top = verts_b[SMPL_LANDMARK_INDICES["HEAD_TOP"], up_axis]
        heels = verts_b[
            [SMPL_LANDMARK_INDICES["LEFT_HEEL"], SMPL_LANDMARK_INDICES["RIGHT_HEEL"]],
            up_axis
        ].mean()
        return torch.abs(head_top - heels)

    def _measure_waist_smpl(self, verts_b, joints_b):
        origin = 0.5 * (
            verts_b[SMPL_LANDMARK_INDICES["BELLY_BUTTON"]] +
            verts_b[SMPL_LANDMARK_INDICES["BACK_BELLY_BUTTON"]]
        )
        normal = joints_b[SMPL_JOINT2IND["spine3"]] - joints_b[SMPL_JOINT2IND["pelvis"]]
        return self._soft_slice_circumference(verts_b, origin, normal)

    def _measure_hip_smpl(self, verts_b, joints_b):
        origin = verts_b[SMPL_LANDMARK_INDICES["PUBIC_BONE"]]
        normal = joints_b[SMPL_JOINT2IND["spine3"]] - joints_b[SMPL_JOINT2IND["pelvis"]]
        return self._soft_slice_circumference(verts_b, origin, normal)

    def _measure_biceps_smpl(self, verts_b, joints_b):
        origin = verts_b[SMPL_LANDMARK_INDICES["RIGHT_BICEP"]]
        normal = joints_b[SMPL_JOINT2IND["right_shoulder"]] - joints_b[SMPL_JOINT2IND["right_elbow"]]
        return self._soft_slice_circumference(verts_b, origin, normal)

    def _measure_thigh_smpl(self, verts_b, joints_b):
        origin = verts_b[SMPL_LANDMARK_INDICES["LEFT_THIGH"]]
        normal = joints_b[SMPL_JOINT2IND["spine3"]] - joints_b[SMPL_JOINT2IND["pelvis"]]
        return self._soft_slice_circumference(verts_b, origin, normal)

    def measure_smpl_anthropometrics(self, verts, joints):
        """
        verts:  (B, V, 3)
        joints: (B, J, 3)
        returns dict[str, Tensor(B,)]
        """
        out = {
            "height": [],
            "waist_circ": [],
            "hip_circ": [],
            "biceps_circ": [],
            "thigh_circ": [],
        }

        B = verts.shape[0]
        for b in range(B):
            vb = verts[b]
            jb = joints[b]

            out["height"].append(self._measure_height_smpl(vb))
            out["waist_circ"].append(self._measure_waist_smpl(vb, jb))
            out["hip_circ"].append(self._measure_hip_smpl(vb, jb))
            out["biceps_circ"].append(self._measure_biceps_smpl(vb, jb))
            out["thigh_circ"].append(self._measure_thigh_smpl(vb, jb))

        for k in out:
            out[k] = torch.stack(out[k], dim=0)

        return out

    @torch.no_grad()
    def scan_measurement_targets(self, th_scan_meshes):
        """
        Stable, no-grad targets extracted from the scan mesh.
        """
        measures = [self.anthropometric_measurements(v) for v in th_scan_meshes.verts_list()]
        keys = ["height", "waist_circ", "hip_circ", "biceps_circ", "thigh_circ"]

        out = {}
        device = th_scan_meshes.verts_list()[0].device
        for k in keys:
            vals = []
            for m in measures:
                v = m[k]
                if not math.isnan(v):
                    vals.append(v)
                else:
                    vals.append(0.0)
            out[k] = torch.tensor(vals, dtype=torch.float32, device=device)

        return out
    
    

    @staticmethod
    def mesh_volume(verts, faces):
        """Compute absolute mesh volume from vertices/faces."""
        tri = verts[faces]  # (F, 3, 3)
        v0, v1, v2 = tri[:, 0], tri[:, 1], tri[:, 2]
        signed = torch.sum(v0 * torch.cross(v1, v2, dim=1), dim=1) / 6.0
        return torch.abs(torch.sum(signed))

    @torch.no_grad()
    def compute_fit_score(self, th_scan_meshes, smpl, chamfer_samples=10000):
        """Compute quality metrics for SMPL-to-scan alignment."""
        verts, _, _, _ = smpl()
        th_smpl_meshes = Meshes(
            verts=verts,
            faces=torch.stack([smpl.faces] * len(verts), dim=0)
        )

        s2m = point_mesh_face_distance(
            th_smpl_meshes,
            Pointclouds(points=th_scan_meshes.verts_list())
        )
        m2s = point_mesh_face_distance(
            th_scan_meshes,
            Pointclouds(points=th_smpl_meshes.verts_list())
        )

        smpl_samples = sample_points_from_meshes(th_smpl_meshes, num_samples=chamfer_samples)
        scan_points_list = []
        for scan_verts in th_scan_meshes.verts_list():
            n_scan = scan_verts.shape[0]
            if n_scan >= chamfer_samples:
                idx = torch.randperm(n_scan, device=scan_verts.device)[:chamfer_samples]
            else:
                idx = torch.randint(0, n_scan, (chamfer_samples,), device=scan_verts.device)
            scan_points_list.append(scan_verts[idx])
        scan_points = torch.stack(scan_points_list, dim=0)
        chamfer, _ = chamfer_distance(smpl_samples, scan_points)

        # Volume metrics (% difference scan vs SMPL)
        smpl_vols = []
        for smpl_verts in th_smpl_meshes.verts_list():
            smpl_vols.append(self.mesh_volume(smpl_verts, smpl.faces))
        scan_vols = []
        for scan_verts, scan_faces in zip(th_scan_meshes.verts_list(), th_scan_meshes.faces_list()):
            scan_vols.append(self.mesh_volume(scan_verts, scan_faces))

        smpl_vols = torch.stack(smpl_vols)
        scan_vols = torch.stack(scan_vols)
        eps = torch.tensor(1e-8, device=scan_vols.device, dtype=scan_vols.dtype)
        vol_diff_pct = 100.0 * torch.abs(smpl_vols - scan_vols) / torch.clamp(scan_vols, min=eps)

        # Anthropometric metrics (scan vs SMPL)
        scan_measures = [self.anthropometric_measurements(v) for v in th_scan_meshes.verts_list()]
        smpl_measures = [self.anthropometric_measurements(v) for v in th_smpl_meshes.verts_list()]

        measure_keys = ['height', 'waist_circ', 'hip_circ', 'biceps_circ', 'thigh_circ']
        anthropometrics = {}
        for key in measure_keys:
            scan_vals = [m[key] for m in scan_measures if not math.isnan(m[key])]
            smpl_vals = [m[key] for m in smpl_measures if not math.isnan(m[key])]
            scan_mean = float(np.mean(scan_vals)) if len(scan_vals) > 0 else float('nan')
            smpl_mean = float(np.mean(smpl_vals)) if len(smpl_vals) > 0 else float('nan')
            if math.isnan(scan_mean) or math.isnan(smpl_mean) or abs(scan_mean) < 1e-8:
                diff_pct = float('nan')
            else:
                diff_pct = abs(smpl_mean - scan_mean) * 100.0 / abs(scan_mean)
            anthropometrics[key] = {
                'scan': scan_mean,
                'smpl': smpl_mean,
                'diff_pct': diff_pct
            }

        anthropometric_levels_rel = smpl_measures[0].get('levels_rel', {}) if len(smpl_measures) > 0 else {}

        geom = s2m + m2s
        score = 1.0 / (1.0 + geom + chamfer)

        return {
            's2m': float(s2m.detach().cpu().item()),
            'm2s': float(m2s.detach().cpu().item()),
            'chamfer': float(chamfer.detach().cpu().item()),
            'geom': float(geom.detach().cpu().item()),
            'scan_volume': float(torch.mean(scan_vols).detach().cpu().item()),
            'smpl_volume': float(torch.mean(smpl_vols).detach().cpu().item()),
            'volume_diff_pct': float(torch.mean(vol_diff_pct).detach().cpu().item()),
            'fit_score': float(score.detach().cpu().item()),
            'anthropometrics': anthropometrics,
            'anthropometric_levels_rel': anthropometric_levels_rel
        }


    def compose_smpl_joints(self, J, face, hands, th_pose_3d):
        if th_pose_3d is None:
            return J
        if th_pose_3d.shape[1] == 25:
            joints = J
        else:
            joints = torch.cat([J, face, hands], 1)
        return joints

    def optimize_pose_only(self, th_scan_meshes, smpl, iterations,
                           steps_per_iter, th_pose_3d, prior_weight=None):
        # split_smpl = SMPLHPyTorchWrapperBatchSplitParams.from_smplh(smpl).to(self.device)
        split_smpl = SMPLPyTorchWrapperBatchSplitParams.from_smpl(smpl).to(self.device)
        optimizer = torch.optim.Adam([split_smpl.trans, split_smpl.top_betas, split_smpl.global_pose], 0.02,
                                     betas=(0.9, 0.999))

        # Get loss_weights
        weight_dict = self.get_loss_weights()

        iter_for_global = 5
        for it in range(iter_for_global + iterations):
            loop = tqdm(range(steps_per_iter))
            if it < iter_for_global:
                # Optimize global orientation
                print('Optimizing SMPL global orientation')
                loop.set_description('Optimizing SMPL global orientation')
            elif it == iter_for_global:
                # Now optimize full SMPL pose
                print('Optimizing SMPL pose only')
                loop.set_description('Optimizing SMPL pose only')
                optimizer = torch.optim.Adam([split_smpl.trans, split_smpl.top_betas, split_smpl.global_pose,
                                              split_smpl.body_pose], 0.02, betas=(0.9, 0.999))
            else:
                loop.set_description('Optimizing SMPL pose only')

            for i in loop:
                optimizer.zero_grad()
                # Get losses for a forward pass
                loss_dict = self.forward_step_pose_only(split_smpl, th_pose_3d, prior_weight)
                # Get total loss for backward pass
                tot_loss = self.backward_step(loss_dict, weight_dict, it/2)
                tot_loss.backward()
                optimizer.step()

                l_str = 'Iter: {}'.format(i)
                for k in loss_dict:
                    l_str += ', {}: {:0.4f}'.format(k, weight_dict[k](loss_dict[k], it).mean().item())
                    loop.set_description(l_str)

                if self.debug:
                    self.viz_fitting(split_smpl, th_scan_meshes)

        self.copy_smpl_params(smpl, split_smpl)

        print('** Optimised smpl pose **')

    def copy_smpl_params(self, smpl, split_smpl):
        # Put back pose, shape and trans into original smpl
        smpl.pose.data = split_smpl.pose.data
        smpl.betas.data = split_smpl.betas.data
        smpl.trans.data = split_smpl.trans.data

    def forward_step_pose_only(self, smpl, th_pose_3d, prior_weight):
        """
        Performs a forward step, given smpl and scan meshes.
        Then computes the losses.
        currently no prior weight implemented for smplh
        """
        # Get pose prior
        prior = get_prior(self.model_root, smpl.gender)

        # losses
        loss = dict()
        # loss['pose_obj'] = batch_get_pose_obj(th_pose_3d, smpl, init_pose=False)
        # 3D joints loss
        J, face, hands = smpl.get_landmarks()
        
        joints = self.compose_smpl_joints(J, face, hands, th_pose_3d)
        loss['pose_pr'] = torch.mean(prior(smpl.pose))
        loss['betas'] = torch.mean(smpl.betas ** 2)
        j3d_loss = batch_3djoints_loss(th_pose_3d, joints)
        loss['pose_obj'] = j3d_loss
        return loss

    def get_loss_weights(self):
        """Set loss weights"""
        s2m_w = getattr(self, 's2m_weight', 20. ** 2)
        m2s_w = getattr(self, 'm2s_weight', 20. ** 2)
        betas_w = getattr(self, 'betas_weight', 10. ** 0.0)
        chamfer_w = getattr(self, 'chamfer_weight', 10. ** 0)

        loss_weight = {'s2m': lambda cst, it: s2m_w * cst * (1 + it),
                       'm2s': lambda cst, it: m2s_w * cst / (1 + it),
                       'betas': lambda cst, it: betas_w * cst / (1 + it),
                       # nova loss geométrica (mais conservadora para não degradar pose)
                       'chamfer': lambda cst, it: chamfer_w * cst * (it / (1 + it)),
                       'offsets': lambda cst, it: 10. ** -1 * cst / (1 + it),
                       'pose_pr': lambda cst, it: 10. ** -5 * cst / (1 + it),
                       'hand': lambda cst, it: 10. ** -5 * cst / (1 + it),
                       'lap': lambda cst, it: cst / (1 + it),
                       'pose_obj': lambda cst, it: 10. ** 2 * cst / (1 + it),
                       'meas_height': lambda cst, it: 15.0 * cst,
                       'meas_waist': lambda cst, it: 10.0 * cst,
                       'meas_hip': lambda cst, it: 10.0 * cst,
                       'meas_biceps': lambda cst, it: 5.0 * cst,
                       'meas_thigh': lambda cst, it: 5.0 * cst
                       }
        return loss_weight
        
    def mask_scan_vertices(self, scan_mesh, x_quantile_hands=0.90):
        filtered_points = []
        for verts in scan_mesh.verts_list():
            x_abs = torch.abs(verts[:, 0])
            x_thr = torch.quantile(x_abs, x_quantile_hands)
            keep = x_abs < x_thr
            filtered_points.append(verts[keep])
        return Pointclouds(points=filtered_points)


def main(args):
    fitter = SMPLHFitter(args.model_root, debug=args.display, hands=args.hands)
    fitter.iterations = args.iterations
    fitter.steps_per_iter = args.steps_per_iter
    fitter.pose_shape_lr = args.pose_shape_lr
    fitter.s2m_weight = args.s2m_weight
    fitter.m2s_weight = args.m2s_weight
    fitter.betas_weight = args.betas_weight
    fitter.chamfer_weight = args.chamfer_weight
    fitter.fit([args.scan_path], [args.pose_file], args.gender, args.save_path)


if __name__ == "__main__":
    import argparse
    from utils.configs import load_config
    parser = argparse.ArgumentParser(description='Run Model')
    parser.add_argument('scan_path', type=str, help='path to the 3d scans')
    parser.add_argument('pose_file', type=str, help='3d body joints file')
    parser.add_argument('save_path', type=str, help='save path for all scans')
    parser.add_argument('-g', '--gender', type=str, default='neutral', choices=['male', 'female', 'neutral'])
    parser.add_argument('--display', default=False, action='store_true')
    parser.add_argument("--config-path", "-c", type=Path, default="config.yml",
                        help="Path to yml file with config")
    parser.add_argument('-hands', default=False, action='store_true', help='use SMPL+hand model or not')
    parser.add_argument('--iterations', type=int, default=8, help='pose+shape outer iterations')
    parser.add_argument('--steps-per-iter', type=int, default=40, help='optimizer steps per outer iteration')
    parser.add_argument('--pose-shape-lr', type=float, default=0.02, help='Adam lr for pose+shape')
    parser.add_argument('--s2m-weight', type=float, default=20. ** 2, help='base weight for scan-to-mesh loss')
    parser.add_argument('--m2s-weight', type=float, default=20. ** 2, help='base weight for mesh-to-scan loss')
    parser.add_argument('--betas-weight', type=float, default=10. ** 0.0, help='base weight for betas regularization')
    parser.add_argument('--chamfer-weight', type=float, default=10. ** 0, help='base weight for chamfer loss')
    args = parser.parse_args()

    # args.scan_path = 'data/mesh_1/scan.obj'
    # args.pose_file = 'data/mesh_1/3D_joints_all.json'
    # args.display = True
    # args.save_path = 'data/mesh_1'
    # args.gender = 'male'
    config = load_config(args.config_path)
    args.model_root = Path(config["SMPL_MODELS_PATH"])

    main(args)
