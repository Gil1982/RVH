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
            anthropometric_landmarks = fit_score.get('anthropometric_landmarks', {})
            with open(os.path.join(save_path, 'anthropometric_landmarks.json'), 'w') as f:
                json.dump(anthropometric_landmarks, f, indent=2)
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

        # regularização de shape
        loss['betas'] = torch.mean(smpl.betas ** 2)

        # prior de pose
        loss['pose_pr'] = torch.mean(prior(smpl.pose))

        if self.hands:
            hand_prior = HandPrior(self.model_root, type='grab')
            loss['hand'] = torch.mean(hand_prior(smpl.pose))

        J, face, hands = smpl.get_landmarks()

        if th_pose_3d is not None:
            # 3D joints loss
            joints = self.compose_smpl_joints(J, face, hands, th_pose_3d)
            j3d_loss = batch_3djoints_loss(th_pose_3d, joints)
            loss['pose_obj'] = j3d_loss

        # Anthropometric consistency loss (scan vs SMPL), guided by SMPL landmarks
        measure_keys = ['height', 'chest_circ', 'waist_circ', 'hip_circ', 'biceps_circ', 'thigh_circ']
        anthropo_terms = []
        for b, (scan_verts, smpl_verts) in enumerate(zip(th_scan_meshes.verts_list(), verts)):
            levels_rel = self._landmark_levels_rel_from_joints(smpl_verts, J[b])
            scan_m = self.anthropometric_measurements(scan_verts, levels_rel=levels_rel)
            smpl_m = self.anthropometric_measurements(smpl_verts, levels_rel=levels_rel)
            for k in measure_keys:
                scan_v = scan_m[k]
                smpl_v = smpl_m[k]
                if torch.isfinite(scan_v) and torch.isfinite(smpl_v):
                    anthropo_terms.append(torch.abs(smpl_v - scan_v) / (torch.abs(scan_v) + 1e-6))

        if len(anthropo_terms) > 0:
            loss['anthro'] = torch.stack(anthropo_terms).mean()

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

    def _landmark_levels_rel_from_joints(self, verts, joints):
        """Landmark-guided anthropometric levels (inspired by SMPL landmark definitions)."""
        up_axis, _, _ = self._infer_body_axes(verts)
        up = verts[:, up_axis]
        up_min = torch.min(up)
        h = torch.clamp(torch.max(up) - up_min, min=1e-8)

        idx = self._anthropometric_joint_indices(joints)
        pelvis_i = idx['pelvis']
        l_hip_i, r_hip_i = idx['left_hip'], idx['right_hip']
        spine2_i = idx['spine2']
        l_shoulder_i, r_shoulder_i = idx['left_shoulder'], idx['right_shoulder']
        l_elbow_i, r_elbow_i = idx['left_elbow'], idx['right_elbow']
        l_knee_i, r_knee_i = idx['left_knee'], idx['right_knee']

        pelvis = joints[pelvis_i, up_axis]
        hip_mid = 0.5 * (joints[l_hip_i, up_axis] + joints[r_hip_i, up_axis])
        chest_mid = 0.5 * (joints[l_shoulder_i, up_axis] + joints[r_shoulder_i, up_axis])
        waist_mid = 0.5 * (pelvis + joints[spine2_i, up_axis])

        biceps_l = 0.5 * (joints[l_shoulder_i, up_axis] + joints[l_elbow_i, up_axis])
        biceps_r = 0.5 * (joints[r_shoulder_i, up_axis] + joints[r_elbow_i, up_axis])
        thigh_l = 0.5 * (joints[l_hip_i, up_axis] + joints[l_knee_i, up_axis])
        thigh_r = 0.5 * (joints[r_hip_i, up_axis] + joints[r_knee_i, up_axis])

        def to_rel(v):
            rel = (v - up_min) / h
            rel = torch.clamp(rel, 0.02, 0.98)
            return float(rel.detach().cpu().item())

        return {
            'chest': to_rel(chest_mid),
            'waist': to_rel(waist_mid),
            'hip': to_rel(hip_mid),
            'biceps_left': to_rel(biceps_l),
            'biceps_right': to_rel(biceps_r),
            'thigh_left': to_rel(thigh_l),
            'thigh_right': to_rel(thigh_r),
        }

    @staticmethod
    def _anthropometric_joint_indices(joints):
        """
        Resolve the joint index convention used by `smpl.get_landmarks()`.

        BODY_25/OpenPose layout places shoulders at 2/5, elbows at 3/6, pelvis at 8,
        hips at 9/12 and knees at 10/13. Older SMPL body-joint layouts use the
        canonical 24-joint indexing.
        """
        n_joints = int(joints.shape[0])
        if n_joints >= 25:
            return {
                'pelvis': 8,
                'spine2': 1,
                'right_shoulder': 2,
                'right_elbow': 3,
                'right_hip': 9,
                'right_knee': 10,
                'left_shoulder': 5,
                'left_elbow': 6,
                'left_hip': 12,
                'left_knee': 13,
            }

        return {
            'pelvis': 0,
            'left_hip': 1,
            'right_hip': 2,
            'left_knee': 4,
            'right_knee': 5,
            'spine2': 6,
            'left_shoulder': 16,
            'right_shoulder': 17,
            'left_elbow': 18,
            'right_elbow': 19,
        }

    def _anthropometric_landmark_points_from_joints(self, joints):
        """Return joint-based reference points used for anthropometric levels."""
        idx = self._anthropometric_joint_indices(joints)
        pelvis_i = idx['pelvis']
        l_hip_i, r_hip_i = idx['left_hip'], idx['right_hip']
        l_knee_i, r_knee_i = idx['left_knee'], idx['right_knee']
        spine2_i = idx['spine2']
        l_shoulder_i, r_shoulder_i = idx['left_shoulder'], idx['right_shoulder']
        l_elbow_i, r_elbow_i = idx['left_elbow'], idx['right_elbow']

        def midpoint(a, b):
            return 0.5 * (joints[a] + joints[b])

        return {
            'chest': {
                'point': midpoint(l_shoulder_i, r_shoulder_i),
                'landmarks': {
                    'left_shoulder': joints[l_shoulder_i],
                    'right_shoulder': joints[r_shoulder_i],
                }
            },
            'waist': {
                'point': midpoint(pelvis_i, spine2_i),
                'landmarks': {
                    'pelvis': joints[pelvis_i],
                    'spine2': joints[spine2_i],
                }
            },
            'hip': {
                'point': midpoint(l_hip_i, r_hip_i),
                'landmarks': {
                    'left_hip': joints[l_hip_i],
                    'right_hip': joints[r_hip_i],
                }
            },
            'biceps_left': {
                'point': midpoint(l_shoulder_i, l_elbow_i),
                'landmarks': {
                    'left_shoulder': joints[l_shoulder_i],
                    'left_elbow': joints[l_elbow_i],
                }
            },
            'biceps_right': {
                'point': midpoint(r_shoulder_i, r_elbow_i),
                'landmarks': {
                    'right_shoulder': joints[r_shoulder_i],
                    'right_elbow': joints[r_elbow_i],
                }
            },
            'thigh_left': {
                'point': midpoint(l_hip_i, l_knee_i),
                'landmarks': {
                    'left_hip': joints[l_hip_i],
                    'left_knee': joints[l_knee_i],
                }
            },
            'thigh_right': {
                'point': midpoint(r_hip_i, r_knee_i),
                'landmarks': {
                    'right_hip': joints[r_hip_i],
                    'right_knee': joints[r_knee_i],
                }
            },
        }

    def _circumference_at_rel_height_torch(self, verts, rel_height, band_rel=0.01, side=None):
        """Differentiable circumference proxy from a soft horizontal slice."""
        up_axis, lr_axis, depth_axis = self._infer_body_axes(verts)
        up = verts[:, up_axis]
        lr = verts[:, lr_axis]
        depth = verts[:, depth_axis]

        up_min = torch.min(up)
        up_max = torch.max(up)
        h = torch.clamp(up_max - up_min, min=1e-8)
        up_target = up_min + rel_height * h
        band = torch.clamp(band_rel * h, min=1e-5)

        w = torch.exp(-0.5 * ((up - up_target) / band) ** 2)
        if side == 'left':
            w = w * torch.sigmoid(40.0 * lr)
        elif side == 'right':
            w = w * torch.sigmoid(-40.0 * lr)

        w_sum = torch.clamp(torch.sum(w), min=1e-8)
        w = w / w_sum

        mu_lr = torch.sum(w * lr)
        mu_depth = torch.sum(w * depth)
        radial = torch.sqrt((lr - mu_lr) ** 2 + (depth - mu_depth) ** 2 + 1e-8)
        mean_r = torch.sum(w * radial)
        return 2.0 * torch.tensor(np.pi, device=verts.device, dtype=verts.dtype) * mean_r

    def anthropometric_measurements(self, verts, levels_rel=None):
        """Anthropometric measures guided by landmarks/levels."""
        up_axis, _, _ = self._infer_body_axes(verts)
        up = verts[:, up_axis]
        height = torch.max(up) - torch.min(up)

        if levels_rel is None:
            levels_rel = {
                'chest': 0.78,
                'waist': 0.62,
                'hip': 0.53,
                'biceps_left': 0.74,
                'biceps_right': 0.74,
                'thigh_left': 0.42,
                'thigh_right': 0.42,
            }

        chest = self._circumference_at_rel_height_torch(verts, float(levels_rel['chest']), band_rel=0.012)
        waist = self._circumference_at_rel_height_torch(verts, float(levels_rel['waist']), band_rel=0.012)
        hip = self._circumference_at_rel_height_torch(verts, float(levels_rel['hip']), band_rel=0.012)

        biceps_l = self._circumference_at_rel_height_torch(verts, float(levels_rel['biceps_left']), band_rel=0.010, side='left')
        biceps_r = self._circumference_at_rel_height_torch(verts, float(levels_rel['biceps_right']), band_rel=0.010, side='right')
        thigh_l = self._circumference_at_rel_height_torch(verts, float(levels_rel['thigh_left']), band_rel=0.012, side='left')
        thigh_r = self._circumference_at_rel_height_torch(verts, float(levels_rel['thigh_right']), band_rel=0.012, side='right')

        return {
            'height': height,
            'chest_circ': chest,
            'waist_circ': waist,
            'hip_circ': hip,
            'biceps_circ': 0.5 * (biceps_l + biceps_r),
            'thigh_circ': 0.5 * (thigh_l + thigh_r),
            'levels_rel': levels_rel,
        }

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

        # Anthropometric metrics (scan vs SMPL), landmark-guided
        J, _, _ = smpl.get_landmarks()
        scan_measures = []
        smpl_measures = []
        for b, (scan_v, smpl_v) in enumerate(zip(th_scan_meshes.verts_list(), th_smpl_meshes.verts_list())):
            levels_rel = self._landmark_levels_rel_from_joints(smpl_v, J[b])
            scan_measures.append(self.anthropometric_measurements(scan_v, levels_rel=levels_rel))
            smpl_measures.append(self.anthropometric_measurements(smpl_v, levels_rel=levels_rel))

        measure_keys = ['height', 'chest_circ', 'waist_circ', 'hip_circ', 'biceps_circ', 'thigh_circ']
        anthropometrics = {}
        for key in measure_keys:
            scan_vals = [float(m[key].detach().cpu().item()) for m in scan_measures if torch.isfinite(m[key])]
            smpl_vals = [float(m[key].detach().cpu().item()) for m in smpl_measures if torch.isfinite(m[key])]
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
        anthropometric_landmarks = {}
        if J.shape[0] > 0:
            landmark_dict = self._anthropometric_landmark_points_from_joints(J[0])
            anthropometric_landmarks = {
                measure_name: {
                    'point': [float(v) for v in info['point'].detach().cpu().tolist()],
                    'landmarks': {
                        lm_name: [float(v) for v in lm_point.detach().cpu().tolist()]
                        for lm_name, lm_point in info['landmarks'].items()
                    }
                }
                for measure_name, info in landmark_dict.items()
            }

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
            'anthropometric_levels_rel': anthropometric_levels_rel,
            'anthropometric_landmarks': anthropometric_landmarks
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
        anthro_w = getattr(self, 'anthro_weight', 10. ** 1)

        loss_weight = {'s2m': lambda cst, it: s2m_w * cst * (1 + it),
                       'm2s': lambda cst, it: m2s_w * cst / (1 + it),
                       'betas': lambda cst, it: betas_w * cst / (1 + it),
                       # nova loss geométrica (mais conservadora para não degradar pose)
                       'chamfer': lambda cst, it: chamfer_w * cst * (it / (1 + it)),
                       'anthro': lambda cst, it: anthro_w * cst,
                       'offsets': lambda cst, it: 10. ** -1 * cst / (1 + it),
                       'pose_pr': lambda cst, it: 10. ** -5 * cst / (1 + it),
                       'hand': lambda cst, it: 10. ** -5 * cst / (1 + it),
                       'lap': lambda cst, it: cst / (1 + it),
                       'pose_obj': lambda cst, it: 10. ** 2 * cst / (1 + it)
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
    fitter.anthro_weight = args.anthro_weight
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
    parser.add_argument('--anthro-weight', type=float, default=10. ** 1, help='base weight for anthropometric loss')
    args = parser.parse_args()

    # args.scan_path = 'data/mesh_1/scan.obj'
    # args.pose_file = 'data/mesh_1/3D_joints_all.json'
    # args.display = True
    # args.save_path = 'data/mesh_1'
    # args.gender = 'male'
    config = load_config(args.config_path)
    args.model_root = Path(config["SMPL_MODELS_PATH"])

    main(args)
