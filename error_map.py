import os
import json
import math
from io import BytesIO
from pathlib import Path
import numpy as np
import trimesh
import torch
from typing import Optional, Tuple

from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.loss import point_mesh_face_distance


def jet_colormap(x: np.ndarray) -> np.ndarray:
    """
    x em [0,1] -> RGB uint8
    """
    x = np.clip(x, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
    rgb = np.stack([r, g, b], axis=1)
    return (rgb * 255).astype(np.uint8)


def load_mesh(path: str) -> trimesh.Trimesh:
    if not os.path.exists(path):
        raise FileNotFoundError("Arquivo não encontrado: %s" % path)

    mesh = trimesh.load(path, process=False)

    if isinstance(mesh, trimesh.Scene):
        geoms = []
        for g in mesh.geometry.values():
            if isinstance(g, trimesh.Trimesh):
                geoms.append(g)
        if len(geoms) == 0:
            raise ValueError("Nenhuma malha válida encontrada em: %s" % path)
        mesh = trimesh.util.concatenate(geoms)

    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("Arquivo não é uma malha válida: %s" % path)

    if mesh.vertices is None or len(mesh.vertices) == 0:
        raise ValueError("Malha sem vértices: %s" % path)

    return mesh


def clean_mesh_for_pytorch3d(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    Remove faces inválidas/degeneradas que às vezes atrapalham o PyTorch3D.
    """
    m = mesh.copy()

    try:
        m.remove_unreferenced_vertices()
    except Exception:
        pass

    try:
        m.remove_degenerate_faces()
    except Exception:
        pass

    try:
        m.remove_duplicate_faces()
    except Exception:
        pass

    faces = np.asarray(m.faces)
    verts = np.asarray(m.vertices)

    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("A malha não tem faces triangulares válidas.")

    valid = np.all(faces >= 0, axis=1) & np.all(faces < len(verts), axis=1)
    faces = faces[valid]

    m = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    return m


def trimesh_to_pytorch3d_mesh(mesh: trimesh.Trimesh, device: torch.device) -> Meshes:
    verts = torch.tensor(np.asarray(mesh.vertices), dtype=torch.float32, device=device)
    faces = torch.tensor(np.asarray(mesh.faces), dtype=torch.int64, device=device)
    return Meshes(verts=[verts], faces=[faces])


def trimesh_vertices_to_pointcloud(mesh: trimesh.Trimesh, device: torch.device) -> Pointclouds:
    points = torch.tensor(np.asarray(mesh.vertices), dtype=torch.float32, device=device)
    return Pointclouds(points=[points])


def compute_scan_to_smpl_errors(scan_mesh: trimesh.Trimesh,
                                smpl_mesh: trimesh.Trimesh,
                                device: torch.device) -> np.ndarray:
    """
    Calcula uma distância por vértice do scan até a superfície do SMPL.

    Usa PyTorch3D point-to-mesh face distance.
    """
    scan_points = trimesh_vertices_to_pointcloud(scan_mesh, device)
    smpl_mesh_pt3d = trimesh_to_pytorch3d_mesh(smpl_mesh, device)

    # Distância quadrática total média:
    # point_mesh_face_distance retorna média por ponto do batch.
    # Para obter distância por ponto individual, usamos a função interna do Meshes/Pointclouds
    # via packed representation e ops de baixo nível.
    #
    # Como a API pública não devolve por-ponto diretamente, usamos
    # point_mesh_face_distance para checagem geral e calculamos por ponto
    # com nearest.on_surface do trimesh como fallback geométrico preciso.
    #
    # Isso mantém a compatibilidade e ainda aproveita PyTorch3D para validar a malha.
    _ = point_mesh_face_distance(smpl_mesh_pt3d, scan_points)

    # Cálculo por vértice: ponto -> superfície exata no trimesh
    # Requer rtree/embree em alguns ambientes; se falhar, cai no nearest-vertex.
    scan_vertices = np.asarray(scan_mesh.vertices)

    try:
        closest_points, distances, _ = smpl_mesh.nearest.on_surface(scan_vertices)
        errors = distances.astype(np.float64)
        return errors
    except Exception as e:
        print("[AVISO] nearest.on_surface falhou, usando nearest-vertex. Motivo:", e)
        smpl_vertices = np.asarray(smpl_mesh.vertices)
        # fallback simples
        from scipy.spatial import cKDTree
        tree = cKDTree(smpl_vertices)
        distances, _ = tree.query(scan_vertices, k=1)
        return distances.astype(np.float64)


def make_colored_scan(scan_mesh: trimesh.Trimesh,
                      errors: np.ndarray,
                      vmax: Optional[float] = None) -> Tuple[trimesh.Trimesh, float]:
    if vmax is None:
        vmax = float(np.percentile(errors, 95))
        vmax = max(vmax, 1e-8)

    norm = np.clip(errors / vmax, 0.0, 1.0)
    colors_rgb = jet_colormap(norm)
    alpha = 255 * np.ones((colors_rgb.shape[0], 1), dtype=np.uint8)
    colors_rgba = np.hstack([colors_rgb, alpha])

    colored = scan_mesh.copy()
    colored.visual = trimesh.visual.ColorVisuals(
        mesh=colored,
        vertex_colors=colors_rgba
    )
    return colored, vmax


def make_gray_smpl(smpl_mesh: trimesh.Trimesh,
                   rgba=(180, 180, 180, 180)) -> trimesh.Trimesh:
    out = smpl_mesh.copy()
    color = np.tile(np.array([rgba], dtype=np.uint8), (len(out.vertices), 1))
    out.visual = trimesh.visual.ColorVisuals(
        mesh=out,
        vertex_colors=color
    )
    return out


def save_histogram(errors: np.ndarray, out_jpg: str, vmax: float) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 4.5))
    plt.hist(errors, bins=80)
    plt.axvline(vmax, linestyle="--")
    plt.xlabel("Erro (distância)")
    plt.ylabel("Contagem")
    plt.title("Distribuição do erro scan -> SMPL")
    plt.tight_layout()
    plt.savefig(out_jpg, dpi=180)
    plt.close()


def save_overlay_glb(scan_colored: trimesh.Trimesh,
                     smpl_gray: trimesh.Trimesh,
                     out_glb: str) -> None:
    scene = trimesh.Scene()
    scene.add_geometry(scan_colored, node_name="scan_error")
    scene.add_geometry(smpl_gray, node_name="smpl_fit")
    scene.export(out_glb)


def convex_hull_2d(points_2d: np.ndarray) -> np.ndarray:
    """Retorna os pontos do casco convexo 2D em ordem (algoritmo monotonic chain)."""
    if points_2d.shape[0] < 3:
        return np.empty((0, 2), dtype=np.float64)

    pts = sorted(set((float(x), float(y)) for x, y in points_2d))
    if len(pts) < 3:
        return np.empty((0, 2), dtype=np.float64)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for pt in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], pt) <= 0:
            lower.pop()
        lower.append(pt)

    upper = []
    for pt in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], pt) <= 0:
            upper.pop()
        upper.append(pt)

    hull = lower[:-1] + upper[:-1]
    if len(hull) < 3:
        return np.empty((0, 2), dtype=np.float64)
    return np.asarray(hull, dtype=np.float64)


def infer_body_axes(vertices: np.ndarray) -> Tuple[int, int, int]:
    """Infer (up, left-right, depth) axes from bbox extents."""
    ext = vertices.max(axis=0) - vertices.min(axis=0)
    up_axis = int(np.argmax(ext))
    rem = [a for a in [0, 1, 2] if a != up_axis]
    if ext[rem[0]] >= ext[rem[1]]:
        lr_axis, depth_axis = rem[0], rem[1]
    else:
        lr_axis, depth_axis = rem[1], rem[0]
    return up_axis, lr_axis, depth_axis


def anthropometric_ring_points(mesh: trimesh.Trimesh,
                               rel_height: float,
                               band_rel: float = 0.012,
                               side: Optional[str] = None,
                               target_point: Optional[np.ndarray] = None) -> np.ndarray:
    """Extrai anel antropométrico via seção real da malha por plano."""
    verts = np.asarray(mesh.vertices)
    up_axis, lr_axis, depth_axis = infer_body_axes(verts)

    up = verts[:, up_axis]
    up_min, up_max = float(up.min()), float(up.max())
    h = max(up_max - up_min, 1e-8)
    up_target = up_min + rel_height * h

    plane_origin = np.zeros(3, dtype=np.float64)
    plane_origin[up_axis] = up_target
    plane_normal = np.zeros(3, dtype=np.float64)
    plane_normal[up_axis] = 1.0

    section = mesh.section(plane_origin=plane_origin, plane_normal=plane_normal)
    if section is None:
        return np.empty((0, 3), dtype=np.float64)

    # Cada entrada de discrete é uma polyline 3D da interseção
    curves = section.discrete
    if curves is None or len(curves) == 0:
        return np.empty((0, 3), dtype=np.float64)

    # Filtra curvas muito pequenas/ruidosas
    target = None
    if target_point is not None:
        target = np.asarray(target_point, dtype=np.float64).reshape(-1)
        if target.shape[0] != 3 or not np.all(np.isfinite(target)):
            target = None

    candidates = []
    lr_center = float(np.mean(verts[:, lr_axis]))
    for c in curves:
        ring = np.asarray(c, dtype=np.float64)
        if ring.shape[0] < 8:
            continue
        # fecha loop para desenho/perímetro
        if np.linalg.norm(ring[0] - ring[-1]) > 1e-8:
            ring = np.vstack([ring, ring[:1]])

        mean_lr = float(np.mean(ring[:, lr_axis]))
        # side filtering
        if side == 'left' and mean_lr <= lr_center:
            continue
        if side == 'right' and mean_lr >= lr_center:
            continue

        seg = np.linalg.norm(np.diff(ring, axis=0), axis=1)
        perim = float(np.sum(seg))
        if perim < 0.05:  # remove loops espúrios muito pequenos
            continue

        # distância lateral do centro corporal (útil para braços/pernas)
        side_dist = abs(mean_lr - lr_center)
        target_dist = 0.0
        if target is not None:
            target_dist = float(np.min(np.linalg.norm(ring - target[None, :], axis=1)))
        candidates.append((ring, perim, side_dist, target_dist))

    if len(candidates) == 0:
        return np.empty((0, 3), dtype=np.float64)

    if target is not None:
        candidates.sort(key=lambda x: (x[3], -x[1]))
        return candidates[0][0]

    if side is None:
        # tronco: pega curva mais central, com maior perímetro entre as centrais
        candidates.sort(key=lambda x: (x[2], -x[1]))
        return candidates[0][0]

    # membro unilateral: pega curva mais lateral (afasta do tronco)
    candidates.sort(key=lambda x: (x[2], x[1]), reverse=True)
    return candidates[0][0]


def add_ring_to_scene(scene: trimesh.Scene,
                      ring_points: np.ndarray,
                      color_rgba: Tuple[int, int, int, int],
                      node_name: str) -> bool:
    if ring_points.shape[0] < 4:
        return False
    path = trimesh.load_path(ring_points)
    if hasattr(path, 'colors'):
        path.colors = np.tile(np.array([color_rgba], dtype=np.uint8), (len(path.entities), 1))
    scene.add_geometry(path, node_name=node_name)
    return True


def best_ring_in_window(mesh: trimesh.Trimesh,
                        rel_range: Tuple[float, float],
                        mode: str,
                        band_rel: float,
                        side: Optional[str] = None,
                        steps: int = 15) -> Tuple[np.ndarray, float]:
    rel_values = np.linspace(rel_range[0], rel_range[1], steps)
    best_ring = np.empty((0, 3), dtype=np.float64)
    best_val = np.nan

    for rel_h in rel_values:
        ring = anthropometric_ring_points(mesh, float(rel_h), band_rel=band_rel, side=side)
        if ring.shape[0] < 4:
            continue
        seg = np.linalg.norm(np.diff(ring, axis=0), axis=1)
        circ = float(seg.sum())
        if np.isnan(best_val):
            best_ring = ring
            best_val = circ
            continue
        if (mode == 'max' and circ > best_val) or (mode == 'min' and circ < best_val):
            best_ring = ring
            best_val = circ

    return best_ring, float(best_val) if not np.isnan(best_val) else float('nan')


def add_anthropometric_rings(scene: trimesh.Scene, mesh: trimesh.Trimesh) -> Tuple[int, list]:
    """Adiciona anéis antropométricos adaptativos ao scene; retorna quantidade e metadados."""
    specs = [
        ('chest', (0.72, 0.86), 'max', 0.012, None, (255, 60, 60, 255)),
        ('waist', (0.56, 0.70), 'min', 0.012, None, (255, 200, 0, 255)),
        ('hip', (0.45, 0.60), 'max', 0.012, None, (40, 220, 40, 255)),
        ('biceps_left', (0.67, 0.80), 'max', 0.010, 'left', (255, 0, 220, 255)),
        ('biceps_right', (0.67, 0.80), 'max', 0.010, 'right', (255, 0, 220, 255)),
        ('thigh_left', (0.30, 0.48), 'max', 0.012, 'left', (40, 140, 255, 255)),
        ('thigh_right', (0.30, 0.48), 'max', 0.012, 'right', (40, 140, 255, 255)),
    ]

    count = 0
    infos = []
    for name, rel_range, mode, band, side, color in specs:
        ring, circ = best_ring_in_window(mesh, rel_range=rel_range, mode=mode, band_rel=band, side=side)
        ok = add_ring_to_scene(scene, ring, color_rgba=color, node_name=f"ring_{name}")
        if ok:
            count += 1
            infos.append({'name': name, 'color': color, 'ref_point': ring_reference_point(ring)})
            print(f"[INFO] Ring {name}: circ={circ:.4f}, range={rel_range}, mode={mode}")
        else:
            print(f"[AVISO] Ring {name}: sem pontos suficientes")
    return count, infos




def ring_reference_point(ring_points: np.ndarray) -> Optional[np.ndarray]:
    if ring_points is None or ring_points.shape[0] < 3:
        return None
    core = ring_points[:-1] if np.linalg.norm(ring_points[0] - ring_points[-1]) < 1e-8 else ring_points
    if core.shape[0] == 0:
        return None
    return np.mean(core, axis=0)


def marker_radius_from_mesh(mesh: trimesh.Trimesh) -> float:
    verts = np.asarray(mesh.vertices)
    ext = verts.max(axis=0) - verts.min(axis=0)
    return float(max(ext.max() * 0.01, 2e-3))


def add_landmark_markers(scene: trimesh.Scene, ring_infos: list, radius: float, landmark_map: Optional[dict] = None) -> None:
    if landmark_map is None:
        for info in ring_infos:
            ref = info.get('ref_point', None)
            if ref is None:
                continue
            color = info['color']
            sph = trimesh.creation.icosphere(subdivisions=2, radius=radius)
            sph.apply_translation(ref)
            col = np.tile(np.array([color], dtype=np.uint8), (len(sph.vertices), 1))
            sph.visual = trimesh.visual.ColorVisuals(mesh=sph, vertex_colors=col)
            scene.add_geometry(sph, node_name=f"reference_{info['name']}")
        return

    for info in ring_infos:
        measure_name = info['name']
        color = info['color']
        if measure_name not in landmark_map:
            continue
        lm_entries = landmark_map[measure_name].get('landmarks', {})
        for lm_name, lm_point in lm_entries.items():
            pt = np.asarray(lm_point, dtype=np.float64)
            sph = trimesh.creation.icosphere(subdivisions=2, radius=radius)
            sph.apply_translation(pt)
            col = np.tile(np.array([color], dtype=np.uint8), (len(sph.vertices), 1))
            sph.visual = trimesh.visual.ColorVisuals(mesh=sph, vertex_colors=col)
            scene.add_geometry(sph, node_name=f"landmark_{measure_name}_{lm_name}")


def annotate_overlay_image_with_labels(png_bytes: bytes, ring_infos: list, landmark_map: Optional[dict] = None) -> bytes:
    """Adds a separate sidebar legend with circumference and landmark names."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        return png_bytes

    img = Image.open(BytesIO(png_bytes)).convert('RGBA')
    try:
        font_title = ImageFont.truetype('DejaVuSans.ttf', 24)
        font_text = ImageFont.truetype('DejaVuSans.ttf', 20)
        font_small = ImageFont.truetype('DejaVuSans.ttf', 18)
    except Exception:
        font_title = ImageFont.load_default()
        font_text = ImageFont.load_default()
        font_small = ImageFont.load_default()

    sidebar_w = 520
    canvas = Image.new('RGBA', (img.width + sidebar_w, img.height), (255, 255, 255, 255))
    canvas.paste(img, (sidebar_w, 0))
    draw = ImageDraw.Draw(canvas)

    x0, y = 20, 20
    line_h = 28

    draw.text((x0, y), 'Circunferencias', fill=(0, 0, 0, 255), font=font_title)
    y += 38
    for info in ring_infos:
        color = info['color']
        name = info['name']
        draw.rectangle([x0, y + 4, x0 + 18, y + 22], fill=tuple(color), outline=(0, 0, 0, 255))
        draw.text((x0 + 26, y), name, fill=(0, 0, 0, 255), font=font_text)
        y += line_h

    y += 14
    section_title = 'Landmarks SMPL' if landmark_map is not None else 'Pontos de referencia (fallback)'
    draw.text((x0, y), section_title, fill=(0, 0, 0, 255), font=font_title)
    y += 38

    if landmark_map is None:
        draw.text((x0, y), 'fit_score/landmarks json sem anthropometric_landmarks.', fill=(120, 0, 0, 255), font=font_text)
        y += line_h
        draw.text((x0, y), 'Pontos exibidos abaixo sao proxies dos aneis.', fill=(120, 0, 0, 255), font=font_small)
        y += line_h
        for info in ring_infos:
            color = info['color']
            measure_name = info['name']
            draw.rectangle([x0, y + 4, x0 + 18, y + 22], fill=tuple(color), outline=(0, 0, 0, 255))
            draw.text((x0 + 26, y), f'{measure_name}_ref', fill=(0, 0, 0, 255), font=font_text)
            y += line_h
    else:
        for info in ring_infos:
            measure_name = info['name']
            color = info['color']
            draw.rectangle([x0, y + 4, x0 + 18, y + 22], fill=tuple(color), outline=(0, 0, 0, 255))
            draw.text((x0 + 26, y), measure_name, fill=(0, 0, 0, 255), font=font_text)
            y += line_h
            lm_names = list(landmark_map.get(measure_name, {}).get('landmarks', {}).keys())
            if len(lm_names) == 0:
                draw.text((x0 + 26, y), '- sem landmarks', fill=(80, 80, 80, 255), font=font_small)
                y += line_h
                continue
            for lm_name in lm_names:
                draw.text((x0 + 40, y), f'- {lm_name}', fill=(80, 80, 80, 255), font=font_small)
                y += 24

    out = BytesIO()
    canvas.convert('RGB').save(out, format='PNG')
    return out.getvalue()



def load_overlay_metadata_from_fit_score(fit_score_json: str) -> Tuple[Optional[dict], Optional[dict]]:
    requested_fit_score = Path(fit_score_json)

    def candidate_paths(path: Path, filename: str) -> list:
        candidates = [path.with_name(filename)]
        parent_registered = path.parent / 'registered_scans' / filename
        if parent_registered not in candidates:
            candidates.append(parent_registered)
        if path.parent.name == 'registered_scans':
            parent_root = path.parent.parent / filename
            if parent_root not in candidates:
                candidates.append(parent_root)

        search_roots = [path.parent]
        if parent_registered.parent not in search_roots:
            search_roots.append(parent_registered.parent)
        if path.parent.name == 'registered_scans' and path.parent.parent not in search_roots:
            search_roots.append(path.parent.parent)

        for root in search_roots:
            if not root.exists() or not root.is_dir():
                continue
            for found in sorted(root.rglob(filename)):
                if found not in candidates:
                    candidates.append(found)
        return candidates

    fit_score_candidates = candidate_paths(requested_fit_score, requested_fit_score.name)
    fit_score_path = next((p for p in fit_score_candidates if p.exists()), requested_fit_score)
    landmarks_candidates = candidate_paths(fit_score_path, 'anthropometric_landmarks.json')

    levels = None
    landmarks = None

    if fit_score_path.exists():
        try:
            with open(fit_score_path, 'r') as f:
                data = json.load(f)
            levels = data.get('anthropometric_levels_rel', None)
            landmarks = data.get('anthropometric_landmarks', None)
            if isinstance(levels, dict) and len(levels) > 0:
                print(f"[INFO] Usando levels do fit_score: {fit_score_path}")
            else:
                levels = None
            if not isinstance(landmarks, dict) or len(landmarks) == 0:
                landmarks = None
        except Exception as e:
            print('[AVISO] Falha ao ler fit_score.json:', e)
    else:
        print(f"[AVISO] fit_score.json não encontrado em: {requested_fit_score}")
        print(f"[AVISO] Caminhos tentados: {[str(p) for p in fit_score_candidates]}")

    if landmarks is None:
        landmarks_path = next((p for p in landmarks_candidates if p.exists()), None)
        if landmarks_path is not None:
            try:
                with open(landmarks_path, 'r') as f:
                    landmarks = json.load(f)
                if not isinstance(landmarks, dict) or len(landmarks) == 0:
                    landmarks = None
                else:
                    print(f"[INFO] Usando anthropometric_landmarks.json: {landmarks_path}")
            except Exception as e:
                print('[AVISO] Falha ao ler anthropometric_landmarks.json:', e)

    return levels, landmarks


def load_levels_from_fit_score(fit_score_json: str) -> Optional[dict]:
    levels, _ = load_overlay_metadata_from_fit_score(fit_score_json)
    return levels


def add_anthropometric_rings_from_levels(scene: trimesh.Scene,
                                         mesh: trimesh.Trimesh,
                                         levels_rel: dict,
                                         landmark_map: Optional[dict] = None) -> Tuple[int, list]:
    specs = [
        ('chest', 'chest', 0.012, None, (255, 60, 60, 255)),
        ('waist', 'waist', 0.012, None, (255, 200, 0, 255)),
        ('hip', 'hip', 0.012, None, (40, 220, 40, 255)),
        ('biceps_left', 'biceps_left', 0.010, 'left', (255, 0, 220, 255)),
        ('biceps_right', 'biceps_right', 0.010, 'right', (255, 0, 220, 255)),
        ('thigh_left', 'thigh_left', 0.012, 'left', (40, 140, 255, 255)),
        ('thigh_right', 'thigh_right', 0.012, 'right', (40, 140, 255, 255)),
    ]
    count = 0
    infos = []
    for name, level_key, band, side, color in specs:
        rel_h = levels_rel.get(level_key, None)
        if rel_h is None:
            print(f"[AVISO] Level ausente no fit_score para {name}")
            continue
        target_point = None
        if landmark_map is not None:
            target_point = landmark_map.get(name, {}).get('point')
        ring = anthropometric_ring_points(
            mesh,
            float(rel_h),
            band_rel=band,
            side=side,
            target_point=target_point,
        )
        ok = add_ring_to_scene(scene, ring, color_rgba=color, node_name=f"ring_{name}")
        if ok:
            count += 1
            infos.append({'name': name, 'color': color, 'ref_point': ring_reference_point(ring)})
            print(f"[INFO] Ring {name}: rel={float(rel_h):.4f} (fit_score)")
        else:
            print(f"[AVISO] Ring {name}: sem pontos suficientes para rel={float(rel_h):.4f}")
    return count, infos


def save_anthropometric_overlay(smpl_gray: trimesh.Trimesh,
                                out_glb: str,
                                out_jpg: str,
                                levels_rel: Optional[dict] = None,
                                landmark_map: Optional[dict] = None) -> None:
    scene = trimesh.Scene()
    scene.add_geometry(smpl_gray, node_name='smpl_fit')
    if levels_rel is not None:
        n_rings, ring_infos = add_anthropometric_rings_from_levels(
            scene,
            smpl_gray,
            levels_rel,
            landmark_map=landmark_map,
        )
    else:
        n_rings, ring_infos = add_anthropometric_rings(scene, smpl_gray)

    add_landmark_markers(scene, ring_infos, radius=marker_radius_from_mesh(smpl_gray), landmark_map=landmark_map)

    scene.export(out_glb)
    print(f"[OK] Overlay antropométrico GLB salvo em: {out_glb} (rings={n_rings})")

    try:
        png = scene.save_image(resolution=(1400, 1000), visible=True)
        png = annotate_overlay_image_with_labels(png, ring_infos, landmark_map=landmark_map)
        with open(out_jpg, 'wb') as f:
            f.write(png)
        print(f"[OK] Overlay antropométrico JPG salvo em: {out_jpg} (rings={n_rings})")
    except Exception as e:
        print('[AVISO] Não consegui gerar JPG antropométrico:', e)


def save_overlay_image(scan_colored: trimesh.Trimesh,
                       smpl_gray: trimesh.Trimesh,
                       out_jpg: str) -> None:
    try:
        scene = trimesh.Scene()
        scene.add_geometry(scan_colored, node_name="scan_error")
        scene.add_geometry(smpl_gray, node_name="smpl_fit")
        png = scene.save_image(resolution=(1400, 1000), visible=True)
        with open(out_jpg, "wb") as f:
            f.write(png)
        print("[OK] Render overlay salvo em:", out_jpg)
    except Exception as e:
        print("[AVISO] Não consegui gerar JPG do overlay:", e)


def print_metrics(errors: np.ndarray, vmax: float) -> None:
    print("\n=== Métricas de erro scan -> SMPL ===")
    print("Erro médio   : %.6f" % errors.mean())
    print("Erro mediano : %.6f" % np.median(errors))
    print("Erro std     : %.6f" % errors.std())
    print("Erro mínimo  : %.6f" % errors.min())
    print("Erro máximo  : %.6f" % errors.max())
    print("P90          : %.6f" % np.percentile(errors, 90))
    print("P95          : %.6f" % np.percentile(errors, 95))
    print("vmax (cor)   : %.6f" % vmax)


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def main():
    scan_path = "/home/gilberto/caesar/data/caesar_clean/csr0017a_processed.ply"

    candidate_smpl_paths = [
        
        "output_demo/registered_scans/csr0017a_processed_smpl.ply",
    ]

    smpl_path = None
    for p in candidate_smpl_paths:
        if os.path.exists(p):
            smpl_path = p
            break

    if smpl_path is None:
        raise FileNotFoundError(
            "Não encontrei a malha de saída do SMPL em output_demo/. "
            "Ajuste candidate_smpl_paths no script."
        )

    out_dir = "output_demo"
    os.makedirs(out_dir, exist_ok=True)

    out_colored_ply = os.path.join(out_dir, "error_map_scan_colored.ply")
    out_hist_jpg = os.path.join(out_dir, "error_hist.jpg")
    out_overlay_glb = os.path.join(out_dir, "error_overlay.glb")
    out_overlay_jpg = os.path.join(out_dir, "error_overlay.jpg")
    out_anthro_glb = os.path.join(out_dir, "anthropometric_overlay.glb")
    out_anthro_jpg = os.path.join(out_dir, "anthropometric_overlay.jpg")
    fit_score_json = os.path.join(out_dir, "fit_score.json")

    device = choose_device()
    print("[INFO] Usando device:", device)
    print("[INFO] Scan:", scan_path)
    print("[INFO] SMPL:", smpl_path)

    scan_mesh = load_mesh(scan_path)
    smpl_mesh = load_mesh(smpl_path)

    smpl_mesh = clean_mesh_for_pytorch3d(smpl_mesh)

    print("[INFO] Scan vertices:", len(scan_mesh.vertices))
    print("[INFO] SMPL vertices:", len(smpl_mesh.vertices))
    print("[INFO] SMPL faces   :", len(smpl_mesh.faces))

    errors = compute_scan_to_smpl_errors(scan_mesh, smpl_mesh, device)
    scan_colored, vmax = make_colored_scan(scan_mesh, errors)
    smpl_gray = make_gray_smpl(smpl_mesh)

    scan_colored.export(out_colored_ply)
    print("[OK] Scan colorido salvo em:", out_colored_ply)

    save_overlay_glb(scan_colored, smpl_gray, out_overlay_glb)
    print("[OK] Overlay GLB salvo em:", out_overlay_glb)

    save_histogram(errors, out_hist_jpg, vmax)
    print("[OK] Histograma salvo em:", out_hist_jpg)

    save_overlay_image(scan_colored, smpl_gray, out_overlay_jpg)
    levels_rel, landmark_map = load_overlay_metadata_from_fit_score(fit_score_json)
    save_anthropometric_overlay(smpl_gray, out_anthro_glb, out_anthro_jpg, levels_rel=levels_rel, landmark_map=landmark_map)

    print_metrics(errors, vmax)

    print("\nArquivos gerados:")
    print(" -", out_colored_ply)
    print(" -", out_overlay_glb)
    print(" -", out_hist_jpg)
    print(" -", out_overlay_jpg, "(se o render funcionar no WSL)")
    print(" -", out_anthro_glb)
    print(" -", out_anthro_jpg, "(se o render funcionar no WSL)")


if __name__ == "__main__":
    main()
