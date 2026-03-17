import os
import math
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
                               side: Optional[str] = None) -> np.ndarray:
    """Estimativa de anel 3D para uma circunferência antropométrica."""
    verts = np.asarray(mesh.vertices)
    up_axis, lr_axis, depth_axis = infer_body_axes(verts)

    up = verts[:, up_axis]
    up_min, up_max = float(up.min()), float(up.max())
    h = max(up_max - up_min, 1e-8)
    up_target = up_min + rel_height * h
    band = max(band_rel * h, 1e-5)

    keep = np.abs(up - up_target) <= band
    if side == 'left':
        keep &= verts[:, lr_axis] > 0
    elif side == 'right':
        keep &= verts[:, lr_axis] < 0

    pts = verts[keep][:, [lr_axis, depth_axis]]
    if pts.shape[0] < 16:
        return np.empty((0, 3), dtype=np.float64)

    hull = convex_hull_2d(pts)
    if hull.shape[0] < 3:
        return np.empty((0, 3), dtype=np.float64)

    ring = np.zeros((hull.shape[0], 3), dtype=np.float64)
    ring[:, lr_axis] = hull[:, 0]
    ring[:, depth_axis] = hull[:, 1]
    ring[:, up_axis] = up_target
    ring = np.vstack([ring, ring[:1]])
    return ring


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


def add_anthropometric_rings(scene: trimesh.Scene, mesh: trimesh.Trimesh) -> int:
    """Adiciona anéis antropométricos adaptativos ao scene; retorna quantidade desenhada."""
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
    for name, rel_range, mode, band, side, color in specs:
        ring, circ = best_ring_in_window(mesh, rel_range=rel_range, mode=mode, band_rel=band, side=side)
        ok = add_ring_to_scene(scene, ring, color_rgba=color, node_name=f"ring_{name}")
        if ok:
            count += 1
            print(f"[INFO] Ring {name}: circ={circ:.4f}, range={rel_range}, mode={mode}")
        else:
            print(f"[AVISO] Ring {name}: sem pontos suficientes")
    return count


def save_anthropometric_overlay(smpl_gray: trimesh.Trimesh,
                                out_glb: str,
                                out_jpg: str) -> None:
    scene = trimesh.Scene()
    scene.add_geometry(smpl_gray, node_name='smpl_fit')
    n_rings = add_anthropometric_rings(scene, smpl_gray)
    scene.export(out_glb)
    print(f"[OK] Overlay antropométrico GLB salvo em: {out_glb} (rings={n_rings})")

    try:
        png = scene.save_image(resolution=(1400, 1000), visible=True)
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
    save_anthropometric_overlay(smpl_gray, out_anthro_glb, out_anthro_jpg)

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