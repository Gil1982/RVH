import os
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

    print_metrics(errors, vmax)

    print("\nArquivos gerados:")
    print(" -", out_colored_ply)
    print(" -", out_overlay_glb)
    print(" -", out_hist_jpg)
    print(" -", out_overlay_jpg, "(se o render funcionar no WSL)")


if __name__ == "__main__":
    main()