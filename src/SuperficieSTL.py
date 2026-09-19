import numpy as np
import open3d as o3d


def create_test_surface(nx=200, ny=200, dx=1.0, dy=1.0):
    """
    Genera una superficie sintética Z(x,y) y una máscara booleana.
    """
    x = np.arange(nx) * dx
    y = np.arange(ny) * dy
    X, Y = np.meshgrid(x, y)

    # Ejemplo 1: combinación de gaussiana + plano inclinado
    Z = (
        15.0 * np.exp(-((X - 90)**2 + (Y - 100)**2) / (2 * 25**2))
        + 0.03 * X
        + 0.01 * Y
    )

    # Agregar una segunda protuberancia
    Z += 8.0 * np.exp(-((X - 140)**2 + (Y - 60)**2) / (2 * 15**2))

    # Máscara: todo válido al inicio
    mask = np.ones_like(Z, dtype=bool)

    # Simular un hueco
    hole = ((X - 60)**2 + (Y - 150)**2) < 15**2
    mask[hole] = False

    return X, Y, Z, mask


def heightmap_to_mesh(X, Y, Z, mask):
    """
    Convierte un mapa de altura Z con máscara en una TriangleMesh de Open3D.
    Solo crea triángulos donde los 4 nodos locales son válidos.
    """
    ny, nx = Z.shape

    vertices = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    vertex_valid = mask.ravel()

    triangles = []

    def vid(i, j):
        return i * nx + j

    for i in range(ny - 1):
        for j in range(nx - 1):
            v00 = vid(i, j)
            v01 = vid(i, j + 1)
            v10 = vid(i + 1, j)
            v11 = vid(i + 1, j + 1)

            valid00 = vertex_valid[v00]
            valid01 = vertex_valid[v01]
            valid10 = vertex_valid[v10]
            valid11 = vertex_valid[v11]

            # Triángulo 1: (v00, v10, v11)
            if valid00 and valid10 and valid11:
                triangles.append([v00, v10, v11])

            # Triángulo 2: (v00, v11, v01)
            if valid00 and valid11 and valid01:
                triangles.append([v00, v11, v01])

    triangles = np.asarray(triangles, dtype=np.int32)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)

    # Eliminar vértices no referenciados
    mesh.remove_unreferenced_vertices()

    # Normales
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()

    return mesh


def smooth_mesh(mesh, iterations=5):
    """
    Suavizado laplaciano ligero.
    """
    mesh_smooth = mesh.filter_smooth_laplacian(number_of_iterations=iterations)
    mesh_smooth.compute_vertex_normals()
    mesh_smooth.compute_triangle_normals()
    return mesh_smooth


def export_stl(mesh, filename="surface_test.stl"):
    ok = o3d.io.write_triangle_mesh(filename, mesh)
    if not ok:
        raise RuntimeError(f"No se pudo guardar el archivo STL: {filename}")
    print(f"STL guardado en: {filename}")

def main():
    X, Y, Z, mask = create_test_surface(nx=250, ny=220, dx=0.5, dy=0.5)

    # Exagerar altura para visualizar mejor
    Z = (
    40.0 * np.exp(-((X - 45)**2 + (Y - 50)**2) / (2 * 12**2))
    + 20.0 * np.exp(-((X - 70)**2 + (Y - 30)**2) / (2 * 8**2))
    + 0.05 * X
    + 0.02 * Y
)

    mesh = heightmap_to_mesh(X, Y, Z, mask)
    mesh = smooth_mesh(mesh, iterations=3)

    mesh.paint_uniform_color([0.7, 0.7, 0.7])

    o3d.visualization.draw_geometries(
        [mesh],
        mesh_show_back_face=True,
        mesh_show_wireframe=True
    )

    export_stl(mesh, "surface_test.stl")

if __name__ == "__main__":
    main()