from typing import Any, Callable, cast
import numpy as np
import k3d

K3DPlot = Any


def make_dark_plot(**kwargs: Any) -> K3DPlot:
    """Convenience: create a k3d plot with dark-theme defaults.

    Users can override any default by passing the same keyword in kwargs.
    """

    defaults = dict(
        height=600,
        background_color=0x1E1E1E,
        grid_color=0xD2D2D2,
        label_color=0xF0F0F0,
        grid_visible=True,
        axes=["x", "y", "z"],
    )
    defaults.update(kwargs)
    return cast(Callable[..., K3DPlot], k3d.plot)(**defaults)


def surface_from_function(
    f, x_range, y_range, nx=100, ny=100, scale_z=None, z_min=None, z_max=None
):
    """
    n, R = 1.0, 8.314

    def ideal_gas(V, T):
        return n * R * T / V

    vertices, indices, VV, TT, PP = surface_from_function(
        ideal_gas,
        x_range=(0.2, 5.0),     # V
        y_range=(100, 500),     # T
        nx=120,
        ny=120,
        scale_z=None            # or PP.max()
    )

    plot = k3d.plot()

    plot += k3d.mesh(
        vertices,
        indices,
        flat_shading=False,
        opacity=0.9
    )

    plot.display()
    """
    x = np.linspace(*x_range, nx)
    y = np.linspace(*y_range, ny)

    XX, YY = np.meshgrid(x, y, indexing="xy")
    ZZ = f(XX, YY)

    if scale_z is not None:
        ZZ = ZZ / scale_z

    if z_min is not None or z_max is not None:
        ZZ = np.clip(ZZ, z_min, z_max)

    vertices = np.column_stack([XX.ravel(), YY.ravel(), ZZ.ravel()]).astype(np.float32)

    indices = []
    for i in range(ny - 1):
        for j in range(nx - 1):
            a = i * nx + j
            b = a + 1
            c = a + nx
            d = c + 1

            indices.append([a, b, d])
            indices.append([a, d, c])

    indices = np.array(indices, dtype=np.uint32)

    return vertices, indices, XX, YY, ZZ
