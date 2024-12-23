import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve, bicgstab, spilu
from scipy.sparse.linalg import LinearOperator
import matplotlib.pyplot as plt


class FEMSolver:
    def __init__(self, nx=200, nz=200):
        self.nx = nx
        self.nz = nz
        self.x = None
        self.z = None
        self.protrusion = None

    def solve_potential(self, protrusion):
        """Solve for electric potential using FEM"""
        try:
            # Save protrusion for later use
            self.protrusion = protrusion

            # Generate mesh
            X, Z = self.generate_adaptive_mesh(protrusion)

            # Create stiffness matrix
            K = self.create_stiffness_matrix(X, Z)

            # Apply boundary conditions
            K, f = self.apply_boundary_conditions(K, protrusion)

            # Add small regularization
            K = K + sparse.eye(K.shape[0], format='csr') * 1e-10

            # Solve system
            potential = spsolve(K, f)

            # Reshape solution
            potential = potential.reshape((len(self.z), len(self.x)))

            return X, Z, potential

        except Exception as e:
            print(f"Error in solve_potential: {e}")
            print("Trying alternative solver...")

            # Try iterative solver with preconditioner
            try:
                ilu = spilu(K)
                M = LinearOperator(K.shape, ilu.solve)
                potential, info = bicgstab(K, f, M=M, tol=1e-6, maxiter=1000)

                if info != 0:
                    raise RuntimeError(f"Iterative solver failed to converge (info={info})")

                potential = potential.reshape((len(self.z), len(self.x)))
                return X, Z, potential
            except Exception as e2:
                print(f"Alternative solver also failed: {e2}")
                raise

    def generate_adaptive_mesh(self, protrusion):
        """Generate adaptive mesh with strong refinement near tip and surface"""
        # Base points with logarithmic spacing near surface
        x_min, x_max = 0, 2.5 * protrusion.radius
        z_min, z_max = 0, 1.2 * protrusion.height

        # Create logarithmic spacing for better resolution near surface
        x_half = np.logspace(np.log10(0.1), np.log10(protrusion.radius), self.nx // 2)
        x_second_half = np.linspace(protrusion.radius, x_max, self.nx // 2)
        self.x = np.unique(np.concatenate([x_half, x_second_half]))

        # Similar for z coordinate
        z_dense = np.logspace(np.log10(0.1), np.log10(protrusion.height), self.nz // 2)
        z_sparse = np.linspace(protrusion.height, z_max, self.nz // 2)
        self.z = np.unique(np.concatenate([z_dense, z_sparse]))

        return np.meshgrid(self.x, self.z)

    def create_stiffness_matrix(self, X, Z):
        """Create global stiffness matrix"""
        n = len(self.x) * len(self.z)
        K = sparse.lil_matrix((n, n))

        dx = np.diff(self.x)
        dz = np.diff(self.z)
        dx = np.append(dx, dx[-1])
        dz = np.append(dz, dz[-1])

        for i in range(len(self.x)):
            for j in range(len(self.z)):
                node = i + j * len(self.x)

                if i > 0 and i < len(self.x) - 1 and j > 0 and j < len(self.z) - 1:
                    dxi = (dx[i] + dx[i - 1]) / 2
                    dzi = (dz[j] + dz[j - 1]) / 2

                    K[node, node] = -2 / (dxi ** 2) - 2 / (dzi ** 2)
                    K[node, node - 1] = 1 / (dxi ** 2)  # left
                    K[node, node + 1] = 1 / (dxi ** 2)  # right
                    K[node, node - len(self.x)] = 1 / (dzi ** 2)  # bottom
                    K[node, node + len(self.x)] = 1 / (dzi ** 2)  # top
                else:
                    K[node, node] = 1.0

        return K.tocsr()

    def apply_boundary_conditions(self, K, protrusion):
        """Apply Dirichlet boundary conditions"""
        n = len(self.x) * len(self.z)
        f = np.zeros(n)
        K = K.tolil()

        # Mark nodes inside protrusion
        inside_nodes = []
        for i, x in enumerate(self.x):
            for j, z in enumerate(self.z):
                if self.is_point_in_protrusion(x, z, protrusion):
                    node = i + j * len(self.x)
                    inside_nodes.append(node)

        # Set boundary conditions
        for node in inside_nodes:
            K[node, :] = 0
            K[node, node] = 1
            f[node] = 0

        # Top boundary (external field)
        top_nodes = np.arange(len(self.x)) + (len(self.z) - 1) * len(self.x)
        for node in top_nodes:
            K[node, :] = 0
            K[node, node] = 1
            f[node] = -protrusion.ext_field * self.z[-1]

        # Bottom boundary (ground)
        bottom_nodes = np.arange(len(self.x))
        for node in bottom_nodes:
            K[node, :] = 0
            K[node, node] = 1
            f[node] = 0

        return K.tocsr(), f

    def is_point_in_protrusion(self, x, z, protrusion):
        """Check if point (x,z) is inside protrusion"""
        if z > protrusion.height or z < 0:
            return False
        surface_r = protrusion.get_surface_point(z)
        if surface_r is None:
            return False
        return x <= surface_r

    def calculate_field(self, potential):
        """Calculate electric field from potential"""
        Ex = np.zeros_like(potential)
        Ez = np.zeros_like(potential)

        # Interior points
        for i in range(1, len(self.x) - 1):
            dx = (self.x[i + 1] - self.x[i - 1]) / 2
            Ex[:, i] = -(potential[:, i + 1] - potential[:, i - 1]) / (2 * dx)

        for j in range(1, len(self.z) - 1):
            dz = (self.z[j + 1] - self.z[j - 1]) / 2
            Ez[j, :] = -(potential[j + 1, :] - potential[j - 1, :]) / (2 * dz)

        # Boundaries
        dx_forward = self.x[1] - self.x[0]
        dx_backward = self.x[-1] - self.x[-2]
        dz_forward = self.z[1] - self.z[0]
        dz_backward = self.z[-1] - self.z[-2]

        Ex[:, 0] = -(potential[:, 1] - potential[:, 0]) / dx_forward
        Ex[:, -1] = -(potential[:, -1] - potential[:, -2]) / dx_backward
        Ez[0, :] = -(potential[1, :] - potential[0, :]) / dz_forward
        Ez[-1, :] = -(potential[-1, :] - potential[-2, :]) / dz_backward

        E = np.sqrt(Ex ** 2 + Ez ** 2)
        return E, Ex, Ez

    def plot_results(self, X, Z, potential, E=None):
        """Plot FEM results"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Plot potential
        levels = np.linspace(np.min(potential), np.max(potential), 50)
        cp1 = ax1.contourf(X, Z, potential, levels=levels, cmap='viridis')
        plt.colorbar(cp1, ax=ax1, label='Electric Potential (V)')

        ax1.set_title('Electric Potential')
        ax1.set_xlabel('x (nm)')
        ax1.set_ylabel('z (nm)')

        # Plot field magnitude
        if E is not None:
            E_log = np.log10(E + 1e-10)
            levels_E = np.linspace(np.min(E_log), np.max(E_log), 50)
            cp2 = ax2.contourf(X, Z, E_log, levels=levels_E, cmap='plasma')
            plt.colorbar(cp2, ax=ax2, label='log₁₀(|E|) (V/nm)')
            ax2.set_title('Electric Field Magnitude (log scale)')
            ax2.set_xlabel('x (nm)')
            ax2.set_ylabel('z (nm)')

        plt.tight_layout()
        plt.show()