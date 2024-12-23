import numpy as np

class NanoProtrusion:
    """Base class for nano protrusion calculations"""
    def __init__(self, height, radius, ext_field=1.0):
        self.height = height
        self.radius = radius
        self.ext_field = ext_field

    def get_surface_point(self, z):
        raise NotImplementedError("Subclass must implement get_surface_point()")

    def get_enhancement_factor(self):
        raise NotImplementedError("Subclass must implement get_enhancement_factor()")

    def get_potential(self, r, z):
        raise NotImplementedError("Subclass must implement get_potential()")

class ConicalProtrusion(NanoProtrusion):
    """Conical protrusion with rounded tip"""
    def __init__(self, height, radius, tip_radius, ext_field=1.0):
        super().__init__(height, radius, ext_field)
        self.tip_radius = tip_radius
        self.theta = np.arctan(radius/height)

    def get_surface_point(self, z):
        if z > self.height:
            return None
        elif z > self.height - self.tip_radius:
            # Spherical cap region
            dz = self.height - z
            r = np.sqrt(self.tip_radius**2 - dz**2)
            return min(r, self.radius)
        else:
            # Conical region
            return (self.radius/self.height) * z

    def get_enhancement_factor(self):
        k = (1 - np.sin(self.theta))/(1 + np.sin(self.theta))
        beta = 1/(k * self.tip_radius)
        return beta

    def get_potential(self, r, z):
        k = (1 - np.sin(self.theta))/(1 + np.sin(self.theta))
        if z >= self.height:
            return -self.ext_field * z
        else:
            return -self.ext_field * z * (1 + self.height/(k * self.tip_radius)) * (1 - z/self.height)**k

class ParaboloidProtrusion(NanoProtrusion):
    """Paraboloid protrusion"""
    def __init__(self, height, radius, ext_field=1.0):
        super().__init__(height, radius, ext_field)
        self.k = 2.0  # shape parameter

    def get_surface_point(self, z):
        if z > self.height:
            return None
        return np.sqrt(2 * self.radius * z / self.height) * self.radius

    def get_enhancement_factor(self):
        beta = (1 + self.height/self.radius)**(self.k + 1)
        return beta

    def get_potential(self, r, z):
        if z >= self.height:
            return -self.ext_field * z
        else:
            return -self.ext_field * z * self.get_enhancement_factor()

class EllipsoidProtrusion(NanoProtrusion):
    """Ellipsoid protrusion"""
    def __init__(self, height, radius, ext_field=1.0):
        super().__init__(height, radius, ext_field)
        self.a = height  # semi-major axis
        self.b = radius  # semi-minor axis

    def get_surface_point(self, z):
        if z > self.height:
            return None
        return self.radius * np.sqrt(1 - (z/self.height)**2)

    def get_enhancement_factor(self):
        aspect_ratio = self.height/self.radius
        beta = aspect_ratio * (1 + np.sqrt(1 - 1/aspect_ratio**2))
        return beta

    def get_potential(self, r, z):
        if z >= self.height:
            return -self.ext_field * z
        else:
            return -self.ext_field * z * self.get_enhancement_factor()

class ProtrusionArray:
    """Array of multiple protrusions"""
    def __init__(self, protrusion_type, spacing, nx, ny):
        self.protrusion_type = protrusion_type
        self.spacing = spacing
        self.nx = nx
        self.ny = ny
        self.protrusions = []
        self._generate_array()

    def _generate_array(self):
        for i in range(self.nx):
            for j in range(self.ny):
                x = i * self.spacing
                y = j * self.spacing
                self.protrusions.append({
                    'position': (x, y),
                    'protrusion': self.protrusion_type
                })

    def get_effective_enhancement(self):
        """Calculate effective enhancement factor with screening"""
        single_beta = self.protrusion_type.get_enhancement_factor()
        density = 1/(self.spacing**2)
        screening_factor = 1/(1 + np.sqrt(density) * self.protrusion_type.radius)
        return single_beta * screening_factor