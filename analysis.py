import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from protrusions import ConicalProtrusion, ParaboloidProtrusion, EllipsoidProtrusion


class FieldAnalyzer:
    """Class for analyzing field enhancement results"""

    def __init__(self):
        self.results = {}

    def analyze_height_dependency(self, protrusion_class, radius, heights, **kwargs):
        """Analyze how enhancement factor changes with height"""
        betas = []
        for h in heights:
            protrusion = protrusion_class(height=h, radius=radius, **kwargs)
            betas.append(protrusion.get_enhancement_factor())

        self.results['height_dep'] = {'heights': heights, 'betas': betas}
        return heights, betas

    def analyze_radius_dependency(self, protrusion_class, height, radii, **kwargs):
        """Analyze how enhancement factor changes with radius"""
        betas = []
        for r in radii:
            protrusion = protrusion_class(height=height, radius=r, **kwargs)
            betas.append(protrusion.get_enhancement_factor())

        self.results['radius_dep'] = {'radii': radii, 'betas': betas}
        return radii, betas

    def compare_shapes(self, height, radius, tip_radius):
        """Compare different protrusion shapes"""
        results = {}

        # Conical protrusion
        if tip_radius:
            con = ConicalProtrusion(height, radius, tip_radius)
            results['Conical'] = con.get_enhancement_factor()

        # Paraboloid protrusion
        par = ParaboloidProtrusion(height, radius)
        results['Paraboloid'] = par.get_enhancement_factor()

        # Ellipsoid protrusion
        ell = EllipsoidProtrusion(height, radius)
        results['Ellipsoid'] = ell.get_enhancement_factor()

        self.results['shape_comparison'] = results
        return results

    def optimize_parameters(self, protrusion_class, h_range, r_range, **kwargs):
        """Find optimal parameters for maximum enhancement"""

        def objective(x):
            h, r = x
            # Перевіряємо, чи клас приймає tip_radius
            if protrusion_class.__name__ == 'ConicalProtrusion':
                protrusion = protrusion_class(height=h, radius=r, **kwargs)
            else:
                # Для інших типів протрузій не передаємо tip_radius
                filtered_kwargs = {k: v for k, v in kwargs.items()
                                   if k != 'tip_radius'}
                protrusion = protrusion_class(height=h, radius=r, **filtered_kwargs)
            return -protrusion.get_enhancement_factor()  # negative for maximization

        res = minimize(objective,
                       x0=[(h_range[1] + h_range[0]) / 2, (r_range[1] + r_range[0]) / 2],
                       bounds=[h_range, r_range],
                       method='L-BFGS-B')

        opt_result = {
            'height': float(res.x[0]),
            'radius': float(res.x[1]),
            'beta': float(-res.fun)
        }

        self.results['optimization'] = opt_result
        return opt_result

    def analyze_aspect_ratio(self, protrusion_class, ratios, base_height=100, **kwargs):
        """Analyze effect of aspect ratio (height/radius)"""
        betas = []
        heights = []
        radii = []

        for ratio in ratios:
            radius = base_height / ratio
            protrusion = protrusion_class(height=base_height, radius=radius, **kwargs)
            betas.append(protrusion.get_enhancement_factor())
            heights.append(base_height)
            radii.append(radius)

        self.results['aspect_ratio'] = {
            'ratios': ratios.tolist(),
            'betas': betas,
            'heights': heights,
            'radii': radii
        }

        return ratios, betas


class Visualizer:
    """Class for visualization of results"""

    @staticmethod
    def plot_field_distribution(X, Z, potential):
        """Plot electric potential distribution"""
        plt.figure(figsize=(10, 8))
        plt.contourf(X, Z, potential, levels=20, cmap='viridis')
        plt.colorbar(label='Electric Potential (V)')
        plt.xlabel('r (nm)')
        plt.ylabel('z (nm)')
        plt.title('Electric Potential Distribution')
        plt.show()

    @staticmethod
    def plot_enhancement_factor_vs_height(heights, betas, title="Height Dependency"):
        """Plot enhancement factor vs height"""
        plt.figure(figsize=(10, 6))
        plt.plot(heights, betas, 'b-', linewidth=2)
        plt.xlabel('Height (nm)')
        plt.ylabel('Enhancement Factor β')
        plt.title(title)
        plt.grid(True)
        plt.show()

    @staticmethod
    def plot_radius_dependency(radii, betas, title="Radius Dependency"):
        """Plot enhancement factor vs radius"""
        plt.figure(figsize=(10, 6))
        plt.plot(radii, betas, 'r-', linewidth=2)
        plt.xlabel('Radius (nm)')
        plt.ylabel('Enhancement Factor β')
        plt.title(title)
        plt.grid(True)
        plt.show()

    @staticmethod
    def plot_shape_comparison(results):
        """Plot comparison of different shapes"""
        plt.figure(figsize=(10, 6))
        shapes = list(results.keys())
        betas = list(results.values())

        plt.bar(shapes, betas)
        plt.ylabel('Enhancement Factor β')
        plt.title('Comparison of Different Shapes')
        plt.xticks(rotation=45)
        plt.grid(True, axis='y')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_parameter_map(protrusion_class, heights, radii, **kwargs):
        """Plot 2D map of enhancement factor vs height and radius"""
        H, R = np.meshgrid(heights, radii)
        beta = np.zeros_like(H)

        for i in range(len(heights)):
            for j in range(len(radii)):
                protrusion = protrusion_class(height=H[j, i], radius=R[j, i], **kwargs)
                beta[j, i] = protrusion.get_enhancement_factor()

        plt.figure(figsize=(10, 8))
        plt.contourf(H, R, beta, levels=50, cmap='viridis')
        plt.colorbar(label='Enhancement Factor β')
        plt.xlabel('Height (nm)')
        plt.ylabel('Radius (nm)')
        plt.title('Enhancement Factor Parameter Map')
        plt.show()

    @staticmethod
    def plot_aspect_ratio_dependency(ratios, betas):
        """Plot enhancement factor vs aspect ratio"""
        plt.figure(figsize=(10, 6))
        plt.semilogx(ratios, betas, 'g-', linewidth=2)
        plt.xlabel('Aspect Ratio (height/radius)')
        plt.ylabel('Enhancement Factor β')
        plt.title('Enhancement Factor vs Aspect Ratio')
        plt.grid(True)
        plt.show()