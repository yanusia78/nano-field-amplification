import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import json

from protrusions import (ConicalProtrusion, ParaboloidProtrusion,
                         EllipsoidProtrusion, ProtrusionArray)
from fem_solver import FEMSolver
from analysis import FieldAnalyzer, Visualizer


class NanoFieldAnalysis:
    """Main class for managing field enhancement analysis"""

    def __init__(self, output_dir="results"):
        self.analyzer = FieldAnalyzer()
        self.visualizer = Visualizer()
        self.output_dir = output_dir
        self._setup_output_dir()

    def _setup_output_dir(self):
        """Create output directory if it doesn't exist"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def save_results(self, data, filename):
        """Save results to JSON file"""
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)

    def analyze_single_protrusions(self):
        """Analyze different types of single protrusions"""
        print("\n1. Analyzing single protrusions...")

        # Base parameters
        height = 80  # nm
        radius = 20  # nm
        tip_radius = 5  # nm

        # Compare different shapes
        shapes_comparison = self.analyzer.compare_shapes(
            height=height,
            radius=radius,
            tip_radius=tip_radius
        )

        # Save results
        self.save_results(shapes_comparison, "shape_comparison.json")

        # Plot comparison
        self.visualizer.plot_shape_comparison(shapes_comparison)

        return shapes_comparison

    def optimize_protrusion(self):
        """Optimize protrusion parameters"""
        print("\n2. Optimizing parameters...")

        # Optimization for different shapes
        shapes = {
            'Conical': ConicalProtrusion,
            'Paraboloid': ParaboloidProtrusion,
            'Ellipsoid': EllipsoidProtrusion
        }

        optimization_results = {}
        for shape_name, shape_class in shapes.items():
            optimal_params = self.analyzer.optimize_parameters(
                shape_class,
                h_range=(50, 200),
                r_range=(5, 50),
                tip_radius=5 if shape_name == 'Conical' else None
            )
            optimization_results[shape_name] = optimal_params

        # Save results
        self.save_results(optimization_results, "optimization_results.json")

        return optimization_results

    def run_fem_analysis(self, protrusion):
        """Run FEM analysis for given protrusion"""
        print("\n3. Running FEM analysis...")

        solver = FEMSolver(nx=200, nz=200)
        solver.protrusion = protrusion

        # Calculate potential and field
        X, Z, potential = solver.solve_potential(protrusion)
        E, Ex, Ez = solver.calculate_field(potential)

        # Calculate enhancement factor
        E_max = np.max(E)
        beta_fem = E_max / protrusion.ext_field

        # Plot results
        solver.plot_results(X, Z, potential, E)

        results = {
            'beta_fem': float(beta_fem),
            'E_max': float(E_max)
        }
        self.save_results(results, "fem_results.json")

        return results

    def analyze_arrays(self, protrusion):
        """Analyze arrays of protrusions"""
        print("\n4. Analyzing protrusion arrays...")

        spacings = np.linspace(30, 200, 10)
        array_results = []

        for spacing in spacings:
            array = ProtrusionArray(protrusion, spacing, nx=3, ny=3)
            beta_eff = array.get_effective_enhancement()
            array_results.append({
                'spacing': float(spacing),
                'beta_eff': float(beta_eff)
            })

        # Save results
        self.save_results(array_results, "array_results.json")

        # Plot results
        plt.figure(figsize=(10, 6))
        spacing_values = [res['spacing'] for res in array_results]
        beta_values = [res['beta_eff'] for res in array_results]
        plt.plot(spacing_values, beta_values, 'b-o')
        plt.xlabel('Spacing (nm)')
        plt.ylabel('Effective Enhancement Factor β')
        plt.title('Array Enhancement vs Spacing')
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, 'array_analysis.png'))
        plt.close()

        return array_results

    def generate_report(self, all_results):
        """Generate comprehensive analysis report"""
        print("\n5. Generating report...")

        report = f"""
    Field Enhancement Analysis Report
    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

    1. Shape Comparison:
    """
        # Add shape comparison results
        for shape, beta in all_results['shapes'].items():
            report += f"   {shape}: beta = {beta:.2f}\n"

        report += "\n2. Optimization Results:\n"
        # Add optimization results
        for shape, params in all_results['optimization'].items():
            report += f"""   {shape}:
          Height: {params['height']:.1f} nm
          Radius: {params['radius']:.1f} nm
          Maximum beta: {params['beta']:.1f}
    """

        report += f"""
    3. FEM Analysis Results:
       Maximum field: {all_results['fem']['E_max']:.2f} V/nm
       Numerical beta: {all_results['fem']['beta_fem']:.2f}

    4. Array Analysis:
       Spacing range: {all_results['arrays'][0]['spacing']:.1f} - {all_results['arrays'][-1]['spacing']:.1f} nm
       Maximum beta_eff: {max(res['beta_eff'] for res in all_results['arrays']):.2f}
    """

        # Save report
        try:
            report_path = os.path.join(self.output_dir, 'analysis_report.txt')
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report)
        except Exception as e:
            print(f"Warning: Could not save report to file: {e}")
            print("\nReport content:")
            print(report)

        return report

def main():
    # Initialize analysis
    analysis = NanoFieldAnalysis()

    # Run analyses
    shape_results = analysis.analyze_single_protrusions()
    opt_results = analysis.optimize_protrusion()

    # Create optimal conical protrusion for detailed analysis
    optimal_conical = ConicalProtrusion(
        height=opt_results['Conical']['height'],
        radius=opt_results['Conical']['radius'],
        tip_radius=5
    )

    # Run FEM and array analysis
    fem_results = analysis.run_fem_analysis(optimal_conical)
    array_results = analysis.analyze_arrays(optimal_conical)

    # Collect all results
    all_results = {
        'shapes': shape_results,
        'optimization': opt_results,
        'fem': fem_results,
        'arrays': array_results
    }

    # Generate report
    report = analysis.generate_report(all_results)
    print("\nAnalysis complete. Results saved in 'results' directory.")


if __name__ == "__main__":
    main()