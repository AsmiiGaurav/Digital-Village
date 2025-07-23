import rasterio
import geopandas as gpd
import numpy as np
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from shapely.geometry import box, Polygon
from shapely.ops import unary_union
import pandas as pd
from typing import Dict, List, Tuple, Optional
import warnings
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MPLPolygon
warnings.filterwarnings('ignore')

def shoelace_area(coordinates: List[Tuple[float, float]]) -> float:
   """
   Calculate polygon area using the shoelace formula.

   Parameters:
   -----------
   coordinates : List[Tuple[float, float]]
       List of (x, y) coordinate tuples forming the polygon

   Returns:
   --------
   float
       Area of the polygon
   """
   if len(coordinates) < 3:
       return 0.0

   # Ensure polygon is closed
   if coordinates[0] != coordinates[-1]:
       coordinates = coordinates + [coordinates[0]]

   n = len(coordinates) - 1  # Exclude the duplicate closing point
   area = 0.0

   for i in range(n):
       j = (i + 1) % n
       area += coordinates[i][0] * coordinates[j][1]
       area -= coordinates[j][0] * coordinates[i][1]

   return abs(area) / 2.0

def get_pixel_coordinates(pixel_row: int, pixel_col: int, transform: rasterio.Affine) -> List[Tuple[float, float]]:
   """
   Get the corner coordinates of a pixel.

   Parameters:
   -----------
   pixel_row : int
       Row index of the pixel
   pixel_col : int
       Column index of the pixel
   transform : rasterio.Affine
       Raster transform

   Returns:
   --------
   List[Tuple[float, float]]
       List of corner coordinates [(x, y), ...]
   """
   # Get pixel bounds in geographic coordinates
   left = transform * (pixel_col, pixel_row)
   right = transform * (pixel_col + 1, pixel_row)
   bottom_left = transform * (pixel_col, pixel_row + 1)
   bottom_right = transform * (pixel_col + 1, pixel_row + 1)

   # Return corners in order: top-left, top-right, bottom-right, bottom-left
   return [
       left,           # top-left
       right,          # top-right
       bottom_right,   # bottom-right
       bottom_left     # bottom-left
   ]

def get_intersection_coordinates(plot_geometry, pixel_coords: List[Tuple[float, float]]) -> Optional[List[Tuple[float, float]]]:
   """
   Get the coordinates of intersection between plot geometry and pixel.

   Parameters:
   -----------
   plot_geometry : shapely.geometry
       Plot geometry
   pixel_coords : List[Tuple[float, float]]
       Pixel corner coordinates

   Returns:
   --------
   Optional[List[Tuple[float, float]]]
       Intersection coordinates or None if no intersection
   """
   try:
       # Create pixel polygon
       pixel_polygon = Polygon(pixel_coords)

       # Calculate intersection
       intersection = plot_geometry.intersection(pixel_polygon)

       if intersection.is_empty:
           return None

       # Handle different geometry types
       if hasattr(intersection, 'exterior'):
           # Single polygon
           return list(intersection.exterior.coords[:-1])  # Exclude duplicate last point
       elif hasattr(intersection, 'geoms'):
           # MultiPolygon or GeometryCollection
           all_coords = []
           for geom in intersection.geoms:
               if hasattr(geom, 'exterior'):
                   all_coords.extend(list(geom.exterior.coords[:-1]))
           return all_coords if all_coords else None
       else:
           return None

   except Exception as e:
       print(f"Error calculating intersection: {e}")
       return None

def calculate_precise_area_weighted_et(geotiff_path: str, geojson_path: str,
                                    plot_id_column: str = 'id') -> pd.DataFrame:
   """
   Calculate area-weighted ET values using precise coordinate-based area calculation.

   Parameters:
   -----------
   geotiff_path : str
       Path to the GeoTIFF file containing ET values
   geojson_path : str
       Path to the GeoJSON file containing plot boundaries
   plot_id_column : str
       Column name in GeoJSON that contains plot identifiers

   Returns:
   --------
   pd.DataFrame
       DataFrame with plot IDs and their area-weighted ET values
   """

   # Read the GeoJSON file
   plots_gdf = gpd.read_file(geojson_path)

   # Read the GeoTIFF file
   with rasterio.open(geotiff_path) as src:
       et_data = src.read(1)  # Read first band
       transform = src.transform
       crs = src.crs
       nodata = src.nodata

   # Ensure both datasets have the same CRS
   if plots_gdf.crs != crs:
       plots_gdf = plots_gdf.to_crs(crs)

   results = []

   for idx, plot in plots_gdf.iterrows():
       plot_id = plot[plot_id_column] if plot_id_column in plot else idx
       geometry = plot.geometry

       print(f"Processing plot {plot_id}...")

       # Get the bounding box of the plot
       minx, miny, maxx, maxy = geometry.bounds

       # Convert bounds to pixel coordinates
       ul_col, ul_row = ~transform * (minx, maxy)  # Upper left
       lr_col, lr_row = ~transform * (maxx, miny)  # Lower right

       # Round to get pixel indices
       ul_col, ul_row = int(np.floor(ul_col)), int(np.floor(ul_row))
       lr_col, lr_row = int(np.ceil(lr_col)), int(np.ceil(lr_row))

       # Ensure indices are within raster bounds
       ul_col = max(0, ul_col)
       ul_row = max(0, ul_row)
       lr_col = min(et_data.shape[1], lr_col)
       lr_row = min(et_data.shape[0], lr_row)

       if ul_col >= lr_col or ul_row >= lr_row:
           print(f"Warning: Plot {plot_id} is outside raster bounds")
           results.append({
               'plot_id': plot_id,
               'area_weighted_et': np.nan,
               'total_intersection_area': 0,
               'pixel_count': 0,
               'intersection_details': []
           })
           continue

       # Store intersection details for each pixel
       intersection_details = []
       total_intersection_area = 0

       # Process each pixel in the bounding box
       for row in range(ul_row, lr_row):
           for col in range(ul_col, lr_col):
               # Get pixel value
               pixel_value = et_data[row, col]

               # Skip nodata pixels
               if nodata is not None and pixel_value == nodata:
                   continue
               if np.isnan(pixel_value):
                   continue

               # Get pixel corner coordinates
               pixel_coords = get_pixel_coordinates(row, col, transform)

               # Get intersection coordinates
               intersection_coords = get_intersection_coordinates(geometry, pixel_coords)

               if intersection_coords and len(intersection_coords) >= 3:
                   # Calculate intersection area using shoelace formula
                   intersection_area = shoelace_area(intersection_coords)

                   if intersection_area > 0:
                       intersection_details.append({
                           'pixel_row': row,
                           'pixel_col': col,
                           'pixel_value': pixel_value,
                           'intersection_area': intersection_area,
                           'intersection_coords': intersection_coords,
                           'pixel_coords': pixel_coords
                       })
                       total_intersection_area += intersection_area

       # Calculate area-weighted ET
       if total_intersection_area > 0 and intersection_details:
           area_weighted_sum = 0

           # Calculate weights and weighted sum
           for detail in intersection_details:
               weight = detail['intersection_area'] / total_intersection_area
               detail['weight'] = weight
               area_weighted_sum += detail['pixel_value'] * weight

           area_weighted_et = area_weighted_sum
       else:
           area_weighted_et = np.nan

       results.append({
           'plot_id': plot_id,
           'area_weighted_et': area_weighted_et,
           'total_intersection_area': total_intersection_area,
           'pixel_count': len(intersection_details),
           'intersection_details': intersection_details
       })

   # Create DataFrame with main results
   main_results = pd.DataFrame([{
       'plot_id': r['plot_id'],
       'area_weighted_et': r['area_weighted_et'],
       'total_intersection_area': r['total_intersection_area'],
       'pixel_count': r['pixel_count']
   } for r in results])

   # Store detailed results for visualization
   main_results._detailed_results = results

   return main_results

def visualize_precise_plot_coverage(geotiff_path: str, geojson_path: str, plot_id: str,
                                 results_df: pd.DataFrame, plot_id_column: str = 'id'):
   """
   Visualize precise plot coverage with coordinate-based intersections.
   """
   # Get detailed results
   detailed_results = getattr(results_df, '_detailed_results', None)
   if detailed_results is None:
       print("No detailed results available. Please run calculate_precise_area_weighted_et first.")
       return

   # Find the specific plot results
   plot_result = None
   for result in detailed_results:
       if str(result['plot_id']) == str(plot_id):
           plot_result = result
           break

   if plot_result is None:
       print(f"Plot {plot_id} not found in results.")
       return

   # Read data
   plots_gdf = gpd.read_file(geojson_path)
   plot = plots_gdf[plots_gdf[plot_id_column] == plot_id].iloc[0]

   with rasterio.open(geotiff_path) as src:
       et_data = src.read(1)
       transform = src.transform
       crs = src.crs
       nodata = src.nodata

   # Ensure same CRS
   if plots_gdf.crs != crs:
       plots_gdf = plots_gdf.to_crs(crs)
       plot = plots_gdf[plots_gdf[plot_id_column] == plot_id].iloc[0]

   # Create visualization
   fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))

   # Get plot bounds with buffer
   minx, miny, maxx, maxy = plot.geometry.bounds
   buffer = max((maxx - minx), (maxy - miny)) * 0.2
   plot_minx, plot_miny = minx - buffer, miny - buffer
   plot_maxx, plot_maxy = maxx + buffer, maxy + buffer

   # Convert to pixel coordinates for cropping
   ul_col, ul_row = ~transform * (plot_minx, plot_maxy)
   lr_col, lr_row = ~transform * (plot_maxx, plot_miny)
   ul_col, ul_row = int(np.floor(ul_col)), int(np.floor(ul_row))
   lr_col, lr_row = int(np.ceil(lr_col)), int(np.ceil(lr_row))
   ul_col = max(0, ul_col)
   ul_row = max(0, ul_row)
   lr_col = min(et_data.shape[1], lr_col)
   lr_row = min(et_data.shape[0], lr_row)

   # Create extent
   extent = [plot_minx, plot_maxx, plot_miny, plot_maxy]

   # Crop raster
   subset = et_data[ul_row:lr_row, ul_col:lr_col]
   subset_masked = np.copy(subset).astype(float)
   if nodata is not None:
       subset_masked[subset == nodata] = np.nan

   # Plot 1: ET data with plot overlay
   im1 = ax1.imshow(subset_masked, extent=extent, origin='upper', alpha=0.8, cmap='viridis')

   # Plot plot boundary
   if hasattr(plot.geometry, 'exterior'):
       x, y = plot.geometry.exterior.xy
       ax1.plot(x, y, color='red', linewidth=3, label=f'Plot {plot_id}')
       ax1.fill(x, y, color='red', alpha=0.2)

   # Add pixel grid
   for row in range(ul_row, lr_row + 1):
       for col in range(ul_col, lr_col + 1):
           pixel_coords = get_pixel_coordinates(row, col, transform)
           if pixel_coords:
               pixel_x = [coord[0] for coord in pixel_coords] + [pixel_coords[0][0]]
               pixel_y = [coord[1] for coord in pixel_coords] + [pixel_coords[0][1]]
               ax1.plot(pixel_x, pixel_y, color='white', alpha=0.5, linewidth=1)

   ax1.set_title(f'Plot {plot_id} - ET Data with Pixel Grid', fontsize=14)
   ax1.set_xlabel('X Coordinate')
   ax1.set_ylabel('Y Coordinate')
   ax1.legend()
   plt.colorbar(im1, ax=ax1, label='ET Value', fraction=0.046, pad=0.04)

   # Plot 2: Intersection areas
   ax2.set_xlim(plot_minx, plot_maxx)
   ax2.set_ylim(plot_miny, plot_maxy)

   # Plot plot boundary
   if hasattr(plot.geometry, 'exterior'):
       x, y = plot.geometry.exterior.xy
       ax2.plot(x, y, color='blue', linewidth=3, label=f'Plot {plot_id}')

   # Plot intersection areas with different colors based on weights
   colors = plt.cm.Reds(np.linspace(0.3, 1.0, len(plot_result['intersection_details'])))

   for i, detail in enumerate(plot_result['intersection_details']):
       # Plot pixel boundary
       pixel_coords = detail['pixel_coords']
       pixel_x = [coord[0] for coord in pixel_coords] + [pixel_coords[0][0]]
       pixel_y = [coord[1] for coord in pixel_coords] + [pixel_coords[0][1]]
       ax2.plot(pixel_x, pixel_y, color='black', linewidth=1, alpha=0.5)

       # Plot intersection area
       intersection_coords = detail['intersection_coords']
       if len(intersection_coords) >= 3:
           intersection_x = [coord[0] for coord in intersection_coords]
           intersection_y = [coord[1] for coord in intersection_coords]

           # Create matplotlib polygon
           polygon = MPLPolygon(list(zip(intersection_x, intersection_y)),
                              facecolor=colors[i], alpha=0.7, edgecolor='black', linewidth=1)
           ax2.add_patch(polygon)

           # Add weight text
           centroid_x = np.mean(intersection_x)
           centroid_y = np.mean(intersection_y)
           ax2.text(centroid_x, centroid_y, f'{detail["weight"]:.3f}',
                   ha='center', va='center', fontsize=8, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

   ax2.set_title(f'Plot {plot_id} - Intersection Areas & Weights', fontsize=14)
   ax2.set_xlabel('X Coordinate')
   ax2.set_ylabel('Y Coordinate')
   ax2.legend()
   ax2.set_aspect('equal')

   # Plot 3: Summary statistics
   ax3.axis('off')

   # Prepare summary data
   summary_text = f"=== Plot {plot_id} Summary ===\n\n"
   summary_text += f"Area-weighted ET: {plot_result['area_weighted_et']:.6f}\n"
   summary_text += f"Total intersection area: {plot_result['total_intersection_area']:.4f}\n"
   summary_text += f"Number of intersecting pixels: {plot_result['pixel_count']}\n\n"

   summary_text += "Pixel Details:\n"
   summary_text += f"{'Pixel':<8} {'ET Value':<10} {'Area':<10} {'Weight':<10}\n"
   summary_text += f"{'-'*50}\n"

   for i, detail in enumerate(plot_result['intersection_details']):
       pixel_pos = f"({detail['pixel_row']},{detail['pixel_col']})"
       summary_text += f"{pixel_pos:<8} {detail['pixel_value']:<10.4f} "
       summary_text += f"{detail['intersection_area']:<10.4f} {detail['weight']:<10.4f}\n"

   # Verification
   total_weight = sum(detail['weight'] for detail in plot_result['intersection_details'])
   summary_text += f"\nWeight sum: {total_weight:.6f} (should be 1.0)\n"

   # Simple average for comparison
   pixel_values = [detail['pixel_value'] for detail in plot_result['intersection_details']]
   if pixel_values:
       simple_avg = np.mean(pixel_values)
       summary_text += f"Simple average ET: {simple_avg:.6f}\n"
       summary_text += f"Difference: {abs(plot_result['area_weighted_et'] - simple_avg):.6f}"

   ax3.text(0.05, 0.95, summary_text, transform=ax3.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))

   plt.tight_layout()
   plt.show()

   return fig

def compare_methods(geotiff_path: str, geojson_path: str, plot_id_column: str = 'id'):
   """
   Compare the original and precise methods.
   """
   print("Running original method...")
   # Note: You'll need to import your original function
   # original_results = calculate_area_weighted_et(geotiff_path, geojson_path, plot_id_column)

   print("Running precise method...")
   precise_results = calculate_precise_area_weighted_et(geotiff_path, geojson_path, plot_id_column)

   print("\nPrecise method results:")
   print(precise_results.head())

   return precise_results

# Example usage
if __name__ == "__main__":
   # File paths
   geotiff_path = "/content/5052_01_04_25"
   geojson_path = "/content/plot_5052.json"

   print("Calculating precise area-weighted ET values...")

   # Calculate using precise coordinate-based method
   results = calculate_precise_area_weighted_et(
       geotiff_path,
       geojson_path,
       plot_id_column='id'
   )

   # Display results
   print("\nPrecise Results:")
   print(results)

   # Save results to CSV
   results.to_csv('precise_area_weighted_et.csv', index=False)
   print("\nResults saved to precise_area_weighted_et.csv!")

   # Visualize specific plots
   if len(results) > 0:
       plot_ids = results['plot_id'].head(2).tolist()
       print(f"\nCreating detailed visualizations for plots: {plot_ids}")

       for plot_id in plot_ids:
           print(f"Visualizing plot {plot_id}...")
           visualize_precise_plot_coverage(geotiff_path, geojson_path, str(plot_id),
                                          results, plot_id_column='id')

   print("\nPrecise calculation complete!")
