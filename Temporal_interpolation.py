import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import warnings
import os

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def test_file_access():
   """Test if the Excel file is accessible"""
   test_file = '/content/june-oct.xlsx'

   print("="*60)
   print("CHECKING FILE ACCESS...")
   print("="*60)

   if os.path.exists(test_file):
       print(f"✓ File found: {test_file}")
       return test_file
   else:
       print(f"✗ File not found: {test_file}")

       # Check for other Excel files
       try:
           files = os.listdir('/content/')
           excel_files = [f for f in files if f.endswith(('.xlsx', '.xls'))]

           if excel_files:
               print("\nAvailable Excel files in /content/:")
               for f in excel_files:
                   print(f"  - {f}")
               # Use the first Excel file found
               return f'/content/{excel_files[0]}'
           else:
               print("No Excel files found in /content/")
               print("Please upload your Excel file to the Colab environment")
               return None
       except:
           print("Could not access /content/ directory")
           return None

def load_farm_type_tables(file_path):
   """Load the three farm type tables (TPR, DSR, AWD) from Excel file."""

   # Try to read from different sheet names
   try:
       df_raw = pd.read_excel(file_path, sheet_name='Sheet1', header=None)
   except:
       try:
           df_raw = pd.read_excel(file_path, sheet_name='Sheet2', header=None)
       except:
           df_raw = pd.read_excel(file_path, header=None)

   # Find table boundaries automatically
   table_starts = []

   for idx, row in df_raw.iterrows():
       if any(str(val).lower().strip() == 'date' for val in row.values if pd.notna(val)):
           table_starts.append(idx)

   print(f"Found {len(table_starts)} table headers at rows: {[x+1 for x in table_starts]}")

   # Define table configurations
   farm_type_names = ['TPR', 'DSR', 'AWD']
   farm_tables = {}

   for i, start_row in enumerate(table_starts):
       farm_type = farm_type_names[i] if i < len(farm_type_names) else f'Table_{i+1}'

       # Determine end row
       if i + 1 < len(table_starts):
           end_row = table_starts[i + 1]
       else:
           end_row = len(df_raw)
           for j in range(start_row + 1, len(df_raw)):
               if df_raw.iloc[j].isna().all():
                   has_more_data = False
                   for k in range(j + 1, min(j + 5, len(df_raw))):
                       if not df_raw.iloc[k].isna().all():
                           has_more_data = True
                           break
                   if not has_more_data:
                       end_row = j
                       break

       # Extract headers and data
       headers = df_raw.iloc[start_row].values
       headers = [str(h) if pd.notna(h) else f'col_{i}' for i, h in enumerate(headers)]

       data_start = start_row + 1
       data_rows = df_raw.iloc[data_start:end_row].copy()
       data_rows = data_rows.dropna(how='all')

       if not data_rows.empty:
           data_rows.columns = headers[:len(data_rows.columns)]
           table_df = process_farm_table(data_rows, farm_type)

           if not table_df.empty:
               farm_tables[farm_type] = table_df
               print(f"✓ {farm_type}: {len(table_df)} dekadal periods, {len(table_df.columns)} farms")

   return farm_tables

def process_farm_table(table_data, farm_type):
   """Process individual farm table."""

   table_data = table_data.copy()

   # Find date column
   date_col = None
   for col in table_data.columns:
       if 'date' in str(col).lower():
           date_col = col
           break

   if date_col is None:
       date_col = table_data.columns[0]

   # Convert date column
   try:
       table_data[date_col] = pd.to_datetime(table_data[date_col], errors='coerce')
   except:
       try:
           # Handle Excel serial numbers
           table_data[date_col] = pd.to_datetime(table_data[date_col], origin='1899-12-30', unit='D', errors='coerce')
       except:
           print(f"Warning: Could not convert dates for {farm_type}")
           return pd.DataFrame()

   # Remove rows with invalid dates
   table_data = table_data.dropna(subset=[date_col])

   if table_data.empty:
       return pd.DataFrame()

   # Set date as index
   table_data.set_index(date_col, inplace=True)

   # Convert other columns to numeric
   numeric_columns = []
   for col in table_data.columns:
       if col != date_col:
           try:
               if isinstance(table_data[col], pd.Series):
                   table_data[col] = pd.to_numeric(table_data[col], errors='coerce')
                   if table_data[col].notna().sum() > 0:
                       numeric_columns.append(col)
           except:
               continue

   if not numeric_columns:
       return pd.DataFrame()

   table_data = table_data[numeric_columns]
   table_data = table_data.dropna(axis=1, how='all')

   return table_data

def interpolate_to_daily(dekadal_df, farm_type):
   """Convert dekadal data to daily using cubic spline interpolation."""
   if dekadal_df.empty:
       return pd.DataFrame()

   dekadal_df = dekadal_df.sort_index()
   start_date = dekadal_df.index.min()
   end_date = dekadal_df.index.max()
   daily_dates = pd.date_range(start=start_date, end=end_date, freq='D')

   dekadal_numeric = (dekadal_df.index - start_date).days.values
   daily_numeric = (daily_dates - start_date).days.values

   daily_df = pd.DataFrame(index=daily_dates)

   print(f"  Interpolating {farm_type}: {len(dekadal_df)} → {len(daily_dates)} days")

   successful_farms = []
   for col in dekadal_df.columns:
       try:
           valid_mask = dekadal_df[col].notna()

           if valid_mask.sum() < 2:
               continue

           x_vals = dekadal_numeric[valid_mask]
           y_vals = dekadal_df[col][valid_mask].values

           if len(x_vals) >= 4:
               cs = CubicSpline(x_vals, y_vals, bc_type='natural')
               interpolated = cs(daily_numeric)
           else:
               interpolated = np.interp(daily_numeric, x_vals, y_vals)

           interpolated = np.maximum(interpolated, 0)
           # Round interpolated values to 2 decimal places
           interpolated = np.round(interpolated, 2)
           daily_df[col] = interpolated
           successful_farms.append(col)

       except Exception as e:
           continue

   print(f"  {farm_type}: ✓ {len(successful_farms)} farms interpolated")
   return daily_df

def calculate_daily_averages(daily_dfs):
   """Calculate daily average ET values for each farm type."""

   avg_df = pd.DataFrame()

   for farm_type, daily_df in daily_dfs.items():
       if not daily_df.empty:
           daily_avg = daily_df.mean(axis=1, skipna=True)
           avg_df[farm_type] = daily_avg
           print(f"  {farm_type}: {daily_avg.min():.2f} - {daily_avg.max():.2f} mm/day")

   return avg_df

def plot_et_comparison(avg_df, period_name="Analysis Period", save_png=True):
   """Create comparison plot of ET trends across farm types."""

   plt.figure(figsize=(14, 8))

   colors = {'TPR': '#2E86C1', 'DSR': '#28B463', 'AWD': '#F39C12'}
   line_styles = {'TPR': '-', 'DSR': '--', 'AWD': '-.'}

   for farm_type in avg_df.columns:
       plt.plot(avg_df.index, avg_df[farm_type],
               color=colors.get(farm_type, 'black'),
               linestyle=line_styles.get(farm_type, '-'),
               linewidth=3,
               label=f'{farm_type} (Avg: {avg_df[farm_type].mean():.2f} mm/day)',
               marker='o', markersize=5, alpha=0.8)

   plt.title(f'Evapotranspiration (ET) Trends Comparison\n{period_name}',
             fontsize=16, fontweight='bold', pad=20)
   plt.xlabel('Date', fontsize=12, fontweight='bold')
   plt.ylabel('ET Value (mm/day)', fontsize=12, fontweight='bold')

   plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
   plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
   plt.xticks(rotation=45)

   plt.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
   plt.legend(loc='upper left', frameon=True, fancybox=True, shadow=True, fontsize=11)

   # Statistics box
   stats_text = f"""Statistics Summary:
Total Days: {len(avg_df)}
Date Range: {avg_df.index.min().strftime('%Y-%m-%d')} to {avg_df.index.max().strftime('%Y-%m-%d')}

Farm Type Comparison:"""

   for farm_type in avg_df.columns:
       stats_text += f"\n  {farm_type}: {avg_df[farm_type].mean():.2f} ± {avg_df[farm_type].std():.2f} mm/day"

   plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            fontsize=9, fontfamily='monospace')

   plt.tight_layout()

   # Save plot as PNG
   if save_png:
       plot_filename = 'ET_Farm_Types_Comparison_Nov2024_Apr2025.png'
       plt.savefig(plot_filename, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
       print(f"✓ Plot saved as: {plot_filename}")

   plt.show()

def generate_summary_statistics(avg_df, daily_dfs):
   """Generate detailed summary statistics."""

   summary_stats = []

   for farm_type in avg_df.columns:
       if farm_type in daily_dfs:
           daily_df = daily_dfs[farm_type]
           avg_values = avg_df[farm_type]

           stats = {
               'Farm_Type': farm_type,
               'Num_Farms': len(daily_df.columns),
               'Days_Analyzed': len(daily_df),
               'Mean_ET': round(avg_values.mean(), 3),
               'Std_ET': round(avg_values.std(), 3),
               'Min_ET': round(avg_values.min(), 3),
               'Max_ET': round(avg_values.max(), 3),
               'Total_ET': round(avg_values.sum(), 3),
               'CV_Percent': round((avg_values.std() / avg_values.mean()) * 100, 2)
           }
           summary_stats.append(stats)

   return pd.DataFrame(summary_stats)

def main():
   """Main analysis function"""

   print("🌾 ET TREND COMPARISON ANALYSIS")
   print("="*60)

   # Test file access
   input_file = test_file_access()
   if input_file is None:
       return

   period_name = "November 2024 - April 2025"

   try:
       print("\n📊 LOADING DATA...")
       farm_tables = load_farm_type_tables(input_file)

       if len(farm_tables) == 0:
           print("❌ No valid tables found")
           return

       print("\n🔄 APPLYING SPLINE INTERPOLATION...")
       daily_dfs = {}

       for farm_type, dekadal_df in farm_tables.items():
           daily_df = interpolate_to_daily(dekadal_df, farm_type)
           if not daily_df.empty:
               daily_dfs[farm_type] = daily_df

       if len(daily_dfs) == 0:
           print("❌ No daily data generated")
           return

       print("\n📈 CALCULATING AVERAGES...")
       avg_df = calculate_daily_averages(daily_dfs)

       print("\n📋 SUMMARY STATISTICS:")
       summary_stats = generate_summary_statistics(avg_df, daily_dfs)
       print(summary_stats.to_string(index=False))

       print("\n📊 CREATING PLOT...")
       plot_et_comparison(avg_df, period_name)

       print("\n💾 SAVING RESULTS...")

       # Save files
       try:
           avg_df.reset_index().to_excel('nov_april_daily_averages.xlsx', index=False)
           summary_stats.to_excel('nov_april_statistics.xlsx', index=False)

           with pd.ExcelWriter('nov_april_all_farms.xlsx') as writer:
               for farm_type, daily_df in daily_dfs.items():
                   daily_df.reset_index().to_excel(writer, sheet_name=f'{farm_type}_Daily', index=False)

           print("✓ Files saved successfully!")

       except Exception as e:
           print(f"⚠ Could not save files: {e}")

       print("\n" + "="*60)
       print("🎉 ANALYSIS COMPLETED!")
       print("="*60)

       # Key findings
       print(f"\n🔍 KEY FINDINGS for {period_name}:")
       mean_ets = {}
       for farm_type in avg_df.columns:
           mean_et = avg_df[farm_type].mean()
           mean_ets[farm_type] = mean_et
           print(f"• {farm_type}: {mean_et:.2f} mm/day")

       if len(mean_ets) > 1:
           highest_et = max(mean_ets, key=mean_ets.get)
           lowest_et = min(mean_ets, key=mean_ets.get)

           print(f"\n🏆 Highest ET: {highest_et} ({mean_ets[highest_et]:.2f} mm/day)")
           print(f"🥉 Lowest ET: {lowest_et} ({mean_ets[lowest_et]:.2f} mm/day)")
           print(f"📏 Difference: {mean_ets[highest_et] - mean_ets[lowest_et]:.2f} mm/day")

   except Exception as e:
       print(f"❌ Error: {str(e)}")
       import traceback
       traceback.print_exc()

# Run the analysis
if __name__ == "__main__":
   main()

# If running in Jupyter/Colab, execute directly
main()
