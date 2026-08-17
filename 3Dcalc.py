#!/usr/bin/env python3
"""
Fish Tracking Distance Calculator

This script processes tracking files in a folder, matches main tracking files 
with their reflection counterparts (files ending in '_r'), and calculates total 
distance moved by combining horizontal/lateral movement from the main file with 
vertical movement from the reflection file.

Usage: python 3Dcalc.py [folder_path]
If no folder path is provided, it processes the current directory.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import re

def calculate_distance_2d(x_coords, y_coords):
    """Calculate total distance traveled given x,y coordinates"""
    if len(x_coords) < 2 or len(y_coords) < 2:
        return 0.0
    
    # Calculate differences between consecutive points
    dx = np.diff(x_coords)
    dy = np.diff(y_coords)
    
    # Calculate euclidean distance for each step
    distances = np.sqrt(dx**2 + dy**2)
    
    # Sum all distances
    total_distance = np.sum(distances)
    
    return total_distance

def calculate_distance_1d(coords):
    """Calculate total distance for 1D movement (e.g., vertical only)"""
    if len(coords) < 2:
        return 0.0
    
    # Calculate absolute differences between consecutive points
    distances = np.abs(np.diff(coords))
    
    # Sum all distances
    total_distance = np.sum(distances)
    
    return total_distance

def calculate_distance_up_to_frame(x_coords, y_coords, frame_nums, target_frame):
    """Calculate total distance traveled up to a specific frame number"""
    if len(x_coords) < 2 or len(y_coords) < 2:
        return 0.0, 0
    
    # Find all points up to and including the target frame
    frame_mask = frame_nums <= target_frame
    if not np.any(frame_mask):
        return 0.0, 0
    
    x_subset = x_coords[frame_mask]
    y_subset = y_coords[frame_mask]
    
    if len(x_subset) < 2:
        return 0.0, len(x_subset)
    
    # Calculate differences between consecutive points
    dx = np.diff(x_subset)
    dy = np.diff(y_subset)
    
    # Calculate euclidean distance for each step
    distances = np.sqrt(dx**2 + dy**2)
    
    # Sum all distances
    total_distance = np.sum(distances)
    
    return total_distance, len(x_subset)

def calculate_distance_1d_up_to_frame(coords, frame_nums, target_frame):
    """Calculate total distance for 1D movement up to a specific frame"""
    if len(coords) < 2:
        return 0.0, 0
    
    # Find all points up to and including the target frame
    frame_mask = frame_nums <= target_frame
    if not np.any(frame_mask):
        return 0.0, 0
    
    coords_subset = coords[frame_mask]
    
    if len(coords_subset) < 2:
        return 0.0, len(coords_subset)
    
    # Calculate absolute differences between consecutive points
    distances = np.abs(np.diff(coords_subset))
    
    # Sum all distances
    total_distance = np.sum(distances)
    
    return total_distance, len(coords_subset)

def read_tracking_file(filepath):
    """Read tracking file and return coordinates for included data points only"""
    try:
        # Read the tab-delimited file
        df = pd.read_csv(filepath, sep='\t')
        
        # Remove any accidental header rows that might be mixed in with data
        # These often happen when files are concatenated
        header_indicators = ['TadpoleID', 'TrialID', 'FrameNo', 'Xcentroid', 'Ycentroid']
        
        # Remove rows where any coordinate column contains header text
        if 'Xcentroid' in df.columns:
            # Remove rows where Xcentroid contains text that looks like a header
            mask = ~df['Xcentroid'].astype(str).isin(header_indicators)
            df = df[mask]
        
        if 'Ycentroid' in df.columns:
            # Remove rows where Ycentroid contains text that looks like a header  
            mask = ~df['Ycentroid'].astype(str).isin(header_indicators)
            df = df[mask]
        
        # Also remove any rows where coordinates are exactly the column names
        original_rows = len(df)
        df = df[df['Xcentroid'].astype(str) != 'Xcentroid']
        df = df[df['Ycentroid'].astype(str) != 'Ycentroid'] 
        
        removed_headers = original_rows - len(df)
        if removed_headers > 0:
            print(f"  Removed {removed_headers} accidental header rows")
        
        # Check for required columns
        missing_cols = []
        if 'Xcentroid' not in df.columns:
            missing_cols.append('Xcentroid')
        if 'Ycentroid' not in df.columns:
            missing_cols.append('Ycentroid')
        if 'FrameNo' not in df.columns:
            missing_cols.append('FrameNo')
        
        if missing_cols:
            return None, None, None, f"Missing required columns: {', '.join(missing_cols)}"
        
        # Filter to only include rows where InclusionStatus is "Included"
        if 'InculsionStatus' in df.columns:
            included_df = df[df['InculsionStatus'] == 'Included'].copy()
        elif 'InclusionStatus' in df.columns:  # Handle potential typo in column name
            included_df = df[df['InclusionStatus'] == 'Included'].copy()
        else:
            print(f"Warning: No InclusionStatus column found in {filepath}, using all data")
            included_df = df.copy()
        
        if included_df.empty:
            return None, None, None, "No data points with InclusionStatus='Included' found"
        
        # Convert to numeric, coercing errors to NaN (this handles any remaining text)
        x_coords = pd.to_numeric(included_df['Xcentroid'], errors='coerce')
        y_coords = pd.to_numeric(included_df['Ycentroid'], errors='coerce') 
        frame_nums = pd.to_numeric(included_df['FrameNo'], errors='coerce')
        
        # Convert to numpy arrays
        x_coords = x_coords.values
        y_coords = y_coords.values
        frame_nums = frame_nums.values
        
        # Remove any NaN values (including those created by coercion of remaining text)
        valid_indices = ~(np.isnan(x_coords) | np.isnan(y_coords) | np.isnan(frame_nums))
        x_coords = x_coords[valid_indices]
        y_coords = y_coords[valid_indices]
        frame_nums = frame_nums[valid_indices]
        
        # Check if we have valid data
        if len(x_coords) == 0:
            return None, None, None, "No valid numeric coordinate data found after cleaning"
        
        # Check for data quality issues
        original_count = len(included_df)
        valid_count = len(x_coords)
        invalid_count = original_count - valid_count
        
        if invalid_count > 0:
            print(f"  Note: {invalid_count} rows with non-numeric data were excluded (likely header rows or invalid entries)")
        
        print(f"  Found {valid_count} valid included data points")
        return x_coords, y_coords, frame_nums, None
        
    except FileNotFoundError:
        return None, None, None, f"File not found: {filepath}"
    except pd.errors.EmptyDataError:
        return None, None, None, "File is empty"
    except pd.errors.ParserError as e:
        return None, None, None, f"File parsing error: {str(e)}"
    except Exception as e:
        return None, None, None, f"Unexpected error: {str(e)}"

def parse_filename(filename):
    """Parse filename to extract key components for matching"""
    # Remove extension
    name = Path(filename).stem
    
    # Remove _r suffix if present
    if name.endswith('_r'):
        name = name[:-2]
    
    # Split by underscores and extract key components
    parts = name.split('_')
    
    # Look for patterns in the filename
    tadpole_id = None
    trial_id = None
    position = None
    behavior_type = None
    
    for i, part in enumerate(parts):
        # Find tadpole ID (C followed by numbers)
        if part.startswith('C') and len(part) > 1:
            if i + 1 < len(parts) and parts[i+1].isdigit():
                tadpole_id = f"{part}_{parts[i+1]}"
        
        # Find trial number
        if part == 'Trial' and i + 1 < len(parts):
            trial_id = parts[i+1]
        
        # Find behavior type (ACT or EX)
        if part in ['ACT', 'EX']:
            behavior_type = part
        
        # Get position (last meaningful part before _r)
        if part in ['L', 'R', 'LM', 'RM']:
            position = part
    
    return {
        'tadpole_id': tadpole_id,
        'trial_id': trial_id, 
        'position': position,
        'behavior_type': behavior_type,
        'full_name': filename
    }

def find_file_pairs(folder_path):
    """Find matching pairs of tracking files based on content rather than exact names"""
    folder = Path(folder_path)
    
    # Look for common file extensions
    extensions = ["*.txt", "*.csv", "*.tsv"]
    all_files = []
    for ext in extensions:
        all_files.extend(list(folder.glob(ext)))
    
    print(f"Found {len(all_files)} total files in folder:")
    for file in sorted(all_files)[:10]:  # Show first 10 files
        print(f"  {file.name}")
    if len(all_files) > 10:
        print(f"  ... and {len(all_files) - 10} more files")
    
    if not all_files:
        print("No .txt, .csv, or .tsv files found in the folder!")
        return []
    
    # Separate main files and reflection files
    main_files = []
    reflection_files = []
    
    for file in all_files:
        if file.stem.endswith('_r'):
            reflection_files.append(file)
        else:
            main_files.append(file)
    
    print(f"\nFound {len(main_files)} main files and {len(reflection_files)} reflection files")
    
    # Parse all files
    main_parsed = [parse_filename(f.name) for f in main_files]
    refl_parsed = [parse_filename(f.name) for f in reflection_files]
    
    # Match files based on tadpole_id, trial_id, and position
    pairs = []
    matched_refl = set()
    
    for i, main_info in enumerate(main_parsed):
        main_file = main_files[i]
        
        # Look for matching reflection file
        for j, refl_info in enumerate(refl_parsed):
            if j in matched_refl:
                continue
                
            refl_file = reflection_files[j]
            
            # Check if they match on the key criteria
            if (main_info['tadpole_id'] == refl_info['tadpole_id'] and 
                main_info['trial_id'] == refl_info['trial_id'] and
                main_info['position'] == refl_info['position']):
                
                pairs.append((main_file, refl_file))
                matched_refl.add(j)
                
                print(f"Matched: {main_file.name}")
                print(f"    with: {refl_file.name}")
                print(f"    (Tadpole: {main_info['tadpole_id']}, Trial: {main_info['trial_id']}, Position: {main_info['position']}, Behavior: {main_info['behavior_type']})")
                break
    
    if not pairs:
        print("\nNo matching pairs found based on content matching.")
        print("Showing parsed file info for debugging:")
        print(f"\nMain files:")
        for i, info in enumerate(main_parsed[:3]):
            print(f"  {main_files[i].name} -> Tadpole: {info['tadpole_id']}, Trial: {info['trial_id']}, Position: {info['position']}, Behavior: {info['behavior_type']}")
        print(f"\nReflection files:")
        for i, info in enumerate(refl_parsed[:3]):
            print(f"  {reflection_files[i].name} -> Tadpole: {info['tadpole_id']}, Trial: {info['trial_id']}, Position: {info['position']}, Behavior: {info['behavior_type']}")
        
        # Check a sample file to see what might be wrong
        if main_files:
            print(f"\nTrying to read sample main file for diagnostics...")
            sample_result = read_tracking_file(main_files[0])
            if len(sample_result) == 4 and sample_result[3]:
                print(f"Sample file error: {sample_result[3]}")
        
        if reflection_files:
            print(f"\nTrying to read sample reflection file for diagnostics...")
            sample_result = read_tracking_file(reflection_files[0])
            if len(sample_result) == 4 and sample_result[3]:
                print(f"Sample file error: {sample_result[3]}")
    else:
        print(f"\nFound {len(pairs)} matching pairs!")
    
    return pairs

def process_tracking_files(folder_path):
    """Main function to process all tracking files in a folder"""
    folder_path = Path(folder_path)
    
    if not folder_path.exists():
        print(f"Error: Folder {folder_path} does not exist")
        return
    
    print(f"Processing folder: {folder_path}")
    print("-" * 50)
    
    # Find file pairs
    pairs = find_file_pairs(folder_path)
    
    if not pairs:
        print("No matching file pairs found!")
        print("Looking for files with pattern: filename.txt and filename_r.txt")
        return
    
    results = []
    problem_files = []
    
    for main_file, reflection_file in pairs:
        print(f"\nProcessing: {main_file.name}")
        
        # Parse filename to get behavior type
        main_info = parse_filename(main_file.name)
        behavior_type = main_info['behavior_type']
        
        # Determine target frame based on behavior type
        if behavior_type == 'ACT':
            target_frame = 4500
        elif behavior_type == 'EX':
            target_frame = 6000
        else:
            print(f"Warning: Unknown behavior type '{behavior_type}' for {main_file.name}")
            target_frame = None
        
        # Read main tracking file (horizontal movement) - only included data
        main_x, main_y, main_frames, main_error = read_tracking_file(main_file)
        
        # Read reflection file (vertical movement) - only included data  
        refl_x, refl_y, refl_frames, refl_error = read_tracking_file(reflection_file)
        
        # Check for errors in either file
        if main_x is None or refl_y is None:
            error_details = []
            if main_error:
                error_details.append(f"Main file: {main_error}")
            if refl_error:
                error_details.append(f"Reflection file: {refl_error}")
            
            combined_error = "; ".join(error_details) if error_details else "Unknown read error"
            
            print(f"Skipping {main_file.name} due to read errors")
            problem_files.append({
                'main_file': main_file.name,
                'reflection_file': reflection_file.name,
                'issue': combined_error
            })
            continue
        
        # Calculate full distances (entire recording)
        horizontal_distance_full = calculate_distance_2d(main_x, main_y)
        vertical_distance_full = calculate_distance_1d(refl_y)
        total_distance_full = np.sqrt(horizontal_distance_full**2 + vertical_distance_full**2)
        
        # Calculate distances up to target frame (if specified)
        if target_frame is not None:
            horizontal_distance_target, main_points_used = calculate_distance_up_to_frame(main_x, main_y, main_frames, target_frame)
            vertical_distance_target, refl_points_used = calculate_distance_1d_up_to_frame(refl_y, refl_frames, target_frame)
            total_distance_target = np.sqrt(horizontal_distance_target**2 + vertical_distance_target**2)
        else:
            horizontal_distance_target = None
            vertical_distance_target = None
            total_distance_target = None
            main_points_used = 0
            refl_points_used = 0
        
        # Extract sample info from filename
        tadpole_id = main_info['tadpole_id'] or "Unknown"
        trial_id = main_info['trial_id'] or "Unknown"
        
        result = {
            'filename': main_file.stem,
            'tadpole_id': tadpole_id,
            'trial_id': trial_id,
            'behavior_type': behavior_type,
            'target_frame': target_frame,
            
            # Full recording distances
            'horizontal_distance_full': horizontal_distance_full,
            'vertical_distance_full': vertical_distance_full,
            'total_distance_full': total_distance_full,
            
            # Target frame distances
            'horizontal_distance_target': horizontal_distance_target,
            'vertical_distance_target': vertical_distance_target,
            'total_distance_target': total_distance_target,
            
            # File info
            'main_file': main_file.name,
            'reflection_file': reflection_file.name,
            'included_points_main_full': len(main_x),
            'included_points_reflection_full': len(refl_y),
            'included_points_main_target': main_points_used,
            'included_points_reflection_target': refl_points_used
        }
        
        results.append(result)
        
        print(f"  TadpoleID: {tadpole_id}, Trial: {trial_id}, Behavior: {behavior_type}")
        print(f"  Included points - Main: {len(main_x)}, Reflection: {len(refl_y)}")
        print(f"  FULL RECORDING:")
        print(f"    Horizontal distance: {horizontal_distance_full:.2f} pixels")
        print(f"    Vertical distance: {vertical_distance_full:.2f} pixels")
        print(f"    Total distance: {total_distance_full:.2f} pixels")
        
        if target_frame is not None:
            print(f"  UP TO FRAME {target_frame} ({behavior_type}):")
            print(f"    Points used - Main: {main_points_used}, Reflection: {refl_points_used}")
            print(f"    Horizontal distance: {horizontal_distance_target:.2f} pixels")
            print(f"    Vertical distance: {vertical_distance_target:.2f} pixels")
            print(f"    Total distance: {total_distance_target:.2f} pixels")
    
    # Save results to file
    if results:
        output_file = folder_path / "tracking_distances_summary.txt"
        
        with open(output_file, 'w') as f:
            f.write("Tadpole Tracking Distance Analysis\n")
            f.write("=" * 50 + "\n")
            f.write("Note: Only data points with InclusionStatus='Included' are analyzed\n")
            f.write("ACT behavior: distances calculated up to frame 4500\n")
            f.write("EX behavior: distances calculated up to frame 6000\n\n")
            
            for result in results:
                f.write(f"Sample: {result['filename']}\n")
                f.write(f"TadpoleID: {result['tadpole_id']}\n")
                f.write(f"Trial: {result['trial_id']}\n")
                f.write(f"Behavior Type: {result['behavior_type']}\n")
                f.write(f"Main file: {result['main_file']} ({result['included_points_main_full']} total included points)\n")
                f.write(f"Reflection file: {result['reflection_file']} ({result['included_points_reflection_full']} total included points)\n")
                
                f.write(f"\nFULL RECORDING DISTANCES:\n")
                f.write(f"  Horizontal distance: {result['horizontal_distance_full']:.3f} pixels\n")
                f.write(f"  Vertical distance: {result['vertical_distance_full']:.3f} pixels\n")
                f.write(f"  Total distance: {result['total_distance_full']:.3f} pixels\n")
                
                if result['target_frame'] is not None:
                    f.write(f"\nDISTANCES UP TO FRAME {result['target_frame']} ({result['behavior_type']}):\n")
                    f.write(f"  Points used - Main: {result['included_points_main_target']}, Reflection: {result['included_points_reflection_target']}\n")
                    f.write(f"  Horizontal distance: {result['horizontal_distance_target']:.3f} pixels\n")
                    f.write(f"  Vertical distance: {result['vertical_distance_target']:.3f} pixels\n")
                    f.write(f"  Total distance: {result['total_distance_target']:.3f} pixels\n")
                
                f.write("-" * 50 + "\n")
        
        # Also save as CSV for easier analysis
        csv_file = folder_path / "tracking_distances_summary.csv"
        df_results = pd.DataFrame(results)
        df_results.to_csv(csv_file, index=False)
        
        print(f"\n" + "=" * 50)
        print(f"Results saved to:")
        print(f"  - {output_file}")
        print(f"  - {csv_file}")
        print(f"\nProcessed {len(results)} file pairs successfully!")
        
        # Report any problem files
        if problem_files:
            print(f"\n" + "!" * 50)
            print(f"PROBLEM FILES REPORT:")
            print(f"Found {len(problem_files)} file pairs with issues:")
            for problem in problem_files:
                print(f"  - {problem['main_file']} & {problem['reflection_file']}")
                print(f"    Issue: {problem['issue']}")
            print(f"!" * 50)
        
        # Print summary statistics
        total_samples = len(results)
        
        # Group results by behavior type
        act_results = [r for r in results if r['behavior_type'] == 'ACT']
        ex_results = [r for r in results if r['behavior_type'] == 'EX']
        
        print(f"\nSummary Statistics:")
        print(f"  Total samples: {total_samples}")
        print(f"    ACT samples: {len(act_results)}")
        print(f"    EX samples: {len(ex_results)}")
        
        # Full recording stats
        if results:
            avg_horizontal_full = np.mean([r['horizontal_distance_full'] for r in results])
            avg_vertical_full = np.mean([r['vertical_distance_full'] for r in results])
            avg_total_full = np.mean([r['total_distance_full'] for r in results])
            
            print(f"\n  FULL RECORDING AVERAGES (All samples):")
            print(f"    Average horizontal distance: {avg_horizontal_full:.2f} pixels")
            print(f"    Average vertical distance: {avg_vertical_full:.2f} pixels")
            print(f"    Average total distance: {avg_total_full:.2f} pixels")
        
        # ACT behavior stats
        if act_results:
            act_full_h = np.mean([r['horizontal_distance_full'] for r in act_results])
            act_full_v = np.mean([r['vertical_distance_full'] for r in act_results])
            act_full_t = np.mean([r['total_distance_full'] for r in act_results])
            
            print(f"\n  ACT BEHAVIOR AVERAGES ({len(act_results)} samples):")
            print(f"    Full recording - H: {act_full_h:.2f}, V: {act_full_v:.2f}, Total: {act_full_t:.2f} pixels")
            
            act_target_results = [r for r in act_results if r['horizontal_distance_target'] is not None]
            if act_target_results:
                act_target_h = np.mean([r['horizontal_distance_target'] for r in act_target_results])
                act_target_v = np.mean([r['vertical_distance_target'] for r in act_target_results])
                act_target_t = np.mean([r['total_distance_target'] for r in act_target_results])
                print(f"    Up to frame 4500 - H: {act_target_h:.2f}, V: {act_target_v:.2f}, Total: {act_target_t:.2f} pixels")
        
        # EX behavior stats
        if ex_results:
            ex_full_h = np.mean([r['horizontal_distance_full'] for r in ex_results])
            ex_full_v = np.mean([r['vertical_distance_full'] for r in ex_results])
            ex_full_t = np.mean([r['total_distance_full'] for r in ex_results])
            
            print(f"\n  EX BEHAVIOR AVERAGES ({len(ex_results)} samples):")
            print(f"    Full recording - H: {ex_full_h:.2f}, V: {ex_full_v:.2f}, Total: {ex_full_t:.2f} pixels")
            
            ex_target_results = [r for r in ex_results if r['horizontal_distance_target'] is not None]
            if ex_target_results:
                ex_target_h = np.mean([r['horizontal_distance_target'] for r in ex_target_results])
                ex_target_v = np.mean([r['vertical_distance_target'] for r in ex_target_results])
                ex_target_t = np.mean([r['total_distance_target'] for r in ex_target_results])
                print(f"    Up to frame 6000 - H: {ex_target_h:.2f}, V: {ex_target_v:.2f}, Total: {ex_target_t:.2f} pixels")
    
    else:
        print("No results to save.")
        
        # Still report problem files even if no successful results
        if problem_files:
            print(f"\n" + "!" * 50)
            print(f"PROBLEM FILES REPORT:")
            print(f"Found {len(problem_files)} file pairs with issues:")
            for problem in problem_files:
                print(f"  - {problem['main_file']} & {problem['reflection_file']}")
                print(f"    Issue: {problem['issue']}")
            print(f"!" * 50)

if __name__ == "__main__":
    # Get folder path from command line argument or use specified directory
    if len(sys.argv) > 1:
        folder_path = sys.argv[1]
    else:
        # CHANGE THIS PATH to your tracking files folder
        folder_path = r"C:\path\to\your\tracking\files"  # Windows example
        # folder_path = "/path/to/your/tracking/files"    # Mac/Linux example
        # folder_path = "."                               # Current directory
    
    # Check if required packages are available
    try:
        import numpy as np
        import pandas as pd
    except ImportError as e:
        print(f"Error: Required package not found: {e}")
        print("Please install required packages:")
        print("pip install numpy pandas")
        sys.exit(1)
    
    process_tracking_files(folder_path)