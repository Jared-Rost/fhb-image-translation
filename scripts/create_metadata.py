#!/usr/bin/env python3
"""
Create metadata.jsonl file for ControlNet training from FHB wheat head dataset.

This script reads the Excel file containing severity data and creates a metadata.jsonl
file with image paths, conditioning image paths, and descriptive text prompts based
on infection severity.

Usage:
    python create_metadata.py --excel_file path/to/data.xlsx --images_dir path/to/images --conditioning_dir path/to/masks --output_file metadata.jsonl
"""

import argparse
import json
import os
import pandas as pd
from pathlib import Path


def severity_to_prompt(severity):
    """
    Convert severity percentage to descriptive text prompt.
    
    Args:
        severity (float): Severity percentage (0-100)
    
    Returns:
        str: Descriptive prompt for the image
    """
    # Standardized format: "AINBN3tpDb, [category] Fusarium Head Blight, X%"
    if severity == 0:
        return "AINBN3tpDb, no Fusarium Head Blight, 0.0%"
    elif 0 < severity <= 10:
        return f"AINBN3tpDb, minimal Fusarium Head Blight, {severity:.1f}%"
    elif 10 < severity <= 30:
        return f"AINBN3tpDb, mild Fusarium Head Blight, {severity:.1f}%"
    elif 30 < severity <= 60:
        return f"AINBN3tpDb, moderate Fusarium Head Blight, {severity:.1f}%"
    elif 60 < severity <= 100:
        return f"AINBN3tpDb, severe Fusarium Head Blight, {severity:.1f}%"
    else:
        return f"AINBN3tpDb, unknown Fusarium Head Blight, {severity:.1f}%"


def create_image_filename(isolate, cultivar, replication, head, dai):
    """
    Create image filename from metadata components.
    
    Args:
        isolate (str): Isolate identifier
        cultivar (str): Cultivar identifier  
        replication (str): Replication identifier
        head (str): Head identifier
        dai (str): Days after infection identifier
    
    Returns:
        str: Formatted filename
    """
    return f"{isolate}_{cultivar}_{replication}_{head}_{dai}DAI.png"


def create_conditioning_filename(image_filename):
    """
    Create conditioning image filename from original image filename.
    
    Args:
        image_filename (str): Original image filename
    
    Returns:
        str: Conditioning image filename with _segmentation suffix
    """
    base_name = Path(image_filename).stem
    return f"{base_name}_segmentation.png"


def main():
    parser = argparse.ArgumentParser(description="Create metadata.jsonl for ControlNet training")
    parser.add_argument(
        "--excel_file",
        type=str,
        required=True,
        help="Path to the Excel file containing severity data"
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        required=True,
        help="Directory containing the original wheat head images"
    )
    parser.add_argument(
        "--conditioning_dir", 
        type=str,
        required=True,
        help="Directory containing the conditioning/segmentation images"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="metadata.jsonl",
        help="Output metadata.jsonl file path (default: metadata.jsonl)"
    )
    parser.add_argument(
        "--severity_column",
        type=str,
        default="Severity",
        help="Column name for severity values (default: Severity)"
    )
    
    args = parser.parse_args()
    
    # Read Excel file
    print(f"Reading Excel file: {args.excel_file}")
    try:
        df = pd.read_excel(args.excel_file)
        print(f"Loaded {len(df)} rows from Excel file")
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return
    
    # Print column names for debugging
    print(f"Available columns: {list(df.columns)}")
    
    # Check if required columns exist
    required_columns = ['Isolate', 'Cultivar', 'Replication', 'Head', 'DAI', args.severity_column]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Error: Missing required columns: {missing_columns}")
        return
    
    # Create metadata entries
    metadata_entries = []
    skipped_count = 0
    
    for idx, row in df.iterrows():
        try:
            # Create image filename from metadata
            image_filename = create_image_filename(
                row['Isolate'], 
                row['Cultivar'], 
                row['Replication'], 
                row['Head'], 
                row['DAI']
            )
            
            # Create conditioning image filename
            conditioning_filename = create_conditioning_filename(image_filename)
            
            # Full paths
            image_path = os.path.join(args.images_dir, image_filename)
            conditioning_path = os.path.join(args.conditioning_dir, conditioning_filename)
            
            # Check if files exist
            if not os.path.exists(image_path):
                print(f"Warning: Image not found: {image_path}")
                skipped_count += 1
                continue
                
            if not os.path.exists(conditioning_path):
                print(f"Warning: Conditioning image not found: {conditioning_path}")
                skipped_count += 1
                continue
            
            # Get severity and create standardized prompt
            severity = float(row[args.severity_column])
            prompt = severity_to_prompt(severity)
            
            # Create metadata entry with ./ prefix for relative paths
            rel_image_path = os.path.relpath(image_path, start=os.path.dirname(args.output_file))
            rel_conditioning_path = os.path.relpath(conditioning_path, start=os.path.dirname(args.output_file))

            # Add ./ prefix to make paths explicit
            metadata_entry = {
                "file_name": f"./{rel_image_path}",
                "conditioning_image": f"./{rel_conditioning_path}",
                "text": prompt
            }
            
            metadata_entries.append(metadata_entry)
            
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            skipped_count += 1
            continue
    
    # Write metadata.jsonl file
    print(f"Writing {len(metadata_entries)} entries to {args.output_file}")
    
    with open(args.output_file, 'w') as f:
        for entry in metadata_entries:
            f.write(json.dumps(entry) + '\n')
    
    print(f"Successfully created metadata.jsonl with {len(metadata_entries)} entries")
    if skipped_count > 0:
        print(f"Skipped {skipped_count} entries due to missing files or errors")
    
    # Print some example entries
    print("\nExample entries:")
    for i, entry in enumerate(metadata_entries[:3]):
        print(f"  {i+1}: {entry}")


if __name__ == "__main__":
    main()