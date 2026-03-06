#!/usr/bin/env python3
"""
Demo script showing how to use group_vis.py
"""

import subprocess
import sys
import os

def run_demo():
    """Run demonstration of group_vis.py functionality."""
    
    print("=== Group Visualization Demo ===\n")
    
    # Change to the script directory
    script_dir = "/Users/sanskriti/Documents/GitHub/miniscope_fullstack"
    os.chdir(script_dir)
    
    demos = [
        {
            "name": "Basic Usage",
            "description": "Run with default settings",
            "command": ["python", "scripts/group_vis.py"]
        },
        {
            "name": "Custom Segment Duration", 
            "description": "Extract 6-second segments instead of 4-second",
            "command": ["python", "scripts/group_vis.py", "--segment-duration", "6.0"]
        },
        {
            "name": "CSV Only",
            "description": "Generate only CSV data without interactive plot",
            "command": ["python", "scripts/group_vis.py", "--no-plot", "--csv-output", "demo_data.csv"]
        },
        {
            "name": "Custom Output",
            "description": "Specify custom output filenames",
            "command": ["python", "scripts/group_vis.py", "--output", "my_plot.html", "--csv-output", "my_data.csv"]
        }
    ]
    
    for i, demo in enumerate(demos, 1):
        print(f"{i}. {demo['name']}")
        print(f"   Description: {demo['description']}")
        print(f"   Command: {' '.join(demo['command'])}")
        print()
        
        # Ask user if they want to run this demo
        response = input(f"Run demo {i}? [y/N]: ").lower().strip()
        if response in ['y', 'yes']:
            print(f"\nRunning: {' '.join(demo['command'])}")
            print("-" * 50)
            
            try:
                result = subprocess.run(demo['command'], 
                                      capture_output=True, 
                                      text=True, 
                                      timeout=30)
                
                print("STDOUT:")
                print(result.stdout)
                
                if result.stderr:
                    print("\nSTDERR:")
                    print(result.stderr)
                
                if result.returncode != 0:
                    print(f"\nCommand failed with return code: {result.returncode}")
                else:
                    print("\nCommand completed successfully!")
                    
            except subprocess.TimeoutExpired:
                print("Command timed out after 30 seconds")
            except Exception as e:
                print(f"Error running command: {e}")
            
            print("-" * 50)
            print()
        
        print()
    
    print("Demo complete!")
    print("\nGenerated files:")
    for filename in ["group_fluorescence_traces.html", "group_traces_data.csv", 
                     "demo_data.csv", "my_plot.html", "my_data.csv"]:
        if os.path.exists(filename):
            print(f"  ✓ {filename}")


def show_help():
    """Show available command line options."""
    print("=== Group Visualization Help ===\n")
    
    try:
        result = subprocess.run(["python", "scripts/group_vis.py", "--help"], 
                               capture_output=True, text=True, timeout=10)
        print(result.stdout)
    except Exception as e:
        print(f"Error getting help: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        show_help()
    else:
        run_demo()