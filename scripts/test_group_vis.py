"""
Test suite for group_vis.py
"""

import os
import sys
import numpy as np
import pandas as pd
import tempfile
import shutil
from pathlib import Path

# Add the main directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from scripts.group_vis import (
    extract_drug_type,
    find_most_variable_segment,
    calculate_wave_character,
    load_trace_data,
    process_experiment_folder
)


def test_extract_drug_type():
    """Test drug type extraction from folder names."""
    print("Testing drug type extraction...")
    
    test_cases = [
        ("01_27_15-18_DOF_Dark1", "DOF"),
        ("17_05_47_QUAN_1", "QUAN"),
        ("some_THAR_experiment", "THAR"),
        ("NORMAL_control", "NORMAL"),
        ("random_folder_name", "UNKNOWN"),
        ("dof_lowercase", "DOF"),
        ("QUAN_something_else", "QUAN")
    ]
    
    for folder_name, expected in test_cases:
        result = extract_drug_type(folder_name)
        assert result == expected, f"Expected {expected}, got {result} for {folder_name}"
        print(f"  ✓ {folder_name} -> {result}")
    
    print("Drug type extraction tests passed!")


def test_wave_character():
    """Test wave character calculation."""
    print("Testing wave character calculation...")
    
    # Test with different signals
    flat_signal = np.ones(100)
    noisy_signal = np.random.normal(1, 0.5, 100)
    wave_signal = np.sin(np.linspace(0, 4*np.pi, 100)) + 1
    
    flat_score = calculate_wave_character(flat_signal)
    noisy_score = calculate_wave_character(noisy_signal)
    wave_score = calculate_wave_character(wave_signal)
    
    print(f"  Flat signal score: {flat_score:.6f}")
    print(f"  Noisy signal score: {noisy_score:.6f}")
    print(f"  Wave signal score: {wave_score:.6f}")
    
    # Wave signal should have higher variance than flat signal
    assert wave_score > flat_score, "Wave signal should have higher variance than flat signal"
    
    # Test with NaN values
    signal_with_nans = np.array([1, 2, np.nan, 4, 5])
    nan_score = calculate_wave_character(signal_with_nans)
    assert nan_score > 0, "Should handle NaN values"
    
    print("Wave character calculation tests passed!")


def test_find_most_variable_segment():
    """Test finding the most variable segment."""
    print("Testing most variable segment detection...")
    
    # Create a signal with a variable section in the middle
    t = np.linspace(0, 10, 1000)
    signal = np.ones_like(t)
    
    # Add variable section from t=4 to t=6
    variable_mask = (t >= 4) & (t <= 6)
    signal[variable_mask] += np.sin(20 * t[variable_mask]) * 0.5
    
    start_idx, end_idx = find_most_variable_segment(signal, t, segment_duration=2.0)
    
    # The detected segment should overlap with the variable region
    detected_start_time = t[start_idx]
    detected_end_time = t[end_idx]
    
    print(f"  Variable region: 4.0 - 6.0 seconds")
    print(f"  Detected region: {detected_start_time:.2f} - {detected_end_time:.2f} seconds")
    
    # Should detect somewhere in the variable region
    assert detected_start_time >= 3.0 and detected_start_time <= 7.0, \
        f"Should detect variable region, got {detected_start_time}"
    
    print("Most variable segment detection tests passed!")


def create_test_data():
    """Create test data structure for testing."""
    print("Creating test data...")
    
    # Create temporary directory structure
    temp_dir = tempfile.mkdtemp()
    plots_dir = os.path.join(temp_dir, "plots")
    os.makedirs(plots_dir)
    
    # Create test experiment folders
    test_experiments = [
        ("01_DOF_test", "DOF"),
        ("02_QUAN_experiment", "QUAN"),
        ("03_normal_control", "NORMAL")
    ]
    
    for folder_name, drug_type in test_experiments:
        exp_dir = os.path.join(plots_dir, folder_name)
        os.makedirs(exp_dir)
        
        # Create synthetic fluorescence data
        n_points = 200
        time = np.linspace(0, 10, n_points)
        
        # Create signal with some variability
        signal = 1.0 + 0.1 * np.sin(2 * np.pi * time) + 0.05 * np.random.normal(size=n_points)
        
        # Add more variability to middle section
        mid_mask = (time >= 3) & (time <= 7)
        signal[mid_mask] += 0.2 * np.sin(10 * np.pi * time[mid_mask])
        
        # Create DataFrame
        df = pd.DataFrame({
            'frame': range(n_points),
            'time_s': time,
            'F_roi1': signal * 100,  # Scale up
            'F0_roi1': 100.0,
            'FF0_roi1': signal,
            'FF0_roi1_smooth': signal
        })
        
        # Save CSV
        csv_path = os.path.join(exp_dir, "fluorescence_traces.csv")
        df.to_csv(csv_path, index=False)
    
    print(f"Test data created in: {plots_dir}")
    return temp_dir, plots_dir


def test_process_experiment_folder():
    """Test processing experiment folder."""
    print("Testing experiment folder processing...")
    
    temp_dir, plots_dir = create_test_data()
    
    try:
        # Process the test data
        drug_data = process_experiment_folder(plots_dir)
        
        print(f"Processed drug data keys: {list(drug_data.keys())}")
        
        # Should have found DOF, QUAN, and NORMAL
        expected_drugs = {'DOF', 'QUAN', 'NORMAL'}
        found_drugs = set(drug_data.keys())
        
        assert expected_drugs.issubset(found_drugs), \
            f"Expected drugs {expected_drugs}, found {found_drugs}"
        
        # Each drug should have at least one experiment
        for drug_type in expected_drugs:
            assert len(drug_data[drug_type]) > 0, f"No experiments found for {drug_type}"
            
            # Check experiment structure
            exp = drug_data[drug_type][0]
            required_keys = {'folder', 'roi', 'time', 'signal', 'wave_score', 'original_start_time'}
            assert required_keys.issubset(exp.keys()), f"Missing keys in experiment data"
            
            print(f"  {drug_type}: {len(drug_data[drug_type])} experiments, "
                  f"wave score: {exp['wave_score']:.4f}")
    
    finally:
        # Clean up
        shutil.rmtree(temp_dir)
    
    print("Experiment folder processing tests passed!")


def run_all_tests():
    """Run all tests."""
    print("=== Running Group Visualization Tests ===\n")
    
    try:
        test_extract_drug_type()
        print()
        
        test_wave_character()
        print()
        
        test_find_most_variable_segment()
        print()
        
        test_process_experiment_folder()
        print()
        
        print("=== All Tests Passed! ===")
        return True
        
    except Exception as e:
        print(f"\n=== Test Failed: {e} ===")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)