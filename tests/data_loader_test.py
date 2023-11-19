import os
import yaml
import pytest
import pandas as pd
from src.loader.data_loader import data_loader

def test_data_loader_csv():
    # Create a temporary CSV file
    file_path = "test_data.csv"
    with open(file_path, "w") as f:
        f.write("id,name,age\n1,Alice,25\n2,Bob,30\n")

    # Call the data_loader() function with the CSV file
    data = data_loader(file_path)

    # Check that the loaded data matches the expected data
    expected_data = pd.DataFrame({"id": [1, 2], "name": ["Alice", "Bob"], "age": [25, 30]})
    pd.testing.assert_frame_equal(data, expected_data)

    # Clean up the temporary file
    os.remove(file_path)

def test_data_loader_xlsx():
    # Create a temporary XLSX file
    file_path = "test_data.xlsx"
    data = pd.DataFrame({"id": [1, 2], "name": ["Alice", "Bob"], "age": [25, 30]})
    data.to_excel(file_path, index=False)

    # Call the data_loader() function with the XLSX file
    data = data_loader(file_path)

    # Check that the loaded data matches the expected data
    expected_data = pd.DataFrame({"id": [1, 2], "name": ["Alice", "Bob"], "age": [25, 30]})
    pd.testing.assert_frame_equal(data, expected_data)

    # Clean up the temporary file
    os.remove(file_path)

def test_data_loader_json():
    # Create a temporary JSON file
    file_path = "test_data.json"
    data = pd.DataFrame({"id": [1, 2], "name": ["Alice", "Bob"], "age": [25, 30]})
    data.to_json(file_path, orient="records")

    # Call the data_loader() function with the JSON file
    data = data_loader(file_path)

    # Check that the loaded data matches the expected data
    expected_data = pd.DataFrame({"id": [1, 2], "name": ["Alice", "Bob"], "age": [25, 30]})
    pd.testing.assert_frame_equal(data, expected_data)

    # Clean up the temporary file
    os.remove(file_path)

def test_data_loader_txt():
    # Create a temporary TSV file
    file_path = "test_data.txt"
    with open(file_path, "w") as f:
        f.write("id\tname\tage\n1\tAlice\t25\n2\tBob\t30\n")

    # Call the data_loader() function with the TSV file
    data = data_loader(file_path)

    # Check that the loaded data matches the expected data
    expected_data = pd.DataFrame({"id": [1, 2], "name": ["Alice", "Bob"], "age": [25, 30]})
    pd.testing.assert_frame_equal(data, expected_data)

    # Clean up the temporary file
    os.remove(file_path)

def test_data_loader_error_handling():
    # Call the data_loader() function with a non-existent file
    with pytest.raises(Exception) as e:
        data_loader("nonexistent_file.csv")
    assert str(e.value) == f"Error: File path nonexistent_file.csv does not exist"

    # Call the data_loader() function with an unsupported file type
    
    # Create a temporary YAML file
    file_path = "test_data.yaml"
    data = {"id": [1, 2], "name": ["Alice", "Bob"], "age": [25, 30]}
    with open(file_path, 'w') as file:
        yaml.dump(data, file)

    with pytest.raises(Exception) as e:
        data_loader(file_path)
    assert str(e.value) == f"Error: Unsupported file type .yaml"

    # Clean up the temporary file
    os.remove(file_path)