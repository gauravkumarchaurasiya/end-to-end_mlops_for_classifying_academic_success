import pandas as pd
from typing import List
from pathlib import Path

# Function to infer Pydantic field types based on pandas data types
def infer_pydantic_type(dtype):
    if pd.api.types.is_integer_dtype(dtype):
        return "int"
    elif pd.api.types.is_float_dtype(dtype):
        return "float"
    elif pd.api.types.is_string_dtype(dtype):
        return "str"
    else:
        return "Any"

# Function to generate Pydantic model code
def generate_pydantic_model(class_name: str, columns: List[str], types: List[str]) -> str:
    model_code = f"from pydantic import BaseModel\n\n\nclass {class_name}(BaseModel):\n"
    for column, dtype in zip(columns, types):
        model_code += f'    "{column}": {dtype}\n'
    return model_code

# Read the CSV file to get the column names and types
root_path = Path(__file__).parent
data_path = root_path/"data"/"interim"/"test.csv"
df = pd.read_csv(data_path)
columns = df.columns
types = [infer_pydantic_type(df[col].dtype) for col in columns]

# Generate the Pydantic model code
class_name = "PredictionDataset"
model_code = generate_pydantic_model(class_name, columns, types)

# Write the generated code to a new Python file
output_file = 'data_models.py'
with open(output_file, 'w') as f:
    f.write(model_code)

print(f"Pydantic model code has been written to {output_file}")
