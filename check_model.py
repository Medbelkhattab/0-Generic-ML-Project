import sys
import os
from src.utils import load_object

try:
    # Load the model
    model_path = os.path.join("artifacts", "model.pkl")
    model = load_object(file_path=model_path)

    # Print the Type (The DNA)
    print("\n" + "="*50)
    print("THE WINNER IS:")
    print(type(model))
    print("="*50 + "\n")

except Exception as e:
    print(e)