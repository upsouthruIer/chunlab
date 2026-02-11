from ase.io import read
from ase.visualize import view
import sys

# Check if a filename was provided as a command-line argument
if len(sys.argv) < 2:
    print("Usage: python view.py <filename_of_structure_file>")
    sys.exit(1)

# The first argument after the script name is the structure file (e.g., POSCAR)
filename = sys.argv[1]

try:
    # Read the structure from the provided file (e.g., POSCAR)
    adslab = read(filename)
    
    # Visualize the structure
    view(adslab)
except FileNotFoundError:
    print(f"Error: File '{filename}' not found.")
except Exception as e:
    print(f"An error occurred while reading the file: {e}")