import os

print("Setting up the full project structure (folders and files)...")

# 1. Define all directories
directories = [
    "data",
    "notebooks",
    "utils",
    "code",
    "visuals",
    "visuals/1",
    "visuals/2",
    "visuals/3",
    "visuals/4",
    "visuals/5"
]

# 2. Define all empty files to create
# This includes the .py files and the .ipynb notebooks
files_to_create = [
    "requirements.txt",
    "main.py",
    os.path.join("utils", "__init__.py"),
    os.path.join("utils", "preprocessing.py"),
    os.path.join("utils", "model_helpers.py"),
    os.path.join("code", "code.py"),
    # We add .gitkeep files to ensure empty folders are tracked by Git
    os.path.join("data", ".gitkeep"),
    os.path.join("visuals", "1", ".gitkeep"),
    os.path.join("visuals", "2", ".gitkeep"),
    os.path.join("visuals", "3", ".gitkeep"),
    os.path.join("visuals", "4", ".gitkeep"),
    os.path.join("visuals", "5", ".gitkeep")
]

# 3. Define the notebook files with minimal valid JSON
notebooks = {
    os.path.join("notebooks", "001_data_preprocessing.ipynb"): '{\n "cells": [],\n "metadata": {},\n "nbformat": 4,\n "nbformat_minor": 5\n}',
    os.path.join("notebooks", "002_regression_model.ipynb"): '{\n "cells": [],\n "metadata": {},\n "nbformat": 4,\n "nbformat_minor": 5\n}',
    os.path.join("notebooks", "003_classification_model.ipynb"): '{\n "cells": [],\n "metadata": {},\n "nbformat": 4,\n "nbformat_minor": 5\n}',
    os.path.join("notebooks", "004_model_comparison.ipynb"): '{\n "cells": [],\n "metadata": {},\n "nbformat": 4,\n "nbformat_minor": 5\n}',
    os.path.join("notebooks", "005_advanced_predictive_probability.ipynb"): '{\n "cells": [],\n "metadata": {},\n "nbformat": 4,\n "nbformat_minor": 5\n}'
}

# 4. Create directories
print("\n--- Creating Directories ---")
for directory in directories:
    try:
        os.makedirs(directory, exist_ok=True)
        print(f"Directory created or exists: {directory}")
    except Exception as e:
        print(f"Error creating directory {directory}: {e}")

# 5. Create empty files
print("\n--- Creating Empty Files ---")
for file_path in files_to_create:
    try:
        if not os.path.exists(file_path):
            with open(file_path, 'w') as f:
                pass  # Just create the empty file
            print(f"File created: {file_path}")
        else:
            print(f"File already exists: {file_path}")
    except Exception as e:
        print(f"Error creating file {file_path}: {e}")

# 6. Create empty, valid notebook files
print("\n--- Creating Empty Notebooks ---")
for nb_path, content in notebooks.items():
    try:
        if not os.path.exists(nb_path):
            with open(nb_path, 'w') as f:
                f.write(content)
            print(f"Notebook created: {nb_path}")
        else:
            print(f"Notebook already exists: {nb_path}")
    except Exception as e:
        print(f"Error creating notebook {nb_path}: {e}")

print("\n--- Project structure setup complete. ---")