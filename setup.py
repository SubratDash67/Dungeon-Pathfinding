import os

# Define the project root directory
project_root = "RealTimeDungeonPathfindingOptimizer"

# Define the directory structure and files to be created
structure = {
    "data": ["__init__.py", "dungeon_graphs.py", "trap_guard_simulation.py"],
    "models": ["__init__.py", "gnn_model.py", "gnn_layers.py"],
    "training": ["__init__.py", "trainer.py", "checkpoint.py"],
    "utils": [
        "__init__.py",
        "data_augmentation.py",
        "graph_utils.py",
        "logging_utils.py",
    ],
    "configs": ["default_config.yaml"],
    "tests": ["test_data.py", "test_models.py", "test_training.py", "test_utils.py"],
    "scripts": ["run_training.py", "run_evaluation.py"],
    "docs": ["README.md", "PROJECT_PLAN.md"],
}


# Function to create directories and files
def create_project_structure(root, structure):
    if not os.path.exists(root):
        os.mkdir(root)
    for folder, files in structure.items():
        folder_path = os.path.join(root, folder)
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
        for file in files:
            file_path = os.path.join(folder_path, file)
            if not os.path.exists(file_path):
                with open(file_path, "w") as f:
                    pass


create_project_structure(project_root, structure)

# Output the created structure for verification
created_structure = {}
for folder in structure.keys():
    folder_path = os.path.join(project_root, folder)
    created_structure[folder] = os.listdir(folder_path)

created_structure
