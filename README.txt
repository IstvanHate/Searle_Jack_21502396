
Machine Perception Assignment

Directory Structure:
---------------------
Your submission folder should be named using the convention [lastname]_[firstname]_[studentID] 
Example: trump_donald_12345678
Please remove spaces, dashes and other non-alphabet characters in your lastname and firstname whilst creating this folder.

Inside this folder, you must have the following structure:

- output/ 
    - task1/ : Directory for Task 1 output files.
    - task2/ : Directory for Task 2 output files.
    - task3/ : Directory for Task 3 output files.
    - task4/ : Directory for Task 4 output files.
    
- packages/ : Folder containing any Python packages that are not installable via pip (optional).
- data/ : Folder containing any pre-trained weights/checkpoints required for your models (if any).
- input/ : folder containing example input data from blackboard
- output/ : folder containing task1-4 outputs
- model_training/ : contains scripts for taining models + other model related data. Training data place in this folder WONT be tracked due to .gitignore


Files:
------
- assignment.py : **Do not modify this file**. It handles execution for tasks 1, 2, 3, and 4. 
  Focus on completing the task files (task1.py, task2.py, task3.py, task4.py) to implement your solution.
  
  Example usage:
  `python assignment.py task1 /path/to/images/for/task1`
  `python assignment.py task2 /path/to/images/for/task2`

- task1.py, task2.py, task3.py, task4.py : Complete these files to implement the functionality for each task. Each task should save the output to the designated output folder.
- model_training/Train_CNN_tsk3.ipynb, my_model_defs.py : for training small CNN model for task 3
Train_YOLO_Colab.ipynb : YOLO training script adapted from the Author, whomst I've left his credits in. Mounts google drive and connects to servers, able to completely run online. 
- Train_YOLO_locally.ipynb : jupyter notebook for training YOLO model locally using data on a G Drive, for training models on Curtin pcs
- requirements.txt : List of acceptable Python libraries for your project.
- environment.yml : export of venv installed packages, used for recreating venv with same installs
