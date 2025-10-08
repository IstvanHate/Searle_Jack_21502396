
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
- conda_install.sh: quickly downloading and installing miniconda on lab pcs so I can create venv

File Structure Overview:
------------------------
Searle_Jack_21502396/
│
├── assignment.py
├── config.txt
├── conda_install.sh
├── environment.yml
├── my_model_defs.py
├── requirements.txt
├── README.txt
│
├── data/
│   ├── README.txt
│   ├── digit_classifier.pth
│   ├── task1YOLO.pt
│   ├── task1YOLO.zip
│   ├── task2YOLO.pt
│   ├── task2YOLO.zip
│
├── input/
│   ├── task1/
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   ├── img3.jpg
│   │   ├── img4.jpg
│   │   └── img5.jpg
│   │
│   ├── task2/
│   │   ├── bn1.png
│   │   ├── bn2.png
│   │   ├── bn3.png
│   │   └── bn4.png
│   │
│   ├── task3/
│   │   ├── bn1/
│   │   │   ├── c1.png
│   │   │   ├── c2.png
│   │   │   └── c3.png
│   │   ├── bn2/
│   │   │   ├── c1.png
│   │   │   ├── c2.png
│   │   │   └── c3.png
│   │   ├── bn3/
│   │   │   ├── c1.png
│   │   │   ├── c2.png
│   │   │   └── c3.png
│   │   ├── bn4/
│   │   │   ├── c1.png
│   │   │   ├── c2.png
│   │   │   └── c3.png
│   │
│   ├── task4/
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   ├── img3.jpg
│   │   ├── img4.jpg
│   │   └── img5.jpg
│   │
│   └── validation(1).zip
│
├── output/
│   ├── task1/
│   │   ├── bn2.png
│   │   ├── bn3.png
│   │   ├── bn4.png
│   │   ├── bn5.png
│   │   ├── img1.txt
│   │   └── task1output.txt
│   │
│   ├── task2/
│   │   └── task2output.txt
│   │
│   ├── task3/
│   │   └── task3output.txt
│   │
│   ├── task4/
│       └── task4output.txt
│
├── model_training/
│   ├── Train_CNN_tsk3.ipynb
│   ├── Train_YOLO_Colab.ipynb
│   ├── Train_YOLO_locally.ipynb
│   ├── digit_classifier.pth
│   │
│   ├── my_model/
│   │   ├── task2YOLO.pt
│   │   └── task2YOLO.zip
│   │
│   ├── runs/
│   │   └── detect/
│   │       ├── train/
│   │       │   └── args.yaml
│   │       └── train2/
│   │           ├── BoxF1_curve.png
│   │           ├── BoxPR_curve.png
│   │           ├── BoxP_curve.png
│   │           ├── BoxR_curve.png
│   │           ├── confusion_matrix.png
│   │           ├── confusion_matrix_normalized.png
│   │           ├── labels.jpg
│   │           ├── results.csv
│   │           ├── results.png
│   │           ├── train_batch0.jpg
│   │           ├── train_batch1.jpg
│   │           ├── train_batch2.jpg
│   │           ├── train_batch4550.jpg
│   │           ├── train_batch4551.jpg
│   │           ├── train_batch4552.jpg
│   │           ├── val_batch0_pred.jpg
│   │           ├── val_batch0_labels.jpg
│   │           ├── val_batch1_pred.jpg
│   │           ├── val_batch1_labels.jpg
│   │           ├── val_batch2_pred.jpg
│   │           ├── val_batch2_labels.jpg
│   │           ├── args.yaml
│   │           └── weights/
│   │               ├── best.pt
│   │               └── last.pt
│   │
│   └── __pycache__/
│
├── packages/
│   └── README.txt
│
├── task1.py
├── task2.py
├── task3.py
├── task4.py
└── __pycache__/
    ├── my_model_defs.cpython-313.pyc
    ├── task1.cpython-313.pyc
    ├── task2.cpython-313.pyc
    ├── task3.cpython-313.pyc
    └── task4.cpython-313.pyc

Setting up virtual environment
conda env create -f environment.yml
conda activate COMP3007_venv
