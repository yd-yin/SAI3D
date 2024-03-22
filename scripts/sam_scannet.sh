BASE_DIR="/home/yuzheng/code/SAI3D"   # Change this to your project directory
REL_DATA_DIR="data/ScanNet"
VIEW_FREQ=5  # this means sample 1 view every 5 views

DATA_DIR="${BASE_DIR}/${REL_DATA_DIR}"

export PYTHONPATH="${BASE_DIR}":$PYTHONPATH

cd Semantic-SAM

python -m sam_auto_generation \
--data_dir="${DATA_DIR}" \
--view_freq=${VIEW_FREQ} 