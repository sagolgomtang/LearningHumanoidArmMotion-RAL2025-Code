git clone --recursive <parent-repo-url>
cd LearningHumanoidArmMotion-RAL2025-Code
git switch tocabi
git submodule update --init --recursive

cd IsaacLab
conda create -n ral python=3.10 -y
conda activate ral

pip install "isaacsim[all,extscache]==4.5.0" --extra-index-url https://pypi.nvidia.com
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
./isaaclab.sh --install none

cd ..
pip install -e ./rsl_rl
pip install -e ./cusadi
pip install -r requirements.txt
pip install "casadi==3.7.0" "numpy==1.26.4" pygame
