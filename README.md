
# How to run this program on Linux Ubuntu system

## Prerequisites

### Clone the Repository

Download the project from Gitlab
```
cd ~/repos
```
```
git clone https://gitlab.com/sainsbury-wellcome-centre/delab/techdev/ultrasound-recording-module.git
```
Enter the project directory 
```
cd ~/repos/ultrasound-recording-module/sound_calibration
```
### Install pyenv
Update your package lists, password required
```
sudo apt update
```
Install the required build dependencies:
```
sudo apt install -y make build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev \
libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python3-openssl git
```
Run curl command to downlad pyenv
```
curl -fsSL https://pyenv.run | bash
```
Configure Your Shell Environment
Run these commands one by one in terminal
```
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc
```
Reload shell
```
source ~/.bashrc
```
Verify installation, expect pyenv 2.x.x
```
pyenv --version
```

### Install python
Install specific version
```
pyenv install 3.12.4
```
```
pyenv local 3.12.4
```

### Virtual Environment
Create virtual environment
```
python3 -m venv venv
```
Activate the environment, expect (venv)....:/repos/ultrasound../
```
source venv/bin/activate
```
### Install Python Dependencies
```
pip install -r requirements.txt
```

### Before running the program

#### Setup expected before running:
-  A calibrated USB mic (e.g. UMIK-1) in the inner box in front of the
- pokewall, between the two speakers.
- The uncalibrated rig mic (camera mic) in its fixed location.
- The UMIK calibration file in sample_data/7101790.txt.
- The standard sweep in sample_data/256k…mono.wav.

## Run the Program

#### First run this code:
```
python3 calibrate_mic.py
```
1. Enter rig ID e.g. 373110
2. Select audio devices:
   - Speaker device = [12] USB PnP Sound Device: Audio (hw:1,0)
   - Calibrated USB mic = [14] Umik-1  Gain: 18dB: USB Audio (hw:3,0)
   - Uncalibrated rig mic = [13] USB: Audio (hw:2,0)
3. Wait for the program to run

#### Then run this immediately after:
```
python3 calibrate_speaker.py
```
1. Enter rig mic cal file, select the one according to the rig ID by enter the number in[], e.g. [0] will be the newest file
2. Select audio devices
   - Speaker device = [12] USB PnP Sound Device: Audio (hw:1,0)
   - Rig mic = [13] USB: Audio (hw:2,0)
3. Wait for the program to run
4. The app will re-runs that cycle automatically at 02:00 every day (no further user input required)