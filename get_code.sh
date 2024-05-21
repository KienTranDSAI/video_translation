git lfs install
git clone https://huggingface.co/vinai/PhoWhisper-medium
git clone https://huggingface.co/vinai/vinai-translate-vi2en-v2

git clone https://github.com/neonbjb/tortoise-tts.git

cd tortoise-tts
python setup.py install
pip install -r requirements.txt
cd ../
git clone https://github.com/soumik-kanad/diff2lip.git
cd diff2lip
pip install -r requirements.txt
cd ../