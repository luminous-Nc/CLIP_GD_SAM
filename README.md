## Steps for clip
```
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
pip install packaging==21.3 
pip install git+https://github.com/openai/CLIP.git
pip install -r requirements.txt    
```
## Steps for GroundingDINO
```
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO/
pip install -e .
```

## Steps for SAM
```
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install opencv-python pycocotools matplotlib onnxruntime onnx
```