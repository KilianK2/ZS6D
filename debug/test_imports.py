# test_imports.py
try:
    import torch
    print("Successfully imported torch")
except ImportError:
    print("Error importing torch")

try:
    import torchvision
    print("Successfully imported torchvision")
except ImportError:
    print("Error importing torchvision")

try:
    import torchaudio
    print("Successfully imported torchaudio")
except ImportError:
    print("Error importing torchaudio")

try:
    import tqdm
    print("Successfully imported tqdm")
except ImportError:
    print("Error importing tqdm")

try:
    import timm
    print("Successfully imported timm")
except ImportError:
    print("Error importing timm")

try:
    import matplotlib
    print("Successfully imported matplotlib")
except ImportError:
    print("Error importing matplotlib")

try:
    import sklearn
    print("Successfully imported scikit-learn")
except ImportError:
    print("Error importing scikit-learn")

try:
    import cv2  # OpenCV
    print("Successfully imported opencv-python")
except ImportError:
    print("Error importing opencv-python")

try:
    import pydensecrf
    print("Successfully imported pydensecrf")
except ImportError:
    print("Error importing pydensecrf")

try:
    import transforms3d
    print("Successfully imported transforms3d")
except ImportError:
    print("Error importing transforms3d")

try:
    import PIL  # Pillow
    print("Successfully imported Pillow")
except ImportError:
    print("Error importing Pillow")

try:
    import plyfile
    print("Successfully imported plyfile")
except ImportError:
    print("Error importing plyfile")

try:
    import trimesh
    print("Successfully imported trimesh")
except ImportError:
    print("Error importing trimesh")

try:
    import imageio
    print("Successfully imported imageio")
except ImportError:
    print("Error importing imageio")

try:
    import png  # pypng
    print("Successfully imported pypng")
except ImportError:
    print("Error importing pypng")

try:
    import vispy
    print("Successfully imported vispy")
except ImportError:
    print("Error importing vispy")

try:
    import OpenGL.GL
    print("Successfully imported PyOpenGL")
except ImportError:
    print("Error importing PyOpenGL")

try:
    import pyglet
    print("Successfully imported pyglet")
except ImportError:
    print("Error importing pyglet")

try:
    import numba
    print("Successfully imported numba")
except ImportError:
    print("Error importing numba")

try:
    import jupyter
    print("Successfully imported jupyter")
except ImportError:
    print("Error importing jupyter")
