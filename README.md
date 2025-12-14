# ICP-NN (Finite Differences)

A simple **ICP (Iterative Closest Points)** implementation using a manual “neural network”, where gradients are computed via **finite difference derivatives**.

## Idea
- Affine alignment between two 2D polygons (translation, X/Y/Z rotation, and scale)
- Gradient-based optimization **without autograd**
- Gradients approximated using finite differences
- Loss function combining **polygon area difference + squared error**

## Features
- 100% Python + NumPy
- No ML frameworks
- Real-time visualization (matplotlib + OpenCV)
- Educational and experimental

## Dependencies
```bash
pip install numpy shapely matplotlib opencv-python
```
