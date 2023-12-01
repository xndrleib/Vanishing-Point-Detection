# Vanishing-point-detection

This repository contains code written for a competition in vanishing point detection in images using [Hough transform](https://en.wikipedia.org/wiki/Hough_transform).

## Project Description

The task of detecting the vanishing point in images is of great importance and has a wide range of applications. It arises in the development of systems for autonomous vehicles (localization on the road surface, self-calibration), in the analysis of scene lighting (analysis of color histograms), and in correcting projective distortions.

800 labeled images of the road surface from a vehicle camera, containing coordinates of the vanishing point, are used as data for constructing and testing the algorithm. During testing, the images are augmented by rotation.

## Response Format

As a result, the algorithm returns a JSON file in the format:
```
{
    "file1.jpg": [x1, y1], 
    "file2.jpg": [x2, y2]
}
```

## Test Sample Generation

The test sample is generated using the script test_generation/test_generation.py:

```bash
python test_generation/test_generation.py --s path_to_dataset --d path_to_save_new_dataset --num num_of_imgs_to_generate --seed seed
```

It is assumed that the data folder is structured as follows:

```
├── dataset
    ├── markup.json
    ├── source
```

## Solution Quality Assessment

The quality of the solution is assessed using an angular metric. The proximity of the predicted point $A$ to the true 
value $B$ is determined by the angle $\alpha$ between the vectors $\xi$ and $\eta$ drawn to these points 
from the point $O$, as demonstrated in the figure below.


$$\alpha = \arccos \frac{\left \langle \xi, \, \eta \right \rangle}{\|\xi\|\|\eta\|},$$

where

```math
\xi = 
\begin{bmatrix} 
A_x \\ 
A_y \\ 
0 
\end{bmatrix} 
- 
\begin{bmatrix} n/2 \\ 
m/2 \\ 
\sqrt{\left( n/2 \right)^2 + \left( m/2 \right)^2} 
\end{bmatrix}, \quad
\eta = 
\begin{bmatrix} 
B_x \\ 
B_y \\ 
0
\end{bmatrix}
-
\begin{bmatrix} 
n/2 \\ 
m/2 \\ 
\sqrt{\left( n/2 \right)^2 + \left( m/2 \right)^2}
\end{bmatrix}.
```

![Metrics](metrics.png)