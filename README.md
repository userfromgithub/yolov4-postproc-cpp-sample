# yolov4-postproc-cpp-sample
The postprocess of the YOLOv4 ( C++ Version )
## Introduction
### How does YOLO-v4 work ?
YOLO-v4 consists of 5 main steps, namely, data input, preprocessing, inference, post-processing, and drawing.
Here we mainly focus on the last two parts, post-processing and drawing.

### Purpose of writing this code
Since we know that the runtime of Python is somehow relatively slower than other types of programming languages, especially, C and C++, just to name a few. Therefore, I tried to rewrite the yolo-v4 post-process Python version with C++ so as to test whether the total runtime of post-processing and result drawing can be sped up or not.

## Result in each threshold
Here we use the three layer outputs obtained after conducting YOLO-v4 inference incl. <i>conv2d_58_Conv2D_YoloRegion</i>, <i>conv2d_66_Conv2D_YoloRegion</i>, and <i>conv2d_74_Conv2D_YoloRegion</i> to implement YOLO-v4 post-processing with C++.

<div align="center">
  <p><strong>input image</strong></p>
  <img src="https://github.com/userfromgithub/yolo-v4-postprocess/blob/main/build/obj_input.jpg" alt="thre99 image" style="display: block; margin: auto;"/>
</div>

<p align="center"><strong>threshold 0.99</strong></p>
<img src="https://github.com/userfromgithub/yolo-v4-postprocess/blob/main/drawing-results/Screenshot%20from%202023-08-30%2008-15-38.png" alt="thre99 image">
<p align="center"><strong>threshold 0.6</strong></p>
<img src="https://github.com/userfromgithub/yolo-v4-postprocess/blob/main/drawing-results/Screenshot%20from%202023-08-30%2008-19-04.png" alt="thre60 image">
<p align="center"><strong>threshold 0.1</strong></p>
<img src="https://github.com/userfromgithub/yolo-v4-postprocess/blob/main/drawing-results/Screenshot%20from%202023-08-30%2008-19-16.png" alt="thre10 image">

## Performace metrics Python vs. C++ (time: millisecond)
### threshold 0.99
| | Reshape | Filter | NMS | Total post-process | Drawing | Total runtime |
|:-------:|:-----:|:-----:|:------:|:------:|:------:|:------:|
| **Python**   | 1.1339 | 2.8736 | 0.0996 | $${\color{green}4.1906}$$ | 1.5921 | $${\color{green}5.7828}$$ |
| **C++**  | $${\color{red}4.423}$$ | 2.529 | 0.002 | 7.195 | $${\color{orange}0.932}$$ | 8.128 |

### threshold 0.6
| | Reshape | Filter | NMS | Total post-process | Drawing | Total runtime |
|:-------:|:-----:|:-----:|:------:|:------:|:------:|:------:|
| **Python**   | 0.4694 | 3.3707 | 1.9178 | $${\color{green}5.8209}$$ | 9.7203 | 15.5413 |
| **C++**  | $${\color{red}4.453}$$ | 2.583 | 0.026 | 7.321 | $${\color{orange}5.409}$$ | $${\color{green}12.731}$$ |

### threshold 0.1
| | Reshape | Filter | NMS | Total post-process | Drawing | Total runtime |
|:-------:|:-----:|:-----:|:------:|:------:|:------:|:------:|
| **Python**   | 1.0774 | 7.1897 | 13.7035 | 22.1300 | 28.6309 | 50.7609 |
| **C++**  | $${\color{red}4.579}$$ | 2.826 | 0.161 | $${\color{green}7.892}$$ | $${\color{orange}17.956}$$ | $${\color{green}25.848}$$ |

## Issue
During the testing of the code, we have discovered that the runtime of the "transpose" function in "reshape" section took up around 1/2 of the total process runtime.
Since Python Numpy module uses BLAS and LAPACK to execute matrix, vector, and linear algebra-related operations, we come up with the idea of solving this issue with Xtensor-Blas module.

## Improvement
**Improvement 1**<br>
Here the <i><strong>transpose</strong></i> function is replaced with the code below:
```
xt::xarray<float> transpose(xt::xarray<float>& predictions) {
    
    xt::xarray<float>::shape_type shape = {predictions.shape()[0], predictions.shape()[2], predictions.shape()[3], predictions.shape()[1]};
    xt::xarray<float> new_predictions(shape);

    for (std::size_t n = 0; n < predictions.shape()[0]; n++) {
        for (std::size_t h = 0; h < predictions.shape()[2]; h++) {
            for (std::size_t w = 0; w < predictions.shape()[3]; w++) {
                for (std::size_t c = 0; c < predictions.shape()[1]; c++) {
                    new_predictions(n, h, w, c) = predictions(n, c, h, w);
                }
            }
        }
    }
    return new_predictions;
}
```
## Performace metrics in C++ after Improvement 1 (best record so far)(time: millisecond)
<table align="center">
  <tr>
    <td colspan=6 align="center"><strong>0.99</strong></td>
    <td colspan=6 align="center"><strong>0.6</strong></td>
    <td colspan=6 align="center"><strong>0.1</strong></td>
  </tr>
  <tr>
    <td colspan=3>Reshape</td> 
    <td colspan=3>Total runtime</td>
    <td colspan=3>Reshape</td> 
    <td colspan=3>Total runtime</td>
    <td colspan=3>Reshape</td> 
    <td colspan=3>Total runtime</td>
  </tr>
  <tr>
    <td colspan=2>Before</td>
    <td colspan=2>4.423</td>
    <td colspan=2>8.128</td>
    <td colspan=2>Before</td>
    <td colspan=2>4.453</td>
    <td colspan=2>12.731</td>
    <td colspan=2>Before</td>
    <td colspan=2>4.579</td>
    <td colspan=2>25.848</td>
  </tr>
  <tr>
    <td colspan=2>After</td>
    <td colspan=2>$${\color{green}3.894}$$</td>
    <td colspan=2>$${\color{green}7.633}$$</td>
     <td colspan=2>After</td>
    <td colspan=2>$${\color{green}3.861}$$</td>
    <td colspan=2>$${\color{green}12.078}$$</td>
    <td colspan=2>After</td>
    <td colspan=2>$${\color{green}3.860}$$</td>
    <td colspan=2>$${\color{green}24.575}$$</td>
  </tr>
</table>

## How to run it?
### Step 1.
**Install Xtensor**<br>
First, install xtl
```
cd /opt
git clone https://github.com/xtensor-stack/xtl.git
cd xtl
cmake -D CMAKE_INSTALL_PREFIX=/opt/xtl
```

Install Xtensor
```
git clone https://github.com/xtensor-stack/xtensor.git
cd xtensor
cmake -DCMAKE_INSTALL_PREFIX=/opt/xtensor
make install
```

### Step 2. 
**To run code**<br>
First, locate to <i>build</i> directory, and
```
cmake ../project -DCMAKE_INSTALL_PREFIX=/opt/ ..
make
```

### Step 3.
execute <i>post-process.cpp</i>
```
./pp obj_input.jpg
```

### Step 4.
When the code successfully run, the result will be:
<img src="https://github.com/userfromgithub/yolo-v4-postprocess/blob/main/drawing-results/Screenshot%20from%202023-08-31%2016-12-07.png" alt="thre99 image">

## Reference
https://superfastpython.com/what-is-blas-and-lapack-in-numpy/ <br>
https://max-c.notion.site/C-Numpy-Python-NPY-efe8a325aacb43ec9827f86185220fdc
