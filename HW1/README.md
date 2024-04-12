# Scale Invariant Feature Detection (SIFD) & Joint Bilateral Filter (JBF)

## How to build environment?

```bash
> conda create --name cv_hw1 python=3.10
> conda activate cv_hw1
> pip install -r requirements.txt
```

## Part 1. Scale Invariant Feature Detection (SIFD)

1. Use Difference of Gaussian (DoG) Filter

|                                                  Octave 1                                                  |                                                  Octave 2                                                  |
|:----------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------:|
| <img src="https://github.com/jxes993409/2024-Spring-Computer-Vision/blob/main/HW1/part1/images/DoG_1_1.png" alt="DoG_1_1.png" style="width: 50%; height: 50%"/> | ![image](https://github.com/jxes993409/2024-Spring-Computer-Vision/blob/main/HW1/part1/images/DoG_2_1.png) |
| <img src="https://github.com/jxes993409/2024-Spring-Computer-Vision/blob/main/HW1/part1/images/DoG_1_2.png" alt="DoG_1_2.png" style="width: 50%; height: 50%"/> | ![image](https://github.com/jxes993409/2024-Spring-Computer-Vision/blob/main/HW1/part1/images/DoG_2_2.png) |
| <img src="https://github.com/jxes993409/2024-Spring-Computer-Vision/blob/main/HW1/part1/images/DoG_1_3.png" alt="DoG_1_3.png" style="width: 50%; height: 50%"/> | ![image](https://github.com/jxes993409/2024-Spring-Computer-Vision/blob/main/HW1/part1/images/DoG_2_3.png) |
| <img src="https://github.com/jxes993409/2024-Spring-Computer-Vision/blob/main/HW1/part1/images/DoG_1_4.png" alt="DoG_1_4.png" style="width: 50%; height: 50%"/> | ![image](https://github.com/jxes993409/2024-Spring-Computer-Vision/blob/main/HW1/part1/images/DoG_2_4.png) |

2. Find local extremum

| Threshold = 1                                                                                                       |                                                    Threshold = 2                                                    | Threshold = 3                                                                                                       |
| ------------------------------------------------------------------------------------------------------------------- |:-------------------------------------------------------------------------------------------------------------------:| ------------------------------------------------------------------------------------------------------------------- |
| ![image](https://github.com/jxes993409/2024-Spring-Computer-Vision/blob/main/HW1/part1/images/2_keypoints_th_1.png) | ![image](https://github.com/jxes993409/2024-Spring-Computer-Vision/blob/main/HW1/part1/images/2_keypoints_th_2.png) | ![image](https://github.com/jxes993409/2024-Spring-Computer-Vision/blob/main/HW1/part1/images/2_keypoints_th_3.png) |

## Part2. Joint Bilateral Filter (JBF)

Use grayscale imgae as guidance to compute the output

|          |                                              cv2.COLOR_BGR2GRAY                                               |                                                 0.0, 0.0, 1.0                                                 |                                                 0.0, 1.0, 0.0                                                 | 0.1, 0.0, 0.9                                                                                                 | 0.1, 0.4, 0.5                                                                                                 |                                                 0.8, 0.2, 0.0                                                 |
| -------- |:-------------------------------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------------------------:| ------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |:-------------------------------------------------------------------------------------------------------------:|
| Guidance | ![image](https://github.com/jxes993409/2024-Spring-Computer-Vision/blob/main/HW1/part2/images/guidance_0.png) | ![image](https://github.com/jxes993409/2024-Spring-Computer-Vision/blob/main/HW1/part2/images/guidance_1.png) | ![image](https://github.com/jxes993409/2024-Spring-Computer-Vision/blob/main/HW1/part2/images/guidance_2.png) | ![image](https://github.com/jxes993409/2024-Spring-Computer-Vision/blob/main/HW1/part2/images/guidance_3.png) | ![image](https://github.com/jxes993409/2024-Spring-Computer-Vision/blob/main/HW1/part2/images/guidance_4.png) | ![image](https://github.com/jxes993409/2024-Spring-Computer-Vision/blob/main/HW1/part2/images/guidance_5.png) |
| Result   |   ![image](https://github.com/jxes993409/2024-Spring-Computer-Vision/blob/main/HW1/part2/images/jbf_0.png)    |   ![image](https://github.com/jxes993409/2024-Spring-Computer-Vision/blob/main/HW1/part2/images/jbf_1.png)    |   ![image](https://github.com/jxes993409/2024-Spring-Computer-Vision/blob/main/HW1/part2/images/jbf_2.png)    | ![image](https://github.com/jxes993409/2024-Spring-Computer-Vision/blob/main/HW1/part2/images/jbf_3.png)      | ![image](https://github.com/jxes993409/2024-Spring-Computer-Vision/blob/main/HW1/part2/images/jbf_4.png)      |   ![image](https://github.com/jxes993409/2024-Spring-Computer-Vision/blob/main/HW1/part2/images/jbf_5.png)    |
| Cost     |                                                    1207799                                                    |                                                    1439568                                                    |                                                    1305961                                                    | 1386209                                                                                                       | 1277424                                                                                                       |                                                    1127895                                                    |