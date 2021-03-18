# Linear Regression model using LSE or Newton's method
## Description
Implement a regularized linear regression model using LSE or Newton's method

## Required Dataset
* Dataset
    * Single Independent Variable X
    * Dependent variable Y
* Example
```
data = [-5.0, 51.76405234596766,
    -4.795918367346939, 45.42306433039972,
    -4.591836734693878, 41.274448104888755,
    -3.979591836734694, 26.636216497466364,
    -3.571428571428571, 20.256806057008426,
    -2.9591836734693877, 11.618429243797276,
    -2.7551020408163263, 10.450525068812203,
    -1.7346938775510203, 1.8480982318414874,
    -1.3265306122448979, -1.0405349639051173,
    -0.9183673469387754, -4.614630798757861,
    -0.7142857142857144, -1.3871977310902517,
    -0.3061224489795915, -1.9916444039966117,
    0.1020408163265305, -0.912924608376358,
    0.7142857142857144, 6.63482003068499,
    1.1224489795918373, 9.546867459016372,
    1.7346938775510203, 15.72016146597016,
    1.9387755102040813, 20.62251683859554,
    2.5510204081632653, 33.48059725819715,
    2.959183673469388, 40.76391965675495,
    3.979591836734695, 66.8997605629381,
    4.387755102040817, 78.44316465660981,
    4.591836734693878, 86.99156782355371,
    5.0, 99.78725971978604]
```

## How to Run
```
    python main.py # 默認模式
```
* --n : the number of polynomial bases
* --lambd: lambda of regularized (only for LSE case) 
* --optimizer : optimizer （LSE, Newton or both. default is both）
* --isplot : whether to visualize the output of the model

## Result
* Case 1: n = 2, = 0
```
    LSE:
    Fitting line: 4.43295031008X^1 + 29.3064047061
    Total error: 16335.123165  

    Newton's Method:
    Fitting line: 4.43295031008X^1 + 29.3064047061
    Total error: 16335.123165
```
![n2lambda0](https://github.com/kenyo3023/Regularized-Linear-Regression-using-LSE-and-Netwon/blob/main/images/n2lambda0.png)

* Case 2: n = 3, = 0
```
    LSE:
    Fitting line: 3.02385339349X^2 + 4.90619026386X^1
    -0.231401756088
    Total error: 26.5599594993

    Newton's Method:
    Fitting line: 3.02385339349X^2 + 4.90619026386X^1
    -0.231401756088
    Total error: 26.5599594993
```
![n3lambda0](https://github.com/kenyo3023/Regularized-Linear-Regression-using-LSE-and-Netwon/blob/main/images/n3lambda0.png)

* Case 3: n = 3, = 10000
```
    LSE:
    Fitting line: 0.8345332827X^2 + 0.0931481983192X^1
    + 0.0469506992735
    Total error: 22649.738493

    Newton's Method:
    Fitting line: 3.02385339349X^2 + 4.90619026386X^1
    -0.231401756088
    Total error: 26.5599594993
```
![n3lambda10000](https://github.com/kenyo3023/Regularized-Linear-Regression-using-LSE-and-Netwon/blob/main/images/n3lambda10000.png)