## multivariate linear regression

`y = w0 + w1 * x1 + w2 * x2 + w3 * x3`

* Use two methods to calculate w0,w1,w2,w3:
  * Explicit analytical solution by matrix: ols_matrix
    * 公式 [ols wiki](https://en.wikipedia.org/wiki/Ordinary_least_squares)
    * 原理是使SRR最小化, 得到的是 ols estimator
  * stochastic gradient descent: sgd
    * 参考 [Andrew Ng 的 notes](http://cs229.stanford.edu/notes/cs229-notes1.pdf)
* files:
  * 数据: `data.csv`
  * 开发文档: [multi-linrg.ipynb](https://github.com/bambooom/multi-linrg/blob/master/multi-linrg.ipynb)
  * 代码: [multi-linrg.py](https://github.com/bambooom/multi-linrg/blob/master/multi-linrg.py)
* Solutions:

```
1st method - analytical solution by matrix
w0 = 2.030762
w1 = 2.973967
w2 = -0.541390
w3 = 0.971329
2nd method - stochastic gradient descent
w0 = 2.030762
w1 = 2.973971
w2 = -0.541381
w3 = 0.971339
```
