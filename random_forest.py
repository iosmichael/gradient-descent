'''
high level understanding of random forest analysis:
deep trees are hard to interpret and have high variance. the inference is quite boxy.

trees notoriously have low bias and high variance
to lower the variance there are three different techniques that we can use:
1. bagging (use bootstrapping techniques to average and reduce variance)
2. random forests (build shallow trees to prevent overfitting and decorrelate individual training datasets to maximize the effectiveness of the averaging)
3. boosting (intentionally calculate the residual functions, prone to overfitting)
4. gradient boosting (use gradient descent to average functions)
5. xgboosting (kaggle winner chosen algorithms)
'''
