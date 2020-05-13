# TravelSalesdrome Problem

1. Please change the botName in your account  to "demo-defbot", or change the botName to be the same as yours.

2. Please upload to your Branch, not directly to the Master Branch.

   

## 现存的问题

1. 对一个规模为n的问题，Noisy Chaotic Neural Network 需要一个n*n的矩阵Y来进行迭代和更新。对于n较大的情况，计算成本相对高昂；
2. 初始化的时候Y矩阵为随机选取获得，没有利用到对手和上次迭代的信息；
3. 迭代后是否存在进一步剪枝优化的可能；
4. 是否存在更好的算法。

