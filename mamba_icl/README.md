### Mamba
##### 通过试探性实验，即使用含有三种函数类型测试点的混合数据集来训练同一个模型，我们发现：
*模型可能隐含具有对非线性结构更友好的 inductive bias，这或许导致其在处理 Gaussian 和 Dynamical 类函数时的 ICL 表现优于 Linear 函数。*

#### To-do List
- 将epoch设置为120，并采用early stopping策略
- 三类函数分开训练三个模型