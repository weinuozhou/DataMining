# 机器学习概论

?> 机器学习(machine learning):研究如何通过计算的手段，利用经验来改善系统自身的性能。在计算机上从数据中产生**模型**的算法，即学习算法(learning algorithm)

<div style="text-align: center;"><img alt='202404082114110' src='https://cdn.jsdelivr.net/gh/weno861/image@main/img/202404082114110.png' width=500px> </div>

> **没有免费的午餐定理**（No Free Lunch Theorem)：对于一个学习算法`A`，如果在某些问题上它比算法 `B` 好，那么必然存在另一些问题，在那些问题中 `B` 比 `A` 更好

因此要**谈论算法的优劣必须基于具体的学习问题**

## 基本概念

### 特征空间

* 输入空间：所有输入的可能取值
* 输出空间：所有输出的可能取值
* 特征向量表示每个具体的输入， 所有特征向量构成特征空间，特征空间的**每一个维度对应一种特征**

### 样本表示

通常输入实例用 $\vec{x}$ 表示，真实标记用 $\vec{y}$ 表示，模型的预测值用 $\hat{y}$ 表示

所有的向量均为列向量，其中输入实例 $\vec{x}$ 的特征向量记作（假设特征空间为 $n$ 维）:
$$
\vec{x}=\left[ \begin{array}{c}
	x^{\left( 1 \right)}\\
	x^{\left( 2 \right)}\\
	\cdot \cdot \cdot\\
	x^{\left( n \right)}\\
\end{array} \right] 
$$

* $\vec{y}$ 为连续的，这一类问题称为**回归**问题
* $\vec{y}$ 为离散的，这一类问题称为**分类**问题

## 机器学习三大要素

机器学习的三要素分别是**模型**、**策略**和**算法**

### 模型

模型是关于未知函数 $f$ 的假设，即 $f\left( \cdot \right)$ 的一个**假设空间**，记作 $F$，例如线性回归模型、决策树模型等

将学习过程看作一个在**解空间中进行搜索**的过程，搜索目标就是找到与训练集匹配的解

### 策略

策略是关于如何选择最优模型的准则，即**损失函数**，记作 $L$，例如均方误差、$0-1$损失等

#### 损失函数

对于给定的输入 $\vec{x}$ ，由模型预测的输出值 $\hat{y}$ 与真实的标记值 $\vec{y}$ 可能不一致。此时，用损失函数度量错误的程度，也称作代价函数

* $0-1$损失函数:
$$
L\left( f\left( \vec{x} \right),y \right)=\left\{ \begin{array}{c}
	0,y=f\left( \vec{x} \right)\\
	1,y\neq f\left( \vec{x} \right)\\
\end{array} \right. 
$$
* 均方误差损失函数:
$$
L\left( f\left( \vec{x} \right),y \right)=\left( f\left( \vec{x} \right)-y \right)^{2} 
$$
* 交叉熵损失函数:
$$
L\left( f\left( \vec{x} \right),y \right)=-y\log f\left( \vec{x} \right)-(1-y)\log \left( 1-f\left( \vec{x} \right) \right) 
$$
* 对数似然损失函数:
$$
L\left( f\left( \vec{x} \right),y \right)=-\log f\left( \vec{x} \right) 
$$	
* KL散度损失函数:
$$
L\left( f\left( \vec{x} \right),y \right)=y\log \frac{y}{f\left( \vec{x} \right)}+(1-y)\log \frac{1-y}{1-f\left( \vec{x} \right)} 
$$

#### 正则化项

正则化项用于**控制模型的复杂度**，防止过拟合，例如$L1$正则化、$L2$正则化等

$$
L_1=\left| \left| \vec{w} \right| \right|_{1}=\sum_{i=1}^{n}\left| w_{i} \right| \\
L_2=\left| \left| \vec{w} \right| \right|_{2}=\sqrt{\sum_{i=1}^{n}w_{i}^{2}} 
$$

### 算法

算法指学习模型的具体计算方法。通常采用数值计算的方法求解，如：梯度下降法,牛顿法等等