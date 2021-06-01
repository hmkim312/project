---
title: Mnist Fashion으로 알아보는 Deep Learning
author: HyunMin Kim
date: 2021-05-10 00:00:00 0000
categories: [Data Science, Deep Learning]
tags: [MLP, CNN, Decision Tree,  Random Forest, Mnist Fashion]
---

# Mnist Fashion으로 알아보는 Deep Learning

---
## 목차

0. 개요
1. 패키지 목록
2. 데이터 로드
3. 데이터 라벨링
4. 데이터 시각화
5. 머신러닝, 딥러닝을 위한 데이터 처리
6. Machine Learning
7. Deep Learning
8. 결과 및 회고

---

### 0. 개요 - Mnist Fashoin Image

<img src ="https://user-images.githubusercontent.com/60168331/117663987-35e42e00-b1dc-11eb-88c7-64db3b4fbff0.png">

- Mnist Fashion Image는 운동화, 셔츠, 샌들과 같은 의류 이미지들의 모음입니다.
- 총 10 가지의 Class로 이루어져 있으며, 이미지는 28×28 픽셀이며 총 70,000 장으로로 이루어져 있습니다.
- 그 중 60,000장의 데이터는 Train 데이터이고, 10,000장은 Test 데이터입니다.

---

### 0. 개요 - 계획

- 실제 이미지 데이터를 불러오고, 시각화하고 머신러닝과 딥러닝의 알고리즘으로 예측 모델을 생성합니다.
- 머신러닝은 Decision Tree와 RandomForest를 사용합니다.
- 딥러닝의 Multi Layer Perceptron와 Convolution Neural Network를 사용합니다.
- 마지막으로 가장 성능이 좋은 모델로 Test 데이터를 예측하여 결과를 CSV파일로 저장합니다.

---

### 1. 패키지 목록

```python
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import os
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

np.set_printoptions(threshold=np.inf, linewidth=500)
```

## 1. 데이터 로드
---
### 1.1 데이터 로드 - Label Data 

```python
y_train = np.array(pd.read_csv('Fashion_MNIST/train.csv')['class'])
y_test = pd.read_csv('./Fashion_MNIST/test_sample.csv')
```
- Data는 <https://github.com/hmkim312/datas/tree/main/mnist_fashion>에 있습니다.
- y label은 train.csv파일에 class라는 컬럼에 있습니다.
- y값은 id값의 순서대로 들어가져 있습니다.


### 1.2 데이터 로드 - Train Data

```python
train_path = './Fashion_MNIST/images/train/'
test_path = './Fashion_MNIST/images/test/'

X_train = np.array([mpimg.imread(train_path + str(i) + '.png') for i in range(0,60000)])
# for i in range(0,60000):
#     image_path = str(i) + '.png'
#     image = mpimg.imread(train_path + str(i) + '.png')
#     train_images.append(image)

X_test = np.array([mpimg.imread(test_path + str(i) + '.png') for i in range(0,10000)])
# for i in range(0,10000):
#     image_path = str(i) + '.png'
#     image = mpimg.imread(test_path + str(i) + '.png')
#     test_images.append(image)
```

- Train 데이터는 한장 한장 실제 이미지로 있습니다.
- 따라서 이미지를 불러오는데 별도의 전처리를 해야합니다.
- matplotlib의 image를 사용하여 불러왔고, 파일명이 곧 id이 이므로, 순서대로 가져오면 0번 ~ 6만번, 0번 ~ 1만번의 데이터를 가져옵니다.


### 1.3 데이터 로드 - Data Shape

```python
X_train.shape, y_train.shape, X_test.shape
```
    ((60000, 28, 28), (60000,), (10000, 28, 28))

- X_train 데이터는 60,000장에 28 by 28 데이터로 Chanel은 1개로 Gray 스케일이며, Chanel에 대한 정보는 생략되어 있습니다.
- y_train 데이터는 X_train의 라벨입니다. 총 6만개의 데이터의 라벨로 구성되어 있습니다.
- Test 데이터는 10,000장의 28 by 28로 Train 데이터와 이미지 크기는 똑같습니다.


## 2. 데이터 라벨링
---

### 2.1 데이터 라벨링 - 종류

```python
mnist_fashion_labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag','Ankle Boot']
```

- Mnist Fashion Image의 라벨은 총 10가지로 0 ~ 9까지 구성되어 있으며, 각각의 숫자는 다음의 클래스를 의미합니다.


### 2.2 데이터 라벨링 - 종류

|Label|Class|Label|Class
|:---:|---|:---:|---|
|0|T-shirt/top(티셔츠)|1|Trouser(바지)
|2|Pullover(풀오버스웨터)|3|Dress(드레스)
|4|Coat(코드)|5|Sandal(샌들)
|6|Shirt(셔츠)|7|Sneaker(운동화)
|8|Bag(가방)|9|Ankle boot(발목 부츠)


## 3. 데이터 시각화
---

### 3.1 데이터 시각화 - 코드

```python
figure = plt.figure(figsize=(15, 10))
for index, i in enumerate(np.random.randint(0, X_train.shape[0], 15)):
    ax = figure.add_subplot(3, 5, index + 1, xticks=[], yticks=[])
    ax.imshow(X_train[i], cmap = 'gray')
    ax.set_title(f"{y_train[i]} : {mnist_fashion_labels[y_train[i]]}")
plt.show()
```

### 3.2 데이터 시각화

<img src ="https://user-images.githubusercontent.com/60168331/117663990-37155b00-b1dc-11eb-8bd5-b2fd0bb28dc9.png">

    
- Train 데이터에서 랜덤하게 15개 데이터를 골라 시각화 하였습니다.
- 28 by 28의 1채널이라 화질이 그렇게 좋은 이미지는 아니지만, 눈으로 보아도 어떤 의류 사진인지 알수 있는것도 있으나, 아닌것도 있습니다.


#### 3.3 데이터 시각화 - 코드

```python
plt.figure(figsize=(12,12))
plt.imshow(X_train[8703], cmap ='gray')
plt.title(f'{y_train[8703]} : {mnist_fashion_labels[y_train[8703]]}')
plt.xticks([])
plt.yticks([])
plt.show()
```

#### 3.4 데이터 시각화

<img src = "https://user-images.githubusercontent.com/60168331/117663993-37adf180-b1dc-11eb-882a-d3933c5667de.png">

- Train Data를 하나 골라 시각화 해보았습니다.
- 이미지는 6번 Shirt(셔츠) 입니다.

#### 3.5 데이터 시각화 - Array

    [[  0.   0.   0.   0.   0.   0.   3.   5.   0.   0.   0. 137. 172. 139. 183. 142.   0.   0.   2.   1.   0.   0.   1.   0.   0.   0.   0.   0.]
     [  0.   0.   0.   0.   0.   3.   0.   0.   0.   0.   0. 224. 171. 169. 208. 229.   0.   0.   0.   0.   5.   2.   0.   0.   0.   0.   0.   0.]
     [  0.   0.   0.   0.   0.   0.   0.   0.  33.  76. 179. 199. 173. 213. 201. 150. 169.  89.  12.   0.   0.   0.   0.   0.   0.   0.   0.   0.]
     [  0.   0.   1.   0.   0.  73. 163. 189. 211. 191. 168. 168. 199. 192. 192. 167. 180. 197. 186. 155.  79.  10.   0.   0.   1.   0.   0.   0.]
     [  0.   0.   0.   0. 105. 197. 183. 187. 184. 167. 152. 171. 162. 159. 147. 166. 170. 162. 152. 167. 180. 174. 100.   0.   1.   0.   0.   0.]
     [  0.   0.   0.   4. 190. 179. 177. 160. 166. 171. 169. 166. 165. 171. 165. 162. 161. 162. 157. 156. 160. 168. 175.  14.   0.   3.   0.   0.]
     [  0.   0.   0.  73. 192. 178. 176. 188. 166. 172. 182. 166. 169. 186. 171. 168. 167. 147. 184. 168. 162. 157. 186.  76.   0.   2.   0.   0.]
     [  0.   0.   0. 125. 199. 193. 192. 167. 180. 161. 158. 166. 157. 159. 169. 153. 159. 157. 148. 168. 164. 170. 185. 125.   0.   0.   0.   0.]
     [  0.   0.   0. 163. 181. 187. 230. 179. 177. 159. 157. 165. 157. 170. 168. 155. 162. 167. 134. 148. 177. 184. 177. 152.   0.   0.   0.   0.]
     [  0.   0.   0. 190. 183. 154. 228. 204. 195. 183. 173. 182. 175. 179. 155. 174. 189. 178. 162. 161. 219. 171. 171. 168.   0.   0.   0.   0.]
     [  0.   0.   0. 197. 185. 163. 216. 206. 185. 181. 165. 187. 171. 155. 167. 168. 161. 171. 173. 151. 141. 190. 169. 202.   0.   0.   0.   0.]
     [  0.   0.   0. 206. 172. 182. 179. 195. 194. 180. 198. 175. 182. 203. 181. 174. 176. 153. 197. 153.  71. 211. 169. 168.  24.   0.   0.   0.]
     [  0.   0.  27. 215. 175. 194. 172. 186. 194. 169. 196. 190. 171. 190. 183. 178. 182. 158. 208. 164.  50. 230. 171. 183.  57.   0.   0.   0.]
     [  0.   0.  39. 225. 187. 212. 126. 155. 218. 158. 164. 179. 175. 162. 160. 168. 179. 155. 173. 159.  53. 253. 179. 187.  78.   0.   0.   0.]
     [  0.   0.  59. 219. 177. 234.  98. 100. 224. 168. 172. 177. 174. 165. 162. 161. 179. 160. 168. 146.  33. 235. 182. 184. 120.   0.   0.   0.]
     [  0.   0.  87. 216. 178. 241.  53.  99. 242. 178. 174. 187. 189. 190. 160. 172. 201. 166. 173. 145.  15. 225. 177. 181. 153.   0.   0.   0.]
     [  0.   0. 115. 209. 176. 238.   1. 107. 226. 173. 170. 180. 164. 162. 169. 167. 172. 171. 160. 162.   0. 223. 185. 176. 176.   0.   0.   0.]
     [  0.   0. 128. 190. 174. 236.   0. 143. 217. 172. 197. 188. 180. 196. 173. 169. 179. 169. 179. 189.   0. 203. 194. 171. 176.   0.   0.   0.]
     [  0.   0. 145. 191. 186. 220.   0. 162. 221. 166. 197. 193. 173. 185. 206. 171. 174. 171. 173. 202.   0. 187. 210. 173. 186.  21.   0.   0.]
     [  0.   0. 160. 183. 192. 200.   0. 167. 213. 176. 159. 189. 172. 159. 168. 187. 161. 170. 149. 192.   0. 140. 222. 170. 192.  51.   0.   0.]
     [  0.   0. 160. 184. 189. 164.   0. 170. 215. 173. 170. 179. 168. 174. 150. 186. 181. 174. 149. 194.   0.  66. 223. 160. 188.  63.   0.   0.]
     [  0.   0. 151. 186. 201. 106.   0. 181. 208. 186. 178. 185. 180. 184. 146. 194. 209. 170. 166. 202.   0.   0. 255. 168. 204.  58.   0.   0.]
     [  0.   0. 141. 188. 230.  61.   0. 191. 201. 182. 179. 184. 173. 156. 175. 169. 187. 179. 159. 175.   0.   0. 218. 187. 171.  56.   0.   0.]
     [  0.   0. 118. 186. 214.   0.   0. 197. 200. 178. 196. 181. 181. 192. 177. 181. 184. 181. 179. 191.   8.   0. 123. 200. 170.  50.   0.   0.]
     [  0.   0. 124. 185. 196.   0.   0. 205. 203. 185. 202. 203. 175. 204. 190. 178. 189. 177. 189. 210.  20.   0.  48. 194. 171.  62.   0.   0.]
     [  0.   0. 166. 208. 186.   0.   0. 221. 201. 183. 172. 196. 179. 169. 163. 163. 187. 173. 167. 175.  34.   0.  10. 186. 176. 130.   0.   0.]
     [  0.   0. 142. 156. 167.   0.   0. 178. 211. 200. 201. 207. 197. 187. 178. 181. 219. 202. 176. 224.  32.   0.   0. 186. 140. 101.   0.   0.]
     [  0.   0.   0.   0.   0.   0.   0.   0.  53.  93.  98. 119. 132. 135. 121. 123. 126. 108.  79.  24.   0.   0.   0.   0.   0.   0.   0.   0.]]

- 검은곳은 0이고 색깔이 있는 부분에 숫자가 있는것을 알수 있습니다.
- 이러한 숫자의 패턴으로 알고리즘들이 패턴을 찾고, 어떤 이미지인지 예측할수 있습니다.

## 4. 머신러닝, 딥러닝을 위한 데이터 처리
---

#### 4.1 머신러닝, 딥러닝을 위한 데이터 처리 - Validation Data Split

```python
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, stratify = y_train, test_size = 0.2, random_state = 87)
X_train.shape, X_valid.shape, y_train.shape, y_valid.shape
```

((48000, 28, 28), (12000, 28, 28), (48000,), (12000,))

- Train 데이터 6만장을 실제 모델 학습 데이터와 모델의 검증 데이터로 나누겠습니다.
- 학습 데이터를 나누는 이유는 생성된 모델의 성능이 과적합인지, 성능은 제대로 나오는지 등을 확인하기 위해서 입니다.
- Train 데이터에서 검증용 데이터와 학습용 데이터로 한번더 나누어서 모델 학습은 학습용 데이터로 진행합니다.
- 생성된 모델을 검증용 데이터로 모델의 성능을 확인합니다.
- 모델 성능이 확인되면 마지막으로 제출해야할 실제 y값이 없는 테스트 데이터를 예측 후 해당 결과를 csv파일로 저장하여 제출합니다.
- 즉, 6만장 데이터를 4만8천장의 모델 학습용 데이터와, 1만 2천장의 모델 검증용 데이터로 나누어 모델을 생성 한 후 생성된 모델을 1만장의 test 데이터를 예측합니다. 


#### 4.2 머신러닝, 딥러닝을 위한 데이터 처리 - Stratify

```python
unique, counts = np.unique(y_train, return_counts=True)
dict(zip(unique, counts))
```

    {0: 4800,
     1: 4800,
     2: 4800,
     3: 4800,
     4: 4800,
     5: 4800,
     6: 4800,
     7: 4800,
     8: 4800,
     9: 4800}


- 또한 split 과정에서 stratify 옵션을 넣어 label의 class가 모두 동일한 비율로 들어가게 만들었습니다.


#### 4.3 머신러닝, 딥러닝을 위한 데이터 처리 - Scale

```python
X_train.min(),  X_train.max()
```
    (0.0, 1.0)



- 제공받은 데이터는 최소값이 0, 최대값이 1이므로 MinMax Scaler가 적용되어 있어, 따로 Scale을 하지 않습니다.

## 5. Machine Learning
---

### 5.1 Machine Learning - 데이터 처리


```python
skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=87)
X_train_reshape = X_train.reshape(X_train.shape[0], 28 * 28) # 48000, 784
X_valid_reshape = X_valid.reshape(X_valid.shape[0], 28 * 28) # 12000, 784
X_test_reshape = X_test.reshape(X_test.shape[0], 28 * 28) # 10000, 784
```

- 머신러닝을 하기전에 28 by 28의 데이터를 flatten하게 48000 * 784와 12000 * 28형식으로 reshape 해주었습니다. 
- 즉, 6만개의 데이터가 있고 총 784개의 특성이 있는 형식입니다.
- 또한 학습에 Cross Validation을 위해 StratifiedKFold를 생성합니다.


### 5.2 Machine Learning - DecisionTree 학습

```python
clf = DecisionTreeClassifier(random_state=87)
cv_score = cross_val_score(clf, X_train_reshape, y_train, cv = skfold)
print('\nAccuracy: {:.4f}'.format(cv_score.mean()))
```    
    Accuracy: 0.7900

- 가장 먼저 DecisionTree로 학습해보았습니다.
- KFold로 검증한 평균 Accuracy는 0.79정도가 나옵니다.


### 5.3 Machine Learning - DecisionTree 검증

```python
clf.fit(X_train_reshape, y_train)
print('\nAccuracy: {:.4f}'.format(clf.score(X_valid_reshape, y_valid)))
```
    Accuracy: 0.7896


- Decision Tree를 사용하여 만든 모델의 검증결과는 Accuracy 0.7896 입니다.
- 모델의 성능이 좋은편은 아닌듯 합니다.


### 5.4 Machine Learning- RandomForest 학습

```python
rf = RandomForestClassifier(random_state = 87)
cv_score = cross_val_score(rf, X_train_reshape, y_train, cv = skfold)
print('\nAccuracy: {:.4f}'.format(cv_score.mean()))
```

    Accuracy: 0.8799

- RandomForest는 Decision Tree들이 모인 앙상블 기법입니다.
- 보통 Decision Tree보다 성능이 좋은것으로 알려져 있습니다.
- KFold를 진행하여, 평균 Accuracy를 확인해보니 0.8799가 나옵니다. 확실히 Decision Tree보다 성능이 좋습니다.


### 5.4 Machine Learning - RandomForest 검증

```python
rf.fit(X_train_reshape, y_train)
print('\nAccuracy: {:.4f}'.format(rf.score(X_valid_reshape, y_valid)))
```
    Accuracy: 0.8805


- RandomForest를 사용하여 생성한 모델의 검증 결과는 Accuracy 0.8805 입니다.
- 어느정도 준수한 성능을 보여줍니다.


## 6. Deep Learning
---

### 6.1 Deep Learning - 데이터 처리


```python
y_train_one_hot = tf.keras.utils.to_categorical(y_train, 10)
y_valid_one_hot = tf.keras.utils.to_categorical(y_valid, 10)
```

- keras의 유틸을 사용하여, y 라벨을 one-hot encoding 해줍니다.
- One-Hot encodig이란 *9* 라고 표현된 라벨을 [0,0,0,0,0,0,0,0,0,1]의 벡터로 바꾸어주는 것입니다.
- Deep Learning은 각 클래스별로 확률을 출력해주기 때문에 꼭 필요한 작업 입니다.

### 6.2 Deep Learning - Multi Layer Perceptron Model 

```python
mlp_model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape = (28, 28)),
    tf.keras.layers.Dense(1000, activation = 'relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(800, activation = 'relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(500, activation = 'relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(300, activation = 'relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(200, activation = 'relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(100, activation = 'relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(50, activation = 'relu'),
    tf.keras.layers.Dense(10, activation = 'softmax'),
])
```

### 6.2 Deep Learning - Multi Layer Perceptron Model

- 처음 input size는 train 데이터인 (48000, 28, 28)입니다.
- 위의 28, 28 데이터를 평평하게 (flatten) 784로 바꾸어 줍니다.
- 히든 레이어에서 1000 -> 800 -> 500 -> 300 -> 200 -> 100 -> 50 개의 노드가 출력되고  마지막 레이어에서 10개의 노드를 출력합니다.  10개는 y라벨의 갯수 즉, class 입니다.
- 마지막 layer에 softmax 활성화 함수를 사용하여 10개의 노드에 대한 확률을 출력하고, 가장 높은값을 예측값으로 사용합니다.

### 6.3 Deep Learning - Multi Layer Perceptron Summary


    Model: "sequential_3"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    flatten_3 (Flatten)          (None, 784)               0         
    _________________________________________________________________
    dense_20 (Dense)             (None, 1000)              785000    
    _________________________________________________________________
    dropout_17 (Dropout)         (None, 1000)              0         
    _________________________________________________________________
    dense_21 (Dense)             (None, 800)               800800    
    _________________________________________________________________
    dropout_18 (Dropout)         (None, 800)               0         
    _________________________________________________________________
    dense_22 (Dense)             (None, 500)               400500    
    _________________________________________________________________
    dropout_19 (Dropout)         (None, 500)               0         
    _________________________________________________________________
    dense_23 (Dense)             (None, 300)               150300    
    _________________________________________________________________
    dropout_20 (Dropout)         (None, 300)               0         
    _________________________________________________________________
    dense_24 (Dense)             (None, 200)               60200     
    _________________________________________________________________
    dropout_21 (Dropout)         (None, 200)               0         
    _________________________________________________________________
    dense_25 (Dense)             (None, 100)               20100     
    _________________________________________________________________
    dropout_22 (Dropout)         (None, 100)               0         
    _________________________________________________________________
    dense_26 (Dense)             (None, 50)                5050      
    _________________________________________________________________
    dense_27 (Dense)             (None, 10)                510       
    =================================================================
    Total params: 2,222,460
    Trainable params: 2,222,460
    Non-trainable params: 0
    _________________________________________________________________


### 6.3 Deep Learning - Multi Layer Perceptron Summary

- layer는 각 층에서 행하는 type이고, output shape는 해당 층을 거치면 나오게 되는 output에 대한 size 입니다.
- Param은 해당 layer에서 가치는 파라미터 수이며,
- 파라미터의 수는 (입력 데이터 차원 + 1) * 뉴런 수로 계산합니다. 
- ex : dense = (784 +1) * 392 = 307720개

### 6.4 Deep Learning - Multi Layer Perceptron compile

```python
mlp_model.compile(optimizer = 'Adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

- loss : 손실함수, 모델의 최적화에 사용되는 목적함수입니다. 모델과 데이터 분류종류에 따라 MSE, categorical_crossentropy, sparse_categorical_crossentropy, binary_crossentropy 등 이 있습니다.
- optimizer : 최적의 가중치를 검색하는 데 사용되는 최적화 알고리즘. adam이 대표적입니다.
- metrics : 모델의 성능을 저장하는 지표를 의미하며, 이번에는 Accuracy를 사용합니다. 리스트 형태로 여러개도 저장가능합니다.

### 6.5 Deep Learning - Multi Layer Perceptron 학습

```python
checkpoint = ModelCheckpoint(filepath='model.weights.best.mlp.develop.hdf5', verbose=0, save_best_only=True)
earlystopping = EarlyStopping(monitor='val_loss', patience=50)
mlp_history = mlp_model.fit(X_train, y_train_one_hot, epochs=500, batch_size=500,
                            validation_split=0.2, callbacks=[checkpoint, earlystopping], verbose=0)
```

- batch : 전체 데이터셋 중 batch의 크기만큼 학습시키것을 의미합니다.
- epochs : 전체 데이터셋을 학습시키는 수입니다.
- ModelCheckpoint : epoch마다 성능이 좋아지면 모델을 저장합니다.(model.weights.best.mlp.develop.hdf5 라는 이름으로 저장됨)
- EarlyStopping : epoch 마다 설정한 성능이 좋아지지 않으면 자동으로 학습을 종료시킵니다. 위에서는 val_loss 이며, epoch 20회동안 나아지지 않으면 멈춥니다.
- validation_split : 입력받은 데이터의 20%를 사용하여 매 epochs 마다 검증을 실행합니다.

### 6.6 Deep Learning - Multi Layer Perceptron 학습시각화 코드

```python
fig, loss_ax = plt.subplots(figsize=(12, 12))
acc_ax = loss_ax.twinx()

loss_ax.plot(mlp_history.history['loss'], 'y', label='train loss')
loss_ax.plot(mlp_history.history['val_loss'], 'r', label='val loss')
loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
loss_ax.legend(loc='upper right')

acc_ax.plot(mlp_history.history['accuracy'], 'b', label='train acc')
acc_ax.plot(mlp_history.history['val_accuracy'], 'g', label='val acc')
acc_ax.set_ylabel('accuracy')
acc_ax.legend(loc='upper left')

plt.show()
```

### 6.6 Deep Learning - Multi Layer Perceptron 학습시각화
    
<img src = "https://user-images.githubusercontent.com/60168331/117663994-37adf180-b1dc-11eb-9caa-4f57bb0899a9.png">
    
- epoch가 늘어날때마다 학습 데이터의 loss는 줄어드는데, 검증 데이터의 loss는 점점 늘어나고 있습니다.
- 이는 학습이 진행될수록 점점 train 데이터에 과적합되어 다른 데이터는 잘 예측하지 못한다고 볼수 있습니다.
- 검증 loss가 줄지 않아 early stopping이 적용되어 약 80 epoch전에 학습이 종료된것으로 보입니다.


### 6.7 Deep Learning - Multi Layer Perceptron 예측

```python
mlp_model.load_weights('model.weights.best.mlp.develop.hdf5')
print('\nAccuracy: {:.4f}'.format(mlp_model.evaluate(X_valid, y_valid_one_hot)[1]))
```

    375/375 [==============================] - 0s 885us/step - loss: 0.3000 - accuracy: 0.8953
    
    Accuracy: 0.8953


- load_weights로 가장 학습이 잘된 모델의 가중치를 가져와서 검증을 해봅니다.
- Accuracy는 0.8953로 사실 RandomForest보다는 나은 성능을 보여줍니다.

### 6.8 Deep Learning - Convolution Neural Network 데이터 전처리

```python
X_train_cnn = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_valid_cnn = X_valid.reshape(X_valid.shape[0], 28, 28, 1)
X_test_cnn = X_test.reshape(X_test.shape[0], 28, 28, 1)
```

- cnn 모델에 넣기 위해 data를 reshape 해줍니다.
- 사진이미지를 넣을때 마지막에 1인 chanel입니다. Mnist Fashion image는 gray scale이므로 1채널이라, 마지막에 1을 붙여줍니다.

### 6.9 Deep Learning - Convolution Neural Network Model

```python
cnn_model = tf.keras.Sequential([
    keras.layers.Conv2D(filters = 64, kernel_size = 2, padding = 'same', activation = 'relu', input_shape = (28, 28, 1)),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(pool_size = 2),
    keras.layers.Dropout(0.3),
    
    keras.layers.Conv2D(filters = 64, kernel_size = 2, padding = 'same', activation = 'relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(pool_size = 2),
    keras.layers.Dropout(0.3),
    
    keras.layers.Conv2D(filters = 128, kernel_size = 2, padding = 'same', activation = 'relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(pool_size = 2),
    keras.layers.Dropout(0.3),
    
    keras.layers.Conv2D(filters = 128, kernel_size = 2, padding = 'same', activation = 'relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(pool_size = 2),
    keras.layers.Dropout(0.3),
    
    keras.layers.Flatten(), # Flaatten으로 이미지를 일차원으로 바꿔줌
    keras.layers.Dense(1024, activation = 'relu'),
    keras.layers.Dense(512, activation = 'relu'),
    keras.layers.Dense(256, activation = 'relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation = 'softmax')
])
```

### 6.9 Deep Learning - Convolution Neural Network Model

- **input_shape** : 샘플 수를 제외한 입력 형태를 정의 합니다. 모델에서 첫 레이어일 때만 정의하면 됩니다. (행, 열, 채널 수)로 정의합니다. 흑백영상인 경우에는 채널이 1이고, 컬러(RGB)영상인 경우에는 채널을 3으로 설정합니다.
- **activation** : 활성화 함수를 설정
    - linear : 디폴트 값, 입력뉴런과 가중치로 계산된 결과값이 그대로 출력으로 나옵니다.
    - relu : rectifier 함수, 은익층에 주로 쓰입니다. 0이하는 0으로 만들고 그 이상은 그대로 출력합니다.
    - sigmoid : 시그모이드 함수, 이진 분류 문제에서 출력층에 주로 쓰입니다. 0 혹은 1로 출력합니다.
    - softmax : 소프트맥스 함수, 다중 클래스 분류 문제에서 출력층에 주로 쓰입니다. 0과 1사이의 값으로 출력되며, 모두 더한값은 1이 되므로, 확률처럼 사용합니다.
- **filter(kernel)** : 이미지의 특징을 찾기위한 파라미터, 해당 filter가 이미지를 움직이며 특징을 잡아냄, 해당 특징이 featuremap, filter의 종류에 따라 가로선 filter, 세로선 filter등이 있는데 cnn에선 해당 필터를 자동으로 생성함
- **featuremap** : input 이미지에서 filter로 만들어진 해당 이미지의 특성을 가진 map
- **filters** : input 이미지에서 featuremap을 생성 하는 filter의 갯수
- **padding** : 외곽의 값을 0으로 채워넣어서 filter들로 만들어진 featuremap 기존의 이미지의 크기와 같게 할지의 여부 same은 같게, valid는 다르게, same으로 하면 filter가 이미지 사이즈에 맞게 featuremap을 만듬.
- **pooling** : 계속 filter가 이미지를 움직이며 featuremap을 만들고 paddind이 same이라면 계속 같은 크기의 featuremap이 생성되게 됨. 이를 방지하기 위해 pooling 레이어를 거쳐 이미지 사이즈를 줄임, pool_size는 이미지에서 줄여지는 값
    - maxpooling : pooling 영역에서 가장 큰 값만 남기는것
    - averagepoolig : pooling 영역의 모든 데이터의 평균값을 구하여 남김
- **dropout** : 이미지의 일부분을 drop시켜 학습하는데 어려움을 줌, 이로 인해 과적함을 막을 수 있습니다.
- **flatten** : 앞에서 만든 (7, 7, 64)의 배열을 7 * 7 * 64하여  3136의 1줄의 배열로 평평하게 만드는것 입니다.
    - (2번의 pooling으로 이미지 사이즈가 작아짐, 28, 28, 64 -> 14, 14, 64 -> 7, 7, 64)
- **dense** : 평평한 데이터가 들어오면, 해당 데이터를 dense레이어를 지나 맨앞의 사이즈로 줄여줌, 마지막은 10개 사이즈가 나오고, 이를 softmax함수로 활성화하여 0 ~ 9까지의 클래스를 예측할수 있게 해줍니다.


### 6.9 Deep Learning - Convolution Neural Network summary

    Model: "sequential_4"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_4 (Conv2D)            (None, 28, 28, 64)        320       
    _________________________________________________________________
    batch_normalization_4 (Batch (None, 28, 28, 64)        256       
    _________________________________________________________________
    max_pooling2d_4 (MaxPooling2 (None, 14, 14, 64)        0         
    _________________________________________________________________
    dropout_23 (Dropout)         (None, 14, 14, 64)        0         
    _________________________________________________________________
    conv2d_5 (Conv2D)            (None, 14, 14, 64)        16448     
    _________________________________________________________________
    batch_normalization_5 (Batch (None, 14, 14, 64)        256       
    _________________________________________________________________
    max_pooling2d_5 (MaxPooling2 (None, 7, 7, 64)          0         
    _________________________________________________________________
    dropout_24 (Dropout)         (None, 7, 7, 64)          0         
    _________________________________________________________________
    conv2d_6 (Conv2D)            (None, 7, 7, 128)         32896     
    _________________________________________________________________
    batch_normalization_6 (Batch (None, 7, 7, 128)         512       
    _________________________________________________________________
    max_pooling2d_6 (MaxPooling2 (None, 3, 3, 128)         0         
    _________________________________________________________________
    dropout_25 (Dropout)         (None, 3, 3, 128)         0         
    _________________________________________________________________
    conv2d_7 (Conv2D)            (None, 3, 3, 128)         65664     
    _________________________________________________________________
    batch_normalization_7 (Batch (None, 3, 3, 128)         512       
    _________________________________________________________________
    max_pooling2d_7 (MaxPooling2 (None, 1, 1, 128)         0         
    _________________________________________________________________
    dropout_26 (Dropout)         (None, 1, 1, 128)         0         
    _________________________________________________________________
    flatten_4 (Flatten)          (None, 128)               0         
    _________________________________________________________________
    dense_28 (Dense)             (None, 1024)              132096    
    _________________________________________________________________
    dense_29 (Dense)             (None, 512)               524800    
    _________________________________________________________________
    dense_30 (Dense)             (None, 256)               131328    
    _________________________________________________________________
    dropout_27 (Dropout)         (None, 256)               0         
    _________________________________________________________________
    dense_31 (Dense)             (None, 10)                2570      
    =================================================================
    Total params: 907,658
    Trainable params: 906,890
    Non-trainable params: 768
    _________________________________________________________________


### 6.9 Deep Learning - Convolution Neural Network compile

```python
cnn_model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])
```

- loss, optimizer, metrics는 MLP때와 동일 합니다.

### 6.9 Deep Learning - Convolution Neural Network 학습

```python
checkpointer = ModelCheckpoint(filepath='model.weights.best.cnn.hdf5', verbose=0, save_best_only=True)
earlystopping = EarlyStopping(monitor='val_loss', patience=50)
history = cnn_model.fit(X_train_cnn, y_train_one_hot, batch_size=500, epochs=500, verbose=0, validation_split=0.2, callbacks=[checkpointer, earlystopping])
```

- Epochs와 Batch_size를 지정해주고 CNN 모델을 학습시켰습니다.
- MLP 떄와 마찬가지로 Early Stopping 기능과 checkpoint 기능으로 성능이 개선되지않으면 학습을 멈추고, 가장 베스트 모델을 저장 시킵니다.


### 6.10 Deep Learning - Convolution Neural Network 학습 시각화 코드

```python
fig, loss_ax = plt.subplots(figsize=(12, 12))
acc_ax = loss_ax.twinx()

loss_ax.plot(history.history['loss'], 'y', label='train loss')
loss_ax.plot(history.history['val_loss'], 'r', label='val loss')
loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
loss_ax.legend(loc='upper right')

acc_ax.plot(history.history['accuracy'], 'b', label='train acc')
acc_ax.plot(history.history['val_accuracy'], 'g', label='val acc')
acc_ax.set_ylabel('accuracy')
acc_ax.legend(loc='upper left')

plt.show()
```

### 6.10 Deep Learning - Convolution Neural Network 학습 시각
    
<img src = "https://user-images.githubusercontent.com/60168331/117663996-38468800-b1dc-11eb-9042-a617940e4bb6.png">
    


- 앞에서 학습했던 MLP 모델과 비교하여 loss도 굉장히 잘 떨어지고, train과 valid간의 차이도 많이 없어보입니다.
- accuracy도 train과 valid간에 오르는것이 보입니다.

### 6.11 Deep Learning - Convolution Neural Network 검증

```python
cnn_model.load_weights('model.weights.best.cnn.hdf5')
print('\nAccuracy: {:.4f}'.format(cnn_model.evaluate(X_valid_cnn, y_valid_one_hot)[1]))
```

    375/375 [==============================] - 0s 1ms/step - loss: 0.2169 - accuracy: 0.9252
    
    Accuracy: 0.9252


- CNN의 성능이 여태까지의 모델중에 좋게나왔습니다.
- MLP도 Accuracy가 0.9를 넘지못했는데, CNN은 0.9252로 0.9의 Acuuracy를 넘었습니다.
- 확실히 이미지쪽은 CNN이 좋은듯 합니다.


### 6.10 Deep Learning - Convolution Neural Network 예측 후 저장

```python
cnn_predict = np.argmax(cnn_model.predict(X_test_cnn), axis=-1)
y_test_cnn = y_test.copy()
y_test_cnn['class'] = cnn_predict
y_test_cnn.to_csv('./Fashion_MNIST/keyonbit_cnn_test.csv', index = False)
```
- 마지막으로 CNN의 모델로 X_test 데이터를 예측 한뒤 cnn_test.csv에 저장합니다.


## 7. 결과 정리 및 회고
---

### 7.1 결과 정리 및 회고 - 결과 정리

- Decision Tree - Accuracy : 0.7898
- RandomForest - Accuracy : 0.8805
- Multi Layer Perceptron(MLP) - Accuracy : 0.8953
- Convolution Neural Network(CNN) - Accuracy : 0.9252
- 위 결과에서 알수 있듯이 CNN이 가장 좋은 성능을 보였으며 Loss, Accuracy 그래프도 굉장히 안정적으로 보여서 만족합니다.
- 저희는 CNN의 모델을 사용하여 예측한 결과를 제출할 예정이고, Test Accuracy도 검증 Accuracy와 비슷한 0.92 정도의 Accuracy를 예상합니다.
- ML 알고리즘 중 다른 (XGBoost, LGBM 등)도 사용하여 모델을 만들어보고 싶고, MLP, CNN도 Layer를 더 효율적으로 쌓아 성능 개선을 해보고 싶습니다.


### 7.2 결과 정리 및 회고 - 회고

- Mnist Fashion Image는 ML, DL을 공부하신다면 한번씩 해보길 추천하는 기초적인 이미지 데이터입니다.
- 머릿속에서 잘 정리가 되지않았던 ML, DL 관련 알고리즘과 명령어들을 한번씩 더 볼수 있고, 정리가 되었던 기회였습니다.
- 향후 ML, DL 관련 프로젝트를 진행할때 이번에 공부한 기초적인 내용이 많은 도움이 될수 있을듯 합니다.