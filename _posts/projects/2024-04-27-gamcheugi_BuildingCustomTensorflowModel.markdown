---
layout: post
title:  "Tensorflow를 이용한 모델 구현"
date:   2024-04-27 01:15:16 +0900
categories: projects
tags: gamcheugi computervision
---

논문 링크 = <a href=https://www.frontiersin.org/articles/10.3389/frai.2021.796825/full>LINK TO SOURCE</a>


구현체가 공개되지 않은 논문의 내용을 기반으로 Tensorflow를 이용해 CNN 모델을 구현해보았다. 
<br>
![alt text]({{"/assets/images/2024-04-27-BuildingTensorflowModel/0.PNG" | relative_url}})  
<br>
모델은 (244, 244, 3), (244, 244, 1) 짜리 두 개의 이미지 인풋 채널을 활용하였고, Feature Extraction을 위해 Pretrained VGG16 Layer를 사용하고, Prediction Space의 좌표로 매핑하는 구조다. 

Prediction Space는, GazeCapture 데이터셋을 발표한 "Eye Tracking for Everyone(2016)에서 발표한 개념으로, 
PoG의 Label을 실제 물리적인 컴퓨터 화면에 매핑하는 것이 아니라, 해당 프레임을 촬영한 카메라의 위치를 원점(0,0) 으로 놓는 가상의 평면이다. 이 경우 각 기종 별로 해상도 차이에 상관 없이 동일한 Answer space를 가질 수 있는 장점이 있다. 

(사실 논문을 처음 읽고 한참 구현 중간에 가서 깨달은 개념이다. 처음에는 그냥 "마법처럼" 화면 좌표로 매핑되는 것인줄 알았다)

레이어가 어쨌든 적고, 10 epoch만 학습했을 때 0.12cm의 SOTA에 가까운 정확도에 반해서 해당 논문을 토대로 구현해보기로 했다. 

<h3>1</h3>
![alt text]({{"/assets/images/2024-04-27-BuildingTensorflowModel/model_diagram.png" | relative_url}})  

모든 레이어 별로 상세한 parameter가 적혀 있지는 않았고, 모든 개별 레이어가 표시된 다이어그램도 없어서 논문의 대략적인 설명을 듣고 구현한 결과 Trainable Parameter가 1억개가 넘어가는 거대한 모델이 되어버렸다. (하다 못해 Trainable Parameter 개수라도 나와 있었다면 대조가 가능했었으나 다른 단서가 없었다.)
당장 가용 가능한 수단이 Colab밖에 없어서 논문에 나온 묘사를 직접적으로 반하지 않는 선에서 파라미터를 줄여서 총 329,506개로 줄인 버전으로 테스트 해보기로 했다. 
Loss는 논문에서 사용한대로 MSE이다. 

모델 학습이 제대로 이루어지는지 샘플 700개를 넣어본 결과. 
![alt text]({{"/assets/images/2024-04-27-BuildingTensorflowModel/result1.PNG" | relative_url}})  
너무나도 당연한 얘기지만 Loss와 Val loss가 20만대에서 거의 동일한 것을 미루어보아 Underfitting 됐다는 지표. 추가 샘플을 넣었을 때 loss가 하락할지 궁금해서 추가로 확보한 자체 데이터 1700장 가량을 더 넣어서 진행해봤다. 

![alt text]({{"/assets/images/2024-04-27-BuildingTensorflowModel/result2.PNG" | relative_url}})  
본의 수십분의 일로 경량화를 한 모델인데 학습 자체가 진전한다는 사실에는 고무적이나 여전히 정확도가 매우 낮으므로 데이터를 더 확보해야 할 것. 이쯤되면 데이터 요구량이 자체 수집으로는 감당하기 힘든 정도이므로 외부 데이터가 무조건 필요해질 것. (당연한 얘기지만 파라미터가 35만개 가량인데 2300장으로 학습이 이루어지는 걸 바라는 거 자체가 말도 안되긴 한다)

더불어 Colab Pro의 T4 GPU 기준 7h 4m 50s 소요됨. 아마도 데이터 로딩하는 부분에서 심한 bottleneck이 있는 것으로 추정됨. 논문에서 사용했던 데이터셋 중 하나인 GazeCapture
데이터로 학습을 지속할 경우 데이터가 지금보다 훨씬 많아지므로, 더 효율적인 데이터 로딩이 필요할 것 같아 Tensorflow의 tf.data API를 이용해서 Load on-the-fly 구조를 만들었다. 

```python 
def parse_data(face_image_path, face_binary_path, gazepoint_path):
    """Load data based on data path."""
    # Convert EagerTensors to strings
    face_image_path = face_image_path.numpy().decode('utf-8')
    face_binary_path = face_binary_path.numpy().decode('utf-8')
    gazepoint_path = gazepoint_path.numpy().decode('utf-8')

    # Load the face image
    face_image = np.load(face_image_path)
    # Load the face binary and reshape
    face_binary = np.load(face_binary_path).reshape((244, 244, 1))
    # Load the gazepoint
    with open(gazepoint_path, 'r') as f:
        gazepoint = json.load(f)

    return face_image, face_binary, gazepoint


def tf_parse_function(face_image_path, face_binary_path, gazepoint_path):
    face_image, face_binary, gazepoint = tf.py_function(parse_data, [face_image_path, face_binary_path, gazepoint_path], [tf.float32, tf.float32, tf.float32])
    face_image.set_shape((244, 244, 3))
    face_binary.set_shape((244, 244, 1))
    gazepoint.set_shape((2,))
    return {"channel1_input": face_image, "channel2_input": face_binary}, gazepoint


def create_dataset(face_image_paths, face_binary_paths, gazepoint_paths, batch_size=16):
    face_image_paths = tf.constant(face_image_paths)
    face_binary_paths = tf.constant(face_binary_paths)
    gazepoint_paths = tf.constant(gazepoint_paths)

    # Create a Dataset from the file paths
    dataset = tf.data.Dataset.from_tensor_slices((face_image_paths, face_binary_paths, gazepoint_paths))

    # Apply interleaving to read multiple files in parallel
    dataset = dataset.interleave(
        lambda x, y, z: tf.data.Dataset.from_tensors((x, y, z)).map(tf_parse_function, num_parallel_calls=tf.data.AUTOTUNE),
        cycle_length=4,  # Number of input elements processed concurrently
        block_length=1,  # Number of consecutive elements to pull from an input element before moving on to the next input element
        num_parallel_calls=tf.data.AUTOTUNE  # Parallel calls for interleaving
    )

    # Cache the dataset to improve performance
    dataset = dataset.cache()

    # Shuffle the dataset
    dataset = dataset.shuffle(buffer_size=len(face_image_paths))

    # Batch the dataset
    dataset = dataset.batch(batch_size)

    # Prefetch the dataset to improve training performance
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset
```

<h4>텐서플로우 데이터 로딩 함수</h4>
tf.Data API에서 tf.Tensor 데이터형을 강제하기 때문에 형변환 문제 때문에 구현에 많은 시간이 걸렸다. 

<font color='red'>parse_data()</font>함수는 사전에 불러놓은 데이터 샘플 이름을 기반으로 데이터 객체를 로딩하는 함수이다.   
<font color='red'>tf_parse_function()</font>은 실제 prefetch나 caching등 로딩 효율을 위한 작업을 적용하는 <font color='red'>tf.py_function()</font>을 batch 단위로 적용하는데 이용되는 일종의 wrapper function이다. 
tf.py_function은 실제 tf.data.Dataset 객체를 생성하고 configure하는 역할을 한다.

한 문장으로 요약하면:  
tf.py_function을 tf.data.Dataset 객체를 생성하는데, initialization 과정에서 parse_data()를 통해 불러들인 데이터에 자동으로 tf_parse_function을 매핑해준다. 
tf.data.Dataset은 Parallel reading, Caching, Prefetching 등의 추가 작업을 통해 데이터를 읽는 속도를 높이게 된다. 

해당 Loading function을 적용한 이후, 이전에 약 7시간 걸렸던 학습 작업이 20분 내로 끝났다. 
<br>
<br>
<br>
<h4><b>GazeCapture 데이터셋</b></h4>
GazeCapture Dataset <a href="https://gazecapture.csail.mit.edu/download.php"></a>은 압축용량이 136GB, 2,445,504개의 프레임으로 이루어져 있다. 
참고 중인 논문에서는 <b>"알 수 없는 기준으로"</b> 이 중 약 0.65% 정도에 해당되는 15,960개만을 사용하였다. 논문 저자들이 대체 어떤 기준으로 필터링을 했는지는 모르겠지만, 일단은 육안으로 점검해서 너무 흐리게 나오거나, 아예 landmark를 찾을 수 없을 정도로 화면각에서 벗어난 피험자들을 삭제하고, dlib을 적용했을 때 Face에 검출되는 프레임들만 골라서 65,125장의 Dataset subset을 만들었다. 

이후 GazeCapture에서 추출한 65,125개의 Train dataset을 Google Drive에 (압축파일 째) 업로드했다. 

Unzipping하는 과정에서 런타임 디스크 용량이 부족해서 압축해제를 할 수가 없었다. 대신 임기응변으로 파이썬에 있는 ZipFile 패키지에 zip파일을 디렉토리처럼 다룰 수 있는 기능이 있어서 해당 패키지를 사용해 zip file에서 데이터를 불러올 수 있도록 DataLoader 코드를 조금 손을 보았다. 이렇게 zipfile을 그대로 다루는 거 자체도 어느정도 손해가 있을 것 같은데 나중에 파악해보기로 했다. 

<font color="red">
그리고 새벽 02:00시부터 다음날 12:20 자정(약 22시간)까지 돌았지만, 첫 번째 Epoch를 다 돌지 못하고, 약 40개의 Computing Unit을 소모한 끝에 런타임이 터져버렸다. 
</font>

이왕 데이터셋과 모델 소스코드를 받은 김에(멘탈도 추스를 겸) 자체 모델 개발은 잠깐 보류하고 iTracker 모델 포팅 작업을 해보기로 했다. 
<br>
<br>
<br>
<h4><b> 후기 및 개선사항</b></h4>

-> 데이터셋을 더 작은 단위로 나누어서 돌릴 것(1만 장 정도)
-> 새로 만든 콜백 함수 적용하기 (CustomCallback for Email notification)
