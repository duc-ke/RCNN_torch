# RCNN_torch
Object detection(물체탐지)분야의 RCNN(2013) ( Regions with Convolutional Neuron Networks features ) architectures를 pytorch로 구현함.

**본 repository는 @zhujian 님의 `object-detection-algorithm/R-CNN` [github](https://github.com/object-detection-algorithm/R-CNN)을 바탕으로 공부하여 약간의 코드 수정, 한글 및 영어 주석 및 코드 리펙토링을 가미하였음을 밝힙니다.**

RCNN 논문 'Rich feature hierarchies for accurate object detection and semantic segmentation'[(paper link)](https://arxiv.org/abs/1311.2524)과 거의 유사한 방식으로 학습하고 inference할 수 있도록 pytorch 프레임 워크로 구현하였으므로 이해하기 어려웠던 데이터준비 및 bounding box regression loss와 같은 부분을 좀더 명확히 이해할 수 있을 것임.
