# -HoVer-Net: segmentation and classification of cell nuclear & Graph construc-
## 1. 简介

HoVer-Net在乳腺癌数据集上的细胞分割分类，并基于细胞邻接关系建图

论文引用：https://www.nature.com/articles/s41467-023-42504-y  "Single-cell morphological and topological atlas reveals the ecosystem diversity of human breast cancer"

原文代码：https://github.com/fuscc-deep-path/sc_MTOP   
（本项目只展示部分代码）



## 2. 相关内容介绍

### 2.1 HoVer-Net介绍

HoVer-Net（Holistic Nested Network）是一种用于全自动病理图像细胞分割和分类的深度学习模型。它最初由 Simon Graham 等人 在 2019 年的论文 “HoVer-Net: Simultaneous Segmentation and Classification of Nuclei in Histopathology Images” 中提出，主要用于数字病理学，特别是癌症检测任务。

**HoVer-Net 可以同时完成 3 个任务：**

  - 细胞核分割（Nuclear Instance Segmentation）——检测并分割病理图像中的细胞核。
     
  - 细胞核分类（Nuclear Classification）——区分不同类型的细胞核，如肿瘤细胞、炎症细胞、上皮细胞等。
  
  - HoVer 分支（Horizontal-Vertical Distance Map Prediction）——预测细胞核的水平（H）和垂直（V）方向的距离图，从而提升分割精度。

---------------

**HoVer-Net的网络结构**

HoVer-Net 采用了**编码器-解码器**架构，其主要组成部分包括：

  - 主干网络（Backbone）：采用**预训练的 ResNet-50**提取深度特征。 

   - 三个解码头（Decoders）：
     
    	- 分割分支（Segmentation Branch）：输出像素级的细胞核掩码（mask）。
       
    	- HoVer 分支（HoVer Branch）：预测每个像素的水平和垂直位移，帮助分割重叠的细胞核。
       
    	- 分类分支（Classification Branch）：预测每个细胞核的类别。
    
---------------
** HoVer-Net关键创新
- 传统细胞分割方法难以区分相互粘连的细胞核，而HoVer-Net通过**预测细胞核的水平/垂直方向位移**来实现更精准的分割
- 以前的病理图像分割方法通常只能进行分割，而HoVer-Net能够同时进行分割和分类，提高了生物医学图像分析的自动化程度。

---------------
** HoVer-Net的局限性
尽管其在病理图像分割和分类任务上表现优异，但仍存在一些挑战：
- 需要大量高质量标注数据（人工标注很费时）
- 计算资源消耗大（尤其是高分辨率病理图像）
- hoVer机制可能对极端密集区域对细胞分割效果有限。
 
## 2. 实现

### 2.1 HoVer-Net分割分类
相关代码：```CellSegment.py```
#### 2.1.1 WSI读取与Mask处理
- 通过```get_file_handler```读取 WSI 文件（病理切片）。
- 根据 mask 文件判断是否已有组织区域分割结果，如果没有，则通过 Otsu 阈值法自动生成 mask。
```
gray = cv2.cvtColor(wsi_thumb_rgb, cv2.COLOR_RGB2GRAY)
_, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
mask = morphology.remove_small_objects(mask == 0, min_size=8 * 8, connectivity=2)
mask = morphology.remove_small_holes(mask, area_threshold=64 * 64)
mask = morphology.binary_dilation(mask, morphology.disk(8))
```
#### 2.1.2 HoVer-Net预测
- ```__get_raw_prediction(chunk_info_list, patch_info_list)``` 执行 HoVer-Net 推理，获取核分割预测。
- ```pred_inst.npy``` 存储实例分割结果，```pred_map.npy``` 存储预测的概率图。
#### 2.1.3 后处理
HoVer-Net 预测时，由于使用滑动窗口切片，导致核可能被截断，因此需要 3 轮修正：

（1）中央区域的处理：

  - 直接合并分割结果：
    ```
    self.wsi_inst_map[tile_tl[0] : tile_br[0], tile_tl[1] : tile_br[1]] = pred_inst
    ```

（2）边界区域修正

  - 通过对比已有的```roi_inst```（原来的实例分割）和```pred_inst```（新预测的分割）：
  ```
  roi_edge = np.concatenate(
      [roi_inst[[0, -1], :].flatten(), roi_inst[:, [0, -1]].flatten()]
  )
  ```
  - 确保边界实例不会被错误覆盖

（3）角落区域修正
  - 处理交叉区域，防止实例ID发生冲突：
    ```
    pred_inst[pred_inst > 0] += wsi_max_id
    ```

####2.1.4 结果
- 以```json```形式存储：
  ```
  self.__save_json(json_path, self.wsi_inst_info, mag=self.proc_mag)
  ```
  ![QQ_1739505467907](https://github.com/user-attachments/assets/229f46f2-c778-4bfc-9309-21cba7ecd1fa)

### 2.2 细胞形态学、纹理特征提取，细胞邻接关系建图
相关代码：```WSIGraph.py```
#### 2.2.1 形态学特征提取

主要通过 ```SingleMorphFeatures() ```和 ```getRegionPropFromContour()```完成，提取的特征包括：
- 面积 (Area)：```regionProps.area```
- 边界框面积 (BBox Area)：```regionProps.bbox_area```
- 偏心率 (Eccentricity)：```regionProps.eccentricity```
- 圆形度 (Circularity)
- 伸长率 (Elongation)
- Extent：Area/BBox Area
- 长轴、短轴 (MajorAxisLength, MinorAxisLength)
- 周长 (Perimeter)
- Solidity

此外，曲率 (getCurvature()) 计算了：
- 平均曲率、标准差、最大最小曲率
- 凸起/凹陷个数 ```(n_protrusion, n_indentation)```

实现方式：
	-	```getRegionPropFromContour()``` 通过 ```cv2.drawContours()``` 在二值图上绘制细胞轮廓，并用 ```regionprops()``` 计算区域属性。
 - ```getCurvature()``` 计算轮廓的曲率，利用二阶导数估算局部曲率特征。

#### 2.2.2 细胞纹理特征（GLCM纹理）
由 SingleGLCMFeatures() 计算灰度共生矩阵 (GLCM)，提取以下特征：
- ASM (Angular Second Moment)
- 对比度 (Contrast)
- 相关性 (Correlation)
- 熵 (Entropy)
- 均值、方差 (Average, Variance)
- 均匀性 (Homogeneity)
- 细胞内强度均值、标准差、最大最小值 (IntensityMean, IntensityStd, IntensityMax, IntensityMin)
实现方式：
- ```getCellMask()``` 生成对应的细胞二值掩码。
- ```skfeat.graycomatrix()``` 计算GLCM，```graycoprops()```提取特征。
- ```mygreycoprops()```计算熵、方差等自定义 GLCM 特征。

  #### 2.2.3 细胞图建模

由 getGraphDisKnnFeatures() 和 getSingleGraphFeatures() 计算基于细胞邻接关系的图特征，包括：
- 边长统计 (minEdgeLength, meanEdgeLength)
- 度 (Degree)
- 接近中心性 (Closeness)
- 介数中心性 (Betweenness)
- 核数 (Coreness)
- 离心率 (Eccentricity)
- 调和中心性 (HarmonicCentrality)
- 聚类系数 (Clustering Coefficient)

实现方式：
- ```getSingleGraphFeatures()```通过```subgraph.degree() ```计算图的基本属性

  #### 2.2.4 总结
  - 输入：HoverNet生成的细胞轮廓（contours）和边界框（bboxes）
  - 处理：计算形态学特征、纹理特征与基于细胞邻接的图特征
  - 输出：各类细胞特征字典，可用于生存分析和分类任务
  ![QQ_1739505277166](https://github.com/user-attachments/assets/1e51bc42-c483-45bf-bf5b-b94b00951759)


  
