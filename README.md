# 3D Reconstruction

本仓库主要用于2D-T1图像到3D-T1图像的三维重建

## 环境搭建
本项目基于Python环境，运行下述命令安装所需python库
```
pip install -r requirements.txt
```

## 运行
支持dicom数据格式以及nii数据格式的MRI影像的三维重建
### nii格式
如果输入为nii或者nii.gz数据，则需要将要转换的数据组织在一个目录下，如：
```
nii_dir
├── file_1.nii.gz
├── file_2.nii.gz
├── file_3.nii
├── file_4.nii
└── ...
```
然后用如下命令运行重建：
```
python reconstruct.py \
--path_images ./nii_dir # 需要重建的nii或nii.gz图像目录 \
--path_predictions ./results # 保存重建后的结果的目录 \
--gpu # 是否使用GPU进行推理，如果不指定，则默认使用CPU，GPU的对单个MRI的处理速度比CPU快5s左右 \
```

### dicom格式
如果输入为dicom格式，需要将数据组织成如下形式：
```
├── dicom_dir
│   ├── dicom_1
│   │   ├── 00001.dcm
│   │   ├── 00002.dcm
│   │   ├── 00003.dcm
│   │   ├── 00004.dcm
│   │   ├── 00005.dcm
│   │   ├── 00011.dcm
│   │   └── ...
│   └── dicom_2
│       ├── IM000000
│       ├── IM000001
│       ├── IM000002
│       ├── IM000003
│       ├── IM000004
│       ├── IM000005
│       ├── ...

```
如果输入是dicom数据，需要在运行脚本时注明几个参数：
```
python reconstruct.py \
--path_images ./dicom_dir # 需要重建的dicom图像目录 \
--path_predictions ./results # 重建后的图像的保存目录 \
--gpu # 是否使用GPU进行推理，如果不指定，则默认使用CPU，GPU的对单个MRI的处理速度比CPU快5s左右 \
--dicom # 表明现在输入dicom格式的数据，仅在转换dicom数据的时候指定
--output_folder ./tmp # 保存中间过程中将dicom转换为nii的暂存目录
--keep_nii # 表明保留生成的中间nii文件，若不指定该参数，则默认不保留
```
