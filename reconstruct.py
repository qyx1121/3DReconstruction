import os
import os.path as osp
import numpy as np
import onnxruntime as ort
from reconstruction.lab2im import edit_volumes, utils
from argparse import ArgumentParser
import pydicom
import dicom2nifti
import shutil


parser = ArgumentParser()
parser.add_argument("--path_images", type=str, default="example",
                    help="images to super-resolve / synthesize. Can be the path to a single image or to a folder")
parser.add_argument("--path_predictions", type=str, default="results",
                    help="path where to save the synthetic 1mm MP-RAGEs. Must be the same type "
                         "as path_images (path to a single image or to a folder)")
parser.add_argument("--gpu", action="store_true", help="enforce running with CPU rather than GPU.")
parser.add_argument("--dicom", action="store_true", help="whether the input image is dicom")
parser.add_argument("--output_folder", default="tmp", help="the output folder for saving the .nii converted from .dicom")
parser.add_argument("--keep_nii", action="store_true", help="whether to keep intermediate .nii files")
parser.add_argument("--output_type", default="nii", choices=["nii", "dicom"], help="the type of the output data, nii or dicom")

args = vars(parser.parse_args())
args['dicom'] = True
args['output_type'] = "dicom"
path_images = osp.abspath(args['path_images'])
basename = osp.basename(path_images)
path_predictions = osp.abspath(args['path_predictions'])
provider = 'CUDAExecutionProvider' if args['gpu'] else 'CPUExecutionProvider'
model = ort.InferenceSession("./reconstruction/model/SynthSR.onnx", providers=[provider])

if args["dicom"]:
    os.makedirs(args['output_folder'], exist_ok=True)
    dicom_paths = [osp.join(args['path_images'], i) for i in os.listdir(args['path_images'])]
    images_to_segment = []
    headers = []
    if osp.isdir(dicom_paths[0]): ### 路径为dicom文件夹
        path_predictions = [osp.join(path_predictions, osp.basename(dcm_p) + "_Reconsturct.nii") for dcm_p in dicom_paths]
        for dcm_p in dicom_paths:
            dicom2nifti.convert_directory(dcm_p, args['output_folder'])
            dicom_files = [osp.join(dcm_p, i) for i in os.listdir(dcm_p)]
            dcm_f = dicom_files[0]
            hdr = pydicom.dcmread(dcm_f, stop_before_pixels=True)
            headers.append(hdr)
    else:
        dicom2nifti.convert_directory(args['path_images'], args['output_folder'])
        path_predictions = [osp.join(path_predictions, osp.basename(args['path_images']) + "_Reconsturct.nii")]
        dicom_files = [osp.join(args['path_images'], i) for i in os.listdir(args['path_images'])]
        dcm_f = dicom_files[0]
        hdr = pydicom.dcmread(dcm_f, stop_before_pixels=True)
        headers.append(hdr)

    images_to_segment = [osp.join(args['output_folder'], i) for i in os.listdir(args['output_folder'])]

else:
    if ('.nii.gz' not in basename) & ('.nii' not in basename) & ('.mgz' not in basename) & ('.npz' not in basename):
        if osp.isfile(path_images):
            raise Exception('extension not supported for %s, only use: nii.gz, .nii, .mgz, or .npz' % path_images)
        images_to_segment = utils.list_images_in_folder(path_images)
        utils.mkdir(path_predictions)
        path_predictions = [osp.join(path_predictions, osp.basename(image)).replace('.nii', '_Reconsturct.nii') for
                            image in images_to_segment]
        path_predictions = [seg_path.replace('.mgz', '_Reconstruct.mgz') for seg_path in path_predictions]
        path_predictions = [seg_path.replace('.npz', '_Reconstruct.npz') for seg_path in path_predictions]

    else:
        assert osp.isfile(path_images), "files does not exist: %s " \
                                            "\nplease make sure the path and the extension are correct" % path_images
        images_to_segment = [path_images]
        path_predictions = [path_predictions]

print('Found %d images' % len(images_to_segment))
for id, (path_image, path_prediction) in enumerate(zip(images_to_segment, path_predictions)):
    print('  Working on image %d ' % (id + 1))
    
    if not args['dicom']:
        print('  ' + path_image)
        img_name = osp.basename(path_image)
    im, aff, hdr = utils.load_volume(path_image, im_only=False, dtype='float')
    vols = min(im.shape)

    if vols >= 150:
        #3D数据不需要三维重建
        utils.save_volume(im, aff, None, path_prediction)
    else:
        # im表示图像数组
        # aff表示图像在参考空间的位置
        # hdr表示图像头部信息
        im, aff = edit_volumes.resample_volume(im, aff, [1.0, 1.0, 1.0]) # reshape到 1mm x 1mm x 1mm

        im, aff2 = edit_volumes.align_volume_to_ref(im, aff, aff_ref=np.eye(4), return_aff=True, n_dims=3)
        im = im - np.min(im)
        im = im / np.max(im)
        I = im[np.newaxis, ..., np.newaxis]
        W = (np.ceil(np.array(I.shape[1:-1]) / 32.0) * 32).astype('int')  # 32的倍数
        idx = np.floor((W - I.shape[1:-1]) / 2).astype('int')
        S = np.zeros([1, *W, 1])
        S[0, idx[0]:idx[0] + I.shape[1], idx[1]:idx[1] + I.shape[2], idx[2]:idx[2] + I.shape[3], :] = I
        output = model.run(None, {"unet_input":S.astype(np.float32)})

        pred = np.squeeze(output)
        pred = 255 * pred
        pred[pred < 0] = 0
        pred[pred > 128] = 128
        pred = pred[idx[0]:idx[0] + I.shape[1], idx[1]:idx[1] + I.shape[2], idx[2]:idx[2] + I.shape[3]]

        dcm_hdr = headers[id] if args['output_type'] == "dicom" else None
        utils.save_volume(pred, aff2, hdr, path_prediction, args['output_type'], dcm_hdr = dcm_hdr)

if args['dicom'] and not args['keep_nii']:
    shutil.rmtree(args['output_folder'])