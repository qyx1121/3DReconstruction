import os
from os.path import abspath

import glob
import numpy as np
import nibabel as nib
import pydicom as pyd

import nii2dcm.nii
import nii2dcm.svr
from nii2dcm.dcm_writer import (
    transfer_nii_hdr_series_tags,
    transfer_nii_hdr_instance_tags,
    transfer_ref_dicom_series_tags,
    write_slice
)



def list_images_in_folder(path_dir, include_single_image=True, check_if_empty=True):
    """List all files with extension nii, nii.gz, mgz, or npz within a folder."""
    basename = os.path.basename(path_dir)
    if include_single_image & \
            (('.nii.gz' in basename) | ('.nii' in basename) | ('.mgz' in basename) | ('.npz' in basename)):
        assert os.path.isfile(path_dir), 'file %s does not exist' % path_dir
        list_images = [path_dir]
    else:
        if os.path.isdir(path_dir):
            list_images = sorted(glob.glob(os.path.join(path_dir, '*nii.gz')) +
                                 glob.glob(os.path.join(path_dir, '*nii')) +
                                 glob.glob(os.path.join(path_dir, '*.mgz')) +
                                 glob.glob(os.path.join(path_dir, '*.npz')))
        else:
            raise Exception('Folder does not exist: %s' % path_dir)
        if check_if_empty:
            assert len(list_images) > 0, 'no .nii, .nii.gz, .mgz or .npz image could be found in %s' % path_dir
    return list_images

def mkdir(path_dir):
    """Recursively creates the current dir as well as its parent folders if they do not already exist."""
    if path_dir[-1] == '/':
        path_dir = path_dir[:-1]
    if not os.path.isdir(path_dir):
        list_dir_to_create = [path_dir]
        while not os.path.isdir(os.path.dirname(list_dir_to_create[-1])):
            list_dir_to_create.append(os.path.dirname(list_dir_to_create[-1]))
        for dir_to_create in reversed(list_dir_to_create):
            os.mkdir(dir_to_create)

def load_volume(path_volume, im_only=True, squeeze=True, dtype=None, aff_ref=None):
    """
    Load volume file.
    :param path_volume: path of the volume to load. Can either be a nii, nii.gz, mgz, or npz format.
    If npz format, 1) the variable name is assumed to be 'vol_data',
    2) the volume is associated with an identity affine matrix and blank header.
    :param im_only: (optional) if False, the function also returns the affine matrix and header of the volume.
    :param squeeze: (optional) whether to squeeze the volume when loading.
    :param dtype: (optional) if not None, convert the loaded volume to this numpy dtype.
    :param aff_ref: (optional) If not None, the loaded volume is aligned to this affine matrix.
    The returned affine matrix is also given in this new space. Must be a numpy array of dimension 4x4.
    :return: the volume, with corresponding affine matrix and header if im_only is False.
    """
    
    if isinstance(path_volume, nib.nifti1.Nifti1Image):
        x = path_volume
        aff = x.affine
        header = x.header
        if squeeze:
            volume = np.squeeze(np.asanyarray(x.dataobj))
        else:
            volume = np.asanyarray(x.dataobj)
    
    else:
        assert path_volume.endswith(('.nii', '.nii.gz', '.mgz', '.npz')), 'Unknown data file: %s' % path_volume
        if path_volume.endswith(('.nii', '.nii.gz', '.mgz')):
            x = nib.load(path_volume)
            if squeeze:
                volume = np.squeeze(np.asanyarray(x.dataobj))
            else:
                volume = np.asanyarray(x.dataobj)
            aff = x.affine
            header = x.header
        else:  # npz
            volume = np.load(path_volume)['vol_data']
            if squeeze:
                volume = np.squeeze(volume)
            aff = np.eye(4)
            header = nib.Nifti1Header()
        

    if dtype is not None:
        if 'int' in dtype:
            volume = np.round(volume)
        volume = volume.astype(dtype=dtype)

    # align image to reference affine matrix
    if aff_ref is not None:
        from . import edit_volumes_old  # the import is done here to avoid import loops
        n_dims, _ = get_dims(list(volume.shape), max_channels=10)
        volume, aff = edit_volumes_old.align_volume_to_ref(volume, aff, aff_ref=aff_ref, return_aff=True, n_dims=n_dims)

    if im_only:
        return volume
    else:
        return volume, aff, header


def save_volume(volume, aff, header, path, output_type, dcm_hdr = None, res=None, dtype=None, n_dims=3):
    """
    Save a volume.
    :param volume: volume to save
    :param aff: affine matrix of the volume to save. If aff is None, the volume is saved with an identity affine matrix.
    aff can also be set to 'FS', in which case the volume is saved with the affine matrix of FreeSurfer outputs.
    :param header: header of the volume to save. If None, the volume is saved with a blank header.
    :param path: path where to save the volume.
    :param res: (optional) update the resolution in the header before saving the volume.
    :param dtype: (optional) numpy dtype for the saved volume.
    :param n_dims: (optional) number of dimensions, to avoid confusion in multi-channel case. Default is None, where
    n_dims is automatically inferred.
    """

    mkdir(os.path.dirname(path))
    if '.npz' in path:
        np.savez_compressed(path, vol_data=volume)
    else:
        if header is None:
            header = nib.Nifti1Header()
        if isinstance(aff, str):
            if aff == 'FS':
                aff = np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
        elif aff is None:
            aff = np.eye(4)
        nifty = nib.Nifti1Image(volume, aff, header)
        if dtype is not None:
            if 'int' in dtype:
                volume = np.round(volume)
            volume = volume.astype(dtype=dtype)
            nifty.set_data_dtype(dtype)
        if res is not None:
            if n_dims is None:
                n_dims, _ = get_dims(volume.shape)
            res = reformat_to_list(res, length=n_dims, dtype=None)
            nifty.header.set_zooms(res)
        
        if output_type == "nii":
            nib.save(nifty, path)
        else:
            path = path.replace(".nii", "")
            os.makedirs(path, exist_ok=True)
            run_nii2dcm(nifty, path, header = dcm_hdr)

def get_dims(shape, max_channels=10):
    """Get the number of dimensions and channels from the shape of an array.
    The number of dimensions is assumed to be the length of the shape, as long as the shape of the last dimension is
    inferior or equal to max_channels (default 3).
    :param shape: shape of an array. Can be a sequence or a 1d numpy array.
    :param max_channels: maximum possible number of channels.
    :return: the number of dimensions and channels associated with the provided shape.
    example 1: get_dims([150, 150, 150], max_channels=10) = (3, 1)
    example 2: get_dims([150, 150, 150, 3], max_channels=10) = (3, 3)
    example 3: get_dims([150, 150, 150, 15], max_channels=10) = (4, 1), because 5>3"""
    if shape[-1] <= max_channels:
        n_dims = len(shape) - 1
        n_channels = shape[-1]
    else:
        n_dims = len(shape)
        n_channels = 1
    return n_dims, n_channels

def reformat_to_list(var, length=None, load_as_numpy=False, dtype=None):
    """This function takes a variable and reformat it into a list of desired
    length and type (int, float, bool, str).
    If variable is a string, and load_as_numpy is True, it will be loaded as a numpy array.
    If variable is None, this function returns None.
    :param var: a str, int, float, list, tuple, or numpy array
    :param length: (optional) if var is a single item, it will be replicated to a list of this length
    :param load_as_numpy: (optional) whether var is the path to a numpy array
    :param dtype: (optional) convert all item to this type. Can be 'int', 'float', 'bool', or 'str'
    :return: reformatted list
    """

    # convert to list
    if var is None:
        return None
    var = load_array_if_path(var, load_as_numpy=load_as_numpy)
    if isinstance(var, (int, float, np.int, np.int32, np.int64, np.float, np.float32, np.float64)):
        var = [var]
    elif isinstance(var, tuple):
        var = list(var)
    elif isinstance(var, np.ndarray):
        if var.shape == (1,):
            var = [var[0]]
        else:
            var = np.squeeze(var).tolist()
    elif isinstance(var, str):
        var = [var]
    elif isinstance(var, bool):
        var = [var]
    if isinstance(var, list):
        if length is not None:
            if len(var) == 1:
                var = var * length
            elif len(var) != length:
                raise ValueError('if var is a list/tuple/numpy array, it should be of length 1 or {0}, '
                                 'had {1}'.format(length, var))
    else:
        raise TypeError('var should be an int, float, tuple, list, numpy array, or path to numpy array')

    # convert items type
    if dtype is not None:
        if dtype == 'int':
            var = [int(v) for v in var]
        elif dtype == 'float':
            var = [float(v) for v in var]
        elif dtype == 'bool':
            var = [bool(v) for v in var]
        elif dtype == 'str':
            var = [str(v) for v in var]
        else:
            raise ValueError("dtype should be 'str', 'float', 'int', or 'bool'; had {}".format(dtype))
    return var

def load_array_if_path(var, load_as_numpy=True):
    """If var is a string and load_as_numpy is True, this function loads the array writen at the path indicated by var.
    Otherwise it simply returns var as it is."""
    if (isinstance(var, str)) & load_as_numpy:
        assert os.path.isfile(var), 'No such path: %s' % var
        var = np.load(var)
    return var

def run_nii2dcm(input_nii_path, output_dcm_path, dicom_type="MR", header = None, ref_dicom_file=None):
    """
    Execute NIfTI to DICOM conversion

    :param input_nii_path: input .nii/.nii.gz file
    :param output_dcm_path: output DICOM directory
    :param dicom_type: specified by user on command-line
    :param ref_dicom: reference DICOM file for transferring Attributes
    """

    # load NIfTI
    if isinstance(input_nii_path, str):
        nii = nib.load(input_nii_path)
    else:
        nii = input_nii_path

    # get pixel data from NIfTI
    # TODO: create method in nii class
    nii_img = nii.get_fdata()
    nii_img = nii_img.astype("uint16")  # match DICOM datatype

    # get NIfTI parameters
    nii2dcm_parameters = nii2dcm.nii.Nifti.get_nii2dcm_parameters(nii)

    # initialise nii2dcm.dcm object
    # --dicom_type specified on command line
    if dicom_type is None:
        dicom = nii2dcm.dcm.Dicom('nii2dcm_dicom.dcm')

    if dicom_type is not None and dicom_type.upper() in ['MR', 'MRI']:
        dicom = nii2dcm.dcm.DicomMRI('nii2dcm_dicom_mri.dcm')

    if dicom_type is not None and dicom_type.upper() in ['SVR']:
        dicom = nii2dcm.svr.DicomMRISVR('nii2dcm_dicom_mri_svr.dcm')
        nii_img = nii.get_fdata()
        nii_img[nii_img < 0] = 0  # set background pixels = 0 (negative in SVRTK)
        nii_img = nii_img.astype("uint16")

    # load reference DICOM object
    # --ref_dicom_file specified on command line
    ref_dicom = None
    if ref_dicom_file is not None:
        ref_dicom = pyd.dcmread(ref_dicom_file)
    elif header is not None:
        ref_dicom = header

    # transfer Series tags from NIfTI
    transfer_nii_hdr_series_tags(dicom, nii2dcm_parameters)

    # transfer tags from reference DICOM
    # IMPORTANT: this deliberately happens last in the DICOM tag manipulation process so that any tag values transferred
    # from the reference DICOM override any values initialised by nii2dcm
    if ref_dicom is not None:
        transfer_ref_dicom_series_tags(dicom, ref_dicom)

    """
    Write DICOM files
    - Transfer NIfTI parameters and write slices, instance-by-instance
    """
    print('nii2dcm: writing DICOM files ...')  # TODO use logger

    for instance_index in range(0, nii2dcm_parameters['NumberOfInstances']):

        # Transfer Instance tags
        transfer_nii_hdr_instance_tags(dicom, nii2dcm_parameters, instance_index)

        # Write slice
        write_slice(dicom, nii_img, instance_index, output_dcm_path)

    print(f'nii2dcm: DICOM files written to: {abspath(output_dcm_path)}')  # TODO use logger