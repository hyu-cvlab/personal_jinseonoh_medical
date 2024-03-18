# import dicom2nifti
import os
import shutil
import pydicom
import numpy as np
###소희님 코드에서 수정안됨
# dicom2nifti.convert_directory(dicom_directory, output_folder, compression=True, reorient=True)
def CenterCrop(image, output_size):

    # pad the sample if necessary
    if image.shape[0] <= output_size[0] or image.shape[1] <= output_size[1]:
        pw = max((output_size[0] - image.shape[0]) // 2 + 3, 0)
        ph = max((output_size[1] - image.shape[1]) // 2 + 3, 0)
#         pd = max((output_size[2] - label.shape[2]) // 2 + 3, 0)
        image = np.pad(image, [(pw, pw), (ph, ph)],
                       mode='constant', constant_values=0)
#         label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)],
#                        mode='constant', constant_values=0)

    (w, h) = image.shape

    w1 = int(round((w - output_size[0]) / 2.))
    h1 = int(round((h - output_size[1]) / 2.))
#     d1 = int(round((d - output_size[2]) / 2.))

    image = image[w1:w1 + output_size[0], h1:h1 +
                                                  output_size[1]]

    return image#, label

# dicom_directory = '/home/has/Results/dcm_list/case_000/ANO_0087'
dicom_directory = '/data/hanyang_Prostate/50_example/original/rawData_dcm_51-300/51-100'
# dicom_directory = '/home/has/Datasets/CV_CT/slice_5'
dicom_folder = next(os.walk(dicom_directory))[1]
dicom_folder.sort()
# print(dicom_folder)
"""
['00856195', '00872925', '01072851', '01089153', '01091012', '01135944', '01156496', '01390654', '01483511', '01501178', '01524803', '01583741', '01604829', '01615505', '01900496', '01954399', '01985402', '02016760', '02036673', '02040923', '02046067', '02131577', '02174749', '02197251', '02212889', '02213631', '02302519', '02466559', '02595553', '02601355', '02655698', '02749848', '02753249', '02794471', '02801588', '02952212', '02953294', '02956329', '02957614', '02957615', '02957616', '02957618', '02958197', '02958407', '02958409', '02959028', '02959029', '02959981', '02959982', '02960707']
"""

rst_path = '/data/vnet_30000_fold4/input_dcm/51-100'
if os.path.isdir(rst_path) == False:
    os.makedirs(rst_path)
count_patient = 1
for case in dicom_folder:
    print('case',count_patient,":", case)
    dicom_f1_n = next(os.walk(os.path.join(dicom_directory, case)))[1]
    dicom_f1 = os.path.join(dicom_directory, case, dicom_f1_n[0])
#     print("dicom_f1_n",dicom_f1_n)  # dicom_f1_n ['0005_20220216_083153']


    input_path = dicom_f1
    #input_path = os.path.join (dicom_f2 , input_path [:])
#     print('input_path', input_path)  # input_path /data/hanyang_Prostate/50_example/original/rawData_dcm_51-300/51-100/00856195/0005_20220216_083153
    
    file_list = sorted(os.listdir(input_path))[461:589]
#     print("file_list\n",len(file_list),file_list)
#     print('file_list[0]', os.path.join(input_path, file_list[0]))  # data/hanyang_Prostate/50_example/original/rawData_dcm_51-300/51-100/00856195/0005_20220216_083153/IN_00001_0000275404.dcm
    # 디렉토리 내의 모든 DICOM 파일 경로를 리스트로 가져오기
#     dicom_files = [os.path.join(input_path, file) for file in file_list]

    # ImagePositionPatient 값들을 저장할 리스트
    image_positions = []
    initial_z = 0
    slice_gap = 0.8
    # DICOM 파일 읽기
#     dicom_data = pydicom.dcmread(os.path.join(input_path, file_list[461]))  # 426~625 [200] -> 461 ~ 589[128]

    # DICOM 파일들을 읽어서 ImagePositionPatient 값을 가져오기
    for file in file_list:
        file_path = os.path.join(input_path, file)
        print('file_path : ', file_path)
        dicom_data = pydicom.dcmread(file_path)
#         print("dicom_data.ImagePositionPatient[2]",dicom_data.ImagePositionPatient[2])
#         dicom_data.ImagePositionPatient[2] = initial_z * slice_gap
#         print("dicom_data.ImagePositionPatient[2]",dicom_data.ImagePositionPatient[2])
        image_positions.append(dicom_data.ImagePositionPatient[2])  # Z-axis value

        # Pixel Data 가져오기 (HU 조정을 위해)
        image_array = dicom_data.pixel_array.astype(np.int16)  # 픽셀 데이터를 int16 타입으로 변환
        print('image_array.shape', image_array.shape)
        print('print(np.min(image_array), np.max(image_array)):',np.min(image_array), np.max(image_array))

        image_array = np.where(image_array < -100, -100, image_array)
        image_array = np.where(image_array > 200, 200, image_array)

        print('print(np.min(image_array), np.max(image_array)):',np.min(image_array), np.max(image_array))

        # 현재 이미지의 크기 가져오기
        height, width = image_array.shape

        # center crop할 크기
        crop_width = crop_height = 256

        # center crop을 위한 좌표 계산
        left = (width - crop_width) // 2
        top = (height - crop_height) // 2
        right = left + crop_width
        bottom = top + crop_height

        # 이미지를 center crop하여 새로운 크기로 변경
        image_array = image_array[top:bottom, left:right]


        # 새로운 DICOM 객체 생성
    #     new_dicom = pydicom.Dataset(dicom_data)  # 기존 DICOM 데이터 복사

        # PixelData 및 이미지 크기 조정
    #     new_dicom.PixelData = cropped_image.tobytes()
    #     new_dicom.Rows, new_dicom.Columns = crop_height, crop_width

    # #     image_array = CenterCrop(image_array, output_size=[256,256])
    #     print("cropped_image.shape", cropped_image.shape)

        # 수정된 픽셀 데이터를 DICOM 객체에 설정
    #     image_array = image_array.astype(np.int16)  # 픽셀 값을 다시 int16 타입으로 변환
#         dicom_data.PixelData = image_array.tobytes()
        dicom_data.Rows, dicom_data.Columns = crop_height, crop_width




        # 현재 Image Orientation 값 확인
        current_image_orientation = dicom_data.ImageOrientationPatient
        print("현재 Image Orientation 값:", current_image_orientation)
        # x, y 축 반전
        flipped_image = np.flipud(np.fliplr(image_array))

        # DICOM 파일의 헤더 정보로부터 RAI 정보 가져오기
#         orientation = dicom_data.ImageOrientationPatient
        row_cosine = np.array(current_image_orientation[:3])
        column_cosine = np.array(current_image_orientation[3:])

        # RAI 방향이 유지되도록 이미지 데이터 반전
        if row_cosine[0] < 0:
            flipped_image = np.flipud(flipped_image)
        if column_cosine[1] < 0:
            flipped_image = np.fliplr(flipped_image)

        # DICOM 파일에 이미지 데이터 적용
        dicom_data.PixelData = flipped_image.tobytes()

#         # 변경할 Image Orientation 값 설정 (예시: 상하 및 좌우 대칭)
#         new_image_orientation = [-1.0, 0.0, 0.0, 0.0, -1.0, 0.0]  # 상하 및 좌우 대칭

#         # Image Orientation 값을 변경
#         dicom_data.ImageOrientationPatient = new_image_orientation

        # 변경된 Image Orientation 값 확인
        print("변경된 Image Orientation 값:", dicom_data.ImageOrientationPatient)

        # 현재 Image Position 값 확인
        current_image_position = dicom_data.ImagePositionPatient
        print("현재 Image Position 값:", current_image_position)

        # 변경할 Image Position 값 설정 (예시: 1, 1, 1)
        new_image_position = [0, 0, 0]

        # Image Position 값을 변경
        dicom_data.ImagePositionPatient = new_image_position

        # 변경된 Image Position 값 확인
        print("변경된 Image Position 값:", dicom_data.ImagePositionPatient)

        # 현재 Spacing 값 확인
        current_spacing = dicom_data.PixelSpacing
        print("현재 Spacing 값:", current_spacing)

        # 변경할 Spacing 값 설정 (예시: 1mm 간격)
        new_spacing = [0.8, 0.8]

        # Spacing 값을 변경
        dicom_data.PixelSpacing = new_spacing
        # 변경된 Spacing 값 확인
        print("변경된 Spacing 값:", dicom_data.PixelSpacing)

        dicom_data.save_as(os.path.join(rst_path, case, file))
        print('chch')
#         initial_z += 1
    count_patient += 1
    # ImagePositionPatient 값들을 정렬
    image_positions.sort()

    # 슬라이스 간격 계산
    slice_thicknesses = [abs(image_positions[i] - image_positions[i-1]) for i in range(1, len(image_positions))]

    # SliceThickness 값들 출력
    print("Slice Thicknesses:", slice_thicknesses)


    output_f_path = os.path.join(rst_path,'{}'.format((case)))
#     print('output_f_path', output_f_path)  # output_f_path /data/vnet_30000_fold4/input_dcm/51-100/00856195
    if os.path.isdir(output_f_path)==False:
        os.makedirs(output_f_path)

    # output_path = os.path.join(output_f_path, 'imaging.nii.gz')
    # print(output_path)
#     dicom2nifti.convert_directory(input_path, output_f_path, compression=True, reorient=False)
    # dicom2nifti.convert_directory(dicom_directory, output_path, compression=True, reorient=True)
    # icom2nifti.dicom_series_to_nifti(dicom_directory +'/'+dicom_list, output_path, reorient_nifti=True)

    output_before = next(os.walk(output_f_path))[2]
    print('output_before',output_before)
    output_before_path = os.path.join(output_f_path, output_before[0])
    print('output_before_path',output_before_path)
    # print(output_before_path)
#     os.rename(output_before_path, os.path.join(rst_path,'{}.nii.gz'.format(case)))
    # print(output_f_path)
    print("")
    





# for dicom_list in dicom_folder:
#     print(dicom_list)
#     output_path = '/home/has/Datasets/(has)CT_nii/case_' + '{0:05d}'.format(dicom_folder.index(dicom_list))
#     os.makedirs(output_path)
#     input_path = dicom_directory + '/' + dicom_list
#     print(input_path)
#     dicom2nifti.convert_directory(input_path, output_path, compression=True, reorient=True)
#     # dicom2nifti.convert_directory(dicom_directory, output_path, compression=True, reorient=True)
#     # icom2nifti.dicom_series_to_nifti(dicom_directory +'/'+dicom_list, output_path, reorient_nifti=True)

import cv2
import os
import numpy as np
import nibabel as nib
import monai
import torch
from scipy.ndimage import binary_erosion, binary_dilation


def CenterCrop(image, label, output_size):

    # pad the sample if necessary
    if label.shape[0] <= output_size[0] or label.shape[1] <= output_size[1] or label.shape[2] <= \
            output_size[2]:
        pw = max((output_size[0] - label.shape[0]) // 2 + 3, 0)
        ph = max((output_size[1] - label.shape[1]) // 2 + 3, 0)
        pd = max((output_size[2] - label.shape[2]) // 2 + 3, 0)
        image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)],
                       mode='constant', constant_values=0)
        label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)],
                       mode='constant', constant_values=0)

    (w, h, d) = image.shape

    w1 = int(round((w - output_size[0]) / 2.))
    h1 = int(round((h - output_size[1]) / 2.))
    d1 = int(round((d - output_size[2]) / 2.))

    label = label[w1:w1 + output_size[0], h1:h1 +
                                                  output_size[1], d1:d1 + output_size[2]]
    image = image[w1:w1 + output_size[0], h1:h1 +
                                                  output_size[1], d1:d1 + output_size[2]]

    return image, label

def erode_dilate_3d(image, iterations=1):
    eroded = binary_erosion(image, iterations=iterations)
    dilated = binary_dilation(eroded, iterations=iterations)
    return dilated

def dilate_erode_3d(image, iterations=1):
    dilated = binary_dilation(image, iterations=iterations)
    eroded = binary_erosion(dilated, iterations=iterations)
    return eroded



##trim된 데이터의 image
data_path = '/data/hanyang_Prostate/50_example/image_nifti_wholebody/1-50'#'/home/psh/data/hanyang_Prostate/50_example/image_nifti_wholebody/1-50'
## PZ trim
label_path = '/data/hanyang_Prostate/50_example/trim/data_pre_before/label_nii_class1_trim'
## TZ trim
# label_path = '/data/hanyang_Prostate/50_example/trim/data_pre_before/label_nii_class2_trim'#'/home/psh/data/hanyang_Prostate/50_example/trim/data_pre_before/label_nii_class2_trim'

data_file_list = sorted(os.listdir(data_path))
label_file_list = sorted(os.listdir(label_path))


for k in range(len(data_file_list)):

    ct_path = os.path.join(data_path, data_file_list[k])
    ct = nib.load(ct_path)           # w,h,d
    affine = ct.affine
    ct = ct.get_fdata()

    mask_path = os.path.join(label_path, label_file_list[k])
    mask = nib.load(mask_path)      # w,h,d
    # affine_mask = mask.affine
    mask = mask.get_fdata()             ##imgae , GT 크기 맞나 확인

    ## img & label 한 쌍인지 확인
    ct_name = ct_path.split('/')[-1].split('.')[0]
    mask_name = mask_path.split('/')[-1].split('_')[1]
    if ct_name != mask_name :
        print(ct_path)
        continue

    ct = np.transpose(ct, (2, 0, 1))        # d,w,h
    mask = np.transpose(mask, (2, 0, 1))        # d,w,h

    # if affine_mask.any() != affine.any() :
    #    print('stop')

    # ct data z modi
    ct = ct[:200, :, :]
    mask = mask[:200, :, :]

    d, w, h = int(ct.shape[0]),int(ct.shape[1]),int(ct.shape[2])

    print('{}, d:{}, w:{}, h:{}'.format(data_file_list[k], d, w, h))

    # if c < 208:
    #    z_padding = 208-c
    #    ct = np.pad(ct, ((0, z_padding), (0, 0), (0, 0)), 'constant')
    #    mask = np.pad(mask, ((0, z_padding), (0, 0), (0, 0)), 'constant')

#     ## HU 조정 (-100 ~ 200)
    for i in range(len(ct)):
        ct[i] = np.where(ct[i] < -100, -100, ct[i])
        ct[i] = np.where(ct[i] > 200, 200, ct[i])

#     # 원하는 Window Level 및 Window Width 설정
#     window_level = 45
#     window_width = 450

#     # Window 설정 범위 계산
#     lower_bound = window_level - window_width / 2
#     upper_bound = window_level + window_width / 2

#     # HU 값을 Window 범위 내로 조정
#     ct = np.where(ct < lower_bound, lower_bound, ct)
#     ct = np.where(ct > upper_bound, upper_bound, ct)

#     # Min-Max 정규화
#     ct = (ct - lower_bound) / (upper_bound - lower_bound)

    ## normalization
#     ct = (ct - ct.min()) / (ct.max() - ct.min())
    ## standard deviation
    #ct = (ct - np.mean(ct)) / np.std(ct)  ##normalize
    ct = ct.astype(np.float32)
    print(ct.min(), ct.max())


    ## crop center
    ct,mask = CenterCrop(ct,mask,output_size=[200,350,350])
    morph_mask = dilate_erode_3d(mask, iterations=5)
    morph_mask = (morph_mask > 0).astype(np.uint8)  # 이진화 및 데이터 타입 변경


    print('{}, d:{}, w:{}, h:{}'.format(data_file_list[k], ct.shape[0], ct.shape[1], ct.shape[2]))

    c_path = '/data/hanyang_Prostate/50_example/trim/sl_data_wo_norm_morphology_5/centerCrop_350_350_200/image'
    m_path = '/data/hanyang_Prostate/50_example/trim/sl_data_wo_norm_morphology_5/centerCrop_350_350_200/label_trim'#'/home/psh/data/hanyang_Prostate/50_example/trim/ssl_data/centerCrop_200/label_trim'

    if not os.path.exists(c_path):
        os.makedirs(c_path)
    if not os.path.exists(m_path):
        os.makedirs(m_path)

    c_path2 = c_path + '/{}'.format((data_file_list[k]))
    m_path2 = m_path + '/{}.nii.gz'.format((label_file_list[k].split('_')[1]))

    ct_t = np.transpose(ct, (1, 2, 0))
    mask_t = np.transpose(morph_mask, (1, 2, 0))

    ct_t = nib.Nifti1Image(ct_t, affine)
    mask_t = nib.Nifti1Image(mask_t, affine)
    nib.save(ct_t, c_path2)
    nib.save(mask_t, m_path2)
    print('__________________________________________')





