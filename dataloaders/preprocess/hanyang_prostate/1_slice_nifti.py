import os
import nibabel as nib
import numpy as np

ori_directory = '/data/hanyang_Prostate/50_example/image_nifti_wholebody/1-50'
data_list = sorted(os.listdir(ori_directory))

rst_path = '/data/hanyang_Prostate/50_example/image_nifti_wholebody_2d/1-50'
if os.path.isdir(rst_path) == False:
    os.makedirs(rst_path)

for case in data_list:
    patient = case[:-7]
    # 원본 NIfTI 파일 로드
    input_file = os.path.join(ori_directory,case)
    img = nib.load(input_file)
    data = img.get_fdata()
    
    # 슬라이스할 z 축 수 설정
    slices = data.shape[2]  # 원하는 슬라이스 수

#     # 저장 디렉토리 설정
#     output_dir = 'sliced_nii_files'
#     os.makedirs(output_dir, exist_ok=True)
    output_dir = os.path.join(rst_path,patient)
    if os.path.isdir(output_dir) == False:
        os.makedirs(output_dir)
        
    # 각 슬라이스 추출 및 저장
    z_step = data.shape[2] // slices
    for i in range(slices):
        start_z = i * z_step
        end_z = (i + 1) * z_step if i < slices - 1 else data.shape[2]
        sliced_data = data[:, :, start_z:end_z]
 
        # 슬라이스된 데이터를 NIfTI 형식으로 저장
        output_file = os.path.join(output_dir, f'{patient}_{i}.nii.gz')
#         print(output_file)
        sliced_img = nib.Nifti1Image(sliced_data, img.affine)
        nib.save(sliced_img, os.path.join(output_file))
print("image done")

# label
ori_directory = '/data/hanyang_Prostate/50_example/trim/data_pre_before/label_nii_class1_trim'
data_list = sorted(os.listdir(ori_directory))

rst_path = '/data/hanyang_Prostate/50_example/trim/data_pre_before_2d/label_nii_class1_trim'
if os.path.isdir(rst_path) == False:
    os.makedirs(rst_path)

for case in data_list:
    patient = case[3:-27]
#     print(patient)
    # 원본 NIfTI 파일 로드
    input_file = os.path.join(ori_directory,case)
    img = nib.load(input_file)
    data = img.get_fdata()
    
    # 슬라이스할 z 축 수 설정
    slices = data.shape[2]  # 원하는 슬라이스 수
#     print('slices',slices)

#     # 저장 디렉토리 설정
#     output_dir = 'sliced_nii_files'
#     os.makedirs(output_dir, exist_ok=True)
    output_dir = os.path.join(rst_path,patient)
    if os.path.isdir(output_dir) == False:
        os.makedirs(output_dir)
        
    # 각 슬라이스 추출 및 저장
    z_step = data.shape[2] // slices
    for i in range(slices):
        start_z = i * z_step
        end_z = (i + 1) * z_step if i < slices - 1 else data.shape[2]
        sliced_data = data[:, :, start_z:end_z]
 
        # 슬라이스된 데이터를 NIfTI 형식으로 저장
        output_file = os.path.join(output_dir, f'{patient}_{i}.nii.gz')
#         print(output_file)
        sliced_img = nib.Nifti1Image(sliced_data, img.affine)
        nib.save(sliced_img, os.path.join(output_file))
print("label done")