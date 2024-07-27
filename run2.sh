#!bin/bash
CUDA_VISIBLE_DEVICES=4,5,6,7 \
./tools/dist_train.sh \
configs/patchfusion_depthanything/ours_depth_anythingv2_prompt2_vits_inter4k_slice4.py 4 28000 \
--work-dir ./work_dir/depthanything_vitl_ver2_inter4k \
--log-name inter4k_vits_slice22_prompt4_ver2 \
--tag inter4k,vits
