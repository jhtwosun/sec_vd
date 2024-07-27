#!bin/bash

./tools/dist_train.sh \
configs/patchfusion_depthanything/ours_depth_anythingv2_vits_inter4k_slice2.py 4 28001 \
--work-dir ./work_dir/depthanything_vitl_inter4k \
--log-name inter4k_vits_slice22_prompt4 \
--tag inter4k,vits
