#!bin/bash

./tools/infer.sh \
configs/patchfusion_depthanything/ours_depth_anythingv2_vitl_mvs_slice1_infer_wotrain.py 1 28001 \
--work-dir ./work_dir/depthanything_vitl_u4k \
--log-name mvs_vits_slice22_prompt4_infer \
--tag mvs,vits
