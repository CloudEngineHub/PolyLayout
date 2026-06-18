#!/bin/bash

EXPERIMENT=$1
OUTPUT_DIR=$2

#
# Table 1. Room layout estimation.
#

python -m pixloc.run_PolyLayout --experiment $EXPERIMENT --conf pixloc/pixlib/configs/eval_polylayout_ase.yaml --split test --output $OUTPUT_DIR/ase_test.json
python -m pixloc.run_PolyLayout --experiment $EXPERIMENT --conf pixloc/pixlib/configs/eval_polylayout_scannetpp.yaml --split multi_room --output $OUTPUT_DIR/scannetpp_multi_room.json
python -m pixloc.run_PolyLayout --experiment $EXPERIMENT --conf pixloc/pixlib/configs/eval_polylayout_2d3ds.yaml --split test --flatten --output $OUTPUT_DIR/2d3ds_test.json

#
# Table 2. Ablation experiments.
#

# Parameter sharing
python -m pixloc.run_PolyLayout --experiment $EXPERIMENT --conf pixloc/pixlib/configs/eval_polylayout_ase_single_room.yaml --split test --output $OUTPUT_DIR/ablations/param_no_sharing.json
python -m pixloc.run_PolyLayout --experiment $EXPERIMENT --conf pixloc/pixlib/configs/eval_polylayout_ase.yaml --split test --output $OUTPUT_DIR/ablations/param_shared_orientation.json model.optimizer.shared_floor=false model.optimizer.shared_ceiling=false

# Network
python -m pixloc.run_PolyLayout --experiment pixcuboid_scannetpp --conf pixloc/pixlib/configs/eval_polylayout_ase_resnet.yaml --split test --output $OUTPUT_DIR/ablations/network_resnet_feat.json model.optimizer.edge_cost_scale=0.0 model.optimizer.vp_cost_scale=0.0 model.optimizer.perimeter_cost_scale=0.0
python -m pixloc.run_PolyLayout --experiment $EXPERIMENT --conf pixloc/pixlib/configs/eval_polylayout_ase.yaml --split test --output $OUTPUT_DIR/ablations/network_dino_feat.json model.optimizer.edge_cost_scale=0.0 model.optimizer.vp_cost_scale=0.0 model.optimizer.perimeter_cost_scale=0.0
python -m pixloc.run_PolyLayout --experiment pixcuboid_scannetpp --conf pixloc/pixlib/configs/eval_polylayout_ase_resnet.yaml --split test --output $OUTPUT_DIR/ablations/network_resnet.json

# Cost
python -m pixloc.run_PolyLayout --experiment $EXPERIMENT --conf pixloc/pixlib/configs/eval_polylayout_ase.yaml --split test --output $OUTPUT_DIR/ablations/cost_feat_edge_vp.json model.optimizer.perimeter_cost_scale=0.0

# Initialization
python -m pixloc.run_PolyLayout --experiment $EXPERIMENT --conf pixloc/pixlib/configs/eval_polylayout_ase.yaml --split test --output $OUTPUT_DIR/ablations/init_cuboid.json data.init_layout=cuboid_cameras model.vp_opt_cam_margin=2.5
python -m pixloc.run_PolyLayout --experiment $EXPERIMENT --conf pixloc/pixlib/configs/eval_polylayout_ase.yaml --split test --output $OUTPUT_DIR/ablations/init_circle.json model.vp_opt_num_points_quadrant=4 model.vp_opt_cam_margin=1

# Shape
python -m pixloc.run_PolyLayout --experiment $EXPERIMENT --conf pixloc/pixlib/configs/eval_polylayout_ase.yaml --split test --output $OUTPUT_DIR/ablations/shape_no_split.json model.split_layout_thr=0.0
python -m pixloc.run_PolyLayout --experiment $EXPERIMENT --conf pixloc/pixlib/configs/eval_polylayout_ase.yaml --split test --output $OUTPUT_DIR/ablations/shape_no_simp.json model.optimizer.simplify_layout_every=0
