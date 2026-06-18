#!/bin/bash
set -e
root_dir=$(dirname $(dirname $(realpath $0)))
ase_dir=$root_dir/datasets/ase
data_dir=$root_dir/pixloc/pixlib/datasets/ase
while read scene; do
  src_dir=$ase_dir/$scene/rgb_undistorted
  dst_path=$src_dir/line_segments.npz
  checkpoint_path=$root_dir/weights/deeplsd_md.tar
  python -m pixloc.pixlib.extract_line_segments --source $src_dir --destination $dst_path --checkpoint $checkpoint_path
done < <(cat $data_dir/scenes_val.txt $data_dir/scenes_test.txt)
