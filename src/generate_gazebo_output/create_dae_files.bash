#!/bin/bash

for d in $GDL_OBJECT_PATH/* ; do
	cd $d
	meshlabserver -i optimized_tsdf_texture_mapped_mesh.obj -o optimized_tsdf_texture_mapped_mesh.dae -om vc vn fc wt

done