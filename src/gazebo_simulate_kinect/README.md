Order: 
	- Run build_training_data.py
	# FOR THE NEXT STEPS, MAKE SURE TO CHANGE THE DATASET LOCATION IN EACH .PY #
	- Run repack.py (puts the .h5s in a better format)
	- Run aggregate_output_learn_fingertips.py (seperates all grasps into bins)
	- Run condense_labels.py (gets rid of a lot of the smaller bins, and makes the h5 more managable)