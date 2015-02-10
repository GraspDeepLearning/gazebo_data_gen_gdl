def get_histogram_for_dof_values(subdirs):
    dof_values = np.zeros((num_images, NUM_DOF))

    current = 0
    for subdir in subdirs:

        in_dataset_fullpath = INPUT_DIRECTORY + '/' + subdir + "/rgbd_and_labels.h5"
        print in_dataset_fullpath
        in_dataset = h5py.File(in_dataset_fullpath)

        if 'dof_values' not in in_dataset.keys():
            continue

        for i in range(in_dataset['dof_values'].shape[0]):
            dof_values[current] = np.copy(in_dataset['dof_values'][i])
            current += 1

    hist_list = []
    bin_edges_list = []

    for i in range(in_dataset['dof_values'].shape[1]):
        hist, bin_edges = np.histogram(in_dataset['dof_values'][:, i], NUM_DOF_BINS, (0, math.pi))
        hist_list.append(hist)
        bin_edges_list.append(bin_edges)

    return hist_list, bin_edges_list


#helper function to determine what bin a data point belongs in.
def get_bin(data_point, bin_edges):
    bin_id = 0
    for bin_edge in bin_edges:

        #this will never pass on first bin_edge
        if data_point < bin_edge:
            break

        bin_id += 1

    bin_id -= 1

    #sanity check
    assert bin_id >= 0

    return bin_id


def get_num_grasp_types():
    wrist_roll_bins = np.arange(-math.pi, math.pi)
    return math.pow(NUM_DOF_BINS, NUM_DOF) * NUM_DEPTH_BINS * len(wrist_roll_bins)


def get_grasp_type(dof_values, dof_bin_edges_list, d, wrist_roll):

    grasp_type = 0

    for i in range(NUM_DOF):

        dof_value = dof_values[i]
        bin_edges = dof_bin_edges_list[i]

        #this should be between 0 and 3 inclusive
        bin_id = get_bin(dof_value, bin_edges)

        grasp_type += bin_id * math.pow(NUM_DOF_BINS, NUM_DOF_BINS-i-1)

    d_bin = 0
    if d > .02:
        d_bin = 1


    wrist_roll_bins = np.arange(-math.pi, math.pi)
    wrist_roll_bin = get_bin(wrist_roll, wrist_roll_bins)


    grasp_type += d_bin * math.pow(NUM_DOF_BINS, NUM_DOF)
    grasp_type += wrist_roll_bin * math.pow(NUM_DOF_BINS, NUM_DOF) * NUM_DEPTH_BINS

    return grasp_type


