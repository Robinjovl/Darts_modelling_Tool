import h5py


def load_hdf5_to_dict(filename, path='/', decode_strings: list = None):
    """
    Recursively loads HDF5 file contents into a nested dictionary.
    :param hdf5_file: HDF5 file object or filename.
    :param path: Path to start traversing the HDF5 structure from, defaults to root.
    :return: Nested dictionary with the structure and data of the HDF5 file.
    """
    # Open the file if a filename is provided
    if decode_strings is None:
        decode_strings = ['variable_names']
    if isinstance(filename, str):
        with h5py.File(filename, 'r') as f:
            return load_hdf5_to_dict(f)

    result = {}
    for key in filename[path]:
        item = filename[path + key]
        if isinstance(item, h5py.Dataset):
            # Load the entire dataset into the dictionary
            if key in decode_strings:
                result[key] = [x.decode('utf-8') for x in item]
            else:
                result[key] = item[:]
        elif isinstance(item, h5py.Group):
            # Recursively load the group
            result[key] = load_hdf5_to_dict(filename, path + key + '/')
    return result


def add_data_2_h5(m, filename, restart_ts):
    """
    This function allows one, to append to the same reservoir.h5 saved data file
    """

    with h5py.File(filename, 'r+') as file:
        time = file['dynamic/time'][:]
        dataset = file['dynamic/X'][:]

        new_time = time[: restart_ts + 1]
        new_dataset = dataset[: restart_ts + 1, :, :]

        del file['dynamic/time']
        del file['dynamic/X']

        dynamic_group = file['dynamic']

        dynamic_group.create_dataset(
            'time',
            shape=(0,),
            maxshape=(None,),
            dtype=m.output.precision_map[m.output.precision],
        )

        dynamic_group.create_dataset(
            'X',
            shape=(0, m.reservoir.mesh.n_res_blocks, m.physics.n_vars),
            maxshape=(None, m.reservoir.mesh.n_res_blocks, m.physics.n_vars),
            dtype=m.output.precision_map[m.output.precision],
            compression=m.output.compression,
        )

        dynamic_group['time'].resize(new_time.shape)
        dynamic_group['time'][:] = new_time

        dynamic_group['X'].resize(new_dataset.shape)
        dynamic_group['X'][:] = new_dataset

    file.close()

    return 0
