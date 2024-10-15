import h5py

def load_hdf5_to_dict(filename, path='/', decode_strings: list = ['variable_names']):
    """
    Recursively loads HDF5 file contents into a nested dictionary.
    :param hdf5_file: HDF5 file object or filename.
    :param path: Path to start traversing the HDF5 structure from, defaults to root.
    :return: Nested dictionary with the structure and data of the HDF5 file.
    """
    # Open the file if a filename is provided
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