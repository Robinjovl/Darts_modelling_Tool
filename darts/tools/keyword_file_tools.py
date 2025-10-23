import gzip
import os.path as osp
import shutil

import numpy as np

from darts.engines import value_vector


def get_table_keyword(file_name, keyword):
    with open(file_name) as f:
        for line in f:
            if line.strip() == keyword:
                table = []
                while True:
                    row = f.readline()
                    if row.startswith('#') or row.startswith('--'):  # skip comments
                        continue
                    elif row.find('/') != -1:  # end of the table
                        return table
                    else:
                        try:
                            a = np.fromstring(row.strip(), dtype=float, sep=' ')
                        except ValueError:
                            print(
                                "Error processing the file",
                                file_name,
                                "! Can't convert the row to float array:",
                                row,
                            )
                            exit(1)
                        if a.size > 0:
                            table.append(value_vector(a))
                break


def load_single_keyword(file_name, keyword, def_len=1000, cache=0):
    read_data_mode = 0
    pos = 0
    cache_filename = file_name + '.' + keyword + '.cache'

    if cache:
        # if caching is enabled and cache file is already created, read from it
        import os

        if os.path.isfile(cache_filename):
            print(f"Reading {keyword} from {cache_filename}...", end='', flush=True)
            a = np.fromfile(cache_filename)
            print(f" {len(a):d} values have been read.")
            return a

    # start with specified (or default) array length
    a = np.zeros(def_len)
    with open(file_name) as f:
        for line in f:
            s_line = line.strip()

            # requested keyword is not yet detected
            if read_data_mode == 0:
                # to support PETREL files read the first word in the line (there could be comments after the keyword
                # in the same line)
                first_word = s_line.split(maxsplit=1)
                # check if the line is not empty, if so - take the first word
                if first_word:
                    first_word = first_word[0]

                if first_word == keyword:
                    # requested keyword is now detected
                    read_data_mode = 1
                    print(
                        f"Reading {keyword} from {osp.abspath(file_name)}...",
                        end='',
                        flush=True,
                    )
                    continue
                if s_line == 'INCLUDE':
                    path = osp.abspath(osp.dirname(file_name))
                    include = osp.join(path, f.readline().strip(' \\/\n'))
                    a = load_single_keyword(include, keyword, def_len)
                    if a.size > 0:
                        return a
                    else:
                        continue
            # requested keyword is not yet detected or comment found - skip the line
            if (
                not read_data_mode
                or len(s_line) == 0
                or s_line.startswith('#')
                or s_line.startswith('--')
            ):
                continue
            # collect all float values to numpy array
            # check for repeating values
            if s_line.find('*') != -1:
                b = []
                s1 = s_line.split()
                for x in range(s1.__len__()):
                    if s1[x].find('*') != -1:
                        s2 = s1[x].split('*')
                        s2_add = np.ones(int(s2[0]), dtype=float)
                        s2_add.fill(s2[1])
                        b = np.append(b, s2_add)
                    else:
                        try:
                            value = float(s1[x])
                        except ValueError:
                            # in PETREL the trailing slash can be on the same line with numbers
                            # Skip the message if that is the case
                            if s1[x] != '/':
                                print("\n''", s1[x], "'' is not a float, skipping...\n")
                            continue
                        b = np.append(b, value)
            else:
                if s_line.find('/') != -1:  # end of the array
                    break
                try:
                    b = np.fromstring(s_line, dtype=float, sep=' ')
                except ValueError:
                    print(
                        "Error processing the file",
                        file_name,
                        "! Can't convert the row to float array:",
                        s_line,
                    )
                    exit(1)

            # Check if there is still enough place in array
            # if not, enlarge array by a factor of 2
            while pos + b.size > def_len:
                def_len *= 2
                a.resize(def_len, refcheck=False)
            # copy data from b to a
            a[pos : pos + b.size] = b
            pos += b.size

            # break when slash found
            if line.find('/') != -1:
                break
    # shrink the array to actual read length
    a.resize(pos, refcheck=False)

    if cache:
        # if caching is enabled, save to cache file
        a.tofile(cache_filename)
        print(f" {pos:d} values have been read and cached.")
    else:
        print(f" {pos:d} values have been read.")

    return a


def save_few_keywords(fname, keys, data):
    f = open(fname, 'w')
    for id in range(len(keys)):
        f.write(keys[id])
        for i, val in enumerate(data[id]):
            if i % 6 == 0:
                f.write('\n')
            if not isinstance(val, float):
                f.write(str(val))
            else:
                f.write(f"{val:12.10f}")
            f.write('\t')
        f.write('\n' + '/' + '\n')
    f.close()


def compressed_file(fname, verbose=False):
    '''
    Creates a compressed file or uncompresses an archived file
    '''
    fname_gz = fname + '.gz'
    if osp.exists(fname):
        if not osp.exists(fname_gz):
            compress_file(fname, fname_gz, verbose=verbose)
    else:
        if osp.exists(fname_gz):
            decompress_file(fname, fname_gz, verbose=verbose)
        else:
            raise Exception(
                'Cannot find either uncompressed or compressed file: '
                + fname
                + ' or '
                + fname_gz
            )


def compress_file(fname, fname_gz, verbose=False, compresslevel=9):
    if verbose:
        print('Compressing', fname, 'to', fname_gz, '...')
    with open(fname, 'rb') as f_in:
        with gzip.open(fname_gz, 'wb', compresslevel=compresslevel) as f_out:
            shutil.copyfileobj(f_in, f_out)
    if verbose:
        print('Done')


def decompress_file(fname, fname_gz, verbose=False):
    if verbose:
        print('Uncompressing', fname_gz, 'to', fname, '...')
    with gzip.open(fname_gz, 'rb') as f_in:
        with open(fname, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    if verbose:
        print('Done')
