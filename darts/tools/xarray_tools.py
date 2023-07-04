import numpy as np
import pandas as pd
import xarray as xr
import re

class xarray_writer():
    def __init__(self, verbose=False):
        self.inited = False
        self.verbose = verbose

    def init_dims_coords(self,
                  nx: int, ny: int, nz: int,  # grid dimensions
                  ts: np.array,  # timesteps array, days
                  X: np.array, Y: np.array, Z: np.array  # cell center coords
                  ):
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.nt = len(ts)  # number of time steps
        self.ts = ts/365.
        self.X = X
        self.Y = Y
        self.Z = Z

    def create_xarray(self,
                      arrays,  #: Dict[np.array],
                      arrays_2d
                      ):
        if self.inited:
            return
        self.inited = True

        # Initialize DataArray to store results
        data_vars = {}
        for name in arrays.keys():
            data_vars[name] = (['time', 'Z', 'Y', 'X'],
                               np.zeros((self.nt, self.nz, self.ny, self.nx)))

        attrs = {'title': 'Simulation Results'}
        self.data = xr.Dataset(data_vars=data_vars,
                          coords={'time': self.ts, 'X': self.X, 'Y': self.Y, 'Z': self.Z},
                          attrs=attrs)
        for name in arrays.keys():
            self.data[name] = xr.DataArray(np.zeros((self.nt, self.nz, self.ny, self.nx)),
                                      dims=['time', 'Z', 'Y', 'X'])


        # Initialize DataArray to store results
        data_vars = {}
        for name in arrays_2d.keys():
            data_vars[name] = (['time', 'Z', 'Y'],
                               np.zeros((self.nt, self.nz-1, self.ny)))

        attrs = {'title': 'Data on Fault'}
        self.data_2d = xr.Dataset(data_vars=data_vars,
                          coords={'time': self.ts, 'Z': self.Z[:-1], 'Y': self.Y},
                          attrs=attrs)
        for name in arrays_2d.keys():
            self.data_2d[name] = xr.DataArray(np.zeros((self.nt, self.nz-1, self.ny)),
                                      dims=['time', 'Z', 'Y'])
        #print(self.data_2d.keys())

    def merge_xarray(self, time_data):
        # parse header of time_data ("origin", "name", "unit")
        re_time_data = re.compile('(?P<origin>\w*?)[\s:]*(?P<name>[\w\s]+) \(?(?P<unit>[\w\/]+)\)?')

        if time_data == {}:
            time = np.array([0.])
        else:
            time = np.array(time_data['time'])/365.
        ds = xr.Dataset()
        for k, v in time_data.items():
            if re_time_data.match(k):
                origin, name, unit = re_time_data.match(k).groups()
                # substitute spaces with underscores in all names
                name = name.replace(' ', '_')
                origin = origin.replace(' ', '_')
                #TODO add comment
                ds = ds.merge({name: xr.DataArray(
                    data=np.array(time_data[k]).reshape(1, -1) if origin else np.array(time_data[k]),
                    coords={'origin': [origin], 'time': time} if origin else {'time': time},
                    dims=('origin', 'time') if origin else ('time'), attrs={'unit': unit})})
        return ds

    def append_xarray(self,
                      time_data,  # engine.time_data
                      arrays,  #: List[np.array],
                      arrays_2d,  #: List[np.array],
                      ):
        if 'time' not in time_data.keys():
            t = 0
        else:
            t = len(time_data['time']) - 1  # number of currently computed timesteps
        for name in arrays.keys():
            arr = arrays[name]
            if arr is None:
                arr = np.zeros((self.nz, self.ny, self.nx))
            self.data[name][t] = xr.DataArray(arr, dims=['Z', 'Y', 'X'],
                                              coords={'X': self.X, 'Y': self.Y, 'Z': self.Z})
            if self.verbose:
                print('array', name, 'time,years', t/365., 'range:', np.array(self.data[name][t]).min(), '-', np.array(self.data[name][t]).max())

        for name in arrays_2d.keys():
            arr = arrays_2d[name]
            if arr is None:
                arr = np.zeros((self.nz-1, self.ny))
            #print(name, arr.shape)
            self.data_2d[name][t] = xr.DataArray(arr, dims=['Z', 'Y'],
                                              coords={'Y': self.Y, 'Z': self.Z[:-1]})
            if self.verbose:
                print('array', name, 'time,years', int(t/365.), 'range:', np.array(self.data_2d[name][t]).min(), '-', np.array(self.data_2d[name][t]).max())
    def write(self, filename, time_data, arrays, arrays_2d, write_x=False):
        self.create_xarray(arrays, arrays_2d)

        if write_x:
            ds = self.merge_xarray(time_data)
            ds.to_netcdf(filename + '_timedata.nc')

        if write_x:
            self.data.to_netcdf(filename + '_cubes.nc')
            self.data_2d.to_netcdf(filename + '_fault.nc')
        else:
            self.append_xarray(time_data, arrays, arrays_2d)
        #print(r.keys())
        #print(r.dims())
