"""
This script reads each vtk file using ParaView and then saves its cell data to a separate csv file.
Note that the name of the vtk files must start with "solution"
"""

from pathlib import Path

import numpy as np
import pandas as pd
from paraview import servermanager
from paraview.simple import *
from paraview.vtk.numpy_interface import dataset_adapter as dsa

folder = Path(".")  # current directory

for vtk_path in sorted(folder.glob("solution*.vtk")):
    print(f"Processing {vtk_path}")

    data = OpenDataFile(str(vtk_path))

    # Fetch dataset (VTK object) from server to client
    vtkobj = servermanager.Fetch(data)
    wrapped = dsa.WrapDataObject(vtkobj)

    # Store CellID in a dict
    n_cells = wrapped.GetNumberOfCells()
    data_dict = {"CellID": np.arange(n_cells, dtype=np.int64)}

    # Get point data
    points = np.asarray(wrapped.Points)  # shape (n_points, 3)
    data_dict["X"] = points[:, 0][
        :n_cells
    ]  # first column = the X-coordinate of the points

    # Get cell data
    cell_data = wrapped.CellData

    # Store other props in the dict
    for name in cell_data.keys():
        arr = np.asarray(cell_data[name])
        if arr.ndim == 2 and arr.shape[1] > 1:
            for j in range(arr.shape[1]):
                data_dict[f"{name}_{j}"] = arr[:, j]
        else:
            data_dict[name] = arr

    # Save the dict in a csv file
    out_csv = vtk_path.with_suffix(".csv")
    pd.DataFrame(data_dict).to_csv(out_csv, index=False)
    print(f" Saved {out_csv}")
