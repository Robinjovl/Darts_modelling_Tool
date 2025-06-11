from main import run_simulation
import gstools as gs
import numpy as np
import os
from multiprocessing import Pool
from functools import partial
import logging
import sys

def generate_random_field(dir, n_realizations, nx=20, len_scale=8, var=1):
    x = y = range(nx)
    for i in range(n_realizations):
        model = gs.Spherical(dim=2, var=1, len_scale=len_scale)
        srf = gs.SRF(model)
        field = srf((x, y), mesh_type='structured')
        fname = os.path.join(dir, f'porosity_{i}.txt')
        np.savetxt(fname, field)
        # srf.plot()

def setup_logger(folder):
    log_file = os.path.join(folder, 'simulation.log')
    logger = logging.getLogger(f'simulation_{os.path.basename(folder)}')
    logger.setLevel(logging.INFO)
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def run_single_simulation(i, prefix, poro_folder, nx, max_ts, n_obl_mult):
    folder = os.path.join(prefix, f'{i}')
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    # Setup logger for this simulation
    logger = setup_logger(folder)
    logger.info(f"Starting simulation {i}")
    
    try:
        poro_filename = os.path.join(poro_folder, f'porosity_{i}.txt')
        logger.info(f"Using porosity file: {poro_filename}")
        
        run_simulation(domain='2D', nx=nx, output=True, max_ts=max_ts,
                            poro_filename=poro_filename,
                            output_folder=folder,
                            n_obl_mult=n_obl_mult,
                            interpolator='multilinear',
                            minerals=['calcite', 'dolomite'],
                            kinetic_mechanisms=['acidic', 'neutral', 'carbonate'],
                            co2_injection=0.1,
                            platform='cpu')
        logger.info(f"Successfully completed simulation {i}")
    except Exception as e:
        logger.error(f"Error in simulation {i}: {str(e)}")
        raise
    finally:
        # Remove handlers to prevent memory leaks
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)

def run_batch_simulation(n_runs, nx, corr_len, max_ts, poro_folder=None, prefix=None, n_batch=4):
    if poro_folder is None:
        var = 1
        poro_folder = f'spherical_{nx}_{corr_len}_{var}'
        
        if prefix is None:
            prefix = 'output_2D'

        if not os.path.exists(prefix):
            os.makedirs(prefix)
        poro_folder = os.path.join(prefix, poro_folder)

        if not os.path.exists(poro_folder):
            os.makedirs(poro_folder)
        
        generate_random_field(dir=poro_folder, n_realizations=n_runs, nx=nx, len_scale=corr_len, var=var)

    n_obl_mult = 3
    
    # Create a partial function with fixed arguments
    run_sim = partial(run_single_simulation, 
                     prefix=prefix,
                     poro_folder=poro_folder,
                     nx=nx,
                     max_ts=max_ts,
                     n_obl_mult=n_obl_mult)
    
    # Run simulations in parallel with specified batch size
    with Pool(processes=n_batch) as pool:
        pool.map(run_sim, range(n_runs))

if __name__ == '__main__':
    n_runs = 100
    nx = 50
    n_batch = 32
    corr_len = 5
    run_batch_simulation(n_runs=n_runs, nx=nx, corr_len=corr_len, max_ts=4.e-4, 
                            prefix=f'batch_2D_{nx}_{n_runs}_1', n_batch=n_batch)