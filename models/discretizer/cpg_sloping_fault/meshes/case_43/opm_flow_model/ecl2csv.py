from ecl.summary import EclSum
from ecl.eclfile import EclFile
from cpg_tools import save_array
import numpy as np
import sys

def ecl2csv(fname):
    summary = EclSum(fname) 

    days = summary.numpy_vector('TIME')
    fopr = summary.numpy_vector('FOPR')
    fwpr = summary.numpy_vector('FWPR')
    fwir = summary.numpy_vector('FWIR')

    with open('opm_rates.csv', 'w') as f:
        f.writelines('days\tOPM_FOPR\tOPM_FWPR\tOPM_FWIR\n')

        for i in range(days.shape[0]):
            f.writelines(str(days[i]) + '\t' +str(fopr[i]) + '\t' +str(fwpr[i]) + '\t' +str(fwir[i]) + '\n')

def ecl2grdecl(fname, keyword):
    n_time_steps = 10

    # read ACTNUM
    f_egrid = EclFile(fname + '.EGRID')
    actnum = np.array(f_egrid['ACTNUM'])[0]

    # read and export trans
    out_fname = 'tran_opm.grdecl'
    f_init = EclFile(fname + '.INIT')
    save_array(actnum, out_fname, 'ACTNUM')
    tran = f_init['TRANX']
    save_array(np.array(tran)[0], out_fname, 'TRANX', mode='a')
    tran = f_init['TRANY']
    save_array(np.array(tran)[0], out_fname, 'TRANY', mode='a')
    tran = f_init['TRANZ']
    save_array(np.array(tran)[0], out_fname, 'TRANZ', mode='a')
    try:
        tran = f_init['TRANNNC']
        print('number of NNCs', np.array(tran)[0].size)
    except:
        pass

    # NNC
    #a = f_egrid['NNCHEAD'] #?
    if 'NNC1' in f_egrid.keys():
        nnc1 = np.array(f_egrid['NNC1'])[0]
        nnc2 = np.array(f_egrid['NNC2'])[0]
        nnc_tran = np.array(f_init['TRANNNC'])[0]
        n_nnc = nnc_tran.size
        with open('NNC_opm.txt', 'w') as f:
            for i in range(n_nnc):
                f.write(str(nnc1[i])+'\t'+str(nnc2[i])+'\t'+str(nnc_tran[i])+'\n')
                
    # read and export pressure
    f_unrst = EclFile(fname + '.UNRST')

    for step in range(n_time_steps + 1):
        p = f_unrst[keyword][step] 
        out_fname = keyword + '_' + str(step) + '_opm.grdecl'
        save_array(actnum, out_fname, 'ACTNUM')
        save_array(np.array(p), out_fname, keyword, mode='a')


if __name__  == '__main__':
    ecl2csv('MODEL.UNSMRY')
    ecl2grdecl('MODEL', 'PRESSURE')
