from os import stat, mkdir, chdir

def cd_mkdir (work_dir):
    # select working directory
    try:
        stat(work_dir)
    except:
        mkdir(work_dir)

    chdir(work_dir)