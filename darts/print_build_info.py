import datetime
import getpass
import socket
import subprocess
import os

def print_build_info():
    here = os.path.abspath(os.path.dirname(__file__))
    version_info_file = os.path.join(here, 'build_info.txt')
    if os.path.exists(version_info_file):
        with open(version_info_file, 'r') as fp:
            date_time = fp.readline().rstrip()
            user_host = fp.readline().rstrip()
            git_hash = fp.readline().rstrip()
            print ('darts-package built on %s by %s from %s' % (date_time, user_host, git_hash))
    else:
        import subprocess
        try:
            git_hash = subprocess.run(['git', 'describe', '--always', '--dirty'], stdout=subprocess.PIPE, cwd=here)
        except FileNotFoundError:
            print('darts-package is run locally from %s [no git hash info available]', here)
            return
        print('darts-package is imported locally from %s [%s]' % (here, git_hash.stdout.decode('utf-8').rstrip()))


if __name__ == '__main__':
  """
  When this script is excecuted, it will generate 'build_info.txt'
  """

  print("Creating version info file...")
  here = os.path.abspath(os.path.dirname(__file__))
  version_info_file = os.path.join(here, 'build_info.txt')
  with open(version_info_file, 'w') as fp:
    build_date = datetime.datetime.now()
    fp.write(build_date.strftime("%d/%m/%Y %H:%M:%S\n"))

    username = getpass.getuser()
    hostname = socket.gethostname()
    fp.write("%s@%s\n" % (username, hostname))

    git_hash = subprocess.run(['git', 'describe', '--always', '--dirty'], stdout=subprocess.PIPE)
    fp.write(git_hash.stdout.decode('utf-8'))

  print ("Embedded build info:")
  with open(version_info_file, 'r') as f:
    print(f.read())
