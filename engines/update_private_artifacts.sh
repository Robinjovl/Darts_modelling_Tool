# create folder as they dont exist in repo
mkdir -p lib/darts_linear_solvers/lib
mkdir -p lib/darts_linear_solvers/include
# lcd argument in smbclient consumes only absolute path, so use simple cd 
cd lib/darts_linear_solvers/lib
#copy all linux libs 
smbclient -U $2%$3 //$1/darts-private-artifacts/ -c 'prompt OFF;mget *.a' -D darts-linear-solvers-linux-hypre/lib
cd ../include
#copy headers
smbclient -U $2%$3 //$1/darts-private-artifacts/ -c 'prompt OFF;mget *.h' -D darts-linear-solvers-linux-hypre/include
# come back just in case
cd ../../..

