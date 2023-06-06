# create folder as they dont exist in repo
mkdir -p lib/darts_linear_solvers/lib
mkdir -p lib/darts_linear_solvers/include
# lcd argument in smbclient consumes only absolute path, so use simple cd 
cd lib/darts_linear_solvers/lib
#copy all linux libs 
smbclient -U $2%$3 //$1/opendarts-private-artifacts/ -c 'prompt OFF;mget *.a' -D darts-linear-solvers/lib
cd ../include
#copy headers
smbclient -U $2%$3 //$1/opendarts-private-artifacts/ -c 'prompt OFF;mget *.h' -D darts-linear-solvers/include
# come back just in case
cd ../../..

