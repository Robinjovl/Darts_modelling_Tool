# Exit when any command fails
set -e

ODLS="0" # default linear solvers

# update submodules
echo -e "\n- Update submodules: START\n"
rm -rf thirdparty/eigen
git submodule sync --recursive
git submodule update --recursive --remote --init

if [ $# -gt 0 ] # use the first cmd argument if it is passed
then
    ODLS=$1
fi

NT="-j 8" # number of threads used to compile
if [ $# -gt 1 ] # use second cmd argument if it is passed
then
    NT="-j $2"
fi

conf="mt" # "mt" for multithread(openMP), "gpu" - for openMP+CUDA
if [ $# -gt 2 ] # use third cmd argument if it is passed
then
    conf="$3"
fi

echo "ODLS=$ODLS NT=$NT conf=$conf"

which python3-config
export PYTHON_IFLAGS=`python3-config --includes`

# build solvers
if [ $ODLS == "0" ]
then # deprecated linear solvers
	cd engines
	./update_private_artifacts.sh $SMBNAME $SMBLOGIN $SMBPASS
	cd ..
else  #open-darts solvers
	cd solvers/helper_scripts
	./build_linux.sh
	cd ../..
fi

# compile engines
cd engines
make clean
if [ $ODLS == "0" ] #no cmd arguments
then
	make $conf $NT USE_OPENDARTS_LINEAR_SOLVERS=false
else
	make $NT USE_OPENDARTS_LINEAR_SOLVERS=true #open-darts currently works only without openMP
fi

if [ $? == 0 ]
then
    echo "make successfull"
else
    echo "make failed"
    exit 1
fi

cd ..

# compile discretizer
cd discretizer
make clean
if [ $ODLS == "0" ] #no cmd arguments
then
	make $NT USE_OPENDARTS_LINEAR_SOLVERS=false
else
	make $NT USE_OPENDARTS_LINEAR_SOLVERS=true
fi

if [ $? == 0 ]
then
    echo "make successfull"
else
    echo "make failed"
    exit 1
fi

cd ..

# build darts.whl

# generating build info of darts-package
python darts/print_build_info.py

python3 setup.py clean
python3 setup.py build bdist_wheel

# installing
python3 -m pip install .
