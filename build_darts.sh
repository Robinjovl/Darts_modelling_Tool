# Exit when any command fails
set -e

ODLS="0" # default linear solvers are iterative
config=""  # if config=="debug" build debug

# update submodules
echo -e "\n- Update submodules: START\n"
rm -rf thirdparty/eigen
git submodule sync --recursive
git submodule update --recursive --remote --init

if [ $# -gt 0 ] # use the first cmd argument if it is passed
then
    ODLS=$1
fi

# default configuration
if [ $ODLS == "0" ]
then
    config_engines="mt"
elif [ $ODLS == "1" ]
then
    config_engines="" #open-darts (with ODLS=1) currently works only without openMP
else
    echo "Error: Unknown ODLS=$ODLS in the first argument!"
    exit 1
fi

config_discretizer="release"
config_solvers="Release"

if [ $# -gt 1 ] # use the first cmd argument if it is passed
then
    config=$2
    if [ $config == "debug" ]
    then
        config_engines="mt_debug"
        config_discretizer="debug"
        config_solvers="Debug"
    elif [ $config == "release" ]
    then
        : # do nothing
    else
        echo "Error: Unknown configuration=$config in the second argument!"
        exit 1
    fi
fi

NT="-j 8" # number of threads used to compile
if [ $# -gt 2 ] # use the second cmd argument if it is passed
then
    NT="-j $2"
fi

echo "ODLS=$ODLS config=$config NT=$NT"

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
	./build_linux.sh $(config_solvers)
	cd ../..
fi

# compile engines
cd engines
make clean
if [ $ODLS == "0" ]
then
	make $(config_engines) $NT USE_OPENDARTS_LINEAR_SOLVERS=false
else
	make $(config_engines) $NT USE_OPENDARTS_LINEAR_SOLVERS=true 
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
	make $(config_discretizer) $NT USE_OPENDARTS_LINEAR_SOLVERS=false
else
	make $(config_discretizer) $NT USE_OPENDARTS_LINEAR_SOLVERS=true
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
