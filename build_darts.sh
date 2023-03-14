ODLS="0" # default linear solvers
if [ $# -gt 0 ] # use the first cmd argument if it is passed
then 
    ODLS=$1
fi

NT="-j 1" # number of threads used to compile
if [ $# -gt 1 ] # use the second cmd argument if it is passed
then 
    NT="-j $2"
fi

echo "ODLS=$ODLS NT=$NT"

which python3-config
export PYTHON_IFLAGS=`python3-config --includes`

# build linear solvers
rm -rf ./darts-engines/lib/darts_linear_solvers
if [ $ODLS == "0" ] 
then # deprecated linear solvers
	cd darts-engines
	./update_private_artifacts.sh $SMBNAME $SMBLOGIN $SMBPASS
	cd ..
else  #open-darts linear solvers
	cd opendarts_linear_solvers
	cd helper_scripts
	./build_linux.sh
	cd ../..
fi

# compile engines
cd darts-engines
make clean
if [ $ODLS == "0" ] #no cmd arguments
then
	make mt $NT USE_OPENDARTS_LINEAR_SOLVERS=false
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

# build darts.whl
cd darts-package
python3 setup.py clean
python3 setup.py build bdist_wheel

cd ..