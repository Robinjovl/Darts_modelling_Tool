copy CHANGELOG.md darts
python setup.py clean
python setup.py build bdist_wheel

rem get a python version to use in a wheel name 
FOR /F "delims=" %%i IN ('python -c "import sys;print(str(sys.version_info.major) + str(sys.version_info.minor))"') DO (
    SET "pyver=%%i"
)

python -m pip install --upgrade --no-deps --force-reinstall dist/open_darts-1.3.1-cp%pyver%-cp%pyver%-win_amd64.whl
