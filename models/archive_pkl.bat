rem %4 -- $CI_COMMIT_REF_NAME
set odls=%1
set gpu=%2

set fname=pkl_win.zip
echo %fname%

set pklnamebase="perf_win"
if "%odls%"=="-a" (
    set pklname=%pklnamebase%"_iter"
) else (
    set pklname=%pklnamebase%"_odls"
)

if "%gpu%"=="1" (
    set pklname=%pklnamebase%"_gpu"
)

rem # delete pkls from previous pipeline run
if exist %fname% del %fname%

if /I %UPLOAD_PKL% NEQ 1 exit

"C:\Program files\7-Zip\7z.exe" a -r %fname% %pklname%*.pkl"


