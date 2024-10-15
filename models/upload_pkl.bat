rem %4 -- $CI_COMMIT_REF_NAME
set commit=%1
set odls=%2

set fname=pkl_win.zip
echo %fname%

set pklname="perf_win"
if "%odls%"=="-a" (
    set pklname=%pklname%"_iter"
)

rem # delete pkls from previous pipeline run
if exist %fname% del %fname%

if /I %UPLOAD_PKL% NEQ 1 exit

"C:\Program files\7-Zip\7z.exe" a -r %fname% %pklname%*.pkl"


