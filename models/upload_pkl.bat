rem %4 -- $CI_COMMIT_REF_NAME
set srv=%1
set lgn=%2
set pwd=%3
set commit=%4
set odls=%5

if /I %UPLOAD_PKL% NEQ 1 exit

set fname=pkl_win.zip
echo %fname%

set pklname="perf_win"
if "%odls%"=="0" (
    set pklname=%pklname%"_iter"
)

rem # delete pkls from previous pipeline run
if exist %fname% del %fname%

"C:\Program files\7-Zip\7z.exe" a -r %fname% %pklname%.pkl"
rem .\*\ref\%pklname%.pkl

rem net use \\%srv%\darts-private-artifacts %pwd% /user:WORKGROUP\%lgn%

rem copy %fname% \\%srv%\darts-private-artifacts\pkl_win\

rem del %fname%

