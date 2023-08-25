rem %4 -- $CI_COMMIT_REF_NAME
set srv=%1
set lgn=%2
set pwd=%3
set commit=%4
set odls=%5
set py=%6

if /I %UPLOAD_PKL% NEQ 1 exit

set fname=%commit%_%odls%_%py%.zip
echo %fname%

pklname="pkl_win"
if "%odls"=="0" (
    pklname=$pklname"_iter"
)

"C:\Program files\7-Zip\7z.exe" a -r %fname% %pklname%.pkl  .\*\ref\%pklname%.pkl"

net use \\%srv%\darts-private-artifacts %pwd% /user:WORKGROUP\%lgn%

copy %fname% \\%srv%\darts-private-artifacts\pkl_win\

del %fname%

