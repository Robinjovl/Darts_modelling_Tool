rem %4 -- $CI_COMMIT_REF_NAME
set srv=%1
set lgn=%2
set pwd=%3
set commit=%4
set odls=%5

set fname=%commit%_%odls%.zip
echo %fname%

"C:\Program files\7-Zip\7z.exe" a -r %fname% *.pkl
net use \\%srv%\opendarts-private-artifacts %pwd% /user:WORKGROUP\%lgn%
rem if (-not (Test-Path $target_dir)) {mkdir $target_dir}
copy %fname% \\darts-ci.citg.tudelft.nl\opendarts-private-artifacts\pkl_win\

del %fname%