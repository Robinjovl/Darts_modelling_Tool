rem %4 -- $CI_COMMIT_REF_NAME
set srv=%1
set lgn=%2
set pwd=%3
set commit=%4

echo %commit%

7z a -r %commit%.zip *.pkl
net use \\%srv%\opendarts-private-artifacts %pwd% /user:WORKGROUP\%lgn%
rem if (-not (Test-Path $target_dir)) {mkdir $target_dir}
copy %commit%.zip \\darts-ci.citg.tudelft.nl\opendarts-private-artifacts\pkl_win\

del %commit%.zip