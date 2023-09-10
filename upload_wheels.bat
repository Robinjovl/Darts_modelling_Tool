rem if TEST_ALL==1 copy the wheel to smb share

if /I %TEST_ALL% NEQ 1 exit

set commit=%1
set odls=%2

set srv=%4
set lgn=%5
set pwd=%6

set fname=dist\*

net use \\%srv%\darts-private-artifacts %pwd% /user:WORKGROUP\%lgn%

copy %fname% \\%srv%\darts-private-artifacts\wheels\%commit%_ODLS%odls%\windows
