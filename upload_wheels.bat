rem if TEST_ALL==1 copy the wheel to smb share

if /I %TEST_ALL% NEQ 1 exit

set commit=%1
set odls=%2

set fname=dist\*

net use \\%SMBNAME%\darts-private-artifacts %SMBPASS% /user:WORKGROUP\%SMBLOGIN%

copy %fname% \\%SMBNAME%\darts-private-artifacts\wheels\%commit%_ODLS%odls%\windows
