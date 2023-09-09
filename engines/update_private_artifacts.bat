net use \\%1\darts-private-artifacts %3 /user:WORKGROUP\%2
xcopy \\%1\darts-private-artifacts\darts-linear-solvers_vs2022\lib\*.lib   .\lib\darts_linear_solvers\lib\ /y
xcopy \\%1\darts-private-artifacts\darts-linear-solvers_vs2022\include\*.h .\lib\darts_linear_solvers\include\ /y
