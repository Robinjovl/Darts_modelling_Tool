rem %4 -- $CI_COMMIT_REF_NAME
set odls=%1
set gpu=%2

set fname=pkl_win.zip
echo %fname%

set pklnamebase=perf_win
set wellpklnamebase=well_time_data_win
if "%odls%"=="-a" (
    set pklname=%pklnamebase%_iter
    set wellpklname=%wellpklnamebase%_iter
) else (
    set pklname=%pklnamebase%_odls
    set wellpklname=%wellpklnamebase%_odls
)

if "%gpu%"=="1" (
    set pklname=%pklnamebase%_gpu
    set wellpklname=%wellpklnamebase%_gpu
)

rem # delete pkls from previous pipeline run
if exist %fname% del %fname%

if /I %UPLOAD_PKL% NEQ 1 exit

"C:\Program files\7-Zip\7z.exe" a -r %fname% %pklname%*.pkl %wellpklname%*.pkl

rem vtk references (THM_vs_geomech_proxy, displaced_fault_reactivation) are written
rem by save_vtk_ref during the same UPLOAD_PKL=1 run, so they belong in this archive.
rem Unlike the pkl names they carry no _odls/_iter/_gpu suffix: one reference serves
rem every lane. The masks are anchored on ref\ and used without -r, so the run
rem outputs under results\ are not swept in. 7z only warns when a mask matches
rem nothing, so a lane with no vtk references still succeeds.
"C:\Program files\7-Zip\7z.exe" a %fname% "*\ref\*\*.vtu" "*\*\ref\*\*.vtu"
