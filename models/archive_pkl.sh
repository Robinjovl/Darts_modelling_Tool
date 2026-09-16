# create an archive with .pkl files
# NB: the caller MUST quote both arguments (./archive_pkl.sh "$ODLS" "$TEST_GPU").
# Unquoted, an empty ODLS is dropped by word splitting and TEST_GPU slides into $1,
# so a GPU job silently archives the CPU (_odls) references instead of the _gpu ones.
odls=$1
gpu=$2

fname="pkl_lin.tar.gz"
echo $fname

pklnamebase="perf_lin"
wellpklnamebase="well_time_data_lin"
if [[ $odls == "-a" ]]
then
    pklname=$pklnamebase"_iter"
    wellpklname=$wellpklnamebase"_iter"
else
    pklname=$pklnamebase"_odls"
    wellpklname=$wellpklnamebase"_odls"
fi

if [[ $gpu == "1" ]]
then
    pklname=$pklnamebase"_gpu"
    wellpklname=$wellpklnamebase"_gpu"
fi

# Echo the resolved basenames: if the caller mangles the arguments this is the only
# place the mismatch is observable, and archiving the wrong references is silent.
echo "archive_pkl: odls='$odls' gpu='$gpu' -> $pklname*.pkl, $wellpklname*.pkl"

rm -f $fname # delete pkls from previous pipeline run

if [[ "$UPLOAD_PKL" != 1 ]]; then
	exit
fi

shopt -s nullglob
pkl_files=(
    ./*/ref/"$pklname"*.pkl
    ./*/*/ref/"$pklname"*.pkl
    ./*/ref/"$wellpklname"*.pkl
    ./*/*/ref/"$wellpklname"*.pkl
    # vtk references (THM_vs_geomech_proxy, displaced_fault_reactivation) live in
    # ref/<output folder>/<solution>.vtu and are written by save_vtk_ref during the
    # same UPLOAD_PKL=1 run, so they belong in the same archive. Unlike the pkl names
    # they carry no _odls/_iter/_gpu suffix: one reference serves every lane. Matching
    # on the ref/ path is what keeps the run outputs under results/ out of the archive.
    ./*/ref/*/*.vtu
    ./*/*/ref/*/*.vtu
)

if [[ ${#pkl_files[@]} -gt 0 ]]; then
    tar -czf "$fname" "${pkl_files[@]}"
    echo "archive_pkl: archived ${#pkl_files[@]} file(s) into $fname"
else
    echo "archive_pkl: no reference files matched, $fname not created"
fi
