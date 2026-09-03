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
)

if [[ ${#pkl_files[@]} -gt 0 ]]; then
    tar -czf "$fname" "${pkl_files[@]}"
fi
