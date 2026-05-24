# create an archive with .pkl files
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
