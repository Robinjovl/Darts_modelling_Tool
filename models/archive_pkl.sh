# create an archive with .pkl files 
odls=$1
gpu=$2

fname="pkl_lin.tar.gz"
echo $fname

pklnamebase="perf_lin"
if [[ $odls == "-a" ]]
then
    pklname=$pklnamebase"_iter"
else
    pklname=$pklnamebase"_odls"
fi

if [[ $gpu == "1" ]]
then
    pklname=$pklnamebase"_gpu"
fi

rm -f $fname # delete pkls from previous pipeline run

if [[ "$UPLOAD_PKL" != 1 ]]; then
	exit
fi

tar -czf $fname ./*/ref/"$pklname"*.pkl
