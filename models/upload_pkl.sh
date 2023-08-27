# create an archive with .pkl files 
commit=$1
odls=$2

if [ "$UPLOAD_PKL" != 1 ]; then
	exit
fi

fname="pkl_lin.tar.gz"
echo $fname

pklname="perf_lin"
if [ $odls == "0" ]
then
    pklname=$pklname"_iter"
fi

rm -f $fname # delete pkls from previous pipeline run
tar -czf $fname ./*/"$pklname".pkl #./*/ref/"$pklname".pkl
