# create an archive with .pkl files and upload them to $1/opendarts-private-artifacts/pkl
commit=$4
odls=$5
py=$6

if [ "$UPLOAD_PKL" != 1 ]; then
	exit
fi

fname="$commit"_ODLS"$odls"_PY"$py.tar.gz"
echo $fname

pklname="pkl_lin"
if [ $odls == "0" ]
then
    pklname=$pklname"_iter"
fi

tar -czf $fname ./*/"$pklname".pkl ./*/ref/"$pklname".pkl

smbclient -U $2%$3 //$1/darts-private-artifacts -c "put $fname" -D=pkl

rm $fname
