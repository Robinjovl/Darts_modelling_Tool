# create an archive with .pkl files and upload them to $1/opendarts-private-artifacts/pkl
commit=$4
odls=$5
fname="$commit"_"$odls.tar.gz"
echo $fname
tar -czf $fname ./*/*.pkl
#smbclient -U $2%$3 //$1/opendarts-private-artifacts -c "prompt OFF;mkdir pkl"
smbclient -U $2%$3 //$1/opendarts-private-artifacts -c "put $fname" -D=pkl
rm $fname
