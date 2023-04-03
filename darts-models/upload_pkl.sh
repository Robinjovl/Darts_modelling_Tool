commit=$4
odls=%5
fname=$(commit)_$(odls).tar.gz
tar -czf $fname ./*/*.pkl
smbclient -U $2%$3 //$1/opendarts-private-artifacts -c "prompt OFF;mkdir pkl"
smbclient -U $2%$3 //$1/opendarts-private-artifacts -c "put $fname" -D=pkl
rm $fname
