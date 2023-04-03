commit=$4
tar -czf $commit.tar.gz ./*/*.pkl
smbclient -U $2%$3 //$1/opendarts-private-artifacts -c "prompt OFF;mkdir pkl"
smbclient -U $2%$3 //$1/opendarts-private-artifacts -c "put $commit.tar.gz" -D=pkl
rm $commit.tar.gz
