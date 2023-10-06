# if TEST_ALL==1 copy the wheel to smb share
commit=$1
odls=$2

if [ "$TEST_ALL" != 1 ]; then
	exit
fi

fname='dist/*'

smbclient -U $SMBLOGIN%$SMBPASS //$SMBNAME/darts-private-artifacts -c "put $fname" -D=wheels/"$commit"_"ODLS$odls"/linux 
