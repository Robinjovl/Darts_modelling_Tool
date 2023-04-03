tar -czf "$CI_COMMIT_REF_NAME".tar.gz ./*/*.pkl
smbclient -U $2%$3 //$1/opendarts-private-artifacts -c "prompt OFF;mkdir pkl"
smbclient -U $2%$3 //$1/opendarts-private-artifacts -c "put $CI_COMMIT_REF_NAME.tar.gz" -D=pkl


