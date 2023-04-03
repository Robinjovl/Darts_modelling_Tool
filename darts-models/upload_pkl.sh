tar -czf "$CI_COMMIT_REF_NAME".tar.gz ./*/*.pkl
smbclient -U $2%$3 //$1/darts-private-artifacts -c "mkdir $CI_COMMIT_REF_NAME"
smbclient -U $2%$3 //$1/darts-private-artifacts/open-darts/pkl -c "put "$CI_COMMIT_REF_NAME".tar.gz -D="$CI_COMMIT_REF_NAME"

