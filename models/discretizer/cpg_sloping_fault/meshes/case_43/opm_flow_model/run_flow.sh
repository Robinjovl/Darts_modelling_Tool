# remove previous results
rm *.csv *opm.grdecl

# run simulation in OPM
flow model.data

if [ $? == 0 ]
then
  # read binaries and convert to grdecl and csv
  python3 ecl2csv.py
fi
