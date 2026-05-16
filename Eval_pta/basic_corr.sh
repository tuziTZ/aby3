
python build.py

rm ./debug.txt

./bin/frontend -prog -1 -role 0 -Test true &
./bin/frontend -prog -1 -role 1 -Test true &
./bin/frontend -prog -1 -role 2 -Test true &
wait;