# build ABY3
python ./build.py --setup

# fix the bug in the thirdparty library.
FILE_PATH="./thirdparty/libOTe/cryptoTools/cryptoTools/Circuit/BetaLibrary.cpp"
LINE_NUMBER=1203
NEW_TEXT="           G = GateType::na_And;"

sed -i "${LINE_NUMBER}s/.*/${NEW_TEXT}/" "$FILE_PATH"

# then rebuild.
python ./build.py --setup
