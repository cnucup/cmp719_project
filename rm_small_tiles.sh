for i in *.png
do
    size=($(identify -format "%w %h" "$i"))
    (( size[0] < "512" || size[1] < "512" )) && mv "$i" "/destination/folder/path"
done
