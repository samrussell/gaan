COUNT=0
for i in raw/*.png
do
    if test -f "$i" 
    then
       COUNT=`expr $COUNT + 1`
       echo "converting $i"
       convert $i -resize 256x256! processed/$COUNT.png
    fi
done
