search_dir="data/ixi/IXI-T1"

n=1
for entry in "$search_dir"/*
do
    med2image -i "$entry" -r -o "Normal_$n.jpg" -s 180
    ((n=n+1))
    med2image -i "$entry" -r -o "Normal_$n.jpg" -s 156
    ((n=n+1))
done