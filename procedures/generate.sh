for i in `seq 1 16`;
do
  mkdir -p images/train_subject$i
  python3 procedures/generate_images.py train/train_subject$i.mat
done

cat *.csv > labels.csv
