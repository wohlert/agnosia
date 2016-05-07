for i in `seq 1 16`;
do
  python3 procedures/generate_images.py train/train_subject$i.mat
done
