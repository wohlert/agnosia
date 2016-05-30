# This script is intended for the generation of
# image data from EEG/MEG activity for use
# in deep neural network.

#echo "label,name" | tee train_labels.csv test_labels.csv

# Generate training data
for i in `seq 1 16`;
do
    echo "Generating data for train_subject${i}"
    mkdir -p images/train_subject$i
    python3 procedures/generate_images.py train/train_subject$i.mat
done

#cat train_subject*.csv >> train_labels.csv
#rm -f train_subject*.csv

# Generate test data
for i in `seq 17 22`;
do
    echo "Generating data for test_subject${i}"
    mkdir -p images/test_subject$i
    python3 procedures/generate_images.py test/test_subject$i.mat
done

#cat test_subject*.csv >> test_labels.csv
#rm -f test_subject*.csv

# Put data into a tarball for extraction
#tar -cvzf data.tar.gz images train_labels.csv test_labels.csv
