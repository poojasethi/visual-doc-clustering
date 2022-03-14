DATASET=${1?Error: no dataset specified}
MODEL=${2?Error: no model specified}
OUTPUT=${3?Error: no output filename specified}

echo "----------------------Output Results-----------------------" > ./results/ktest/$OUTPUT

for i in {2..100}
do
  echo "Running cluster for ${i} clusters"

  echo "Running cluster for ${i} clusters" >> ./results/ktest/$OUTPUT

  python clustering.py -p datasets/$DATASET/ -r $MODEL -e 768 -s average_all_words_mask_pads -o results/$DATASET/$MODEL/average_all_words_mask_pads/ktest -m models/fine_tuned_related -k ${i} >> ./results/ktest/$OUTPUT

done  

exit
