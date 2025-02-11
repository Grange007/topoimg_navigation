python3 ./pd_controller.py

echo "First process terminated, starting second process..."

python3 ./similarity_compare.py $1 $2 $3 $4

echo "Second process terminated, starting third process..."

bash ./3_fine_adjustment.sh $1 $2 $3 $4