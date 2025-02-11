traj_name=$(yq e '.trajs[1]' ../config/traj_name.yaml)

python3 ./similarity_compare.py --model vint --dir $traj_name

echo "Find next pose terminated, starting vint navigation..."

bash ./5_vint_navigate.sh --model vint --dir $traj_name
