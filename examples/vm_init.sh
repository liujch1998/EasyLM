tpu_name=$1

tpu tulu-ppo ssh $tpu_name "\
    git clone https://github.com/liujch1998/EasyLM.git n-tulu-ppo-jax; \
    cd n-tulu-ppo-jax; \
    git checkout ppo3; \
    ./scripts/tpu_vm_setup.sh; \
"
