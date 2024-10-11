import os
from multiprocessing import Pool


if __name__ == "__main__":
    pool = Pool(processes=1)  # 进程池

    run_file = "train.py"
    train_data_path1 = "./data/MVTec3D-multiview"
    train_data_path2 = "./data/Real3D-multiview"

    dataset1 = "mvtec3d"
    dataset2 = "real3d"
    gpu_id = 1

    sh = f'CUDA_VISIBLE_DEVICES={gpu_id} python {run_file} --save_path ./exps_compare/ --train_data_path {train_data_path1} --dataset {dataset1} \
            --epoch 3'
    print(f'exec {sh}')
    pool.apply_async(os.system, (sh,))

    for epoch_num in [3]:
        checkpoint = './exps_compare/epoch_{}.pth'.format(epoch_num)
        sh = f'CUDA_VISIBLE_DEVICES={gpu_id} python ./test.py --save_path ./result_compare/ \
            --checkpoint_path {checkpoint} --data_path {train_data_path2} --dataset {dataset2}'
        print(f'exec {sh}')
        pool.apply_async(os.system, (sh,))

    pool.close()
    pool.join()








