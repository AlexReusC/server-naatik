import os, shutil

def configure_storage(dataset_path, original_name_dataset):
    try:
        shutil.rmtree(f'./data/{original_name_dataset}')
    except:
        pass
    try:
        shutil.rmtree(f'./data_transformation/joblibs/{original_name_dataset}')
    except:
        pass

    # Creating the main folder in data
    os.makedirs(f'./data/{original_name_dataset}',exist_ok=True)

    # Saving the dataset in its correspondent folder
    with open(dataset_path, 'r') as origin, open(f'./data/{original_name_dataset}/{original_name_dataset}.csv', 'w') as dest:
        results = origin.read()
        dest.write(results)

    # Creating the files for the train and test
    os.makedirs(f'./data/{original_name_dataset}/train',exist_ok=True)
    os.makedirs(f'./data/{original_name_dataset}/test',exist_ok=True)

    # Creating the main folder for the joblibs
    os.makedirs(f'./data_transformation/joblibs/{original_name_dataset}',exist_ok=True)
    os.makedirs(f'./data_transformation/joblibs/{original_name_dataset}/model',exist_ok=True)
    os.makedirs(f'./data_transformation/joblibs/{original_name_dataset}/etl',exist_ok=True)
    os.makedirs(f'./data_transformation/joblibs/{original_name_dataset}/model/mlp',exist_ok=True)
    os.makedirs(f'./data_transformation/joblibs/{original_name_dataset}/model/decission_tree',exist_ok=True)
    os.makedirs(f'./data_transformation/joblibs/{original_name_dataset}/model/benchmark',exist_ok=True)
    os.makedirs(f'./data_transformation/joblibs/{original_name_dataset}/model/random_forest',exist_ok=True)
    os.makedirs(f'./data_transformation/joblibs/{original_name_dataset}/model/cnn',exist_ok=True)


