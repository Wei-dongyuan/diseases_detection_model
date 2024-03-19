class DefaultConfigs(object):
    #1.string parameters
    root = './data_insect/insect/'
    train_data =  root + "/train/"
    test_data = "./aaa/"
    val_data = "no"
    model_name = "resnet50"
    weights = "./checkpoints/"
    best_models = "best_model/"
    epoch_models = weights + 'epoch_model/'
    submit = "./submit/"
    logs = "./logs/"
    gpus = "1"
    train_json = root + "/temp/labels/train_annotations.json"
    valid_json = root + "/temp/labels/valid_annotations.json"

    #2.numeric parameters
    epochs = 40
    batch_size = 1
    img_height = 650
    img_weight = 10
    num_classes = 10
    seed = 888
    lr = 1e-4
    lr_decay = 1e-4
    weight_decay = 1e-4
    fold = 'insect'
config = DefaultConfigs()
