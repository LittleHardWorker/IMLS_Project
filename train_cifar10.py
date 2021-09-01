import tensorflow as tf
from data_reader import DataReader
from loss import nt_xent
from utils import LinearWarmUpCosineDecay, add_to_summary, determine_iterations_per_epoch


@tf.function
def train_step(model, config, image1, image2):
    """Perform one training step using normalized cross entropy loss.

    Args:
        model: a tensorflow keras model
        config: model configurations
        image1: First augmented image batch [B, H, W, CH]
        image2: Second augmented image batch [B, H, W, CH]
    Returns:
        loss: sum of training loss and regularization loss
        gradients: gradients of model weights
    """

    with tf.GradientTape() as tape:
        z1 = model(image1, training=True)
        z2 = model(image2, training=True)
        z1 = tf.math.l2_normalize(z1, axis=1)
        z2 = tf.math.l2_normalize(z2, axis=1)

        loss = nt_xent(z1, z2, config.batch_size, config.temperature, config.zdim)
        reg_loss = tf.add_n(model.losses) if model.losses else 0
        loss = loss + reg_loss

    gradients = tape.gradient(loss, model.trainable_variables)

    return loss, gradients

def pretrain_cifar10(model, config):
    """Pretrains the model based on config settings

    This function first creates an instance of the DataReader class.
    Next a projection head is added to the base model. Weights are restored
    if previously saved and finally the model is trained over the entire 
    dataset for the pre-specified number of epochs.

    Args:
        model: a tensorflow keras model
        config: model configurations
    """
    # print(model.input.shape)
    # print(model.output.shape)

    resnet_output = model.output
    layer1 = tf.keras.layers.GlobalAveragePooling2D(name='GAP')(resnet_output)
    layer2 = tf.keras.layers.Dense(units=config.zdim*2, activation=config.proj_head_act)(layer1)
    model_output = tf.keras.layers.Dense(units=config.zdim)(layer2)
    model = tf.keras.Model(model.input, model_output)
    
    x_train = load_cifar10(config)

    iterations_per_epoch = x_train.shape[0]//config.batch_size
    total_iterations = iterations_per_epoch*config.num_epochs

    learning_rate = LinearWarmUpCosineDecay(total_iterations, config.learning_rate)
    optimizer = tf.keras.optimizers.SGD(learning_rate=config.learning_rate, momentum=0.9)

    checkpoint = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=model)
    manager = tf.train.CheckpointManager(checkpoint, config.pretrain_save_path, max_to_keep=10)

    # restore weights if they exist
    if manager.latest_checkpoint:
        checkpoint.restore(manager.latest_checkpoint)
        print('Restoring weights from {}'.format(manager.latest_checkpoint))
    else:
        print('Training model from scratch')

    summary_writer = tf.summary.create_file_writer(config.pretrain_save_path)

    epoch_loss = [] 
    current_epoch = tf.cast(tf.floor(optimizer.iterations/iterations_per_epoch), tf.int64)
    data = DataReader(config)
    batch = data.read_cifar10(x_train, current_epoch, num_epochs=config.num_epochs)


    # Pretrain Start
    aver_loss_list = []
    epoch_num_list = []
    for (image1, image2), epoch in batch:

        loss, grads = train_step(model, config, image1, image2)
        epoch_loss.append(loss)

        optimizer.__setattr__('lr', learning_rate(optimizer.iterations))
        optimizer.apply_gradients(zip(grads, model.trainable_variables))        

        checkpoint.step.assign_add(1)

        # if checkpoint.step.numpy() % 100 == 0:
        add_to_summary(summary_writer, loss, optimizer.__getattribute__('lr'), image1, image2, checkpoint.step.numpy())
        summary_writer.flush()

        if tf.reduce_all(tf.equal(epoch, current_epoch)):
            current_epoch += 1
            print("Loss after epoch {}: {}".format(current_epoch, sum(epoch_loss)/len(epoch_loss)))
            aver_loss_list.append(sum(epoch_loss)/len(epoch_loss))
            epoch_num_list.append(current_epoch)
            epoch_loss = []

            if current_epoch % 10 == 0:
                save_path = manager.save()
                print("Saved checkpoint for epoch {}: {}".format(current_epoch, save_path))


def load_cifar10(config):
    """Loads training data for cifar10.
    Args: 
        config: model configurations
    Returns:
        x_train: training images
    """

    config.input_size = [32, 32]
    config.crop_size = [32, 32]
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    return x_train
