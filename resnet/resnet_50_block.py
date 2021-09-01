import tensorflow as tf

class ResNet_50_BottleNeck(tf.keras.layers.Layer):
    def __init__(self, filters, strides):
        super(ResNet_50_BottleNeck, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=filters,
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding="same")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filters=filters,
                                            kernel_size=(3, 3),
                                            strides=strides,
                                            padding="same")
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv3 = tf.keras.layers.Conv2D(filters=4 * filters,
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding="same")
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.shortcut_conv = tf.keras.layers.Conv2D(filters=4 * filters,
                                                    kernel_size=(1, 1),
                                                    strides=strides,
                                                    padding="same")
        self.shortcut_bn = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=None, **kwargs):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv3(x)
        x = self.bn3(x, training=training)
        x = tf.nn.relu(x)

        shortcut = self.shortcut_conv(inputs)
        shortcut = self.shortcut_bn(shortcut, training=training)

        output = tf.nn.relu(tf.keras.layers.add([x, shortcut]))
        return output


def build_ResNet_50_block(filters, strides, repeat_num):
    block = tf.keras.Sequential()
    block.add(ResNet_50_BottleNeck(filters=filters,
                                   strides=strides))
    for _ in range(1, repeat_num):
        block.add(ResNet_50_BottleNeck(filters=filters,
                                       strides=1))

    return block