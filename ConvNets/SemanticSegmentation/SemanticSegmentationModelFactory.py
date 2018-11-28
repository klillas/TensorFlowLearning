import tensorflow as tf

class SemanticSegmentationModelFactory:

    def initialize_model_U_net(self, label_size, tf_ph_x, tf_ph_droput_keep_prob):
        model = tf.nn.dropout(tf_ph_x, keep_prob=tf_ph_droput_keep_prob)

        model = self.add_convolution(model=model, filter_size=4)
        model = self.add_convolution(model=model, filter_multiplier=1)
        output_layer1 = model
        model = self.add_max_pooling(model=model, pool_size=[4, 4], strides=4)
        # model = tf.nn.dropout(model, keep_prob=self.tf_ph_droput_keep_prob)

        model = self.add_convolution(model=model, filter_multiplier=2)
        model = self.add_convolution(model=model, filter_multiplier=1)
        output_layer2 = model
        model = self.add_max_pooling(model=model)
        # model = tf.nn.dropout(model, keep_prob=self.tf_ph_droput_keep_prob)

        model = self.add_convolution(model=model, filter_multiplier=2)
        model = self.add_convolution(model=model, filter_multiplier=1)

        ###########Deconvolution############
        model = self.add_deconvolution(model=model, size_multiplier=2)
        model = self.add_convolution(model=model, filter_multiplier=0.5)
        model = tf.concat([model, output_layer2], axis=3)
        model = self.add_convolution(model=model, filter_multiplier=1)
        model = self.add_convolution(model=model, filter_multiplier=0.5)

        model = self.add_deconvolution(model=model, size_multiplier=4)
        model = self.add_convolution(model=model, filter_multiplier=0.5)
        model = tf.concat([model, output_layer1], axis=3)
        model = self.add_convolution(model=model, filter_multiplier=1)
        model = self.add_convolution(model=model, filter_multiplier=0.5)

        model = self.add_convolution(model=model, filter_size=label_size)

        return tf.reshape(model, (-1, model.shape[1] * model.shape[2], label_size), name="fcn_logits")

    def add_convolution(self, model, filter_multiplier=0, filter_size=0, kernel_size=[3, 3], strides=1):
        model_filter_size = model.shape[3].value
        new_filter_size = filter_size
        if filter_multiplier != 0:
            new_filter_size = int(model_filter_size * filter_multiplier)
        model = tf.layers.conv2d(
            inputs=model,
            filters=new_filter_size,
            kernel_size=kernel_size,
            strides=strides,
            padding="same",
            activation=tf.nn.relu
        )
        return model

    def add_max_pooling(self, model, pool_size=[2, 2], strides=2):
        model = tf.layers.max_pooling2d(
            inputs=model,
            pool_size=pool_size,
            strides=strides
        )
        return model

    def add_deconvolution(self, model, size_multiplier):
        model = tf.layers.conv2d_transpose(
            model,
            filters=model.shape[3],
            strides=size_multiplier,
            kernel_size=(size_multiplier, size_multiplier),
            padding='same')
        return model