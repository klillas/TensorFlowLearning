import tensorflow as tf

class SemanticSegmentationModelFactory:

    def initialize_cost(self, load_existing_model, model, labels_one_hot, graph):
        tensor_cost = None
        if load_existing_model == False:
            labels_one_hot_float = tf.to_float(labels_one_hot)
            #tensor_cost = tf.reduce_mean(tf.multiply(-1.0, tf.add(tf.multiply(labels_one_hot_float, tf.clip_by_value(model, 1e-10, 1.0)), tf.multiply(tf.subtract(1.0, labels_one_hot_float), tf.log(tf.subtract(1.0, tf.clip_by_value(model, 1e-10, 1.0)))))))
            #tensor_cost = tf.reduce_mean(tf.multiply(labels_one_hot_float, tf.log(model)))
            #tensor_cost = tf.log(model)
            #tensor_cost = tf.reduce_mean(tf.square(tf.subtract(tf.to_int32(model), labels_one_hot)))

            #cross_entropy = -tf.reduce_sum(labels_one_hot_float * tf.log(tf.clip_by_value(model, 1e-10, 1.0)))
            #tensor_cost = cross_entropy

            cost = tf.multiply(labels_one_hot_float, tf.log(tf.clip_by_value(model, 0.000001, 1.0)))
            cost = tf.add(cost, tf.multiply(tf.subtract(1.0, labels_one_hot_float), tf.log(tf.subtract(1.0, tf.clip_by_value(model, 0.0, 0.999999)))))
            cost_per_logit = tf.multiply(-1.0, cost)
            cost_per_pixel = tf.reduce_sum(cost_per_logit, -1)
            pixel_has_label = tf.reduce_mean(labels_one_hot_float, axis=-1)
            # Remove any cost if no label is set for this pixel (will lead to random label approximation for this pixel and hopefully reduce prediction complexity)
            cost_per_pixel = tf.multiply(cost_per_pixel, pixel_has_label)
            tensor_cost = tf.reduce_mean(cost_per_pixel)

            #tensor_cost = cost

            # Calculate distance from actual labels using cross entropy
            #cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=labels_one_hot, name="cross_entropy")
            # Take mean for total loss
            #tensor_cost = tf.reduce_mean(cross_entropy, name="fcn_loss")

        if load_existing_model == True:
            tensor_cost = graph.get_tensor_by_name("fcn_loss:0")

        return tensor_cost

    def initialize_model_U_net(self, label_size, tf_ph_x, tf_ph_droput_keep_prob):
        model = tf.nn.dropout(tf_ph_x, keep_prob=tf_ph_droput_keep_prob)

        model = self.add_convolution(model=model, filter_size=64)
        model = self.add_convolution(model=model, filter_multiplier=1)
        output_layer1 = model
        model = self.add_max_pooling(model=model, pool_size=[4, 4], strides=4)
        model = tf.nn.dropout(model, keep_prob=tf_ph_droput_keep_prob)

        model = self.add_convolution(model=model, filter_multiplier=2)
        model = self.add_convolution(model=model, filter_multiplier=1)
        output_layer2 = model
        model = self.add_max_pooling(model=model, pool_size=[4, 4], strides=4)
        model = tf.nn.dropout(model, keep_prob=tf_ph_droput_keep_prob)

        model = self.add_convolution(model=model, filter_multiplier=2)
        model = self.add_convolution(model=model, filter_multiplier=1)

        ###########Deconvolution############
        model = self.add_deconvolution(model=model, size_multiplier=4)
        model = self.add_convolution(model=model, filter_multiplier=0.5)
        model = tf.concat([model, output_layer2], axis=3)
        model = self.add_convolution(model=model, filter_multiplier=1)
        model = self.add_convolution(model=model, filter_multiplier=0.5)

        model = self.add_deconvolution(model=model, size_multiplier=4)
        model = self.add_convolution(model=model, filter_multiplier=0.5)
        model = tf.concat([model, output_layer1], axis=3)
        model = self.add_convolution(model=model, filter_multiplier=1)
        model = self.add_convolution(model=model, filter_multiplier=1)

        model = self.add_convolution(model=model, filter_multiplier=0.25)

        model = self.add_convolution(model=model, filter_multiplier=0.25)

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