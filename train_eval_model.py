def train(model, mnist_dataset, learning_rate=0.0005, batch_size=16,
          num_steps=5000):
    """Implements the training loop of stochastic gradient descent.

    Performs stochastic gradient descent with the indicated batch_size and
    learning_rate.

    Args:
        model(KMeans or GaussianMixtureModel): Initialized clustering model.
        mnist_dataset: input_data.
        learning_rate(float): Learning rate.
        batch_size(int): batch size used for training.
        num_steps(int): Number of steps to run the update ops.
    """
    for step in range(0, num_steps):
        batch_x, _ = mnist_dataset.train.next_batch(batch_size)
        model.session.run(
            model.update_op_tensor,
            feed_dict={model.x_placeholder: batch_x,
                       model.learning_rate_placeholder: learning_rate}
        )
        # print(loss)



def eval_model(data, model):