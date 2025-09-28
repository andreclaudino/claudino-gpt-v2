import tensorflow as tf


def generate_text(model, tokenizer, start_string, num_generate=5):
    """
    Generates text using the trained model and tokenizer.

    Args:
        model: The trained Keras model.
        tokenizer: The tokenizer used for encoding and decoding.
        start_string: The initial string to start generating text from.
        num_generate: The number of characters to generate.

    Returns:
        The generated text.
    """
    # Convert start string to tokens (integers)
    input_eval = tokenizer.encode(start_string)
    input_eval = tf.expand_dims(input_eval, 0)

    # Empty string to store the generated text
    text_generated = []

    # Low temperature for more predictable text
    # Higher temperature for more creative text
    temperature = 0.2

    # Here batch size == 1
    model.reset_states()

    for i in range(num_generate):
        predictions = model(input_eval)
        # Access the logits from the model's output
        predictions = predictions.logits
        # remove the batch dimension
        predictions = tf.squeeze(predictions, 0)

        # using a categorical distribution to predict the word returned by the model
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()

        # Pass the predicted word as the next input to the model
        # along with the previous hidden state
        input_eval = tf.expand_dims([predicted_id], 0)

        # Decode the predicted token and add it to the generated text
        text_generated.append(tokenizer.decode([predicted_id]))

    return (start_string + "".join(text_generated))