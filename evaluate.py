import matplotlib.pyplot as plt

def plot_history(history):
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("Model losses during training")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend(["train", "test"], loc="upper right")
    plt.show()

    plt.plot(history.history["factorized_top_k/top_100_categorical_accuracy"])
    plt.plot(history.history["val_factorized_top_k/top_100_categorical_accuracy"])
    plt.title("Model accuracies during training")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend(["train", "test"], loc="upper right")
    plt.show()