import matplotlib.pyplot as plt

def plotGraphs(model, solver):
    solver.train()
    plt.subplot(2,1,1)
    plt.title("Training loss")
    plt.plot(solver.loss_history, "o")
    plt.xlabel('Iteration')

    plt.subplot(2,1,2)
    plt.title('Accuracy')
    plt.plot(solver.train_acc_history, '-o', label='train')
    plt.plot(solver.val_acc_history, '-o', label='val')
    plt.plot([0.5] * len(solver.val_acc_history), 'k--')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')
    plt.gcf().set_size_inches(15,12)
    plt.show()
