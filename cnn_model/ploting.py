from matplotlib import pyplot as plt


class Plotting:

    @staticmethod
    def show_plot(fig_size, imgs, range_num, vertical_axis, horizontal_axis):
        plt.figure(figsize=fig_size)
        for i in range(range_num):
            plt.subplot(vertical_axis, horizontal_axis, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(imgs[i])
            plt.xlabel("d")
        plt.show()

    @staticmethod
    def show_accuracy_plot(accuracy, label1, val_accuracy, label2, x_label, y_label):
        plt.plot(accuracy, label=label1)
        plt.plot(val_accuracy, label=label2)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.ylim([0.5, 1])
        plt.legend(loc='lower right')
