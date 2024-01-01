import numpy as np
import matplotlib.pyplot as plt
import functions

'''
y_test, y_pred = functions.synth_categorical(
    number_of_classes=4,
    examples=200,
    percentage_of_pure_chance=0.5
)

functions.heatmap_confusion_matrix(
    y_test=y_test,
    y_pred=y_pred
)
'''


'''
polynomial_order = 2

x, y_true, y_noisy = functions.synth_continuous(
    start=-20,
    end=20,
    num_examples=1000,
    noise_factor=0.6,
    coefficient_limit=3,
    polynomial_order=polynomial_order
)

plt.figure(figsize=(8, 6))
plt.plot(x, y_true, label=f'True polynomial', color='blue')
plt.scatter(x, y_noisy, label='Noisy data', color='red', alpha=0.5)
plt.title(f'Noisy Data with {polynomial_order}-order Polynomial')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.show()
'''


'''
x = np.arange(0, 20, 1)
y = np.ones(20)

x_train, x_test, x_val, y_train, y_test, y_val = functions.preprocess_data(
    x=x, y=y,
    test_size=0.20,
    val_size=0.20,
    random_state=42
)

print(len(x_train), len(x_test), len(x_val))
print(x_train[:3])
'''


'''
class history():
    def __init__(self, training_loss, validation_loss, training_accuracy, validation_accuracy):
        self.history = {
            'loss': training_loss,
            'val_loss': validation_loss,
            'accuracy': training_accuracy,
            'val_accuracy': validation_accuracy
        }
 
training_loss = np.random.rand(20) * 0.6
validation_loss = training_loss * np.random.uniform(0.5, 1.5)
training_accuracy = 1 - training_loss
validation_accuracy = 1 - validation_loss

history = history(training_loss, validation_loss, training_accuracy, validation_accuracy)

functions.plot_history(
    history=history,
    title='This is a test',
    save_path=False,
    grid=True
)
'''


'''
y_true, y_pred = functions.synth_categorical(percentage_of_pure_chance=0.5)

functions.evaluate_model(
    y_true=y_true,
    y_pred=y_pred,
    class_labels=['mutant', 'non-mutant']
)
'''
