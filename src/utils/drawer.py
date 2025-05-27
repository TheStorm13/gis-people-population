import os

import matplotlib.pyplot as plt
import numpy as np


class Drawer:

    def plot_prediction(self, y_true, y_pred, title="", color="Blues", figsize=(10, 10),
                        path='../../logs'):
        """Визуализация истинных и предсказанных значений."""

        plt.figure(dpi=300, figsize=figsize)

        # Определяем общие границы для обеих осей
        min_val = max(0, min(np.min(y_true), np.min(y_pred)))
        max_val = max(np.max(y_true), np.max(y_pred))

        hb = plt.hexbin(
            y_true,
            y_pred,
            gridsize=100,
            bins='log',
            cmap=color,
            mincnt=1,
            edgecolors='none'
        )

        plt.xlim(min_val, max_val)
        plt.ylim(min_val, max_val)
        plt.minorticks_on()
        plt.xlabel('True Value', fontsize=12)
        plt.ylabel('Predicted Value', fontsize=12)
        plt.title(title, fontsize=14, pad=20)

        # Диагональная линия идеального совпадения
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1.5, alpha=0.7)

        # Цветовая шкала
        cb = plt.colorbar(hb, pad=0.01)
        cb.set_label('log10(count)', fontsize=10)

        plt.tick_params(labelsize=10)
        cb.ax.tick_params(labelsize=10)

        plt.tight_layout(pad=2.0)

        if not os.path.exists(path):
            os.makedirs(path)

        plt.savefig(f'{path}/prediction_{title}.png')
        plt.show()
