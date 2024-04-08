import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pat

class room:
    def __init__(self):
        self.room_dim = [8.4, 5.8, 2.7]

        self.speaker_coords = np.array([
            [0.5, 0.5, 0.5], 
            [4.2, 0.5, 0.5], 
            [7.9, 0.5, 0.5], 
            [0.5, 5.3, 0.5], 
            [4.2, 5.3, 0.5], 
            [7.9, 5.3, 0.5], 
            [0.5, 1.5, 0.5], 
            [0.5, 4.3, 0.5], 
        ])
        

        self.b_control_coords = np.array([
            [4.5 + i, 4.3 + j - 0.5, 1.0] 
            for i in np.linspace(-0.3, 0.3, 3)
            for j in np.linspace(-0.3, 0.3, 3)
        ])

        self.d_control_coords = np.array([
            [4.5 + i, 1.5 + j + 0.5, 1.0] 
            for i in np.linspace(-0.3, 0.3, 3)
            for j in np.linspace(-0.3, 0.3, 3)
        ])

        self.b_validation_coords = np.array([
            [4.5 + i, 4.3 + j - 0.5, 1.0] 
            for i in np.linspace(-0.35, 0.35, 4)
            for j in np.linspace(-0.35, 0.35, 4)
        ])

        self.d_validation_coords = np.array([
            [4.5 + i, 1.5 + j + 0.5, 1.0] 
            for i in np.linspace(-0.35, 0.35, 4)
            for j in np.linspace(-0.35, 0.35, 4)
        ])
        
    def draw(self):
        fig, ax = plt.subplots()
        rect = pat.Rectangle((0,0), self.room_dim[0], self.room_dim[1], linewidth=5, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.set_xlim([0, self.room_dim[0]])
        ax.set_ylim([0, self.room_dim[1]])
        ax.set_xlabel("Room Lx [m]")
        ax.set_ylabel("Room Ly [m]")
        ax.scatter(self.b_control_coords[:, 0], self.b_control_coords[:, 1], marker='.', linewidths=1, s=2, label="Zone 1 Control")
        ax.scatter(self.d_control_coords[:, 0], self.d_control_coords[:, 1], marker='.', linewidths=1, s=2, label="Zone 2 Control ")
        ax.scatter(self.b_validation_coords[:, 0], self.b_validation_coords[:, 1], marker='.', linewidths=1, s=2, label="Zone 1 Validation")
        ax.scatter(self.d_validation_coords[:, 0], self.d_validation_coords[:, 1], marker='.', linewidths=1, s=2, label="Zone 2 Validation")
        ax.scatter(self.speaker_coords[:, 0], self.speaker_coords[:, 1], marker='*', linewidths=1, s=2, label="Loudspeakers")
        ax.set_title("Room used in experimental setup")
        ax.set_aspect('equal')
        ax.legend()
        plt.show()
