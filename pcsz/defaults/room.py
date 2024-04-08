import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pat

class room:
    def __init__(self):
        self.room_dim = [4.3, 6, 2.7]

        self.speaker_coords = np.array([
            [self.room_dim[0] / 2 + i, 0.2, 1.0] 
            for i in np.linspace(-0.38 / 2, 0.38 / 2, 10)
        ])
        

        zone_sep = 1.0
        self.b_control_coords = np.array([
            [self.room_dim[0] / 2 - zone_sep  + i, 2.0 + j, 1.0] 
            for i in np.linspace(-0.1, 0.1, 3)
            for j in np.linspace(-0.1, 0.1, 3)
        ])

        self.d_control_coords = np.array([
            [self.room_dim[0] / 2 + zone_sep + i, 2.0 + j, 1.0] 
            for i in np.linspace(-0.1, 0.1, 3)
            for j in np.linspace(-0.1, 0.1, 3)
        ])

        self.b_validation_coords = np.array([
            [self.room_dim[0] / 2 - zone_sep  + i, 2.0 + j, 1.0] 
            for i in np.linspace(-0.075, 0.075, 4)
            for j in np.linspace(-0.075, 0.075, 4)
        ])

        self.d_validation_coords = np.array([
            [self.room_dim[0] / 2 + zone_sep + i, 2.0 + j, 1.0] 
            for i in np.linspace(-0.075, 0.075, 4)
            for j in np.linspace(-0.075, 0.075, 4)
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